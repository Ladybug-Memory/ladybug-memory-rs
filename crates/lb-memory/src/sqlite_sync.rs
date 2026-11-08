//! SQLite intake + staging + replication into the Ladybug graph.
//!
//! ```text
//! remember(text):
//!   1. append  — raw text lands in the SQLite `memory_intake` log
//!   2. extract — GLiNER2.5 mentions (same extractor as the direct path)
//!   3. stage   — distilled fact staged into `node_*` / `rel_*` tables
//!   4. sync    — staged rows replicated into the native Ladybug graph
//! ```
//!
//! SQLite is a first-class citizen here: every `remember` through this
//! module is first a SQLite row. The graph is a replica, converged by a sync
//! tick — either [`replicate_to_store`] (embedded applier, default) or the
//! `sqlite` extension's own `CALL sqlite_sync` when its `.lbug_extension`
//! is loaded (see `Store::sqlite_sync_tick`).
//!
//! The staging schema follows the mapping conventions of the `sqlite`
//! extension (`~/src/ladybug/extension/sqlite`, see its `docs/cdc_design.md`):
//! `node_*` tables carry a single-column primary key (identity), `rel_*`
//! tables carry `src_*` / `dst_*` foreign keys into mapped node tables.
//! Explicit registration remaps the staging labels onto the real graph:
//!
//! ```sql
//! LOAD EXTENSION 'libsqlite.lbug_extension';
//! CALL sqlite_register_node('<abs path>', 'node_Memory', 'Memory', 'id');
//! CALL sqlite_register_node('<abs path>', 'node_Entity', 'Entity', 'id');
//! CALL sqlite_register_edge('<abs path>', 'rel_MentionedIn', 'MentionedIn', 'src_id', 'dst_id');
//! CALL sqlite_register_edge('<abs path>', 'rel_RelatesTo', 'RelatesTo', 'src_id', 'dst_id');
//! CALL sqlite_register_edge('<abs path>', 'rel_ConflictsWith', 'ConflictsWith', 'src_id', 'dst_id');
//! CALL sqlite_sync('<abs path>');
//! ```
//!
//! Two constraints come straight from the extension's apply path
//! (`extension/sqlite/src/cdc/sync_cypher.cpp`), so the mapped staging
//! tables mirror the native tables **exactly** — no extra columns:
//!
//! * Node upserts are `MERGE … SET n.col = row.col` over *every* staging
//!   column. A bookkeeping column would fail to bind on the native table.
//!   Intake linkage therefore lives in unprefixed side tables, which the
//!   extension's convention inference skips.
//! * Edge upserts carry a `_sqlite_rowid` slot that pre-existing native rel
//!   tables don't have (see the `EXTENSION CHANGE PROPOSALS` section in the
//!   README). The embedded applier has no such requirement and is the
//!   default.
//!
//! Both appliers are idempotent: a durable cursor advances per intake id,
//! staged rows carry their native ids back (`lb_id`), and replays converge.

use anyhow::{Context, Result};
use rusqlite::{params, Connection as SqliteConn};

use crate::store::Store;

/// DDL for the intake log, the extension-mapped staging tables, and the
/// replication bookkeeping. Mapped tables mirror the native `Memory` /
/// `Entity` / `MentionedIn` / `RelatesTo` / `ConflictsWith` schema exactly
/// (SQLite-compatible types); everything else is unprefixed and invisible
/// to the extension's convention inference.
const INTAKE_SCHEMA: &str = "
-- Raw intake log: one row per remember call, in arrival order.
CREATE TABLE IF NOT EXISTS memory_intake(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  content TEXT NOT NULL,
  user_id TEXT NOT NULL DEFAULT '',
  project TEXT NOT NULL DEFAULT '',
  session_id TEXT NOT NULL DEFAULT '',
  importance INTEGER NOT NULL DEFAULT 5,
  created_at TEXT NOT NULL DEFAULT ''
);
-- Staged graph, 1:1 with the native schema (see module docs).
CREATE TABLE IF NOT EXISTS node_Memory(
  id INTEGER PRIMARY KEY,
  content TEXT NOT NULL,
  category TEXT NOT NULL DEFAULT 'general',
  user_id TEXT NOT NULL DEFAULT '',
  project TEXT NOT NULL DEFAULT '',
  session_id TEXT NOT NULL DEFAULT '',
  importance INTEGER NOT NULL DEFAULT 5,
  status TEXT NOT NULL DEFAULT 'active'
);
CREATE TABLE IF NOT EXISTS node_Entity(
  id INTEGER PRIMARY KEY,
  canonical_name TEXT NOT NULL UNIQUE,
  entity_type TEXT NOT NULL DEFAULT 'concept',
  aliases TEXT NOT NULL DEFAULT '[]'
);
CREATE TABLE IF NOT EXISTS rel_MentionedIn(
  src_id INTEGER NOT NULL REFERENCES node_Entity(id),
  dst_id INTEGER NOT NULL REFERENCES node_Memory(id),
  mention_text TEXT NOT NULL DEFAULT '',
  confidence REAL NOT NULL DEFAULT 0.5,
  span_start INTEGER NOT NULL DEFAULT 0,
  span_end INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS rel_RelatesTo(
  src_id INTEGER NOT NULL REFERENCES node_Entity(id),
  dst_id INTEGER NOT NULL REFERENCES node_Entity(id),
  relation_type TEXT NOT NULL DEFAULT 'coOccurs',
  confidence REAL NOT NULL DEFAULT 0.5
);
CREATE TABLE IF NOT EXISTS rel_ConflictsWith(
  src_id INTEGER NOT NULL REFERENCES node_Memory(id),
  dst_id INTEGER NOT NULL REFERENCES node_Memory(id),
  cues TEXT NOT NULL DEFAULT '[]'
);
-- Intake linkage + native-id backfill (unprefixed: extension skips these).
CREATE TABLE IF NOT EXISTS stage_memory_map(
  staged_id INTEGER PRIMARY KEY,
  intake_id INTEGER NOT NULL,
  fact_idx INTEGER NOT NULL DEFAULT 0,
  lb_id INTEGER
);
CREATE TABLE IF NOT EXISTS stage_entity_map(
  staged_id INTEGER PRIMARY KEY,
  intake_id INTEGER NOT NULL,
  lb_id INTEGER
);
CREATE TABLE IF NOT EXISTS stage_edge_map(
  rel_table TEXT NOT NULL,
  rowid INTEGER NOT NULL,
  intake_id INTEGER NOT NULL,
  PRIMARY KEY (rel_table, rowid)
);
CREATE TABLE IF NOT EXISTS memory_provenance(
  intake_id INTEGER NOT NULL,
  fact_idx INTEGER NOT NULL DEFAULT 0,
  provenance TEXT NOT NULL DEFAULT '{}',
  PRIMARY KEY (intake_id, fact_idx)
);
CREATE TABLE IF NOT EXISTS staged_outcome(
  intake_id INTEGER PRIMARY KEY,
  outcome TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS __lb_intake_cursor(
  id INTEGER PRIMARY KEY CHECK (id = 1),
  applied_intake_id INTEGER NOT NULL DEFAULT 0
);
";

/// Where the `sqlite` extension comes from for an extension-driven tick.
#[derive(Debug, Clone, Copy)]
pub enum ExtensionSource<'a> {
    /// `INSTALL sqlite` + `LOAD EXTENSION sqlite` (official repo, latest
    /// 0.22.x; cached under `~/.lbdb/extension`, idempotent).
    Repo,
    /// A locally built `.lbug_extension` file (dev override).
    File(&'a str),
}

/// One sync tick result, mirroring `CALL sqlite_sync` columns plus the path
/// taken (`via`: `"extension"` or `"embedded"`).
#[derive(Debug, Clone)]
pub struct SyncStats {
    pub tables_synced: i64,
    pub rows_upserted: i64,
    pub rows_deleted: i64,
    pub applied_intake: i64,
    pub status: String,
    pub via: String,
}

/// The SQLite side of the pipeline: intake log + staging tables + cursor.
pub struct IntakeDb {
    conn: SqliteConn,
    path: String,
}

impl IntakeDb {
    /// Open (creating) the intake database. Parent dirs are created.
    /// Enables WAL mode so the C++ WAL tailer can replicate from it.
    pub fn open(path: &str) -> Result<Self> {
        if path == ":memory:" {
            return Self::open_ephemeral();
        }
        if let Some(parent) = std::path::Path::new(path).parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        let conn = SqliteConn::open(path).with_context(|| format!("opening sqlite intake at {path}"))?;
        conn.execute_batch("PRAGMA journal_mode=WAL;")?;
        conn.execute_batch(INTAKE_SCHEMA)?;
        // Conflict edges point at native graph ids that live outside the
        // staging tables, so enforcement would reject them. The REFERENCES
        // metadata stays (the `sqlite` extension resolves edge endpoints
        // through it); consistency here is maintained by the applier joins.
        conn.execute_batch("PRAGMA foreign_keys=OFF;")?;
        // Sync identity in the extension is the normalized absolute path.
        let abs = std::path::Path::new(path)
            .canonicalize()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|_| path.to_string());
        Ok(Self { conn, path: abs })
    }

    /// In-memory intake (tests / ephemeral stores). The C++ extension needs a
    /// file, so this pairs with the embedded applier only.
    pub fn open_ephemeral() -> Result<Self> {
        let conn = SqliteConn::open_in_memory()?;
        conn.execute_batch(INTAKE_SCHEMA)?;
        conn.execute_batch("PRAGMA foreign_keys=OFF;")?;
        Ok(Self { conn, path: ":memory:".into() })
    }

    pub fn path(&self) -> &str {
        &self.path
    }

    /// Append raw text to the intake log. Returns the intake id.
    pub fn append_intake(
        &self,
        content: &str,
        user_id: &str,
        project: &str,
        session_id: &str,
        importance: i64,
    ) -> Result<i64> {
        self.conn.execute(
            "INSERT INTO memory_intake(content, user_id, project, session_id, importance, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, datetime('now'))",
            params![content, user_id, project, session_id, importance],
        )?;
        Ok(self.conn.last_insert_rowid())
    }

    pub fn pending_intake_ids(&self) -> Result<Vec<i64>> {
        let cursor = self.cursor()?;
        self.collect("SELECT id FROM memory_intake WHERE id > ?1 ORDER BY id", cursor, |row| {
            row.get(0)
        })
    }

    pub fn pending_count(&self) -> Result<i64> {
        let cursor = self.cursor()?;
        Ok(self.conn.query_row(
            "SELECT COUNT(*) FROM memory_intake WHERE id > ?1",
            [cursor],
            |row| row.get(0),
        )?)
    }

    pub fn cursor(&self) -> Result<i64> {
        let v: i64 = self
            .conn
            .query_row("SELECT applied_intake_id FROM __lb_intake_cursor WHERE id = 1", [], |row| {
                row.get(0)
            })
            .unwrap_or(0);
        Ok(v)
    }

    /// Collect all rows of a one-intake query. (Keeps the rusqlite
    /// `Statement` borrows in named locals; chaining `query_map()?.collect()`
    /// as a block-tail expression trips drop-order E0597 under `anyhow`.)
    fn collect<T>(
        &self,
        sql: &str,
        intake_id: i64,
        f: impl FnMut(&rusqlite::Row<'_>) -> rusqlite::Result<T>,
    ) -> Result<Vec<T>> {
        let mut stmt = self.conn.prepare(sql)?;
        let mapped = stmt.query_map([intake_id], f)?;
        let out: Vec<T> = mapped.collect::<std::result::Result<Vec<T>, rusqlite::Error>>()?;
        Ok(out)
    }

    /// Collect all rows of a parameter-free query.
    fn collect_all<T>(
        &self,
        sql: &str,
        f: impl FnMut(&rusqlite::Row<'_>) -> rusqlite::Result<T>,
    ) -> Result<Vec<T>> {
        let mut stmt = self.conn.prepare(sql)?;
        let mapped = stmt.query_map([], f)?;
        let out: Vec<T> = mapped.collect::<std::result::Result<Vec<T>, rusqlite::Error>>()?;
        Ok(out)
    }

    /// Advance the cursor past `id` (post-tick bookkeeping).
    pub(crate) fn mark_applied(&self, id: i64) -> Result<()> {
        self.set_cursor(id)
    }

    fn set_cursor(&self, id: i64) -> Result<()> {
        self.conn.execute(
            "INSERT INTO __lb_intake_cursor(id, applied_intake_id) VALUES (1, ?1)
             ON CONFLICT(id) DO UPDATE SET applied_intake_id = excluded.applied_intake_id",
            [id],
        )?;
        Ok(())
    }

    /// Stage one distilled memory fact. Returns the staging id.
    #[allow(clippy::too_many_arguments)]
    pub fn stage_memory(
        &self,
        intake_id: i64,
        fact_idx: i64,
        content: &str,
        category: &str,
        user_id: &str,
        project: &str,
        session_id: &str,
        importance: i64,
        status: &str,
    ) -> Result<i64> {
        self.conn.execute(
            "INSERT INTO node_Memory(content, category, user_id, project, session_id, importance, status)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![content, category, user_id, project, session_id, importance, status],
        )?;
        let staged = self.conn.last_insert_rowid();
        self.conn.execute(
            "INSERT INTO stage_memory_map(staged_id, intake_id, fact_idx) VALUES (?1, ?2, ?3)",
            params![staged, intake_id, fact_idx],
        )?;
        Ok(staged)
    }

    /// Stage an entity; canonical names are unique per intake file so repeat
    /// mentions of one entity reuse one staged row in both appliers.
    /// `native_id` (when the entity already exists in the graph) becomes the
    /// staging id, so both appliers converge on one identity. Returns the
    /// staging id.
    pub fn stage_entity(
        &self,
        intake_id: i64,
        canonical: &str,
        entity_type: &str,
        aliases_json: &str,
        native_id: Option<i64>,
    ) -> Result<i64> {
        if let Some(native) = native_id {
            self.conn.execute(
                "INSERT INTO node_Entity(id, canonical_name, entity_type, aliases)
                 VALUES (?1, ?2, ?3, ?4)
                 ON CONFLICT(canonical_name) DO UPDATE SET aliases = excluded.aliases",
                params![native, canonical, entity_type, aliases_json],
            )?;
            // Resolve the surviving row id (insert or the conflicting row).
            let staged: i64 = self.conn.query_row(
                "SELECT id FROM node_Entity WHERE canonical_name = ?1",
                [canonical],
                |row| row.get(0),
            )?;
            self.conn.execute(
                "INSERT INTO stage_entity_map(staged_id, intake_id, lb_id) VALUES (?1, ?2, ?3)
                 ON CONFLICT(staged_id) DO UPDATE SET lb_id = excluded.lb_id",
                params![staged, intake_id, native],
            )?;
            return Ok(staged);
        }
        self.conn.execute(
            "INSERT INTO node_Entity(canonical_name, entity_type, aliases)
             VALUES (?1, ?2, ?3)
             ON CONFLICT(canonical_name) DO NOTHING",
            params![canonical, entity_type, aliases_json],
        )?;
        let staged: i64 = self.conn.query_row(
            "SELECT id FROM node_Entity WHERE canonical_name = ?1",
            [canonical],
            |row| row.get(0),
        )?;
        self.conn.execute(
            "INSERT INTO stage_entity_map(staged_id, intake_id) VALUES (?1, ?2)
             ON CONFLICT(staged_id) DO NOTHING",
            params![staged, intake_id],
        )?;
        Ok(staged)
    }

    fn track_edge(&self, rel_table: &str, intake_id: i64) -> Result<()> {
        let rowid = self.conn.last_insert_rowid();
        self.conn.execute(
            "INSERT INTO stage_edge_map(rel_table, rowid, intake_id) VALUES (?1, ?2, ?3)
             ON CONFLICT(rel_table, rowid) DO NOTHING",
            params![rel_table, rowid, intake_id],
        )?;
        Ok(())
    }

    pub fn stage_mention(
        &self,
        intake_id: i64,
        entity_staged_id: i64,
        memory_staged_id: i64,
        mention_text: &str,
        confidence: f64,
        span_start: i64,
        span_end: i64,
    ) -> Result<()> {
        self.conn.execute(
            "INSERT INTO rel_MentionedIn(src_id, dst_id, mention_text, confidence, span_start, span_end)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![entity_staged_id, memory_staged_id, mention_text, confidence, span_start, span_end],
        )?;
        self.track_edge("rel_MentionedIn", intake_id)
    }

    pub fn stage_cooccur(
        &self,
        intake_id: i64,
        a_staged_id: i64,
        b_staged_id: i64,
        confidence: f64,
    ) -> Result<()> {
        if a_staged_id == b_staged_id {
            return Ok(());
        }
        // Undirected edge, canonical order (matches Store::link_cooccur).
        let (lo, hi) = if a_staged_id < b_staged_id {
            (a_staged_id, b_staged_id)
        } else {
            (b_staged_id, a_staged_id)
        };
        self.conn.execute(
            "INSERT INTO rel_RelatesTo(src_id, dst_id, relation_type, confidence)
             VALUES (?1, ?2, 'coOccurs', ?3)",
            params![lo, hi, confidence],
        )?;
        self.track_edge("rel_RelatesTo", intake_id)
    }

    pub fn stage_conflict(
        &self,
        intake_id: i64,
        new_memory_staged_id: i64,
        old_native_id: i64,
        cues_json: &str,
    ) -> Result<()> {
        self.conn.execute(
            "INSERT INTO rel_ConflictsWith(src_id, dst_id, cues) VALUES (?1, ?2, ?3)",
            params![new_memory_staged_id, old_native_id, cues_json],
        )?;
        self.track_edge("rel_ConflictsWith", intake_id)
    }

    /// The recorded outcome for one intake (`duplicate` vs `written`).
    pub fn outcome(&self, intake_id: i64) -> Result<Option<(String, Option<i64>)>> {
        let raw: Option<String> = self
            .conn
            .query_row("SELECT outcome FROM staged_outcome WHERE intake_id = ?1", [intake_id], |row| {
                row.get(0)
            })
            .ok();
        let Some(raw) = raw else { return Ok(None) };
        let v: serde_json::Value = serde_json::from_str(&raw).unwrap_or(serde_json::Value::Null);
        let kind = v.get("kind").and_then(|k| k.as_str()).unwrap_or("").to_string();
        let of = v.get("of").and_then(|o| o.as_i64());
        Ok(Some((kind, of)))
    }

    pub fn record_outcome(&self, intake_id: i64, outcome_json: &str) -> Result<()> {
        self.conn.execute(
            "INSERT INTO staged_outcome(intake_id, outcome) VALUES (?1, ?2)
             ON CONFLICT(intake_id) DO UPDATE SET outcome = excluded.outcome",
            params![intake_id, outcome_json],
        )?;
        Ok(())
    }

    pub fn record_provenance(&self, intake_id: i64, fact_idx: i64, provenance_json: &str) -> Result<()> {
        self.conn.execute(
            "INSERT INTO memory_provenance(intake_id, fact_idx, provenance) VALUES (?1, ?2, ?3)
             ON CONFLICT(intake_id, fact_idx) DO UPDATE SET provenance = excluded.provenance",
            params![intake_id, fact_idx, provenance_json],
        )?;
        Ok(())
    }

    /// Native memory ids converged for one intake, in fact order.
    /// Used to attach `written` entries to staged reports post-sync.
    pub fn staged_memory_lb_ids(&self, intake_id: i64) -> Result<Vec<i64>> {
        let mut stmt = self.conn.prepare(
            "SELECT s.lb_id FROM stage_memory_map s
             WHERE s.intake_id = ?1 AND s.lb_id IS NOT NULL ORDER BY s.fact_idx",
        )?;
        let mapped = stmt.query_map([intake_id], |row| row.get(0))?;
        let out: Vec<i64> = mapped.collect::<std::result::Result<Vec<i64>, rusqlite::Error>>()?;
        Ok(out)
    }

    /// Record extension-tick identity: under MERGE-by-PK a staged row's
    /// native id is its staging id. Backfills `lb_id` for one intake so the
    /// embedded applier and report attach stay coherent afterwards.
    pub(crate) fn backfill_extension_ids(&self, intake_id: i64) -> Result<()> {
        self.conn.execute(
            "UPDATE stage_memory_map SET lb_id = staged_id WHERE intake_id = ?1 AND lb_id IS NULL",
            [intake_id],
        )?;
        self.conn.execute(
            "UPDATE stage_entity_map SET lb_id = staged_id WHERE intake_id = ?1 AND lb_id IS NULL",
            [intake_id],
        )?;
        Ok(())
    }

    fn memory_lb_id(&self, staged_id: i64) -> Result<Option<i64>> {
        Ok(self
            .conn
            .query_row("SELECT lb_id FROM stage_memory_map WHERE staged_id = ?1", [staged_id], |row| {
                row.get(0)
            })
            .unwrap_or(None))
    }

    fn set_memory_lb_id(&self, staged_id: i64, lb_id: i64) -> Result<()> {
        self.conn.execute("UPDATE stage_memory_map SET lb_id = ?1 WHERE staged_id = ?2", params![lb_id, staged_id])?;
        Ok(())
    }

    fn entity_lb_id(&self, staged_id: i64) -> Result<Option<i64>> {
        Ok(self
            .conn
            .query_row("SELECT lb_id FROM stage_entity_map WHERE staged_id = ?1", [staged_id], |row| {
                row.get(0)
            })
            .unwrap_or(None))
    }

    fn set_entity_lb_id(&self, staged_id: i64, lb_id: i64) -> Result<()> {
        self.conn.execute("UPDATE stage_entity_map SET lb_id = ?1 WHERE staged_id = ?2", params![lb_id, staged_id])?;
        Ok(())
    }
}

/// Embedded fallback applier: replicates staged rows for every intake past
/// the cursor into `store`, mirroring the extension's apply order (nodes
/// before edges). Idempotent: per-intake cursor gating plus `lb_id`
/// backfill, so replays converge.
pub fn replicate_to_store(intake: &IntakeDb, store: &Store) -> Result<SyncStats> {
    let mut upserted: i64 = 0;
    let mut applied: i64 = 0;
    let pending = intake.pending_intake_ids()?;
    let mut known = store.list_entities()?;

    for intake_id in pending {
        // Duplicate outcomes decided pre-stage only need the strengthening
        // touch (already applied live at remember time; a cursor-gated
        // replay re-touching is a harmless +1, same as at-least-once
        // redelivery).
        let outcome: String = intake
            .conn
            .query_row("SELECT outcome FROM staged_outcome WHERE intake_id = ?1", [intake_id], |row| {
                row.get(0)
            })
            .unwrap_or_else(|_| "{}".to_string());
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(&outcome) {
            if v.get("kind") == Some(&serde_json::Value::String("duplicate".into())) {
                if let Some(of) = v.get("of").and_then(|o| o.as_i64()) {
                    let _ = store.touch_memory(of, 1);
                }
            }
        }

        // Provenance per fact for full MemoryEntry parity.
        let provs: std::collections::HashMap<i64, String> = intake
            .collect(
                "SELECT fact_idx, provenance FROM memory_provenance WHERE intake_id = ?1",
                intake_id,
                |row| Ok((row.get(0)?, row.get(1)?)),
            )?
            .into_iter()
            .collect();

        // Staged memories for this intake (join keeps the mapped table clean).
        let memories: Vec<(i64, i64, String, String, String, String, String, i64, String)> = intake.collect(
            "SELECT m.id, s.fact_idx, m.content, m.category, m.user_id, m.project, m.session_id,
                    m.importance, m.status
             FROM node_Memory m JOIN stage_memory_map s ON s.staged_id = m.id
             WHERE s.intake_id = ?1 ORDER BY s.fact_idx",
            intake_id,
            |row| {
                Ok((
                    row.get(0)?,
                    row.get(1)?,
                    row.get(2)?,
                    row.get(3)?,
                    row.get(4)?,
                    row.get(5)?,
                    row.get(6)?,
                    row.get(7)?,
                    row.get(8)?,
                ))
            },
        )?;
        let mut mem_lb: std::collections::HashMap<i64, i64> = std::collections::HashMap::new();
        for (staged_id, fact_idx, content, category, user_id, project, session_id, importance, status) in &memories {
            if let Some(existing) = intake.memory_lb_id(*staged_id)? {
                mem_lb.insert(*staged_id, existing);
                continue;
            }
            // Partial-tick convergence: if an extension tick committed this
            // row before failing (statement-by-statement apply), reuse it
            // instead of inserting a twin.
            if let Some(existing) = store.find_exact_memory(content, user_id, project, session_id)? {
                intake.set_memory_lb_id(*staged_id, existing)?;
                mem_lb.insert(*staged_id, existing);
                continue;
            }
            let prov: crate::types::Provenance = provs
                .get(fact_idx)
                .and_then(|s| serde_json::from_str(s).ok())
                .unwrap_or_default();
            let id = store.insert_memory(
                content,
                crate::types::Category::parse(category),
                user_id,
                project,
                session_id,
                *importance,
                crate::types::MemoryStatus::parse(status),
                &prov,
            )?;
            intake.set_memory_lb_id(*staged_id, id)?;
            mem_lb.insert(*staged_id, id);
            upserted += 1;
        }

        // Entity staged ids referenced by this intake's edges, collected
        // from each edge table's map explicitly (clearer than a UNION
        // across tables with different endpoint semantics).
        let ent_staged: Vec<i64> = {
            let mut ids = std::collections::HashSet::new();
            // Only entity endpoints: MentionedIn.src and both RelatesTo
            // ends. ConflictsWith endpoints are memories (new staged row ->
            // old native id), handled in the memory/conflict steps.
            for rel in ["rel_MentionedIn", "rel_RelatesTo"] {
                let (sql, col) = match rel {
                    "rel_MentionedIn" => (
                        "SELECT m.src_id FROM rel_MentionedIn m
                         JOIN stage_edge_map e ON e.rel_table = 'rel_MentionedIn' AND e.rowid = m.rowid
                         WHERE e.intake_id = ?1",
                        "src",
                    ),
                    _ => (
                        "SELECT r.src_id, r.dst_id FROM rel_RelatesTo r
                         JOIN stage_edge_map e ON e.rel_table = 'rel_RelatesTo' AND e.rowid = r.rowid
                         WHERE e.intake_id = ?1",
                        "both",
                    ),
                };
                if col == "both" {
                    let rows: Vec<(i64, i64)> =
                        intake.collect(sql, intake_id, |row| Ok((row.get(0)?, row.get(1)?)))?;
                    for (a, b) in rows {
                        ids.insert(a);
                        ids.insert(b);
                    }
                } else {
                    let rows: Vec<i64> = intake.collect(sql, intake_id, |row| row.get(0))?;
                    for id in rows {
                        ids.insert(id);
                    }
                }
            }
            ids.into_iter().collect()
        };

        // Ensure every referenced entity exists natively (alias growth
        // merges onto the live schema, never forks).
        let mut ent_lb: std::collections::HashMap<i64, i64> = std::collections::HashMap::new();
        for staged_id in ent_staged {
            if let Some(existing) = intake.entity_lb_id(staged_id)? {
                ent_lb.insert(staged_id, existing);
                // Refresh in case the live row changed since staging.
                continue;
            }
            let (canonical, entity_type, aliases_json): (String, String, String) = intake.conn.query_row(
                "SELECT canonical_name, entity_type, aliases FROM node_Entity WHERE id = ?1",
                [staged_id],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )?;
            let staged_aliases: Vec<String> =
                serde_json::from_str(&aliases_json).unwrap_or_else(|_| vec![canonical.clone()]);
            let norm = crate::resolve::normalize(&canonical);
            let native_id = match known.iter().find(|e| e.normalized == norm || e.alias_normalized.contains(&norm)) {
                Some(e) => {
                    let mut aliases = e.aliases.clone();
                    for a in &staged_aliases {
                        if !aliases.iter().any(|x| x == a) {
                            aliases.push(a.clone());
                        }
                    }
                    if !aliases.iter().any(|x| x == &canonical) {
                        aliases.push(canonical.clone());
                    }
                    store.upsert_entity(&canonical, &entity_type, &aliases, Some(e.id))?.0
                }
                None => {
                    let (id, _) = store.upsert_entity(&canonical, &entity_type, &staged_aliases, None)?;
                    upserted += 1;
                    known.push(crate::resolve::KnownEntity {
                        id,
                        normalized: norm.clone(),
                        canonical: canonical.clone(),
                        entity_type: entity_type.clone(),
                        alias_normalized: staged_aliases
                            .iter()
                            .map(|a| crate::resolve::normalize(a))
                            .collect(),
                        aliases: staged_aliases.clone(),
                    });
                    id
                }
            };
            intake.set_entity_lb_id(staged_id, native_id)?;
            ent_lb.insert(staged_id, native_id);
        }
        // Re-seen entities staged under earlier intakes (same canonical → same
        // staged row via UNIQUE): map every backfilled id for edge joins.
        {
            let rows: Vec<(i64, i64)> = intake.collect_all(
                "SELECT staged_id, lb_id FROM stage_entity_map WHERE lb_id IS NOT NULL",
                |row| Ok((row.get(0)?, row.get(1)?)),
            )?;
            for (sid, lb) in rows {
                ent_lb.entry(sid).or_insert(lb);
            }
            let rows: Vec<(i64, i64)> = intake.collect_all(
                "SELECT staged_id, lb_id FROM stage_memory_map WHERE lb_id IS NOT NULL",
                |row| Ok((row.get(0)?, row.get(1)?)),
            )?;
            for (sid, lb) in rows {
                mem_lb.entry(sid).or_insert(lb);
            }
        }

        // Mentions.
        {
            let rows: Vec<(i64, i64, String, f64, i64, i64)> = intake.collect(
                "SELECT m.src_id, m.dst_id, m.mention_text, m.confidence, m.span_start, m.span_end
                 FROM rel_MentionedIn m
                 JOIN stage_edge_map e ON e.rel_table = 'rel_MentionedIn' AND e.rowid = m.rowid
                 WHERE e.intake_id = ?1",
                intake_id,
                |row| {
                    Ok((
                        row.get(0)?,
                        row.get(1)?,
                        row.get(2)?,
                        row.get(3)?,
                        row.get(4)?,
                        row.get(5)?,
                    ))
                },
            )?;
            for (src, dst, text, conf, s, e) in rows {
                if let (Some(eid), Some(mid)) = (ent_lb.get(&src), mem_lb.get(&dst)) {
                    if store.has_mention(*eid, *mid, &text, s, e)? {
                        continue;
                    }
                    store.link_mention(*eid, *mid, &text, conf, s, e)?;
                    upserted += 1;
                }
            }
        }
        // Co-occurrences.
        {
            let rows: Vec<(i64, i64, f64)> = intake.collect(
                "SELECT r.src_id, r.dst_id, r.confidence FROM rel_RelatesTo r
                 JOIN stage_edge_map e ON e.rel_table = 'rel_RelatesTo' AND e.rowid = r.rowid
                 WHERE e.intake_id = ?1",
                intake_id,
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )?;
            for (a, b, conf) in rows {
                if let (Some(x), Some(y)) = (ent_lb.get(&a), ent_lb.get(&b)) {
                    store.link_cooccur(*x, *y, conf)?;
                }
            }
        }
        // Conflicts (new staged memory -> old native memory).
        {
            let rows: Vec<(i64, i64, String)> = intake.collect(
                "SELECT c.src_id, c.dst_id, c.cues FROM rel_ConflictsWith c
                 JOIN stage_edge_map e ON e.rel_table = 'rel_ConflictsWith' AND e.rowid = c.rowid
                 WHERE e.intake_id = ?1",
                intake_id,
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )?;
            for (src, old, cues_json) in rows {
                if let Some(new_id) = mem_lb.get(&src) {
                    if store.has_conflict(*new_id, old)? {
                        continue;
                    }
                    let cues: Vec<String> = serde_json::from_str(&cues_json).unwrap_or_default();
                    store.link_conflict(*new_id, old, &cues)?;
                }
            }
        }

        intake.set_cursor(intake_id)?;
        applied += 1;
    }

    Ok(SyncStats {
        tables_synced: 5,
        rows_upserted: upserted,
        rows_deleted: 0,
        applied_intake: applied,
        status: "ok".into(),
        via: "embedded".into(),
    })
}

/// Derive the default intake path for a Ladybug database file.
pub fn sibling_intake_path(db_path: &str) -> String {
    if db_path == ":memory:" {
        return ":memory:".into();
    }
    let p = std::path::Path::new(db_path);
    match (p.parent(), p.file_stem()) {
        (Some(parent), Some(stem)) => {
            let mut sib = parent.to_path_buf();
            sib.push(format!("{}.intake.sqlite", stem.to_string_lossy()));
            sib.display().to_string()
        }
        _ => format!("{db_path}.intake.sqlite"),
    }
}
