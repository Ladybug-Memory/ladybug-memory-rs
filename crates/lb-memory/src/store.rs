//! LadybugDB-backed graph store (embedded, via the `lbug` crate).
//!
//! One database file holds all scopes; rows carry `user_id` / `project` /
//! `session_id` columns so project/session/global filtering is just a
//! `WHERE` clause. Session *message content* is never stored here — only
//! distilled facts plus provenance references into pi's own session store.

use anyhow::{Context, Result, anyhow};
use lbug::{Connection, Database, SystemConfig, Value};
use serde_json::Value as JsonValue;
use time::OffsetDateTime;

use crate::resolve::KnownEntity;
use crate::types::{Category, MemoryEntry, MemoryStatus, Provenance};

const SCHEMA_STATEMENTS: &[&str] = &[
    "CREATE NODE TABLE IF NOT EXISTS Memory(id SERIAL PRIMARY KEY, content STRING, category STRING, user_id STRING, project STRING, session_id STRING, importance INT64, status STRING, provenance JSON, created_at TIMESTAMP DEFAULT current_timestamp(), updated_at TIMESTAMP DEFAULT current_timestamp())",
    "CREATE NODE TABLE IF NOT EXISTS Entity(id SERIAL PRIMARY KEY, canonical_name STRING, entity_type STRING, aliases JSON, created_at TIMESTAMP DEFAULT current_timestamp(), updated_at TIMESTAMP DEFAULT current_timestamp())",
    "CREATE REL TABLE IF NOT EXISTS MentionedIn(FROM Entity TO Memory, mention_text STRING, confidence DOUBLE, span_start INT64, span_end INT64, valid_from STRING, valid_to STRING, tense STRING, viewpoint STRING, viewpoint_confidence DOUBLE)",
    "CREATE REL TABLE IF NOT EXISTS RelatesTo(FROM Entity TO Entity, relation_type STRING, confidence DOUBLE, valid_from STRING, valid_to STRING, tense STRING, viewpoint STRING, viewpoint_confidence DOUBLE)",
    "CREATE REL TABLE IF NOT EXISTS ConflictsWith(FROM Memory TO Memory, cues JSON, valid_from STRING, valid_to STRING, tense STRING, viewpoint STRING, viewpoint_confidence DOUBLE)",
    "CREATE REL TABLE IF NOT EXISTS Supersedes(FROM Memory TO Memory, reason STRING, valid_from STRING, valid_to STRING, tense STRING, viewpoint STRING, viewpoint_confidence DOUBLE)",
];

/// Edge metadata stamped on every relationship at write time.
/// `valid_from`/`valid_to` are ISO dates (`""` = unknown); `tense` is
/// past|present|future|unknown as stated; `viewpoint` is `"fact"` for
/// uncontroversial statements else `"opinion"`, with `viewpoint_confidence`.
/// Mention/conflict/supersede edges are fact-anchored: their time range is
/// the statement date (asserted-at), since only RelatesTo edges carry
/// independent temporal claims.
#[derive(Debug, Clone, Default)]
pub struct EdgeMeta {
    pub valid_from: String,
    pub valid_to: String,
    pub tense: String,
    pub viewpoint: String,
    pub viewpoint_confidence: f64,
}

/// New rel-table columns, backfilled on pre-existing databases via
/// `ALTER TABLE ... ADD IF NOT EXISTS` (no-op when already present).
const REL_MIGRATIONS: &[&str] = &[
    "valid_from STRING",
    "valid_to STRING",
    "tense STRING",
    "viewpoint STRING",
    "viewpoint_confidence DOUBLE",
];

pub struct Store {
    db: Database,
    fts_ready: bool,
    db_path: Option<String>,
}

impl Store {
    /// Open (creating) the database file. Parent dirs are created.
    pub fn open(path: &str) -> Result<Self> {
        if let Some(parent) = std::path::Path::new(path).parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        let db = Database::new(path, SystemConfig::default())
            .with_context(|| format!("opening ladybug db at {path}"))?;
        let mut store = Self { db, fts_ready: false, db_path: Some(path.to_string()) };
        store.ensure_schema()?;
        Ok(store)
    }

    /// In-memory database (tests).
    pub fn ephemeral() -> Result<Self> {
        let db = Database::in_memory(SystemConfig::default()).context("opening :memory: db")?;
        let mut store = Self { db, fts_ready: false, db_path: None };
        store.ensure_schema()?;
        Ok(store)
    }

    /// The database file path, if file-backed.
    pub fn db_path(&self) -> Option<&str> {
        self.db_path.as_deref()
    }

    /// The SQLite intake database for this store: the sibling
    /// `<stem>.intake.sqlite` for file-backed stores, in-memory otherwise.
    /// This is the first-class write path — memories land here before the
    /// graph (see `sqlite_sync`).
    pub fn intake_db(&self) -> Result<crate::sqlite_sync::IntakeDb> {
        match &self.db_path {
            Some(p) => crate::sqlite_sync::IntakeDb::open(&crate::sqlite_sync::sibling_intake_path(p)),
            None => crate::sqlite_sync::IntakeDb::open_ephemeral(),
        }
    }

    /// Raw Cypher for callers that drive optional integrations (e.g. the
    /// `sqlite` extension's `CALL sqlite_sync`).
    pub fn run_cypher(&self, cypher: &str) -> Result<Vec<Vec<Value>>> {
        self.run(cypher, vec![])
    }

    /// Fetch the `sqlite` extension the official way: `INSTALL sqlite`
    /// (idempotent — downloads the latest 0.22.x build from the extension
    /// repo into `~/.lbdb/extension` only when missing) then `LOAD sqlite`
    /// by name. This is the production path. The host links liblbug shared
    /// (see `.cargo/config.toml`), which is what lets the extension resolve
    /// engine symbols.
    pub fn ensure_sqlite_extension(&self) -> Result<()> {
        self.run("INSTALL sqlite", vec![])?.into_iter().for_each(drop);
        self.run("LOAD sqlite", vec![])?.into_iter().for_each(drop);
        Ok(())
    }

    /// Load a locally built `sqlite` extension binary by explicit path.
    /// Testing/dev override for extension hacking (see README); production
    /// uses [`Self::ensure_sqlite_extension`] (`LOAD sqlite` by name).
    pub fn load_sqlite_extension(&self, ext_path: &str) -> Result<()> {
        let lit = cypher_path_lit(ext_path);
        self.run(&format!("LOAD EXTENSION '{lit}'"), vec![])?.into_iter().for_each(drop);
        Ok(())
    }

    /// Register this store's intake staging tables with the loaded `sqlite`
    /// extension, remapping staging labels onto the native graph tables.
    /// Idempotent (backed by the extension's `__sqlite_cdc_map` MERGE).
    /// Returns the per-table status strings.
    pub fn sqlite_register_all(&self, sqlite_path: &str) -> Result<Vec<String>> {
        let lit = cypher_path_lit(sqlite_path);
        let calls = [
            format!("CALL sqlite_register_node('{lit}', 'node_Memory', 'Memory', 'id') RETURN *"),
            format!("CALL sqlite_register_node('{lit}', 'node_Entity', 'Entity', 'id') RETURN *"),
            format!("CALL sqlite_register_edge('{lit}', 'rel_MentionedIn', 'MentionedIn', 'src_id', 'dst_id') RETURN *"),
            format!("CALL sqlite_register_edge('{lit}', 'rel_RelatesTo', 'RelatesTo', 'src_id', 'dst_id') RETURN *"),
            format!("CALL sqlite_register_edge('{lit}', 'rel_ConflictsWith', 'ConflictsWith', 'src_id', 'dst_id') RETURN *"),
        ];
        let mut out = Vec::new();
        for call in &calls {
            let rows = self.run(call, vec![])?;
            let status = rows
                .first()
                .and_then(|r| r.first())
                .map(|v| format!("{v:?}"))
                .unwrap_or_default();
            out.push(status);
        }
        Ok(out)
    }

    /// One extension-driven replication tick: `CALL sqlite_sync(path)`.
    /// Returns `(tables_synced, rows_upserted, rows_deleted, applied_frames,
    /// status)`. Any engine error is returned verbatim — the caller (see
    /// `Pipeline::remember_via_sqlite`) falls back to the embedded applier.
    pub fn sqlite_sync_tick(&self, sqlite_path: &str) -> Result<(i64, i64, i64, i64, String)> {
        let lit = cypher_path_lit(sqlite_path);
        let rows = self.run(&format!("CALL sqlite_sync('{lit}') RETURN *"), vec![])?;
        let row = rows.first().ok_or_else(|| anyhow!("sqlite_sync returned no row"))?;
        Ok((
            as_i64(&row[0])?,
            as_i64(&row[1])?,
            as_i64(&row[2])?,
            as_i64(&row[3])?,
            as_string(&row[4]).unwrap_or_default(),
        ))
    }

    fn conn(&self) -> Result<Connection<'_>> {
        Connection::new(&self.db).context("opening connection")
    }

    fn ensure_schema(&mut self) -> Result<()> {
        // NLL: scope the connection so `fts_ready` can be updated after.
        let mut fts = false;
        {
            let conn = self.conn()?;
            for stmt in SCHEMA_STATEMENTS {
                conn.query(stmt)?.for_each(drop);
            }
            for table in ["MentionedIn", "RelatesTo", "ConflictsWith", "Supersedes"] {
                for col in REL_MIGRATIONS {
                    let _ = conn
                        .query(&format!("ALTER TABLE {table} ADD IF NOT EXISTS {col}"))
                        .map(|mut r| r.for_each(drop));
                }
            }
            // Full-text search over Memory.content (porter stemmer).
            // INSTALL fetches the extension on first use (cached under
            // ~/.lbdb/extension after); every step tolerates failure so
            // offline / minimal builds fall back to substring search.
            // Requires shared liblbug linking — a statically linked host
            // dead-strips the symbols extensions resolve (see README).
            let _ = conn.query("INSTALL FTS").map(|r| r.for_each(drop));
            if conn.query("LOAD EXTENSION FTS").map(|r| r.for_each(drop)).is_ok() {
                match conn.query(
                    "CALL CREATE_FTS_INDEX('Memory', 'memory_content_fts', ['content'], stemmer := 'porter')",
                ) {
                    Ok(_) => fts = true,
                    Err(e) => {
                        let msg = format!("{e:?}");
                        if msg.contains("already exists") || msg.contains("duplicate") {
                            fts = true;
                        }
                    }
                }
            }
        }
        self.fts_ready = fts;
        Ok(())
    }

    #[cfg(any(test, feature = "test-support"))]
    pub fn fts_ready(&self) -> bool {
        self.fts_ready
    }

    fn run(&self, cypher: &str, params: Vec<(&str, Value)>) -> Result<Vec<Vec<Value>>> {
        let conn = self.conn()?;
        let mut stmt = conn.prepare(cypher)?;
        let res = conn.execute(&mut stmt, params)?;
        Ok(res.collect())
    }

    // ---- writes ----

    #[allow(clippy::too_many_arguments)]
    pub fn insert_memory(
        &self,
        content: &str,
        category: Category,
        user_id: &str,
        project: &str,
        session_id: &str,
        importance: i64,
        status: MemoryStatus,
        provenance: &Provenance,
    ) -> Result<i64> {
        let now = Value::Timestamp(OffsetDateTime::now_utc());
        let prov = serde_json::to_value(provenance).unwrap_or(JsonValue::Null);
        let rows = self.run(
            "CREATE (m:Memory {content: $content, category: $category, user_id: $user, project: $project, session_id: $session, importance: $importance, status: $status, provenance: $prov, created_at: $now, updated_at: $now}) RETURN m.id AS id",
            vec![
                ("content", Value::String(content.to_string())),
                ("category", Value::String(category.as_str().to_string())),
                ("user", Value::String(user_id.to_string())),
                ("project", Value::String(project.to_string())),
                ("session", Value::String(session_id.to_string())),
                ("importance", Value::Int64(importance)),
                ("status", Value::String(status.as_str().to_string())),
                ("prov", Value::Json(prov)),
                ("now", now),
            ],
        )?;
        first_id(&rows)
    }

    pub fn touch_memory(&self, id: i64, importance_bump: i64) -> Result<()> {
        let now = Value::Timestamp(OffsetDateTime::now_utc());
        self.run(
            "MATCH (m:Memory) WHERE m.id = $id SET m.importance = m.importance + $bump, m.updated_at = $now",
            vec![("id", Value::Int64(id)), ("bump", Value::Int64(importance_bump)), ("now", now)],
        )?;
        Ok(())
    }

    pub fn set_status(&self, id: i64, status: MemoryStatus) -> Result<()> {
        let now = Value::Timestamp(OffsetDateTime::now_utc());
        self.run(
            "MATCH (m:Memory) WHERE m.id = $id SET m.status = $status, m.updated_at = $now",
            vec![("id", Value::Int64(id)), ("status", Value::String(status.as_str().to_string())), ("now", now)],
        )?;
        Ok(())
    }

    pub fn delete_memory(&self, id: i64) -> Result<()> {
        self.run("MATCH (m:Memory) WHERE m.id = $id DETACH DELETE m", vec![("id", Value::Int64(id))])?;
        Ok(())
    }

    /// Match-or-create by canonical name. Returns `(id, created)`.
    pub fn upsert_entity(
        &self,
        canonical: &str,
        entity_type: &str,
        aliases: &[String],
        existing_id: Option<i64>,
    ) -> Result<(i64, bool)> {
        let aliases_json: JsonValue =
            JsonValue::Array(aliases.iter().map(|s| JsonValue::String(s.clone())).collect());
        if let Some(id) = existing_id {
            let now = Value::Timestamp(OffsetDateTime::now_utc());
            self.run(
                "MATCH (e:Entity) WHERE e.id = $id SET e.aliases = $aliases, e.updated_at = $now",
                vec![("id", Value::Int64(id)), ("aliases", Value::Json(aliases_json)), ("now", now)],
            )?;
            return Ok((id, false));
        }
        let now = Value::Timestamp(OffsetDateTime::now_utc());
        let rows = self.run(
            "CREATE (e:Entity {canonical_name: $canonical, entity_type: $kind, aliases: $aliases, created_at: $now, updated_at: $now}) RETURN e.id AS id",
            vec![
                ("canonical", Value::String(canonical.to_string())),
                ("kind", Value::String(entity_type.to_string())),
                ("aliases", Value::Json(aliases_json)),
                ("now", now),
            ],
        )?;
        Ok((first_id(&rows)?, true))
    }

    pub fn link_mention(
        &self,
        entity_id: i64,
        memory_id: i64,
        mention_text: &str,
        confidence: f64,
        span_start: i64,
        span_end: i64,
        meta: &EdgeMeta,
    ) -> Result<()> {
        self.run(
            "MATCH (e:Entity), (m:Memory) WHERE e.id = $eid AND m.id = $mid CREATE (e)-[:MentionedIn {mention_text: $text, confidence: $conf, span_start: $s, span_end: $e, valid_from: $vf, valid_to: $vt, tense: $tense, viewpoint: $vp, viewpoint_confidence: $vpc}]->(m)",
            vec![
                ("eid", Value::Int64(entity_id)),
                ("mid", Value::Int64(memory_id)),
                ("text", Value::String(mention_text.to_string())),
                ("conf", Value::Double(confidence)),
                ("s", Value::Int64(span_start)),
                ("e", Value::Int64(span_end)),
                ("vf", Value::String(meta.valid_from.clone())),
                ("vt", Value::String(meta.valid_to.clone())),
                ("tense", Value::String(meta.tense.clone())),
                ("vp", Value::String(meta.viewpoint.clone())),
                ("vpc", Value::Double(meta.viewpoint_confidence)),
            ],
        )?;
        Ok(())
    }

    /// Directed typed edge (`founder-of`, `works-at`, …). Idempotent:
    /// an exact (src, dst, label) edge is never duplicated, so fallback
    /// replays and reconciliations converge.
    pub fn link_typed_relation(
        &self,
        src_id: i64,
        dst_id: i64,
        relation_type: &str,
        confidence: f64,
        meta: &EdgeMeta,
    ) -> Result<()> {
        if src_id == dst_id {
            return Ok(());
        }
        let rows = self.run(
            "MATCH (a:Entity)-[r:RelatesTo]->(b:Entity) WHERE a.id = $src AND b.id = $dst AND r.relation_type = $t RETURN r.confidence LIMIT 1",
            vec![
                ("src", Value::Int64(src_id)),
                ("dst", Value::Int64(dst_id)),
                ("t", Value::String(relation_type.to_string())),
            ],
        )?;
        if !rows.is_empty() {
            return Ok(());
        }
        self.run(
            "MATCH (a:Entity), (b:Entity) WHERE a.id = $src AND b.id = $dst CREATE (a)-[:RelatesTo {relation_type: $t, confidence: $conf, valid_from: $vf, valid_to: $vt, tense: $tense, viewpoint: $vp, viewpoint_confidence: $vpc}]->(b)",
            vec![
                ("src", Value::Int64(src_id)),
                ("dst", Value::Int64(dst_id)),
                ("t", Value::String(relation_type.to_string())),
                ("conf", Value::Double(confidence)),
                ("vf", Value::String(meta.valid_from.clone())),
                ("vt", Value::String(meta.valid_to.clone())),
                ("tense", Value::String(meta.tense.clone())),
                ("vp", Value::String(meta.viewpoint.clone())),
                ("vpc", Value::Double(meta.viewpoint_confidence)),
            ],
        )?;
        Ok(())
    }

    /// Typed edges touching one entity: `(other_canonical, relation_type)`.
    pub fn typed_relations_of(&self, entity_id: i64) -> Result<Vec<(String, String)>> {
        let rows = self.run(
            "MATCH (e:Entity)-[r:RelatesTo]-(o:Entity) WHERE e.id = $id AND r.relation_type <> 'coOccurs' RETURN o.canonical_name, r.relation_type",
            vec![("id", Value::Int64(entity_id))],
        )?;
        rows.iter().map(|r| Ok((as_string(&r[0])?, as_string(&r[1])?))).collect()
    }

    pub fn link_cooccur(&self, a: i64, b: i64, confidence: f64, meta: &EdgeMeta) -> Result<()> {
        if a == b {
            return Ok(());
        }
        let (lo, hi) = if a < b { (a, b) } else { (b, a) };
        let rows = self.run(
            "MATCH (x:Entity)-[r:RelatesTo]-(y:Entity) WHERE x.id = $lo AND y.id = $hi RETURN r.confidence AS c",
            vec![("lo", Value::Int64(lo)), ("hi", Value::Int64(hi))],
        )?;
        if rows.is_empty() {
            self.run(
                "MATCH (x:Entity), (y:Entity) WHERE x.id = $lo AND y.id = $hi CREATE (x)-[:RelatesTo {relation_type: 'coOccurs', confidence: $conf, valid_from: $vf, valid_to: $vt, tense: $tense, viewpoint: $vp, viewpoint_confidence: $vpc}]->(y)",
                vec![
                    ("lo", Value::Int64(lo)),
                    ("hi", Value::Int64(hi)),
                    ("conf", Value::Double(confidence)),
                    ("vf", Value::String(meta.valid_from.clone())),
                    ("vt", Value::String(meta.valid_to.clone())),
                    ("tense", Value::String(meta.tense.clone())),
                    ("vp", Value::String(meta.viewpoint.clone())),
                    ("vpc", Value::Double(meta.viewpoint_confidence)),
                ],
            )?;
        }
        Ok(())
    }

    pub fn link_conflict(&self, new_id: i64, old_id: i64, cues: &[String], meta: &EdgeMeta) -> Result<()> {
        let cues_json: JsonValue =
            JsonValue::Array(cues.iter().map(|s| JsonValue::String(s.clone())).collect());
        self.run(
            "MATCH (a:Memory), (b:Memory) WHERE a.id = $new AND b.id = $old CREATE (a)-[:ConflictsWith {cues: $cues, valid_from: $vf, valid_to: $vt, tense: $tense, viewpoint: $vp, viewpoint_confidence: $vpc}]->(b)",
            vec![
                ("new", Value::Int64(new_id)),
                ("old", Value::Int64(old_id)),
                ("cues", Value::Json(cues_json)),
                ("vf", Value::String(meta.valid_from.clone())),
                ("vt", Value::String(meta.valid_to.clone())),
                ("tense", Value::String(meta.tense.clone())),
                ("vp", Value::String(meta.viewpoint.clone())),
                ("vpc", Value::Double(meta.viewpoint_confidence)),
            ],
        )?;
        Ok(())
    }

    pub fn link_supersedes(&self, new_id: i64, old_id: i64, reason: &str) -> Result<()> {
        let meta = EdgeMeta::default();
        self.run(
            "MATCH (a:Memory), (b:Memory) WHERE a.id = $new AND b.id = $old CREATE (a)-[:Supersedes {reason: $reason, valid_from: $vf, valid_to: $vt, tense: $tense, viewpoint: $vp, viewpoint_confidence: $vpc}]->(b)",
            vec![
                ("new", Value::Int64(new_id)),
                ("old", Value::Int64(old_id)),
                ("reason", Value::String(reason.to_string())),
                ("vf", Value::String(meta.valid_from.clone())),
                ("vt", Value::String(meta.valid_to.clone())),
                ("tense", Value::String(meta.tense.clone())),
                ("vp", Value::String(meta.viewpoint.clone())),
                ("vpc", Value::Double(meta.viewpoint_confidence)),
            ],
        )?;
        Ok(())
    }

    // ---- reads ----

    pub fn list_entities(&self) -> Result<Vec<KnownEntity>> {
        let rows = self.run("MATCH (e:Entity) RETURN e.id, e.canonical_name, e.entity_type, e.aliases", vec![])?;
        rows.iter().map(|row| row_to_known(row)).collect()
    }

    fn scope_where(&self, alias: &str, user: &str, project: &str, session: Option<&str>) -> (String, Vec<(String, Value)>) {
        let mut where_ = format!("{alias}.user_id = $user AND {alias}.project = $project");
        let mut params: Vec<(String, Value)> = vec![
            ("user".into(), Value::String(user.to_string())),
            ("project".into(), Value::String(project.to_string())),
        ];
        if let Some(s) = session {
            where_.push_str(&format!(" AND {alias}.session_id = $session"));
            params.push(("session".into(), Value::String(s.to_string())));
        }
        (where_, params)
    }

    fn params_to_borrowed(params: &[(String, Value)]) -> Vec<(&str, Value)> {
        params.iter().map(|(k, v)| (k.as_str(), v.clone())).collect()
    }

    /// Active facts sharing any of `entity_ids` (for conflict comparison).
    pub fn active_for_entities(
        &self,
        entity_ids: &[i64],
        user: &str,
        project: &str,
        session: Option<&str>,
        limit: usize,
    ) -> Result<Vec<(i64, String)>> {
        if entity_ids.is_empty() {
            return Ok(vec![]);
        }
        let (where_, params) = self.scope_where("m", user, project, session);
        let ids = entity_ids.iter().map(|i| i.to_string()).collect::<Vec<_>>().join(",");
        let cypher = format!(
            "MATCH (e:Entity)-[:MentionedIn]->(m:Memory) WHERE e.id IN [{ids}] AND m.status = 'active' AND {where_} \
             RETURN m.id, m.content, m.importance ORDER BY m.importance DESC LIMIT {}",
            clamp_limit(limit) * 2
        );
        let rows = self.run(&cypher, Self::params_to_borrowed(&params))?;
        // NB: no RETURN DISTINCT + ORDER BY on an unprojected column — the
        // binder drops `m` from scope after DISTINCT. Dedupe in Rust instead.
        let mut seen = std::collections::HashSet::new();
        let mut out = Vec::new();
        for r in &rows {
            let id = as_i64(&r[0])?;
            if seen.insert(id) {
                out.push((id, as_string(&r[1])?));
            }
            if out.len() >= clamp_limit(limit) {
                break;
            }
        }
        Ok(out)
    }

    /// Content search: FTS (porter stemmer) when the extension loaded,
    /// else case-insensitive word-overlap. Score = FTS score, or fraction
    /// of query words present; the caller adds entity boost.
    pub fn search_content(
        &self,
        query: &str,
        user: &str,
        project: &str,
        session: Option<&str>,
        include_non_active: bool,
        limit: usize,
    ) -> Result<Vec<(MemoryEntry, f64)>> {
        let (where_, params) = self.scope_where("m", user, project, session);
        let status_filter = if include_non_active { "" } else { " AND m.status = 'active'" };
        if self.fts_ready {
            let cypher = format!(
                "CALL QUERY_FTS_INDEX('Memory', 'memory_content_fts', $q, top := {}) WITH node AS m, score WHERE {where_}{status_filter} \
                 RETURN m.id, m.content, m.category, m.user_id, m.project, m.session_id, m.importance, m.status, m.provenance, m.created_at, m.updated_at, score ORDER BY score DESC",
                clamp_limit(limit) * 2
            );
            let mut p = Self::params_to_borrowed(&params);
            p.push(("q", Value::String(query.to_string())));
            match self.run(&cypher, p) {
                Ok(rows) => {
                    return rows.iter().map(|r| Ok((row_to_entry(r)?, as_f64(&r[11])?))).collect();
                }
                Err(_) => { /* fall through to substring on FTS failure */ }
            }
        }
        self.search_content_substring(query, &where_, &params, status_filter, limit)
    }

    fn search_content_substring(
        &self,
        query: &str,
        where_: &str,
        params: &[(String, Value)],
        status_filter: &str,
        limit: usize,
    ) -> Result<Vec<(MemoryEntry, f64)>> {
        let words: Vec<String> = query
            .to_lowercase()
            .split(|c: char| !c.is_alphanumeric())
            .filter(|w| w.len() > 1)
            .map(str::to_string)
            .collect();
        if words.is_empty() {
            return Ok(vec![]);
        }
        // Word-OR with per-word params (all user data stays parameterized).
        let ors: Vec<String> = words
            .iter()
            .enumerate()
            .map(|(i, _)| format!("LOWER(m.content) CONTAINS $w{i}"))
            .collect();
        let cypher = format!(
            "MATCH (m:Memory) WHERE ({}) AND {where_}{status_filter} \
             RETURN m.id, m.content, m.category, m.user_id, m.project, m.session_id, m.importance, m.status, m.provenance, m.created_at, m.updated_at \
             ORDER BY m.importance DESC LIMIT {}",
            ors.join(" OR "),
            clamp_limit(limit) * 3
        );
        let mut p = Self::params_to_borrowed(&params);
        // Param names are generated; stash the owned Strings alongside so the
        // borrowed keys stay alive through `run`.
        let names: Vec<String> = (0..words.len()).map(|i| format!("w{i}")).collect();
        for (name, w) in names.iter().zip(words.iter()) {
            p.push((name.as_str(), Value::String(w.clone())));
        }
        let rows = self.run(&cypher, p)?;
        let mut scored: Vec<(MemoryEntry, f64)> = Vec::new();
        for r in &rows {
            let entry = row_to_entry(r)?;
            let lower = entry.content.to_lowercase();
            let matched = words.iter().filter(|w| lower.contains(w.as_str())).count();
            scored.push((entry, matched as f64 / words.len() as f64));
        }
        scored.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap()
                .then(b.0.importance.cmp(&a.0.importance))
                .then(a.0.id.cmp(&b.0.id))
        });
        scored.truncate(clamp_limit(limit));
        Ok(scored)
    }

    /// Facts mentioning one canonical entity.
    pub fn memories_for_entity(
        &self,
        canonical: &str,
        user: &str,
        project: &str,
        session: Option<&str>,
        limit: usize,
    ) -> Result<Vec<(MemoryEntry, f64)>> {
        let (where_, params) = self.scope_where("m", user, project, session);
        let cypher = format!(
            "MATCH (e:Entity)-[r:MentionedIn]->(m:Memory) WHERE e.canonical_name = $canonical AND m.status = 'active' AND {where_} \
             RETURN m.id, m.content, m.category, m.user_id, m.project, m.session_id, m.importance, m.status, m.provenance, m.created_at, m.updated_at, r.confidence \
             ORDER BY r.confidence DESC LIMIT {}",
            clamp_limit(limit)
        );
        let mut p = Self::params_to_borrowed(&params);
        p.push(("canonical", Value::String(canonical.to_string())));
        let rows = self.run(&cypher, p)?;
        rows.iter().map(|r| Ok((row_to_entry(r)?, as_f64(&r[11])?))).collect()
    }

    pub fn entities_matching(&self, fragment: &str, limit: usize) -> Result<Vec<(String, String, i64)>> {
        let rows = self.run(
            &format!(
                "MATCH (e:Entity) WHERE LOWER(e.canonical_name) CONTAINS $frag OPTIONAL MATCH (e)-[r:MentionedIn]->(:Memory) \
                 WITH e, count(r) AS n RETURN e.canonical_name, e.entity_type, n ORDER BY n DESC LIMIT {}",
                clamp_limit(limit)
            ),
            vec![("frag", Value::String(fragment.to_lowercase()))],
        )?;
        rows.iter()
            .map(|r| Ok((as_string(&r[0])?, as_string(&r[1])?, as_i64(&r[2])?)))
            .collect()
    }

    /// User-scoped, cross-project variants for `global` scope.
    pub fn memories_for_entity_global(
        &self,
        canonical: &str,
        user: &str,
        limit: usize,
    ) -> Result<Vec<(MemoryEntry, f64)>> {
        let cypher = format!(
            "MATCH (e:Entity)-[r:MentionedIn]->(m:Memory) WHERE e.canonical_name = $canonical AND m.status = 'active' AND m.user_id = $user \
             RETURN m.id, m.content, m.category, m.user_id, m.project, m.session_id, m.importance, m.status, m.provenance, m.created_at, m.updated_at, r.confidence \
             ORDER BY r.confidence DESC LIMIT {}",
            clamp_limit(limit)
        );
        let rows = self.run(
            &cypher,
            vec![("canonical", Value::String(canonical.to_string())), ("user", Value::String(user.to_string()))],
        )?;
        rows.iter().map(|r| Ok((row_to_entry(r)?, as_f64(&r[11])?))).collect()
    }

    pub fn search_content_global(
        &self,
        query: &str,
        user: &str,
        limit: usize,
    ) -> Result<Vec<(MemoryEntry, f64)>> {
        if self.fts_ready {
            let cypher = format!(
                "CALL QUERY_FTS_INDEX('Memory', 'memory_content_fts', $q, top := {}) WITH node AS m, score WHERE m.user_id = $user AND m.status = 'active' \
                 RETURN m.id, m.content, m.category, m.user_id, m.project, m.session_id, m.importance, m.status, m.provenance, m.created_at, m.updated_at, score ORDER BY score DESC",
                clamp_limit(limit) * 2
            );
            match self.run(
                &cypher,
                vec![("q", Value::String(query.to_string())), ("user", Value::String(user.to_string()))],
            ) {
                Ok(rows) => {
                    return rows.iter().map(|r| Ok((row_to_entry(r)?, as_f64(&r[11])?))).collect();
                }
                Err(_) => { /* fall through */ }
            }
        }
        let words: Vec<String> = query
            .to_lowercase()
            .split(|c: char| !c.is_alphanumeric())
            .filter(|w| w.len() > 1)
            .map(str::to_string)
            .collect();
        if words.is_empty() {
            return Ok(vec![]);
        }
        let ors: Vec<String> = words
            .iter()
            .enumerate()
            .map(|(i, _)| format!("LOWER(m.content) CONTAINS $w{i}"))
            .collect();
        let cypher = format!(
            "MATCH (m:Memory) WHERE ({}) AND m.user_id = $user AND m.status = 'active' \
             RETURN m.id, m.content, m.category, m.user_id, m.project, m.session_id, m.importance, m.status, m.provenance, m.created_at, m.updated_at \
             ORDER BY m.importance DESC LIMIT {}",
            ors.join(" OR "),
            clamp_limit(limit) * 3
        );
        let mut p: Vec<(&str, Value)> = vec![("user", Value::String(user.to_string()))];
        let names: Vec<String> = (0..words.len()).map(|i| format!("w{i}")).collect();
        for (name, w) in names.iter().zip(words.iter()) {
            p.push((name.as_str(), Value::String(w.clone())));
        }
        let rows = self.run(&cypher, p)?;
        let mut scored: Vec<(MemoryEntry, f64)> = Vec::new();
        for r in &rows {
            let entry = row_to_entry(r)?;
            let lower = entry.content.to_lowercase();
            let matched = words.iter().filter(|w| lower.contains(w.as_str())).count();
            scored.push((entry, matched as f64 / words.len() as f64));
        }
        scored.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap()
                .then(b.0.importance.cmp(&a.0.importance))
                .then(a.0.id.cmp(&b.0.id))
        });
        scored.truncate(clamp_limit(limit));
        Ok(scored)
    }

    pub fn list_global(&self, user: &str, limit: usize) -> Result<Vec<MemoryEntry>> {
        let rows = self.run(
            &format!(
                "MATCH (m:Memory) WHERE m.user_id = $user AND m.status = 'active' \
                 RETURN m.id, m.content, m.category, m.user_id, m.project, m.session_id, m.importance, m.status, m.provenance, m.created_at, m.updated_at \
                 ORDER BY m.category, m.importance DESC LIMIT {}",
                clamp_limit(limit)
            ),
            vec![("user", Value::String(user.to_string()))],
        )?;
        rows.iter().map(row_to_entry).collect()
    }

    /// Exact content+scope match, any status. Used by the SQLite fallback
    /// applier to converge after a partial extension tick (the extension
    /// applies statement-by-statement, so a failed tick can leave committed
    /// node rows behind): a staged `written` fact that already exists
    /// natively is the tick's own row, never a coincidence — the pipeline's
    /// decide step keeps true duplicates out of staging.
    pub fn find_exact_memory(
        &self,
        content: &str,
        user: &str,
        project: &str,
        session: &str,
    ) -> Result<Option<i64>> {
        let rows = self.run(
            "MATCH (m:Memory) WHERE m.content = $c AND m.user_id = $u AND m.project = $p AND m.session_id = $s RETURN m.id LIMIT 1",
            vec![
                ("c", Value::String(content.to_string())),
                ("u", Value::String(user.to_string())),
                ("p", Value::String(project.to_string())),
                ("s", Value::String(session.to_string())),
            ],
        )?;
        rows.first().map(|r| as_i64(&r[0])).transpose()
    }

    /// Whether this exact mention edge already exists (fallback replay guard).
    pub fn has_mention(
        &self,
        entity_id: i64,
        memory_id: i64,
        mention_text: &str,
        span_start: i64,
        span_end: i64,
    ) -> Result<bool> {
        let rows = self.run(
            "MATCH (e:Entity)-[r:MentionedIn]->(m:Memory) WHERE e.id = $eid AND m.id = $mid AND r.mention_text = $t AND r.span_start = $s AND r.span_end = $e RETURN r.confidence LIMIT 1",
            vec![
                ("eid", Value::Int64(entity_id)),
                ("mid", Value::Int64(memory_id)),
                ("t", Value::String(mention_text.to_string())),
                ("s", Value::Int64(span_start)),
                ("e", Value::Int64(span_end)),
            ],
        )?;
        Ok(!rows.is_empty())
    }

    /// Whether a ConflictsWith edge already runs from `new_id` to `old_id`.
    pub fn has_conflict(&self, new_id: i64, old_id: i64) -> Result<bool> {
        Ok(self.conflicts_of(new_id)?.iter().any(|(id, _)| *id == old_id))
    }

    pub fn get_memory(&self, id: i64) -> Result<Option<MemoryEntry>> {
        let rows = self.run(
            "MATCH (m:Memory) WHERE m.id = $id RETURN m.id, m.content, m.category, m.user_id, m.project, m.session_id, m.importance, m.status, m.provenance, m.created_at, m.updated_at",
            vec![("id", Value::Int64(id))],
        )?;
        rows.first().map(row_to_entry).transpose()
    }

    pub fn list_by_category(
        &self,
        user: &str,
        project: &str,
        session: Option<&str>,
        include_non_active: bool,
        limit: usize,
    ) -> Result<Vec<MemoryEntry>> {
        let (where_, params) = self.scope_where("m", user, project, session);
        let status_filter = if include_non_active { "" } else { " AND m.status = 'active'" };
        let cypher = format!(
            "MATCH (m:Memory) WHERE {where_}{status_filter} \
             RETURN m.id, m.content, m.category, m.user_id, m.project, m.session_id, m.importance, m.status, m.provenance, m.created_at, m.updated_at \
             ORDER BY m.category, m.importance DESC LIMIT {}",
            clamp_limit(limit)
        );
        let rows = self.run(&cypher, Self::params_to_borrowed(&params))?;
        rows.iter().map(row_to_entry).collect()
    }

    pub fn entities_for_memory(&self, memory_id: i64) -> Result<Vec<String>> {
        let rows = self.run(
            "MATCH (e:Entity)-[:MentionedIn]->(m:Memory) WHERE m.id = $id RETURN e.canonical_name",
            vec![("id", Value::Int64(memory_id))],
        )?;
        rows.iter().map(|r| as_string(&r[0])).collect()
    }

    pub fn counts(&self) -> Result<(i64, i64, i64, i64)> {
        let mem: i64 = as_i64(&self.run("MATCH (m:Memory) RETURN count(m)", vec![])?[0][0])?;
        let ent: i64 = as_i64(&self.run("MATCH (e:Entity) RETURN count(e)", vec![])?[0][0])?;
        let men: i64 = as_i64(&self.run("MATCH ()-[r:MentionedIn]->() RETURN count(r)", vec![])?[0][0])?;
        let rev: i64 = as_i64(
            &self.run("MATCH (m:Memory) WHERE m.status = 'needs_review' OR m.status = 'quarantined' RETURN count(m)", vec![])?[0][0],
        )?;
        Ok((mem, ent, men, rev))
    }

    pub fn needs_review(&self, limit: usize) -> Result<Vec<MemoryEntry>> {
        let rows = self.run(
            &format!(
                "MATCH (m:Memory) WHERE m.status = 'needs_review' OR m.status = 'quarantined' \
                 RETURN m.id, m.content, m.category, m.user_id, m.project, m.session_id, m.importance, m.status, m.provenance, m.created_at, m.updated_at \
                 ORDER BY m.created_at DESC LIMIT {}",
                clamp_limit(limit)
            ),
            vec![],
        )?;
        rows.iter().map(row_to_entry).collect()
    }

    /// Whole-graph dump for visualization: all entities + memories and
    /// every relationship edge, with demo-scale caps. Entity and Memory ids
    /// share one SERIAL space per table but not across tables — disambiguate
    /// with `SnapshotNodeKind` when building graph keys.
    pub fn graph_snapshot(&self, node_limit: usize, edge_limit: usize) -> Result<GraphSnapshot> {
        let node_limit = node_limit.clamp(1, 2000);
        let edge_limit = edge_limit.clamp(1, 5000);
        let mut nodes = Vec::new();
        let mut links = Vec::new();

        let rows = self.run(
            &format!(
                "MATCH (e:Entity) RETURN e.id, e.canonical_name, e.entity_type LIMIT {node_limit}"
            ),
            vec![],
        )?;
        for r in &rows {
            nodes.push(SnapshotNode {
                id: as_i64(&r[0])?,
                kind: SnapshotNodeKind::Entity,
                name: as_string(&r[1]).unwrap_or_default(),
                label: as_string(&r[2]).unwrap_or_else(|_| "concept".into()),
                status: String::new(),
            });
        }
        let rows = self.run(
            &format!(
                "MATCH (m:Memory) RETURN m.id, m.content, m.category, m.status LIMIT {node_limit}"
            ),
            vec![],
        )?;
        for r in &rows {
            let content = as_string(&r[1]).unwrap_or_default();
            let name: String = content.chars().take(48).collect();
            nodes.push(SnapshotNode {
                id: as_i64(&r[0])?,
                kind: SnapshotNodeKind::Memory,
                name,
                label: as_string(&r[2]).unwrap_or_else(|_| "general".into()),
                status: as_string(&r[3]).unwrap_or_default(),
            });
        }

        // Undirected match returns each RelatesTo once.
        let edge_queries = [
            (
                format!(
                    "MATCH (e:Entity)-[r:MentionedIn]->(m:Memory) RETURN e.id, m.id, r.mention_text LIMIT {edge_limit}"
                ),
                "MentionedIn",
                SnapshotNodeKind::Entity,
                SnapshotNodeKind::Memory,
            ),
            (
                format!(
                    "MATCH (a:Entity)-[r:RelatesTo]-(b:Entity) RETURN a.id, b.id, r.relation_type LIMIT {edge_limit}"
                ),
                "RelatesTo",
                SnapshotNodeKind::Entity,
                SnapshotNodeKind::Entity,
            ),
            (
                format!(
                    "MATCH (a:Memory)-[r:ConflictsWith]->(b:Memory) RETURN a.id, b.id, r.cues LIMIT {edge_limit}"
                ),
                "ConflictsWith",
                SnapshotNodeKind::Memory,
                SnapshotNodeKind::Memory,
            ),
            (
                format!(
                    "MATCH (a:Memory)-[r:Supersedes]->(b:Memory) RETURN a.id, b.id, r.reason LIMIT {edge_limit}"
                ),
                "Supersedes",
                SnapshotNodeKind::Memory,
                SnapshotNodeKind::Memory,
            ),
        ];
        for (cypher, default_label, src_kind, dst_kind) in edge_queries {
            let rows = self.run(&cypher, vec![])?;
            for r in &rows {
                let label = as_string(&r[2]).unwrap_or_else(|_| default_label.into());
                let label = if label.is_empty() { default_label.into() } else { label };
                // Cues JSON reads noisy on an edge label; keep the kind.
                let label = if default_label == "ConflictsWith" {
                    default_label.into()
                } else if label.len() > 40 {
                    label.chars().take(40).collect()
                } else {
                    label
                };
                links.push(SnapshotLink {
                    src: (as_i64(&r[0])?, src_kind),
                    dst: (as_i64(&r[1])?, dst_kind),
                    label,
                });
            }
        }
        Ok(GraphSnapshot { nodes, links })
    }

    /// All RelatesTo edges as `(src_canonical, dst_canonical, relation_type)`.
    /// Test/audit support (demo graph inspection included).
    #[cfg(any(test, feature = "test-support"))]
    pub fn relates_to_labels(&self) -> Result<Vec<(String, String, String)>> {
        let rows = self.run(
            "MATCH (a:Entity)-[r:RelatesTo]->(b:Entity) RETURN a.canonical_name, b.canonical_name, r.relation_type",
            vec![],
        )?;
        rows.iter().map(|r| Ok((as_string(&r[0])?, as_string(&r[1])?, as_string(&r[2])?))).collect()
    }

    /// Full RelatesTo edge dump for tests: (src, dst, type, confidence,
    /// valid_from, valid_to, tense, viewpoint, viewpoint_confidence).
    #[cfg(any(test, feature = "test-support"))]
    pub fn relates_to_full(
        &self,
    ) -> Result<Vec<(String, String, String, f64, String, String, String, String, f64)>> {
        let rows = self.run(
            "MATCH (a:Entity)-[r:RelatesTo]->(b:Entity) RETURN a.canonical_name, b.canonical_name, r.relation_type, r.confidence, r.valid_from, r.valid_to, r.tense, r.viewpoint, r.viewpoint_confidence",
            vec![],
        )?;
        rows.iter()
            .map(|r| {
                Ok((
                    as_string(&r[0])?,
                    as_string(&r[1])?,
                    as_string(&r[2])?,
                    as_f64(&r[3])?,
                    as_string(&r[4]).unwrap_or_default(),
                    as_string(&r[5]).unwrap_or_default(),
                    as_string(&r[6]).unwrap_or_default(),
                    as_string(&r[7]).unwrap_or_default(),
                    as_f64(&r[8]).unwrap_or(0.0),
                ))
            })
            .collect()
    }

    /// Most recently touched entities, newest first (by `updated_at`).
    /// Discourse context for anaphora and implicit-argument resolution.
    pub fn recent_entities(&self, limit: usize) -> Result<Vec<KnownEntity>> {
        let rows = self.run(
            &format!(
                "MATCH (e:Entity) RETURN e.id, e.canonical_name, e.entity_type, e.aliases ORDER BY e.updated_at DESC LIMIT {}",
                limit.clamp(1, 50)
            ),
            vec![],
        )?;
        rows.iter().map(|row| row_to_known(row)).collect()
    }

    pub fn conflicts_of(&self, memory_id: i64) -> Result<Vec<(i64, Vec<String>)>> {
        let rows = self.run(
            "MATCH (a:Memory)-[c:ConflictsWith]->(b:Memory) WHERE a.id = $id RETURN b.id, c.cues",
            vec![("id", Value::Int64(memory_id))],
        )?;
        rows.iter()
            .map(|r| {
                let cues = match &r[1] {
                    Value::Json(JsonValue::Array(arr)) => {
                        arr.iter().filter_map(|v| v.as_str().map(str::to_string)).collect()
                    }
                    _ => vec![],
                };
                Ok((as_i64(&r[0])?, cues))
            })
            .collect()
    }
}

/// Escape a filesystem path for embedding in a single-quoted Cypher literal.
fn cypher_path_lit(path: &str) -> String {
    path.replace('\\', "\\\\").replace('\'', "''")
}

/// One node of [`Store::graph_snapshot`].
#[derive(Debug, Clone)]
pub struct SnapshotNode {
    /// Native row id (unique per table, not across tables — pair with `kind`).
    pub id: i64,
    pub kind: SnapshotNodeKind,
    /// Display name: canonical name (entities) or truncated content (memories).
    pub name: String,
    /// Entity type / memory category (drives color).
    pub label: String,
    /// Memory lifecycle status; empty for entities.
    pub status: String,
}

/// Table a [`SnapshotNode`] / [`SnapshotLink`] endpoint lives in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SnapshotNodeKind {
    Entity,
    Memory,
}

/// One edge of [`Store::graph_snapshot`]: native endpoint ids + their tables.
#[derive(Debug, Clone)]
pub struct SnapshotLink {
    pub src: (i64, SnapshotNodeKind),
    pub dst: (i64, SnapshotNodeKind),
    pub label: String,
}

/// Whole-graph dump; see [`Store::graph_snapshot`].
#[derive(Debug, Clone, Default)]
pub struct GraphSnapshot {
    pub nodes: Vec<SnapshotNode>,
    pub links: Vec<SnapshotLink>,
}

fn clamp_limit(limit: usize) -> usize {
    limit.clamp(1, 100)
}

fn first_id(rows: &[Vec<Value>]) -> Result<i64> {
    rows.first().map(|r| as_i64(&r[0])).transpose()?.ok_or_else(|| anyhow!("no id returned"))
}

pub(crate) fn row_to_known(row: &Vec<Value>) -> Result<KnownEntity> {
    let id = as_i64(&row[0])?;
    let canonical = as_string(&row[1])?;
    let entity_type = as_string(&row[2])?;
    let aliases: Vec<String> = match &row[3] {
        Value::Json(JsonValue::Array(arr)) => arr.iter().filter_map(|v| v.as_str().map(str::to_string)).collect(),
        _ => vec![canonical.clone()],
    };
    let aliases = if aliases.is_empty() { vec![canonical.clone()] } else { aliases };
    Ok(KnownEntity {
        id,
        normalized: crate::resolve::normalize(&canonical),
        canonical,
        entity_type,
        alias_normalized: aliases.iter().map(|a| crate::resolve::normalize(a)).collect(),
        aliases,
    })
}

pub fn as_i64(v: &Value) -> Result<i64> {
    match v {
        Value::Int64(i) => Ok(*i),
        Value::Int32(i) => Ok(*i as i64),
        Value::Int16(i) => Ok(*i as i64),
        Value::Int8(i) => Ok(*i as i64),
        Value::UInt64(i) => Ok(*i as i64),
        Value::UInt32(i) => Ok(*i as i64),
        Value::UInt16(i) => Ok(*i as i64),
        Value::UInt8(i) => Ok(*i as i64),
        other => Err(anyhow!("expected int, got {other:?}")),
    }
}

pub fn as_string(v: &Value) -> Result<String> {
    match v {
        Value::String(s) => Ok(s.clone()),
        other => Err(anyhow!("expected string, got {other:?}")),
    }
}

pub fn as_f64(v: &Value) -> Result<f64> {
    match v {
        Value::Double(f) => Ok(*f),
        Value::Float(f) => Ok(*f as f64),
        Value::Int64(i) => Ok(*i as f64),
        other => Err(anyhow!("expected double, got {other:?}")),
    }
}

fn as_timestamp_string(v: &Value) -> String {
    match v {
        Value::Timestamp(ts) => ts.format(&time::format_description::well_known::Rfc3339).unwrap_or_default(),
        Value::String(s) => s.clone(),
        _ => String::new(),
    }
}

fn row_to_entry(row: &Vec<Value>) -> Result<MemoryEntry> {
    let provenance: Provenance = match &row[8] {
        Value::Json(j) => serde_json::from_value(j.clone()).unwrap_or_default(),
        _ => Provenance::default(),
    };
    let session = as_string(&row[5]).unwrap_or_default();
    Ok(MemoryEntry {
        id: as_i64(&row[0])?,
        content: as_string(&row[1])?,
        category: Category::parse(&as_string(&row[2]).unwrap_or_default()),
        user_id: as_string(&row[3]).unwrap_or_default(),
        project: as_string(&row[4]).unwrap_or_default(),
        session_id: if session.is_empty() { None } else { Some(session) },
        importance: as_i64(&row[6]).unwrap_or(5),
        status: MemoryStatus::parse(&as_string(&row[7]).unwrap_or_default()),
        provenance,
        created_at: as_timestamp_string(&row[9]),
        updated_at: as_timestamp_string(&row[10]),
    })
}
