//! The five-step memory pipeline.
//!
//! ```text
//! remember(text):
//!   1. extract   — GLiNER2.5 per logical chunk (spans rebased to the fact)
//!   2. compare   — mentions vs. the stored Entity schema
//!   3. resolve   — merge onto canonical nodes (aliases grow, names don't fork)
//!   4. decide    — duplicate / conflict / low-confidence / new
//!   5. write     — Memory node + MentionedIn + co-occurrence + provenance
//! ```
//!
//! Only distilled facts are written. Chunks are ephemeral; session message
//! content stays in pi's sqlite — provenance keeps `message_refs` (ids), not
//! text.

use std::collections::{HashMap, HashSet};

use anyhow::Result;
use regex::Regex;

use crate::chunker::{LogicalChunker, split_sentences};
use crate::classify::{classify, is_acknowledgment};
use crate::conflicts::{FactVerdict, decide};
use crate::model::Extractor;
use crate::resolve::{KnownEntity, compare, resolve};
use crate::store::Store;
use crate::types::{
    Category, ComparedEntity, FactDecision, MemoryEntry, MemoryStatus, Provenance, RememberReport,
    ResolvedEntity, ScoredMention, Scope, SearchHit,
};

/// Steps 2-4 output: everything `commit_plan` (direct write) or
/// `stage_plan` (SQLite staging) needs, with no graph writes yet.
struct FactPlan {
    fact_text: String,
    extracted: Vec<ScoredMention>,
    compared: Vec<ComparedEntity>,
    resolved: Vec<ResolvedEntity>,
    category: Category,
    status: MemoryStatus,
    decision: FactDecision,
    conflicts: Vec<(i64, Vec<String>)>,
    duplicate_of: Option<(i64, f32)>,
    importance: i64,
}

/// Labels for model-based fact categorization (must match `Category`).
/// Used only when rule cues abstain (`General`) — deterministic first,
/// model as fallback.
const CATEGORY_TASK: &str = "category";
fn category_labels() -> Vec<String> {
    ["identity", "preferences", "goals", "projects", "decisions", "technical", "relationships", "routines", "lessons", "work", "general"]
        .iter()
        .map(|s| s.to_string())
        .collect()
}

pub struct PipelineOptions {
    /// GLiNER threshold for keeping a mention.
    pub extract_threshold: f32,
    /// Default importance for new facts.
    pub importance: i64,
    /// Max chunk size for extraction scoping.
    pub max_chunk: usize,
}

impl Default for PipelineOptions {
    fn default() -> Self {
        Self { extract_threshold: 0.5, importance: 5, max_chunk: 2000 }
    }
}

pub struct Pipeline<'s> {
    store: &'s Store,
    opts: PipelineOptions,
}

impl<'s> Pipeline<'s> {
    pub fn new(store: &'s Store, opts: PipelineOptions) -> Self {
        Self { store, opts }
    }

    // ---- step 1 ----

    /// Extract mentions per chunk, rebasing spans to fact-relative offsets.
    /// Returns `(fact_text, fact_relative_mentions)` per fact.
    fn extract_facts(
        &self,
        extractor: &mut dyn Extractor,
        text: &str,
        per_sentence: bool,
    ) -> Result<Vec<(String, Vec<ScoredMention>)>> {
        if !per_sentence {
            return Ok(vec![self.extract_one(extractor, text, 0)?]);
        }
        let mut facts = Vec::new();
        for (sent, base) in split_sentences(text) {
            if sent.trim().is_empty() {
                continue;
            }
            // Worthiness pre-filter: bare acknowledgments carry no durable
            // content. Explicit `remember` never reaches this branch.
            if is_acknowledgment(&sent) {
                continue;
            }
            let (fact_text, mentions) = self.extract_one(extractor, &sent, base)?;
            if !mentions.is_empty() {
                facts.push((fact_text, mentions));
            }
        }
        Ok(facts)
    }

    /// Extract over chunked text; `char_base` rebases chunk spans into the
    /// caller's coordinate space, then into fact-relative space.
    fn extract_one(
        &self,
        extractor: &mut dyn Extractor,
        fact_text: &str,
        char_base: usize,
    ) -> Result<(String, Vec<ScoredMention>)> {
        let to_char = byte_to_char_map(fact_text);
        let chunker = LogicalChunker { max_unit_size: self.opts.max_chunk, ..Default::default() };
        let mut best: HashMap<(String, String), ScoredMention> = HashMap::new();
        for unit in chunker.chunk(fact_text) {
            // Chunk byte offsets -> fact char offsets.
            let chunk_char_start = byte_to_char(&to_char, unit.start);
            for mut m in extractor.extract(&unit.text, self.opts.extract_threshold)? {
                // Generic surfaces ("User" in every preference) carry no
                // graph information; drop them before resolution.
                if is_generic_surface(&m.text) {
                    continue;
                }
                // Mention spans are relative to the chunk text as extracted
                // (gliner may append terminal punctuation when normalizing;
                // entity spans are unaffected). Clamp defensively.
                let chunk_len = unit.text.chars().count();
                if m.start > chunk_len || m.end > chunk_len + 1 {
                    continue;
                }
                m.start = (char_base + chunk_char_start + m.start).saturating_sub(char_base);
                m.end = (char_base + chunk_char_start + m.end).saturating_sub(char_base);
                let key = (m.text.clone(), m.entity_type.clone());
                best.entry(key)
                    .and_modify(|prev| {
                        if m.confidence > prev.confidence {
                            *prev = m.clone();
                        }
                    })
                    .or_insert(m);
            }
        }
        let mut mentions: Vec<ScoredMention> = best.into_values().collect();
        augment_code_identifiers(fact_text, &to_char, &mut mentions);
        mentions.sort_by(|a, b| {
            b.confidence.partial_cmp(&a.confidence).unwrap().then(a.start.cmp(&b.start))
        });
        Ok((fact_text.to_string(), mentions))
    }

    // ---- full remember ----

    /// Explicit remember: the whole text is one fact.
    pub fn remember(
        &self,
        extractor: &mut dyn Extractor,
        text: &str,
        scope: &Scope,
        importance: Option<i64>,
        provenance: Provenance,
    ) -> Result<Vec<RememberReport>> {
        let facts = self.extract_facts(extractor, text, false)?;
        self.write_facts(extractor, facts, scope, importance, provenance)
    }

    /// Auto-capture: distill to one fact per entity-bearing sentence.
    /// Sentences without entities are dropped — that is the anti-duplication
    /// rule (pi's sqlite keeps the transcript; we keep facts).
    pub fn remember_auto(
        &self,
        extractor: &mut dyn Extractor,
        text: &str,
        scope: &Scope,
        provenance: Provenance,
    ) -> Result<Vec<RememberReport>> {
        let facts = self.extract_facts(extractor, text, true)?;
        let mut prov = provenance;
        prov.auto_captured = true;
        self.write_facts(extractor, facts, scope, None, prov)
    }

    fn write_facts(
        &self,
        extractor: &mut dyn Extractor,
        facts: Vec<(String, Vec<ScoredMention>)>,
        scope: &Scope,
        importance: Option<i64>,
        provenance: Provenance,
    ) -> Result<Vec<RememberReport>> {
        let known: Vec<KnownEntity> = self.store.list_entities()?;
        let mut reports = Vec::new();
        // New entities created earlier in this call are visible to later facts.
        let mut known = known;
        for (fact_text, extracted) in facts {
            reports.push(self.write_one(extractor, fact_text, extracted, scope, importance, &provenance, &mut known)?);
        }
        Ok(reports)
    }

    fn write_one(
        &self,
        extractor: &mut dyn Extractor,
        fact_text: String,
        extracted: Vec<ScoredMention>,
        scope: &Scope,
        importance: Option<i64>,
        provenance: &Provenance,
        known: &mut Vec<KnownEntity>,
    ) -> Result<RememberReport> {
        let plan = self.plan_one(extractor, fact_text, extracted, scope, importance, known)?;
        self.commit_plan(plan, scope, provenance, known)
    }

    /// Steps 2-4 of the pipeline: compare + resolve against the live schema,
    /// categorize, decide. Pure planning — no graph writes — so the same plan
    /// feeds either the direct commit or the SQLite staging path.
    fn plan_one(
        &self,
        extractor: &mut dyn Extractor,
        fact_text: String,
        extracted: Vec<ScoredMention>,
        scope: &Scope,
        importance: Option<i64>,
        known: &[KnownEntity],
    ) -> Result<FactPlan> {
        // Steps 2-3: compare + resolve against the live schema.
        let compared = compare(&extracted, known);
        let resolved = resolve(&compared, known);
        // Categorize: deterministic cues first, model fallback when they
        // abstain. The model is already loaded in the remember path, so
        // this is one extra encoder pass (~20 ms), not a new dependency.
        let mut category = classify(&fact_text);
        if category == Category::General {
            if let Ok(scores) = extractor.classify(&fact_text, CATEGORY_TASK, &category_labels()) {
                if let Some((best, _)) = scores.iter().max_by(|a, b| a.1.partial_cmp(&b.1).unwrap()) {
                    category = Category::parse(best);
                }
            }
        }
        let session = scope.session_id.as_deref();

        // Step 4: decide.
        let best_confidence = extracted.iter().map(|m| m.confidence).fold(0.0f32, f32::max);
        let mut decision: FactDecision;
        let mut status: MemoryStatus;
        let mut conflicts: Vec<(i64, Vec<String>)> = Vec::new();
        let mut duplicate_of: Option<(i64, f32)> = None;

        if extracted.is_empty() {
            decision = FactDecision::LowConfidence { reason: "no entity extracted above threshold".into() };
            status = MemoryStatus::Quarantined;
        } else {
            let entity_ids: Vec<i64> = resolved.iter().filter_map(|r| r.id).collect();
            let similar = self.store.active_for_entities(
                &entity_ids,
                &scope.user_id,
                &scope.project,
                session,
                10,
            )?;
            match decide(&fact_text, &similar) {
                FactVerdict::New => {
                    decision = FactDecision::New { reason: "no similar active fact".into() };
                    status = MemoryStatus::Active;
                }
                FactVerdict::Duplicate { of, similarity } => {
                    decision = FactDecision::Duplicate { of, similarity };
                    duplicate_of = Some((of, similarity));
                    status = MemoryStatus::Active; // existing stays active; nothing written
                }
                FactVerdict::Conflict { with } => {
                    let cues: Vec<String> = with.iter().flat_map(|(_, c)| c.clone()).collect();
                    decision = FactDecision::Conflict { with: with.iter().map(|(id, _)| *id).collect(), cues };
                    conflicts = with;
                    status = MemoryStatus::NeedsReview;
                }
            }
            // Weak extraction overall → quarantine even a "new" fact.
            if best_confidence < 0.35 && matches!(decision, FactDecision::New { .. }) {
                decision = FactDecision::LowConfidence {
                    reason: format!("best entity confidence {best_confidence:.2} below 0.35"),
                };
                status = MemoryStatus::Quarantined;
            }
        }

        Ok(FactPlan {
            fact_text,
            extracted,
            compared,
            resolved,
            category,
            status,
            decision,
            conflicts,
            duplicate_of,
            importance: importance.unwrap_or(self.opts.importance),
        })
    }

    /// Step 5 of the pipeline: write the graph + provenance.
    fn commit_plan(
        &self,
        plan: FactPlan,
        scope: &Scope,
        provenance: &Provenance,
        known: &mut Vec<KnownEntity>,
    ) -> Result<RememberReport> {
        let FactPlan {
            fact_text,
            extracted,
            compared,
            resolved,
            category,
            status,
            decision,
            conflicts,
            duplicate_of,
            importance,
        } = plan;
        let session = scope.session_id.as_deref();
        // Step 5: write the graph + provenance.
        let written: Option<MemoryEntry>;
        if let Some((of, _)) = duplicate_of {
            self.store.touch_memory(of, 1)?;
            written = self.store.get_memory(of)?;
        } else {
            let id = self.store.insert_memory(
                &fact_text,
                category,
                &scope.user_id,
                &scope.project,
                session.unwrap_or(""),
                importance,
                status,
                provenance,
            )?;
            // Persist resolved entities (new nodes + alias growth), then edges.
            let mut entity_ids: Vec<i64> = Vec::new();
            // normalized name-or-alias -> (entity id, mention confidence).
            let mut node_of: HashMap<String, (i64, f32)> = HashMap::new();
            for r in &resolved {
                let (eid, created) = self.store.upsert_entity(
                    &r.canonical,
                    &r.entity_type,
                    &r.aliases,
                    r.id,
                )?;
                entity_ids.push(eid);
                if created {
                    known.push(KnownEntity {
                        id: eid,
                        canonical: r.canonical.clone(),
                        normalized: crate::resolve::normalize(&r.canonical),
                        entity_type: r.entity_type.clone(),
                        alias_normalized: r.aliases.iter().map(|a| crate::resolve::normalize(a)).collect(),
                        aliases: r.aliases.clone(),
                    });
                }
                let conf = compared
                    .iter()
                    .find(|c| c.normalized == crate::resolve::normalize(&r.canonical))
                    .map(|c| c.mention.confidence)
                    .unwrap_or(0.5);
                node_of.insert(crate::resolve::normalize(&r.canonical), (eid, conf));
                for a in &r.aliases {
                    node_of.entry(crate::resolve::normalize(a)).or_insert((eid, conf));
                }
            }
            for c in &compared {
                if let Some((eid, conf)) = node_of.get(&c.normalized) {
                    self.store.link_mention(
                        *eid,
                        id,
                        &c.mention.text,
                        *conf as f64,
                        c.mention.start as i64,
                        c.mention.end as i64,
                    )?;
                }
            }
            for a in 0..entity_ids.len() {
                for b in (a + 1)..entity_ids.len() {
                    self.store.link_cooccur(entity_ids[a], entity_ids[b], 0.5)?;
                }
            }
            for (old, cues) in &conflicts {
                self.store.link_conflict(id, *old, cues)?;
            }
            written = self.store.get_memory(id)?;
        }

        Ok(RememberReport { extracted, compared, resolved, decision, written })
    }

    // ---- sqlite-first path ----
    //
    // Every remember is first a SQLite row (`memory_intake`); the fact goes
    // through GLiNER, is staged into the extension-convention `node_*` /
    // `rel_*` tables, and a sync tick replicates it into the Ladybug graph.

    /// Explicit remember through SQLite: intake -> GLiNER -> stage -> sync.
    /// Returns per-fact reports plus the sync tick stats.
    pub fn remember_via_sqlite(
        &self,
        extractor: &mut dyn Extractor,
        intake: &crate::sqlite_sync::IntakeDb,
        text: &str,
        scope: &Scope,
        importance: Option<i64>,
        provenance: Provenance,
        extension: Option<crate::sqlite_sync::ExtensionSource<'_>>,
    ) -> Result<(Vec<RememberReport>, crate::sqlite_sync::SyncStats)> {
        let facts = self.extract_facts(extractor, text, false)?;
        let staged = self.stage_facts(extractor, intake, facts, scope, importance, provenance)?;
        let stats = self.sync_intake(intake, extension)?;
        // The tick just wrote the staged facts; attach them so the reports
        // match the direct path shape.
        let reports = self.attach_staged_writes(intake, staged)?;
        Ok((reports, stats))
    }

    /// Auto-capture through SQLite: distill to one fact per entity-bearing
    /// sentence, then stage + sync like `remember_via_sqlite`.
    pub fn remember_auto_via_sqlite(
        &self,
        extractor: &mut dyn Extractor,
        intake: &crate::sqlite_sync::IntakeDb,
        text: &str,
        scope: &Scope,
        provenance: Provenance,
        extension: Option<crate::sqlite_sync::ExtensionSource<'_>>,
    ) -> Result<(Vec<RememberReport>, crate::sqlite_sync::SyncStats)> {
        let facts = self.extract_facts(extractor, text, true)?;
        let mut prov = provenance;
        prov.auto_captured = true;
        let staged = self.stage_facts(extractor, intake, facts, scope, None, prov)?;
        let stats = self.sync_intake(intake, extension)?;
        let reports = self.attach_staged_writes(intake, staged)?;
        Ok((reports, stats))
    }

    /// Stage each planned fact; returns `(report, intake_id)` pairs so the
    /// post-sync attach can resolve the written native rows.
    fn stage_facts(
        &self,
        extractor: &mut dyn Extractor,
        intake: &crate::sqlite_sync::IntakeDb,
        facts: Vec<(String, Vec<ScoredMention>)>,
        scope: &Scope,
        importance: Option<i64>,
        provenance: Provenance,
    ) -> Result<Vec<(RememberReport, i64)>> {
        let known: Vec<KnownEntity> = self.store.list_entities()?;
        let mut known = known;
        let mut staged = Vec::new();
        for (fact_text, extracted) in facts {
            let plan = self.plan_one(extractor, fact_text, extracted, scope, importance, &known)?;
            // stage_plan returns the report (`written: None` for
            // replicated facts) plus its intake id; the attach step resolves
            // the native row after the sync tick.
            let (report, intake_id) = self.stage_plan(intake, plan, scope, &provenance, &mut known)?;
            staged.push((report, intake_id));
        }
        Ok(staged)
    }

    /// Stage one planned fact into SQLite: intake row, `node_*` / `rel_*`
    /// rows, outcome + provenance side rows. Duplicate decisions stage no
    /// graph rows — the strengthening touch applies live (idempotent under
    /// cursor-gated replay) and is recorded in the outcome row.
    /// Returns the report (`written` filled for duplicates; `None` for
    /// replicated facts until the sync tick) plus its intake id.
    fn stage_plan(
        &self,
        intake: &crate::sqlite_sync::IntakeDb,
        plan: FactPlan,
        scope: &Scope,
        provenance: &Provenance,
        known: &mut Vec<KnownEntity>,
    ) -> Result<(RememberReport, i64)> {
        let FactPlan {
            fact_text,
            extracted,
            compared,
            resolved,
            category,
            status,
            decision,
            conflicts,
            duplicate_of,
            importance,
        } = plan;
        let session = scope.session_id.as_deref().unwrap_or("");
        let intake_id = intake.append_intake(&fact_text, &scope.user_id, &scope.project, session, importance)?;
        let prov_json = serde_json::to_value(provenance).unwrap_or(serde_json::Value::Null);
        let mut report = RememberReport { extracted, compared, resolved, decision, written: None };
        if let Some((of, similarity)) = duplicate_of {
            // No live write: the sync tick owns the strengthening touch
            // (embedded applier, or the extension pre-pass below), so a
            // cursor-gated replay can converge instead of double-counting.
            report.written = self.store.get_memory(of)?;
            intake.record_outcome(
                intake_id,
                &serde_json::json!({"kind": "duplicate", "of": of, "similarity": similarity}).to_string(),
            )?;
            return Ok((report, intake_id));
        }
        let fact_idx = 0;
        let staged_mem = intake.stage_memory(
            intake_id,
            fact_idx,
            &fact_text,
            category.as_str(),
            &scope.user_id,
            &scope.project,
            session,
            importance,
            status.as_str(),
        )?;
        intake.record_provenance(intake_id, fact_idx, &prov_json.to_string())?;
        // Stage entities (known graph nodes keep their native id so both
        // appliers converge; new ones take staging ids, backfilled on sync).
        // normalized name-or-alias -> staging id.
        let mut node_of: HashMap<String, i64> = HashMap::new();
        let mut entity_ids: Vec<i64> = Vec::new();
        for r in &report.resolved {
            let aliases_json =
                serde_json::to_string(&r.aliases).unwrap_or_else(|_| "[]".to_string());
            let staged = intake.stage_entity(intake_id, &r.canonical, &r.entity_type, &aliases_json, r.id)?;
            entity_ids.push(staged);
            if r.id.is_none() {
                // New canonical name; later facts in this call reuse it.
                known.push(KnownEntity {
                    id: staged,
                    canonical: r.canonical.clone(),
                    normalized: crate::resolve::normalize(&r.canonical),
                    entity_type: r.entity_type.clone(),
                    alias_normalized: r.aliases.iter().map(|a| crate::resolve::normalize(a)).collect(),
                    aliases: r.aliases.clone(),
                });
            }
            node_of.insert(crate::resolve::normalize(&r.canonical), staged);
            for a in &r.aliases {
                node_of.entry(crate::resolve::normalize(a)).or_insert(staged);
            }
        }
        for c in &report.compared {
            if let Some(staged) = node_of.get(&c.normalized) {
                intake.stage_mention(
                    intake_id,
                    *staged,
                    staged_mem,
                    &c.mention.text,
                    c.mention.confidence as f64,
                    c.mention.start as i64,
                    c.mention.end as i64,
                )?;
            }
        }
        for a in 0..entity_ids.len() {
            for b in (a + 1)..entity_ids.len() {
                intake.stage_cooccur(intake_id, entity_ids[a], entity_ids[b], 0.5)?;
            }
        }
        for (old, old_cues) in &conflicts {
            let cues_json = serde_json::to_string(old_cues).unwrap_or_else(|_| "[]".to_string());
            intake.stage_conflict(intake_id, staged_mem, *old, &cues_json)?;
        }
        intake.record_outcome(
            intake_id,
            &serde_json::json!({"kind": "written", "status": status.as_str()}).to_string(),
        )?;
        Ok((report, intake_id))
    }

    /// One replication tick over `intake`: the `sqlite` extension when
    /// `extension` is given and healthy, else the embedded applier.
    /// Extension errors fall back to embedded (reported in `via`) so a
    /// failed INSTALL/LOAD/sync never loses a remember.
    pub fn sync_intake(
        &self,
        intake: &crate::sqlite_sync::IntakeDb,
        extension: Option<crate::sqlite_sync::ExtensionSource<'_>>,
    ) -> Result<crate::sqlite_sync::SyncStats> {
        if intake.pending_count()? == 0 {
            return Ok(crate::sqlite_sync::SyncStats {
                tables_synced: 0,
                rows_upserted: 0,
                rows_deleted: 0,
                applied_intake: 0,
                status: "caught up".into(),
                via: "none".into(),
            });
        }
        if let Some(source) = extension {
            match self.sync_via_extension(intake, source) {
                Ok(stats) => return Ok(stats),
                Err(e) => {
                    let mut stats = crate::sqlite_sync::replicate_to_store(intake, self.store)?;
                    stats.via = format!("embedded(fallback after extension error: {e:#})");
                    return Ok(stats);
                }
            }
        }
        crate::sqlite_sync::replicate_to_store(intake, self.store)
    }

    /// Replicate via the C++ `sqlite` extension: LOAD + explicit label
    /// remapping + `CALL sqlite_sync`. No writes happen here in Rust; the
    /// engine tails the SQLite WAL and applies row ops itself.
    ///
    /// Post-tick bookkeeping keeps the two appliers coherent: under the
    /// extension's MERGE-by-PK semantics a staged row's native id *is* its
    /// staging id, so `lb_id` backfill is exact and the intake cursor can
    /// advance past the converged intakes.
    pub fn sync_via_extension(
        &self,
        intake: &crate::sqlite_sync::IntakeDb,
        source: crate::sqlite_sync::ExtensionSource<'_>,
    ) -> Result<crate::sqlite_sync::SyncStats> {
        if intake.path() == ":memory:" {
            anyhow::bail!("the sqlite extension needs a file-backed intake database");
        }
        match source {
            crate::sqlite_sync::ExtensionSource::Repo => self.store.ensure_sqlite_extension()?,
            crate::sqlite_sync::ExtensionSource::File(path) => self.store.load_sqlite_extension(path)?,
        }
        let _statuses = self.store.sqlite_register_all(intake.path())?;
        // Duplicate outcomes carry no graph rows; apply their strengthening
        // touches in Rust so both appliers converge identically.
        for intake_id in intake.pending_intake_ids()? {
            if let Some((kind, of)) = intake.outcome(intake_id)? {
                if kind == "duplicate" {
                    if let Some(of) = of {
                        let _ = self.store.touch_memory(of, 1);
                    }
                }
            }
        }
        let (tables, upserted, deleted, _applied_frames, status) =
            self.store.sqlite_sync_tick(intake.path())?;
        let pending = intake.pending_intake_ids()?;
        for intake_id in &pending {
            intake.backfill_extension_ids(*intake_id)?;
        }
        if let Some(last) = pending.last() {
            intake.mark_applied(*last)?;
        }
        Ok(crate::sqlite_sync::SyncStats {
            tables_synced: tables,
            rows_upserted: upserted,
            rows_deleted: deleted,
            applied_intake: pending.len() as i64,
            status,
            via: "extension".into(),
        })
    }

    /// Fill `written` on staged reports after a sync tick by reading back
    /// the native rows through the intake side-map. Duplicate reports
    /// already carry their strengthened entry from staging time.
    fn attach_staged_writes(
        &self,
        intake: &crate::sqlite_sync::IntakeDb,
        staged: Vec<(RememberReport, i64)>,
    ) -> Result<Vec<RememberReport>> {
        let mut reports = Vec::with_capacity(staged.len());
        for (mut report, intake_id) in staged {
            if report.written.is_none() {
                let ids = intake.staged_memory_lb_ids(intake_id)?;
                // One memory per intake row in the current staging shape;
                // take the first converged row.
                for id in ids {
                    if let Some(entry) = self.store.get_memory(id)? {
                        report.written = Some(entry);
                        break;
                    }
                }
            } else if let Some((kind, _)) = intake.outcome(intake_id)? {
                if kind == "duplicate" {
                    // Re-read: the sync tick applied the strengthening touch
                    // after staging snapshotted the entry.
                    if let Some(current) = report.written.as_ref().and_then(|w| self.store.get_memory(w.id).ok().flatten()) {
                        report.written = Some(current);
                    }
                }
            }
            reports.push(report);
        }
        Ok(reports)
    }

    // ---- read paths ----

    /// Hybrid recall: entity matches + FTS hits, entity-linked memories first.
    pub fn recall(&self, query: &str, scope: &Scope, limit: usize) -> Result<Vec<SearchHit>> {
        let session = scope.session_id.as_deref();
        let mut hits: Vec<SearchHit> = Vec::new();
        let mut seen: HashSet<i64> = HashSet::new();

        for (canonical, _kind, _n) in self.store.entities_matching(query, 5)? {
            for (entry, conf) in self.store.memories_for_entity(&canonical, &scope.user_id, &scope.project, session, 3)? {
                if seen.insert(entry.id) {
                    hits.push(SearchHit { entry, score: conf as f32 + 0.1, entities: vec![canonical.clone()] });
                }
            }
        }
        for (entry, score) in self.store.search_content(query, &scope.user_id, &scope.project, session, false, limit)? {
            if seen.insert(entry.id) {
                let entities = self.store.entities_for_memory(entry.id).unwrap_or_default();
                hits.push(SearchHit { entry, score: score as f32, entities });
            }
        }
        hits.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        hits.truncate(limit.max(1));
        Ok(hits)
    }

    pub fn tour(&self, scope: &Scope, limit: usize) -> Result<Vec<MemoryEntry>> {
        self.store.list_by_category(&scope.user_id, &scope.project, scope.session_id.as_deref(), false, limit)
    }

    /// Adjudicate a `needs_review` / `quarantined` fact: accept (activate,
    /// optionally superseding the conflicting fact) or drop it.
    pub fn review(&self, id: i64, accept: bool, supersede: Option<i64>) -> Result<MemoryEntry> {
        if !accept {
            self.store.delete_memory(id)?;
            return Err(anyhow::anyhow!("dropped memory {id}"));
        }
        self.store.set_status(id, MemoryStatus::Active)?;
        if let Some(old) = supersede {
            self.store.link_supersedes(id, old, "agent adjudicated conflict")?;
            self.store.set_status(old, MemoryStatus::Superseded)?;
        }
        self.store
            .get_memory(id)?
            .ok_or_else(|| anyhow::anyhow!("memory {id} not found after review"))
    }
}

/// Surfaces too generic to be graph nodes (the subject of every
/// preference sentence, etc.). Compared case-insensitively.
fn is_generic_surface(text: &str) -> bool {
    matches!(
        text.trim().to_lowercase().as_str(),
        "user" | "users" | "person" | "people" | "someone" | "something" | "anything" | "thing" | "things"
        // Bare references to AI models (“models often…”) — the agent talking
        // about itself, not a real-world entity.
        | "model" | "models" | "ai" | "llm" | "llms"
    )
}

/// Deterministic augmenter for code-like identifiers GLiNER's labels miss:
/// `backtick spans` and ALL_CAPS tokens (env vars, constants, acronyms).
/// Fixed 0.65 confidence, `technology` type; skipped where a model mention
/// already covers the span.
fn augment_code_identifiers(fact_text: &str, to_char: &[usize], mentions: &mut Vec<ScoredMention>) {
    let backtick = Regex::new(r"`([^`\n]{1,60})`").expect("backtick regex");
    let caps = Regex::new(r"\b[A-Z][A-Z0-9_]{2,}\b").expect("caps regex");
    let overlaps = |mentions: &[ScoredMention], s: usize, e: usize| mentions.iter().any(|m| s < m.end && m.start < e);
    let mut found: Vec<(usize, usize, String)> = Vec::new();
    for cap in backtick.captures_iter(fact_text) {
        let inner = cap.get(1).expect("inner match");
        let text = inner.as_str().trim().to_string();
        if text.is_empty() || text.contains(' ') && text.len() > 40 {
            continue;
        }
        found.push((byte_to_char(to_char, inner.start()), byte_to_char(to_char, inner.end()), text));
    }
    for m in caps.find_iter(fact_text) {
        found.push((byte_to_char(to_char, m.start()), byte_to_char(to_char, m.end()), m.as_str().to_string()));
    }
    for (s, e, text) in found {
        if overlaps(mentions, s, e) || mentions.iter().any(|m| m.text == text) {
            continue;
        }
        let words = text.split_whitespace().count();
        mentions.push(ScoredMention {
            text,
            entity_type: "technology".to_string(),
            confidence: 0.65,
            start: s,
            end: e,
            // Word grid is the extractor's coordinate space; char spans
            // above are authoritative for storage.
            word_start: 0,
            word_end: words,
        });
    }
}

/// Byte offset → char offset map for a string: `map[b]` is the number of
/// chars fully before byte `b` (continuation bytes map to their char).
fn byte_to_char_map(s: &str) -> Vec<usize> {
    let mut map = vec![0usize; s.len() + 1];
    let mut ci = 0usize;
    for i in 1..=s.len() {
        if i == s.len() || s.is_char_boundary(i) {
            ci += 1;
        }
        map[i] = ci;
    }
    map
}

fn byte_to_char(map: &[usize], byte: usize) -> usize {
    map.get(byte.min(map.len().saturating_sub(1))).copied().unwrap_or(0)
}
