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
use crate::resolve::{KnownEntity, compare, resolve, bind_pronouns, is_pronoun};
use time::OffsetDateTime;
use crate::store::Store;
use crate::types::{
    Category, ComparedEntity, FactDecision, MemoryEntry, MemoryStatus, Provenance, RememberReport,
    ResolvedEntity, ScoredMention, ScoredRelation, Scope, SearchHit,
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
    relations: Vec<ScoredRelation>,
    viewpoint: String,
    viewpoint_confidence: f32,
    valid_from: String,
    valid_to: String,
    tense: String,
}

fn default_entity_labels() -> Vec<String> {
    crate::model::DEFAULT_LABELS.iter().map(|s| s.to_string()).collect()
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
    /// Relation score threshold for keeping a typed edge. Relation scores
    /// are not calibrated to entity confidences (a true edge can score
    /// ~0.6 while its entities score 0.99), so this defaults lower.
    pub relation_threshold: f32,
    /// Default importance for new facts.
    pub importance: i64,
    /// Max chunk size for extraction scoping.
    pub max_chunk: usize,
}

impl Default for PipelineOptions {
    fn default() -> Self {
        Self { extract_threshold: 0.5, relation_threshold: 0.5, importance: 5, max_chunk: 2000 }
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
        // Canonical names in resolution order: discourse context for
        // pronoun binding in later facts of this call.
        let mut recent_order: Vec<String> = Vec::new();
        for (fact_text, extracted) in facts {
            let plan = self.plan_one(extractor, fact_text, extracted, scope, importance, &known, &recent_order)?;
            recent_order.extend(plan.resolved.iter().map(|r| crate::resolve::normalize(&r.canonical)));
            reports.push(self.commit_plan(plan, scope, &provenance, &mut known)?);
        }
        Ok(reports)
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
        recent_order: &[String],
    ) -> Result<FactPlan> {
        // Step 2a: anaphora — bind pronouns to recent entities (in-call
        // resolution order, then the store's most-recently-touched rows).
        // Unresolvable pronouns are dropped and can never fork "He" nodes.
        let store_recents = self.store.recent_entities(10).unwrap_or_default();
        let (kept, links) = bind_pronouns(&extracted, known, recent_order, &store_recents);
        // The kept mentions are the fact from here on (dropped pronouns
        // must not influence decide/extract either).
        let extracted = kept;
        // Steps 2-3: compare + resolve against the live schema. Pronoun
        // links override whatever compare infers from the surface form.
        let mut compared = compare(&extracted, known);
        for (idx, ant) in &links {
            if let Some(c) = compared.get_mut(*idx) {
                c.existing_id = Some(ant.id);
                c.existing_type = Some(ant.entity_type.clone());
            }
        }
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

        // Open-IE relations (step 4b): no declared schema — the scorer
        // judges relatedness under one generic query and phrases are mined
        // from the between-span surface. Skipped when nothing will be
        // written; layouts without heads.onnx yield none and the pair falls
        // back to co-occurrence.
        let mut relations = Vec::new();
        if duplicate_of.is_none() && !resolved.is_empty() {
            if let Ok(rs) = extractor.extract_relations(
                &fact_text,
                &default_entity_labels(),
                &[],
                self.opts.relation_threshold,
            ) {
                relations = rs;
            }
            // Implicit argument completion: a lone person plus a role noun
            // ("... is the CEO") links the most recent organization (else
            // location). The type is the bare role noun.
            let persons: Vec<&ResolvedEntity> =
                resolved.iter().filter(|r| r.entity_type == "person").collect();
            let has_org = resolved
                .iter()
                .any(|r| r.entity_type == "organization" || r.entity_type == "location");
            if duplicate_of.is_none() && relations.is_empty() && persons.len() == 1 && !has_org {
                if let Some((org, role)) = self.implicit_org(&fact_text, known, recent_order, &store_recents) {
                    relations.push(ScoredRelation {
                        relation_type: role,
                        head_text: persons[0].canonical.clone(),
                        head_confidence: 0.6,
                        tail_text: org,
                        tail_confidence: 0.6,
                        score: 0.6,
                    });
                }
            }
        }

        // Viewpoint (step 4c): "fact" for uncontroversial statements,
        // "opinion" otherwise — a zero-shot classification of the fact text.
        // Skipped with nothing to write; errors fall back to unknown.
        let (viewpoint, viewpoint_confidence) =
            if duplicate_of.is_none() && !resolved.is_empty() {
                match extractor.classify(&fact_text, "statement-type", &["fact".to_string(), "opinion".to_string()]) {
                    Ok(scores) => scores
                        .into_iter()
                        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                        .unwrap_or_default(),
                    Err(_) => (String::new(), 0.0),
                }
            } else {
                (String::new(), 0.0)
            };

        // Canonicalize relation surfaces against the Wikidata property
        // inventory: "great ceo of" and "ceo" both become "chief executive
        // officer". Unmatched phrases stay open (raw surface). Applies to
        // scored and implicit relations alike, before anything is written.
        for rel in &mut relations {
            if let Some((_, canonical)) =
                crate::rel_canon::canonicalize_relation(&rel.relation_type)
            {
                rel.relation_type = canonical;
            }
        }

        // Temporal scope (step 4d): explicit dates first, tense + "now" cues
        // as fallback. Bounds are the known-true interval, not the full
        // history: "" means unknown, not unbounded.
        let stmt_date = OffsetDateTime::now_utc().date().to_string();
        let (valid_from, valid_to, tense) = resolve_time_range(&fact_text, &compared, &stmt_date);

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
            relations,
            viewpoint,
            viewpoint_confidence,
            valid_from,
            valid_to,
            tense,
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
            relations,
            viewpoint,
            viewpoint_confidence,
            valid_from,
            valid_to,
            tense,
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
            let meta = crate::store::EdgeMeta {
                valid_from: valid_from.clone(),
                valid_to: valid_to.clone(),
                tense: tense.clone(),
                viewpoint: viewpoint.clone(),
                viewpoint_confidence: viewpoint_confidence as f64,
            };
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
            // Pre-bound surfaces (pronouns) resolve to nodes whose
            // canonical/alias keys differ — index every bound mention.
            for c in &compared {
                if let Some(id) = c.existing_id {
                    node_of.entry(c.normalized.clone()).or_insert((id, c.mention.confidence));
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
                        &meta,
                    )?;
                }
            }
            // Typed relations first: endpoint surfaces resolve through the
            // fact's mentions, falling back to the stored schema (implicit
            // targets like a recent org were never mentioned in this fact,
            // so they only live in `known`).
            let endpoint_id = |surface: &str| -> Option<i64> {
                let norm = crate::resolve::normalize(surface);
                if let Some((id, _)) = node_of.get(&norm) {
                    return Some(*id);
                }
                known.iter().find(|k| k.normalized == norm).map(|k| k.id)
            };
            for rel in &relations {
                if let (Some(hid), Some(tid)) = (endpoint_id(&rel.head_text), endpoint_id(&rel.tail_text)) {
                    if hid != tid {
                        self.store.link_typed_relation(hid, tid, &rel.relation_type, rel.score as f64, &meta)?;
                    }
                }
            }
            for a in 0..entity_ids.len() {
                for b in (a + 1)..entity_ids.len() {
                    self.store.link_cooccur(entity_ids[a], entity_ids[b], 0.5, &meta)?;
                }
            }
            for (old, cues) in &conflicts {
                self.store.link_conflict(id, *old, cues, &meta)?;
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
        let mut recent_order: Vec<String> = Vec::new();
        let mut staged = Vec::new();
        for (fact_text, extracted) in facts {
            let plan = self.plan_one(extractor, fact_text, extracted, scope, importance, &known, &recent_order)?;
            recent_order.extend(plan.resolved.iter().map(|r| crate::resolve::normalize(&r.canonical)));
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
            relations,
            viewpoint,
            viewpoint_confidence,
            valid_from,
            valid_to,
            tense,
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
        let meta = crate::store::EdgeMeta {
            valid_from: valid_from.clone(),
            valid_to: valid_to.clone(),
            tense: tense.clone(),
            viewpoint: viewpoint.clone(),
            viewpoint_confidence: viewpoint_confidence as f64,
        };
        // Pre-bound surfaces (pronouns) resolve to nodes whose
        // canonical/alias keys differ — index every bound mention.
        for c in &report.compared {
            if let Some(id) = c.existing_id {
                node_of.entry(c.normalized.clone()).or_insert(id);
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
                    &meta,
                )?;
            }
        }
        // Same endpoint resolution as the direct path: fact mentions
        // first, stored schema as fallback for implicit targets.
        let endpoint_id = |surface: &str| -> Option<i64> {
            let norm = crate::resolve::normalize(surface);
            if let Some(id) = node_of.get(&norm) {
                return Some(*id);
            }
            known.iter().find(|k| k.normalized == norm).map(|k| k.id)
        };
        for rel in &relations {
            if let (Some(hid), Some(tid)) = (endpoint_id(&rel.head_text), endpoint_id(&rel.tail_text)) {
                if hid != tid {
                    intake.stage_typed_relation(intake_id, hid, tid, &rel.relation_type, rel.score as f64, &meta)?;
                }
            }
        }
        for a in 0..entity_ids.len() {
            for b in (a + 1)..entity_ids.len() {
                intake.stage_cooccur(intake_id, entity_ids[a], entity_ids[b], 0.5, &meta)?;
            }
        }
        for (old, old_cues) in &conflicts {
            let cues_json = serde_json::to_string(old_cues).unwrap_or_else(|_| "[]".to_string());
            intake.stage_conflict(intake_id, staged_mem, *old, &cues_json, &meta)?;
        }
        intake.record_outcome(
            intake_id,
            &serde_json::json!({"kind": "written", "status": status.as_str()}).to_string(),
        )?;
        Ok((report, intake_id))
    }

    /// Implicit organization for a lone person + role noun ("X is the CEO"):
    /// most recent organization in discourse context, else most recent
    /// location. Returns (org canonical, role noun). Nothing is invented:
    /// the person is named, the org comes from context.
    fn implicit_org(
        &self,
        fact_text: &str,
        known: &[KnownEntity],
        recent_order: &[String],
        store_recents: &[KnownEntity],
    ) -> Option<(String, String)> {
        let words: Vec<String> = fact_text
            .split(|c: char| !c.is_alphanumeric())
            .map(|w| w.to_lowercase())
            .filter(|w| !w.is_empty())
            .collect();
        let role = words.iter().find(|w| ROLE_NOUNS.contains(&w.as_str()))?.clone();
        let mut pool: Vec<&KnownEntity> = Vec::new();
        for norm in recent_order.iter().rev() {
            if let Some(k) = known.iter().find(|k| &k.normalized == norm) {
                if (k.entity_type == "organization" || k.entity_type == "location")
                    && !pool.iter().any(|e: &&KnownEntity| e.id == k.id)
                {
                    pool.push(k);
                }
            }
        }
        for k in store_recents {
            if (k.entity_type == "organization" || k.entity_type == "location")
                && !pool.iter().any(|e: &&KnownEntity| e.id == k.id)
                && !pool.iter().any(|e: &&KnownEntity| e.canonical == k.canonical)
            {
                pool.push(k);
            }
        }
        // Organizations beat locations ("CEO of" an org, not a city).
        let org = pool
            .iter()
            .find(|e| e.entity_type == "organization")
            .or_else(|| pool.first())?;
        Some((org.canonical.clone(), role))
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
    /// Typed edges adjacent to a hit's entities contribute their relation
    /// kinds for display, plus a score boost when the query mentions the
    /// relation ("who founded Apple" upranks founder-of memories).
    pub fn recall(&self, query: &str, scope: &Scope, limit: usize) -> Result<Vec<SearchHit>> {
        let session = scope.session_id.as_deref();
        let mut hits: Vec<SearchHit> = Vec::new();
        let mut seen: HashSet<i64> = HashSet::new();
        let id_of: HashMap<String, i64> =
            self.store.list_entities().unwrap_or_default().into_iter().map(|e| (e.canonical, e.id)).collect();

        for (canonical, _kind, _n) in self.store.entities_matching(query, 5)? {
            for (entry, conf) in self.store.memories_for_entity(&canonical, &scope.user_id, &scope.project, session, 3)? {
                if seen.insert(entry.id) {
                    let entities = vec![canonical.clone()];
                    let (boost, relations) = self.describe_relations(&entities, &id_of, query).unwrap_or_default();
                    hits.push(SearchHit {
                        entry,
                        score: conf as f32 + 0.1 + boost,
                        entities,
                        relations,
                    });
                }
            }
        }
        for (entry, score) in self.store.search_content(query, &scope.user_id, &scope.project, session, false, limit)? {
            if seen.insert(entry.id) {
                let entities = self.store.entities_for_memory(entry.id).unwrap_or_default();
                let (boost, relations) = self.describe_relations(&entities, &id_of, query).unwrap_or_default();
                hits.push(SearchHit { entry, score: score as f32 + boost, entities, relations });
            }
        }
        hits.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        hits.truncate(limit.max(1));
        Ok(hits)
    }

    /// Typed relations adjacent to `entity_names`, for display, plus a
    /// cue-match boost against the query. Builds the canonical→id map
    /// internally; the hot path below reuses one map across hits.
    pub fn relations_for(&self, entity_names: &[String], query: &str) -> Result<(f32, Vec<String>)> {
        let id_of: HashMap<String, i64> =
            self.store.list_entities().unwrap_or_default().into_iter().map(|e| (e.canonical, e.id)).collect();
        self.describe_relations(entity_names, &id_of, query)
    }

    /// Typed relations adjacent to `entity_names`, for display, plus a
    /// cue-match boost against the query.
    fn describe_relations(
        &self,
        entity_names: &[String],
        id_of: &HashMap<String, i64>,
        query: &str,
    ) -> Result<(f32, Vec<String>)> {
        let mut boost = 0.0f32;
        let mut out = Vec::new();
        let mut seen_rel = std::collections::HashSet::new();
        for name in entity_names {
            let Some(id) = id_of.get(name) else { continue };
            for (other, rel) in self.store.typed_relations_of(*id)? {
                if seen_rel.insert((other.clone(), rel.clone())) {
                    out.push(format!("{rel} via {other}"));
                }
                if crate::rel_canon::relation_cue_match(&rel, query) {
                    boost += 0.3;
                }
            }
        }
        Ok((boost.min(0.6), out))
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

/// Job-title nouns that license implicit-organization completion for a
/// lone person ("X is the CEO" with no org mentioned). Normalized forms;
/// "co founder" covers the hyphenated surface.
const ROLE_NOUNS: &[&'static str] = &[
    "ceo", "founder", "cofounder", "co founder", "president", "cto", "cio", "cfo", "coo", "chair",
    "chairman", "chairwoman", "director", "vp", "manager", "partner", "chief", "officer", "head",
    "lead", "mayor", "governor", "senator", "king", "queen", "pope", "owner", "boss", "captain",
    "coach", "principal", "dean",
];

/// Resolve a fact's temporal scope: (valid_from, valid_to, tense) with ISO
/// dates ("" = unknown). Explicit dates win; tense words ("was"→past,
/// "is"/"now"→present, "will"→future) and "now"/"former" cues fill the
/// gaps against the statement date. Bounds mean "known true at least
/// during", not the full historical interval.
fn resolve_time_range(fact_text: &str, compared: &[ComparedEntity], stmt_date: &str) -> (String, String, String) {
    let lower = fact_text.to_lowercase();
    let words: Vec<&str> = lower.split(|c: char| !c.is_alphanumeric()).filter(|w| !w.is_empty()).collect();
    let has_any = |cues: &[&str]| words.iter().any(|w| cues.contains(w));

    let tense = if has_any(&["will", "shall"]) {
        "future"
    } else if has_any(&["was", "were", "had", "did", "been", "formerly", "former", "previously", "ex"]) {
        "past"
    } else if has_any(&["is", "are", "am", "now", "currently", "today", "present", "has", "have"]) {
        "present"
    } else {
        ""
    };

    let mut from = String::new();
    let mut to = String::new();
    // Explicit dates from GLiNER date mentions.
    let dates: Vec<String> = compared
        .iter()
        .filter(|c| c.mention.entity_type == "date")
        .map(|c| c.mention.text.clone())
        .collect();
    let year_re = Regex::new(r"\b((?:19|20)\d{2})\b").expect("year regex");
    let month_re = Regex::new(
        r"(?i)\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{1,2},\s*)?(\d{4})\b",
    )
    .expect("month regex");
    let mut years: Vec<i32> = Vec::new();
    for d in &dates {
        if let Some(cap) = year_re.captures(d) {
            if let Ok(y) = cap[1].parse::<i32>() {
                years.push(y);
            }
        }
        if let Some(cap) = month_re.captures(d) {
            let month = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"]
                .iter()
                .position(|m| m.eq_ignore_ascii_case(&cap[1]))
                .map(|i| i + 1)
                .unwrap_or(1) as u32;
            if let Ok(y) = cap[3].parse::<i32>() {
                let last = month_len(month, y);
                let dom: u32 = cap
                    .get(2)
                    .map(|m| m.as_str().trim_end_matches(|c| c == ',' || c == ' ').trim().parse().unwrap_or(1))
                    .unwrap_or(1);
                let dom = dom.clamp(1, last);
                if from.is_empty() {
                    from = format!("{y:04}-{month:02}-{dom:02}");
                }
                to = format!("{y:04}-{month:02}-{last:02}");
            }
        }
    }
    // "between Y1 and Y2" pins both ends.
    if years.len() >= 2 && lower.contains("between") {
        let (mut a, mut b) = (years[0], years[1]);
        if a > b {
            std::mem::swap(&mut a, &mut b);
        }
        from = format!("{a:04}-01-01");
        to = format!("{b:04}-12-31");
    } else {
        for y in &years {
            let before: String = lower
                .find(&y.to_string())
                .map(|pos| lower[..pos].to_string())
                .unwrap_or_default();
            let tail: Vec<&str> = before.split(|c: char| !c.is_alphanumeric()).filter(|w| !w.is_empty()).collect();
            let tail: Vec<&&str> = tail.iter().rev().take(4).collect();
            let has = |cues: &[&str]| tail.iter().any(|w| cues.contains(*w));
            if has(&["from", "since", "after"]) {
                if from.is_empty() {
                    from = format!("{y:04}-01-01");
                }
            } else if has(&["until", "till", "to", "through"]) {
                to = format!("{y:04}-12-31");
            } else if has(&["before"]) {
                to = format!("{:04}-12-31", y - 1);
            } else if from.is_empty() && to.is_empty() {
                // Bare year ("in 1976", or standalone): that calendar year.
                from = format!("{y:04}-01-01");
                to = format!("{y:04}-12-31");
            }
        }
    }
    // Tense/now cues fill remaining gaps against the statement date.
    if tense == "present" && from.is_empty() && has_any(&["now", "today", "currently", "present"]) {
        from = stmt_date.to_string();
    }
    if tense == "past" && to.is_empty() {
        to = stmt_date.to_string();
    }
    (from, to, tense.to_string())
}

fn month_len(month: u32, year: i32) -> u32 {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if year % 4 == 0 && (year % 100 != 0 || year % 400 == 0) => 29,
        2 => 28,
        _ => 30,
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
