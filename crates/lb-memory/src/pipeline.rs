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

use crate::chunker::{LogicalChunker, split_sentences};
use crate::classify::classify;
use crate::conflicts::{FactVerdict, decide};
use crate::model::Extractor;
use crate::resolve::{KnownEntity, compare, resolve};
use crate::store::Store;
use crate::types::{
    FactDecision, MemoryEntry, MemoryStatus, Provenance, RememberReport, ScoredMention, Scope,
    SearchHit,
};

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
        self.write_facts(facts, scope, importance, provenance)
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
        self.write_facts(facts, scope, None, prov)
    }

    fn write_facts(
        &self,
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
            reports.push(self.write_one(fact_text, extracted, scope, importance, &provenance, &mut known)?);
        }
        Ok(reports)
    }

    fn write_one(
        &self,
        fact_text: String,
        extracted: Vec<ScoredMention>,
        scope: &Scope,
        importance: Option<i64>,
        provenance: &Provenance,
        known: &mut Vec<KnownEntity>,
    ) -> Result<RememberReport> {
        // Steps 2-3: compare + resolve against the live schema.
        let compared = compare(&extracted, known);
        let resolved = resolve(&compared, known);
        let category = classify(&fact_text);
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
                importance.unwrap_or(self.opts.importance),
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
    )
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
