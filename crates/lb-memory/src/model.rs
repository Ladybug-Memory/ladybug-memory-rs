//! Extractor abstraction: GLiNER2.5 ONNX behind a trait so the pipeline is
//! testable without the model.

use anyhow::Result;

use crate::types::ScoredMention;

pub const DEFAULT_LABELS: &[&str] =
    &["person", "organization", "location", "product", "event", "date", "technology", "concept"];

pub trait Extractor {
    fn extract(
        &mut self,
        text: &str,
        labels: &[String],
        threshold: f32,
    ) -> Result<Vec<ScoredMention>>;
    /// Classify `text` into `labels` (single-label argmax semantics).
    /// Returns `(label, probability)` in input order; empty means
    /// "no model verdict" (rule-based callers keep their answer).
    fn classify(&mut self, _text: &str, _task: &str, _labels: &[String]) -> Result<Vec<(String, f32)>> {
        Ok(vec![])
    }
    /// Typed relations between the given clean extracted mentions.
    /// Default: unsupported (no typed edges).
    fn extract_relations(
        &mut self,
        _text: &str,
        _entity_labels: &[String],
        _relations: &[crate::types::RelationSpec],
        _mentions: &[crate::types::ScoredMention],
        _threshold: f32,
    ) -> Result<Vec<crate::types::ScoredRelation>> {
        Ok(vec![])
    }
    /// Sentence embedding for vector search (L2-normalized). `None` means
    /// "no embedder" — callers keep lexical behavior. Used to shortlist
    /// canonical-relation candidates from the SQLite inventory.
    fn embed(&mut self, _text: &str) -> Result<Option<Vec<f32>>> {
        Ok(None)
    }
}

/// Lazy GLiNER2.5 extractor: sessions load on first `extract`, never for
/// search-only commands.
pub struct GlinerExtractor {
    snapshot: String,
    model: Option<gliner_onnx::GlinerOnnx>,
}

impl GlinerExtractor {
    pub fn new(snapshot: String) -> Self {
        Self { snapshot, model: None }
    }

    /// Snapshot directory this extractor loads from.
    pub fn snapshot_path(&self) -> &str {
        &self.snapshot
    }

    /// Ensure the snapshot is on disk, downloading it on first run.
    /// Inference entry points call this lazily via `load_or_fetch`; call
    /// it explicitly to pre-warm (e.g. `lbm fetch-model`) without running
    /// the model.
    pub fn ensure_model(&self) -> Result<()> {
        gliner_onnx::fetch::ensure_model(std::path::Path::new(&self.snapshot))
    }
}

impl Extractor for GlinerExtractor {
    fn extract(
        &mut self,
        text: &str,
        labels: &[String],
        threshold: f32,
    ) -> Result<Vec<ScoredMention>> {
        if self.model.is_none() {
            self.model = Some(gliner_onnx::GlinerOnnx::load_or_fetch(&self.snapshot)?);
        }
        let model = self.model.as_mut().expect("just loaded");
        let out = model.extract(text, labels, threshold)?;
        Ok(out
            .mentions
            .into_iter()
            .map(|m| ScoredMention {
                text: m.text,
                entity_type: m.label,
                confidence: m.confidence,
                start: m.start,
                end: m.end,
                word_start: m.word_start,
                word_end: m.word_end,
            })
            .collect())
    }

    fn classify(&mut self, text: &str, task: &str, labels: &[String]) -> Result<Vec<(String, f32)>> {
        if self.model.is_none() {
            self.model = Some(gliner_onnx::GlinerOnnx::load_or_fetch(&self.snapshot)?);
        }
        let model = self.model.as_mut().expect("just loaded");
        Ok(model.classify(text, task, labels, false, 0.5)?.scores)
    }

    fn extract_relations(
        &mut self,
        text: &str,
        entity_labels: &[String],
        relations: &[crate::types::RelationSpec],
        mentions: &[crate::types::ScoredMention],
        threshold: f32,
    ) -> Result<Vec<crate::types::ScoredRelation>> {
        if self.model.is_none() {
            self.model = Some(gliner_onnx::GlinerOnnx::load_or_fetch(&self.snapshot)?);
        }
        let model = self.model.as_mut().expect("just loaded");
        // Layouts without the scorer graph (DanKau 3-graph split) simply
        // yield no typed edges; the pipeline falls back to co-occurrence.
        if !model.supports_relations() {
            return Ok(vec![]);
        }
        let specs: Vec<gliner_onnx::RelationSpec> = relations
            .iter()
            .map(|r| gliner_onnx::RelationSpec {
                relation_type: r.relation_type.clone(),
                head_types: r.head_types.clone(),
                tail_types: r.tail_types.clone(),
                description: r.description.clone(),
                allow_self: r.allow_self,
            })
            .collect();
        // Clean extracted mentions become the scorer's pairs directly
        // (char offsets index this same text). No candidate pools, no
        // top-k selection, no swarm to filter downstream.
        let inputs: Vec<gliner_onnx::MentionInput> = mentions
            .iter()
            .map(|m| gliner_onnx::MentionInput {
                entity_type: m.entity_type.clone(),
                confidence: m.confidence,
                char_start: m.start,
                char_end: m.end,
            })
            .collect();
        let out = model.extract_relations(text, entity_labels, &specs, &inputs, threshold)?;
        Ok(out
            .into_iter()
            .map(|r| crate::types::ScoredRelation {
                relation_type: r.relation_type,
                head_text: r.head.text,
                head_confidence: r.head.confidence,
                tail_text: r.tail.text,
                tail_confidence: r.tail.confidence,
                score: r.score,
            })
            .collect())
    }

    fn embed(&mut self, text: &str) -> Result<Option<Vec<f32>>> {
        if self.model.is_none() {
            self.model = Some(gliner_onnx::GlinerOnnx::load_or_fetch(&self.snapshot)?);
        }
        let model = self.model.as_mut().expect("just loaded");
        Ok(Some(model.embed(text)?))
    }
}

/// Deterministic fake for unit tests (no model, no dylib).
#[cfg(any(test, feature = "test-support"))]
pub struct FakeExtractor {
    pub mentions: Vec<ScoredMention>,
    pub classify_scores: Vec<(String, f32)>,
    pub relations: Vec<crate::types::ScoredRelation>,
    /// Stub embedding returned from `embed` (`None` = no embedder).
    pub embed_vec: Option<Vec<f32>>,
}

#[cfg(any(test, feature = "test-support"))]
impl Default for FakeExtractor {
    fn default() -> Self {
        Self { mentions: vec![], classify_scores: vec![], relations: vec![], embed_vec: None }
    }
}

#[cfg(any(test, feature = "test-support"))]
impl Extractor for FakeExtractor {
    fn extract(
        &mut self,
        _text: &str,
        _labels: &[String],
        _threshold: f32,
    ) -> Result<Vec<ScoredMention>> {
        Ok(self.mentions.clone())
    }

    fn classify(&mut self, _text: &str, _task: &str, _labels: &[String]) -> Result<Vec<(String, f32)>> {
        Ok(self.classify_scores.clone())
    }

    fn extract_relations(
        &mut self,
        _text: &str,
        _entity_labels: &[String],
        _relations: &[crate::types::RelationSpec],
        _mentions: &[crate::types::ScoredMention],
        _threshold: f32,
    ) -> Result<Vec<crate::types::ScoredRelation>> {
        Ok(self.relations.clone())
    }

    fn embed(&mut self, _text: &str) -> Result<Option<Vec<f32>>> {
        Ok(self.embed_vec.clone())
    }
}
