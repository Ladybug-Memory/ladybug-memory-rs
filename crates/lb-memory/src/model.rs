//! Extractor abstraction: GLiNER2.5 ONNX behind a trait so the pipeline is
//! testable without the model.

use anyhow::Result;

use crate::types::ScoredMention;

pub const DEFAULT_LABELS: &[&str] =
    &["person", "organization", "location", "product", "event", "date", "technology", "concept"];

pub trait Extractor {
    fn extract(&mut self, text: &str, threshold: f32) -> Result<Vec<ScoredMention>>;
    /// Classify `text` into `labels` (single-label argmax semantics).
    /// Returns `(label, probability)` in input order; empty means
    /// "no model verdict" (rule-based callers keep their answer).
    fn classify(&mut self, _text: &str, _task: &str, _labels: &[String]) -> Result<Vec<(String, f32)>> {
        Ok(vec![])
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
}

impl Extractor for GlinerExtractor {
    fn extract(&mut self, text: &str, threshold: f32) -> Result<Vec<ScoredMention>> {
        if self.model.is_none() {
            self.model = Some(gliner_onnx::GlinerOnnx::load(&self.snapshot)?);
        }
        let model = self.model.as_mut().expect("just loaded");
        let labels: Vec<String> = DEFAULT_LABELS.iter().map(|s| s.to_string()).collect();
        let out = model.extract(text, &labels, threshold)?;
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
            self.model = Some(gliner_onnx::GlinerOnnx::load(&self.snapshot)?);
        }
        let model = self.model.as_mut().expect("just loaded");
        Ok(model.classify(text, task, labels, false, 0.5)?.scores)
    }
}

/// Deterministic fake for unit tests (no model, no dylib).
#[cfg(any(test, feature = "test-support"))]
pub struct FakeExtractor {
    pub mentions: Vec<ScoredMention>,
    pub classify_scores: Vec<(String, f32)>,
}

#[cfg(any(test, feature = "test-support"))]
impl Extractor for FakeExtractor {
    fn extract(&mut self, _text: &str, _threshold: f32) -> Result<Vec<ScoredMention>> {
        Ok(self.mentions.clone())
    }

    fn classify(&mut self, _text: &str, _task: &str, _labels: &[String]) -> Result<Vec<(String, f32)>> {
        Ok(self.classify_scores.clone())
    }
}
