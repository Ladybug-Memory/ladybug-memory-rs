//! Extractor abstraction: GLiNER2.5 ONNX behind a trait so the pipeline is
//! testable without the model.

use anyhow::Result;

use crate::types::ScoredMention;

pub const DEFAULT_LABELS: &[&str] =
    &["person", "organization", "location", "product", "event", "date", "technology", "concept"];

pub trait Extractor {
    fn extract(&mut self, text: &str, threshold: f32) -> Result<Vec<ScoredMention>>;
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
}

/// Deterministic fake for unit tests (no model, no dylib).
#[cfg(any(test, feature = "test-support"))]
pub struct FakeExtractor {
    pub mentions: Vec<ScoredMention>,
}

#[cfg(any(test, feature = "test-support"))]
impl Extractor for FakeExtractor {
    fn extract(&mut self, _text: &str, _threshold: f32) -> Result<Vec<ScoredMention>> {
        Ok(self.mentions.clone())
    }
}
