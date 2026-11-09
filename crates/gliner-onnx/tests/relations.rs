//! Open-IE smoke tests: no declared schema (empty specs = generic query +
//! mined phrases). Requires the Nico snapshot + ORT.

use gliner_onnx::{GlinerOnnx, MentionInput};

/// Mention helper: char offsets located by substring search.
fn ms(text: &str, entity_type: &str, confidence: f32, surface: &str) -> MentionInput {
    let start = text.find(surface).unwrap_or_else(|| panic!("{surface:?} not in {text:?}"));
    MentionInput {
        entity_type: entity_type.to_string(),
        confidence,
        char_start: start,
        char_end: start + surface.len(),
    }
}

fn snapshot_dir() -> String {
    std::env::var("GLINER_SNAPSHOT").unwrap_or_else(|_| {
        format!(
            "{}/.cache/ladybug-memory/models/gliner2.5-base-v1-nico-onnx",
            std::env::var("HOME").unwrap()
        )
    })
}

fn labels() -> Vec<String> {
    ["person", "organization", "location", "product", "event", "technology", "concept"]
        .iter()
        .map(|s| s.to_string())
        .collect()
}

#[test]
fn open_ie_mines_founded() {
    let mut m = GlinerOnnx::load(snapshot_dir()).expect("load nico snapshot");
    assert!(m.supports_relations());
    // Empty specs = open mode: generic query, phrases mined from text.
    let text = "Steve Jobs founded Apple.";
    let mentions = vec![
        ms(text, "person", 0.99, "Steve Jobs"),
        ms(text, "organization", 0.98, "Apple"),
    ];
    let out = m
        .extract_relations(text, &labels(), &[], &mentions, 0.5)
        .expect("extract_relations");
    eprintln!("{out:#?}");
    let hit = out.iter().find(|r| {
        r.head.text == "Steve Jobs" && r.tail.text == "Apple" && r.relation_type.contains("found")
    });
    assert!(hit.is_some(), "missing open-IE founded edge: {out:?}");
}

#[test]
fn open_ie_separates_unrelated_pairs() {
    let mut m = GlinerOnnx::load(snapshot_dir()).expect("load nico snapshot");
    // Cook and Jobs never interact in this text; only Cook-Apple should link.
    let text = "Tim Cook runs Apple. Steve Jobs was a visionary.";
    let mentions = vec![
        ms(text, "person", 0.99, "Tim Cook"),
        ms(text, "organization", 0.97, "Apple"),
        ms(text, "person", 0.96, "Steve Jobs"),
    ];
    let out = m
        .extract_relations(text, &labels(), &[], &mentions, 0.5)
        .expect("extract_relations");
    eprintln!("{out:#?}");
    assert!(
        out.iter().any(|r| r.head.text == "Tim Cook"
            && r.tail.text == "Apple"
            && (r.relation_type.contains("run"))),
        "missing runs edge: {out:?}"
    );
    // No edge should involve Jobs as an endpoint with Apple (different sentence, no relation).
    assert!(
        !out.iter().any(|r| (r.head.text == "Steve Jobs" || r.tail.text == "Steve Jobs")
            && (r.head.text == "Apple" || r.tail.text == "Apple")),
        "spurious cross-sentence edge: {out:?}"
    );
}

#[test]
fn open_ie_scores_clean_mention_pairs() {
    // With clean mention spans, the direct pair scores well even with
    // other mentions nearby. Date endpoints are the pipeline's job.
    let mut m = GlinerOnnx::load(snapshot_dir()).expect("load nico snapshot");
    let text = "Steve Jobs was CEO of Apple until 2011.";
    let mentions = vec![
        ms(text, "person", 0.99, "Steve Jobs"),
        ms(text, "organization", 0.97, "Apple"),
        ms(text, "date", 0.9, "2011"),
    ];
    let out = m
        .extract_relations(text, &labels(), &[], &mentions, 0.5)
        .expect("extract_relations");
    eprintln!("{out:#?}");
    assert!(
        out.iter().any(|r| r.head.text == "Steve Jobs"
            && r.tail.text == "Apple"
            && r.relation_type.contains("ceo")),
        "missing Jobs->Apple CEO edge: {out:?}"
    );
}

#[test]
fn open_ie_quiet_without_evidence() {
    let mut m = GlinerOnnx::load(snapshot_dir()).expect("load nico snapshot");
    let text = "The sky is blue today.";
    let out = m
        .extract_relations(text, &labels(), &[], &[], 0.5)
        .expect("extract_relations");
    assert!(out.is_empty(), "spurious edges: {out:?}");
}
