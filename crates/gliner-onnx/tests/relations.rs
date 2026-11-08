//! Open-IE smoke tests: no declared schema (empty specs = generic query +
//! mined phrases). Requires the Nico snapshot + ORT.

use gliner_onnx::{GlinerOnnx, RelationSpec};

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
    let out = m
        .extract_relations("Steve Jobs founded Apple.", &labels(), &[], 0.5)
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
    let out = m
        .extract_relations(
            "Tim Cook runs Apple. Steve Jobs was a visionary.",
            &labels(),
            &[],
            0.5,
        )
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
fn open_ie_quiet_without_evidence() {
    let mut m = GlinerOnnx::load(snapshot_dir()).expect("load nico snapshot");
    let out = m
        .extract_relations("The sky is blue today.", &labels(), &[], 0.5)
        .expect("extract_relations");
    assert!(out.is_empty(), "spurious edges: {out:?}");
}
