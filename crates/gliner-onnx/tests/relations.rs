//! Relation smoke test: real `heads.onnx` scorer, no torch oracle.
//!
//! Requires the Nico snapshot + ORT (same as the parity tests).

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
fn founder_of_smoke() {
    let mut m = GlinerOnnx::load(snapshot_dir()).expect("load nico snapshot");
    assert!(m.supports_relations(), "expected the SparseRelationScorer graph");
    let rels = vec![RelationSpec::new("founder-of", &["person"], &["organization"])];
    let out = m
        .extract_relations("Steve Jobs founded Apple.", &labels(), &rels, 0.5)
        .expect("extract_relations");
    eprintln!("{out:#?}");
    let hit = out.iter().find(|r| {
        r.relation_type == "founder-of" && r.head.text == "Steve Jobs" && r.tail.text == "Apple"
    });
    assert!(hit.is_some(), "missing founder-of(Steve Jobs, Apple): {out:?}");
    assert!(hit.unwrap().score > 0.5);
}

#[test]
fn no_relations_without_evidence() {
    let mut m = GlinerOnnx::load(snapshot_dir()).expect("load nico snapshot");
    let rels = vec![RelationSpec::new("founder-of", &["person"], &["organization"])];
    // No person/org relation here; the scorer should stay quiet.
    let out = m
        .extract_relations("The sky is blue today.", &labels(), &rels, 0.5)
        .expect("extract_relations");
    assert!(out.is_empty(), "spurious edges: {out:?}");
}
