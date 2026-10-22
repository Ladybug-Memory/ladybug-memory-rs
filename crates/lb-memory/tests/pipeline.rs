//! Pipeline + store tests on an ephemeral DB with a fake extractor.
//! Run: `cargo test --release -p lb-memory --features test-support`
#![cfg(feature = "test-support")]

use lb_memory::model::FakeExtractor;
use lb_memory::pipeline::{Pipeline, PipelineOptions};
use lb_memory::store::Store;
use lb_memory::types::{FactDecision, MemoryStatus, Provenance, Scope};

fn scope() -> Scope {
    Scope { user_id: "tester".into(), project: "demo".into(), session_id: Some("s1".into()) }
}

fn prov() -> Provenance {
    Provenance {
        session_id: Some("s1".into()),
        message_refs: vec!["msg-1".into()],
        extractor: "gliner2.5".into(),
        model: "test".into(),
        auto_captured: false,
    }
}

fn mention(text: &str, kind: &str, conf: f32) -> lb_memory::types::ScoredMention {
    lb_memory::types::ScoredMention {
        text: text.into(),
        entity_type: kind.into(),
        confidence: conf,
        start: 0,
        end: text.len(),
        word_start: 0,
        word_end: 1,
    }
}

#[test]
fn fts_index_builds_and_searches() {
    let store = Store::ephemeral().unwrap();
    assert!(store.fts_ready(), "FTS extension should load with shared linking");
    store
        .insert_memory(
            "Tim Cook runs Apple.",
            lb_memory::types::Category::Work,
            "u",
            "p",
            "",
            5,
            lb_memory::types::MemoryStatus::Active,
            &Default::default(),
        )
        .unwrap();
    let hits = store.search_content("cook", "u", "p", None, false, 5).unwrap();
    assert_eq!(hits.len(), 1);
    assert!(hits[0].1 > 0.0, "FTS score should be positive");
}

#[test]
fn remember_new_fact_writes_graph() {
    let store = Store::ephemeral().unwrap();
    let pipe = Pipeline::new(&store, PipelineOptions::default());
    let mut ex = FakeExtractor {
        mentions: vec![mention("Tim Cook", "person", 0.99), mention("Apple", "organization", 0.98)],
    };
    let reports = pipe.remember(&mut ex, "Tim Cook runs Apple.", &scope(), None, prov()).unwrap();
    assert_eq!(reports.len(), 1);
    let r = &reports[0];
    assert!(matches!(r.decision, FactDecision::New { .. }), "got {:?}", r.decision);
    assert_eq!(r.resolved.len(), 2);
    assert!(r.resolved.iter().all(|e| e.created));
    let written = r.written.as_ref().unwrap();
    assert_eq!(written.status, MemoryStatus::Active);
    assert_eq!(written.provenance.message_refs, vec!["msg-1".to_string()]);

    let (mem, ent, men, rev) = store.counts().unwrap();
    assert_eq!((mem, ent, men, rev), (1, 2, 2, 0));

    // Recall finds it by entity.
    let hits = pipe.recall("tim cook", &scope(), 5).unwrap();
    assert!(!hits.is_empty());
    assert!(hits[0].entities.iter().any(|e| e == "Tim Cook"));
}

#[test]
fn duplicate_strengthens_instead_of_writing() {
    let store = Store::ephemeral().unwrap();
    let pipe = Pipeline::new(&store, PipelineOptions::default());
    let mut ex = FakeExtractor { mentions: vec![mention("Python", "technology", 0.95)] };
    let first = pipe.remember(&mut ex, "User prefers Python for scripting.", &scope(), None, prov()).unwrap();
    assert!(matches!(first[0].decision, FactDecision::New { .. }));
    let id = first[0].written.as_ref().unwrap().id;

    let second = pipe.remember(&mut ex, "User prefers Python for scripting tasks.", &scope(), None, prov()).unwrap();
    match &second[0].decision {
        FactDecision::Duplicate { of, similarity } => {
            assert_eq!(*of, id);
            assert!(*similarity >= 0.8);
        }
        other => panic!("expected duplicate, got {other:?}"),
    }
    let (mem, _, _, _) = store.counts().unwrap();
    assert_eq!(mem, 1, "duplicate must not add a node");
    let kept = store.get_memory(id).unwrap().unwrap();
    assert_eq!(kept.importance, 6, "duplicate strengthens importance");
}

#[test]
fn conflict_goes_to_review_with_provenance() {
    let store = Store::ephemeral().unwrap();
    let pipe = Pipeline::new(&store, PipelineOptions::default());
    let mut ex = FakeExtractor { mentions: vec![mention("Python", "technology", 0.95)] };
    pipe.remember(&mut ex, "User prefers Python for scripting.", &scope(), None, prov()).unwrap();
    let reports =
        pipe.remember(&mut ex, "User hates Python for scripting.", &scope(), None, prov()).unwrap();
    match &reports[0].decision {
        FactDecision::Conflict { with, cues } => {
            assert_eq!(with.len(), 1);
            assert!(cues.iter().any(|c| c.contains("antonym")));
        }
        other => panic!("expected conflict, got {other:?}"),
    }
    let written = reports[0].written.as_ref().unwrap();
    assert_eq!(written.status, MemoryStatus::NeedsReview);

    // Review queue surfaces it with its cues.
    let queue = store.needs_review(10).unwrap();
    assert_eq!(queue.len(), 1);
    let conflicts = store.conflicts_of(written.id).unwrap();
    assert_eq!(conflicts.len(), 1);

    // Agent accepts + supersedes: old fact retires, history kept.
    let old = conflicts[0].0;
    let entry = pipe.review(written.id, true, Some(old)).unwrap();
    assert_eq!(entry.status, MemoryStatus::Active);
    assert_eq!(store.get_memory(old).unwrap().unwrap().status, MemoryStatus::Superseded);
}

#[test]
fn low_confidence_is_quarantined_not_dropped() {
    let store = Store::ephemeral().unwrap();
    let pipe = Pipeline::new(&store, PipelineOptions::default());
    let mut ex = FakeExtractor { mentions: vec![] };
    let reports = pipe.remember(&mut ex, "Hmm, vague rambling without names.", &scope(), None, prov()).unwrap();
    assert!(matches!(reports[0].decision, FactDecision::LowConfidence { .. }));
    assert_eq!(reports[0].written.as_ref().unwrap().status, MemoryStatus::Quarantined);
    // Hidden from recall.
    assert!(pipe.recall("rambling", &scope(), 5).unwrap().is_empty());
}

#[test]
fn alias_surfaces_merge_onto_canonical() {
    let store = Store::ephemeral().unwrap();
    let pipe = Pipeline::new(&store, PipelineOptions::default());
    let mut ex = FakeExtractor { mentions: vec![mention("LadybugDB", "technology", 0.9)] };
    pipe.remember(&mut ex, "LadybugDB is a graph database.", &scope(), None, prov()).unwrap();
    let mut ex2 = FakeExtractor { mentions: vec![mention("ladybugdb", "technology", 0.88)] };
    let reports = pipe.remember(&mut ex2, "ladybugdb stores the memory graph.", &scope(), None, prov()).unwrap();
    assert!(!reports[0].resolved[0].created, "alias must merge, not fork");
    let entities = store.list_entities().unwrap();
    assert_eq!(entities.len(), 1);
    assert!(entities[0].aliases.iter().any(|a| a == "ladybugdb"));
}
