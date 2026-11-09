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
        classify_scores: vec![],
        relations: vec![],
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
    let mut ex = FakeExtractor { mentions: vec![mention("Python", "technology", 0.95)], classify_scores: vec![], relations: vec![] };
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
    let mut ex = FakeExtractor { mentions: vec![mention("Python", "technology", 0.95)], classify_scores: vec![], relations: vec![] };
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
    let mut ex = FakeExtractor { mentions: vec![], classify_scores: vec![], relations: vec![] };
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
    let mut ex = FakeExtractor { mentions: vec![mention("LadybugDB", "technology", 0.9)], classify_scores: vec![], relations: vec![] };
    pipe.remember(&mut ex, "LadybugDB is a graph database.", &scope(), None, prov()).unwrap();
    let mut ex2 = FakeExtractor { mentions: vec![mention("ladybugdb", "technology", 0.88)], classify_scores: vec![], relations: vec![] };
    let reports = pipe.remember(&mut ex2, "ladybugdb stores the memory graph.", &scope(), None, prov()).unwrap();
    assert!(!reports[0].resolved[0].created, "alias must merge, not fork");
    let entities = store.list_entities().unwrap();
    assert_eq!(entities.len(), 1);
    assert!(entities[0].aliases.iter().any(|a| a == "ladybugdb"));
}

#[test]
fn model_category_fills_rule_gap() {
    // "Tim Cook runs Apple." has no rule cue (General); the fake model
    // verdict promotes it, mirroring the ONNX parities (work=0.942).
    let store = Store::ephemeral().unwrap();
    let pipe = Pipeline::new(&store, PipelineOptions::default());
    let mut ex = FakeExtractor {
        mentions: vec![mention("Tim Cook", "person", 0.99)],
        classify_scores: vec![("work".to_string(), 0.94), ("general".to_string(), 0.68)],
        relations: vec![],
    };
    let reports = pipe.remember(&mut ex, "Tim Cook runs Apple.", &scope(), None, prov()).unwrap();
    assert_eq!(
        reports[0].written.as_ref().unwrap().category,
        lb_memory::types::Category::Work
    );
}

#[test]
fn rules_win_over_model() {
    // Explicit preference cue beats a model verdict.
    let store = Store::ephemeral().unwrap();
    let pipe = Pipeline::new(&store, PipelineOptions::default());
    let mut ex = FakeExtractor {
        mentions: vec![mention("Python", "technology", 0.95)],
        classify_scores: vec![("work".to_string(), 0.99)],
        relations: vec![],
    };
    let reports =
        pipe.remember(&mut ex, "User prefers Python for scripting.", &scope(), None, prov()).unwrap();
    assert_eq!(
        reports[0].written.as_ref().unwrap().category,
        lb_memory::types::Category::Preferences
    );
}

#[test]
fn auto_drops_acknowledgments() {
    let store = Store::ephemeral().unwrap();
    let pipe = Pipeline::new(&store, PipelineOptions::default());
    let mut ex = FakeExtractor {
        mentions: vec![mention("Python", "technology", 0.95)],
        classify_scores: vec![],
        relations: vec![],
    };
    let reports = pipe
        .remember_auto(&mut ex, "Hello. User prefers Python. Thanks!", &scope(), prov())
        .unwrap();
    // "Hello." and "Thanks!" are dropped pre-extraction; only the middle
    // sentence becomes a fact.
    assert_eq!(reports.len(), 1);
    assert!(reports[0].written.as_ref().unwrap().content.contains("Python"));
}

#[cfg(feature = "test-support")]
mod sqlite_first {
    use super::*;
    use lb_memory::sqlite_sync::IntakeDb;
    use lb_memory::store::Store;

    fn intake() -> IntakeDb {
        IntakeDb::open_ephemeral().unwrap()
    }

    #[test]
    fn sqlite_intake_replicates_new_fact() {
        let store = Store::ephemeral().unwrap();
        let pipe = Pipeline::new(&store, PipelineOptions::default());
        let db = intake();
        let mut ex = FakeExtractor {
            mentions: vec![mention("Tim Cook", "person", 0.99), mention("Apple", "organization", 0.98)],
            classify_scores: vec![],
            relations: vec![],
        };
        let (reports, stats) =
            pipe.remember_via_sqlite(&mut ex, &db, "Tim Cook runs Apple.", &scope(), None, prov(), None).unwrap();
        assert_eq!(reports.len(), 1);
        assert_eq!(stats.via, "embedded");
        assert_eq!(stats.applied_intake, 1);
        let written = reports[0].written.as_ref().expect("sync must attach the written entry");
        assert_eq!(written.status, MemoryStatus::Active);
        assert_eq!(written.provenance.message_refs, vec!["msg-1".to_string()]);

        // Same converged graph as the direct path.
        let (mem, ent, men, rev) = store.counts().unwrap();
        assert_eq!((mem, ent, men, rev), (1, 2, 2, 0));

        // Cursor advanced; nothing pending.
        assert_eq!(db.pending_count().unwrap(), 0);
        assert_eq!(db.cursor().unwrap(), 1);

        // Recall finds it.
        let hits = pipe.recall("tim cook", &scope(), 5).unwrap();
        assert!(!hits.is_empty());
    }

    #[test]
    fn sqlite_sync_is_idempotent() {
        let store = Store::ephemeral().unwrap();
        let pipe = Pipeline::new(&store, PipelineOptions::default());
        let db = intake();
        let mut ex = FakeExtractor {
            mentions: vec![mention("Python", "technology", 0.95)],
            classify_scores: vec![],
            relations: vec![],
        };
        let (_, first) =
            pipe.remember_via_sqlite(&mut ex, &db, "User prefers Python for scripting.", &scope(), None, prov(), None)
                .unwrap();
        assert_eq!(first.applied_intake, 1);
        // Second tick converges with nothing to do.
        let second = pipe.sync_intake(&db, None).unwrap();
        assert_eq!(second.applied_intake, 0);
        let (mem, _, _, _) = store.counts().unwrap();
        assert_eq!(mem, 1);
    }

    #[test]
    fn sqlite_duplicate_strengthens() {
        let store = Store::ephemeral().unwrap();
        let pipe = Pipeline::new(&store, PipelineOptions::default());
        let db = intake();
        let mut ex = FakeExtractor {
            mentions: vec![mention("Python", "technology", 0.95)],
            classify_scores: vec![],
            relations: vec![],
        };
        let (first, _) =
            pipe.remember_via_sqlite(&mut ex, &db, "User prefers Python for scripting.", &scope(), None, prov(), None)
                .unwrap();
        let id = first[0].written.as_ref().unwrap().id;
        let (second, _) =
            pipe.remember_via_sqlite(&mut ex, &db, "User prefers Python for scripting tasks.", &scope(), None, prov(), None)
                .unwrap();
        assert!(matches!(second[0].decision, FactDecision::Duplicate { .. }));
        let (mem, _, _, _) = store.counts().unwrap();
        assert_eq!(mem, 1, "duplicate must not add a node");
        // Touched once at stage time (+1 over the insert default of 5).
        assert_eq!(store.get_memory(id).unwrap().unwrap().importance, 6);
    }

    #[test]
    fn sqlite_conflict_goes_to_review() {
        let store = Store::ephemeral().unwrap();
        let pipe = Pipeline::new(&store, PipelineOptions::default());
        let db = intake();
        let mut ex = FakeExtractor {
            mentions: vec![mention("Python", "technology", 0.95)],
            classify_scores: vec![],
            relations: vec![],
        };
        pipe
            .remember_via_sqlite(&mut ex, &db, "User prefers Python for scripting.", &scope(), None, prov(), None)
            .unwrap();
        let (reports, _) =
            pipe.remember_via_sqlite(&mut ex, &db, "User hates Python for scripting.", &scope(), None, prov(), None)
                .unwrap();
        let written = reports[0].written.as_ref().unwrap();
        assert_eq!(written.status, MemoryStatus::NeedsReview);
        assert_eq!(store.conflicts_of(written.id).unwrap().len(), 1);
    }

    #[test]
    fn sqlite_file_backed_intake_sibling() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("mem.lbdb").display().to_string();
        let store = Store::open(&db_path).unwrap();
        let db = store.intake_db().unwrap();
        assert!(db.path().ends_with("mem.intake.sqlite"), "got {}", db.path());
        let pipe = Pipeline::new(&store, PipelineOptions::default());
        let mut ex = FakeExtractor {
            mentions: vec![mention("LadybugDB", "technology", 0.9)],
            classify_scores: vec![],
            relations: vec![],
        };
        let (reports, stats) =
            pipe.remember_via_sqlite(&mut ex, &db, "LadybugDB is a graph database.", &scope(), None, prov(), None)
                .unwrap();
        assert_eq!(stats.via, "embedded");
        assert!(reports[0].written.is_some());
        // WAL mode is on so the C++ tailer can replicate from this file.
        let journal: String = db
            .path()
            .to_string();
        assert!(std::path::Path::new(&journal).exists());
        let (mem, ent, _, _) = store.counts().unwrap();
        assert_eq!((mem, ent), (1, 1));
    }

    #[test]
    fn sqlite_fallback_converges_partial_extension_tick() {
        // The extension applies statement-by-statement, so a failed tick can
        // leave committed node rows behind (observed: nodes land, then the
        // edge batch fails on `_sqlite_rowid`). The fallback must reuse those
        // rows, not twin them.
        use lb_memory::types::{Category, MemoryStatus};
        let store = Store::ephemeral().unwrap();
        let db = IntakeDb::open_ephemeral().unwrap();
        let intake_id = db.append_intake("Ada Lovelace wrote the first algorithm.", "u", "p", "", 5).unwrap();
        let staged_mem = db
            .stage_memory(intake_id, 0, "Ada Lovelace wrote the first algorithm.", "technical", "u", "p", "", 5, "active")
            .unwrap();
        let staged_ent = db.stage_entity(intake_id, "Ada Lovelace", "person", r#"["Ada Lovelace"]"#, None).unwrap();
        db.stage_mention(intake_id, staged_ent, staged_mem, "Ada Lovelace", 1.0, 0, 12, &lb_memory::store::EdgeMeta::default()).unwrap();
        db.record_provenance(intake_id, 0, "{}").unwrap();
        db.record_outcome(intake_id, r#"{"kind":"written","status":"active"}"#).unwrap();
        // Simulate the extension's committed rows from a tick that failed later.
        let mid = store
            .insert_memory(
                "Ada Lovelace wrote the first algorithm.",
                Category::Technical,
                "u",
                "p",
                "",
                5,
                MemoryStatus::Active,
                &Provenance::default(),
            )
            .unwrap();
        let (eid, _) = store.upsert_entity("Ada Lovelace", "person", &["Ada Lovelace".to_string()], None).unwrap();
        store.link_mention(eid, mid, "Ada Lovelace", 1.0, 0, 12, &lb_memory::store::EdgeMeta::default()).unwrap();

        let stats = lb_memory::sqlite_sync::replicate_to_store(&db, &store).unwrap();
        assert_eq!(stats.applied_intake, 1);
        let (mem, ent, men, _) = store.counts().unwrap();
        assert_eq!((mem, ent, men), (1, 1, 1), "no twins from the partial tick");
        assert_eq!(db.pending_count().unwrap(), 0);
    }

    #[test]
    fn extension_tick_never_leaves_partial_state() {
        // No partial state, ever, whichever extension build is under test:
        // - pre-fix builds fail the tick on the `_sqlite_rowid` slot the
        //   pre-existing rel tables lack; per-commit transactions must leave
        //   zero rows behind (pre-tx-fix, nodes committed before the edge
        //   failure: (1, 1, 0));
        // - fixed builds replicate the whole batch (1, 1, 1).
        // Testing-only path: explicit `LOAD EXTENSION '<path>'`.
        // Set LB_SQLITE_EXTENSION to a built libsqlite.lbug_extension.
        let ext = match std::env::var("LB_SQLITE_EXTENSION") {
            Ok(p) => p,
            Err(_) => {
                eprintln!("skip: LB_SQLITE_EXTENSION unset");
                return;
            }
        };
        use lb_memory::sqlite_sync::ExtensionSource;
        let dir = tempfile::tempdir().unwrap();
        let store = Store::open(dir.path().join("t.lbdb").to_str().unwrap()).unwrap();
        let db = store.intake_db().unwrap();
        assert_ne!(db.path(), ":memory:");
        let pipe = Pipeline::new(&store, PipelineOptions::default());
        let intake_id = db.append_intake("Ada wrote the algorithm.", "u", "p", "", 5).unwrap();
        let staged_mem = db.stage_memory(intake_id, 0, "Ada wrote the algorithm.", "technical", "u", "p", "", 5, "active").unwrap();
        let staged_ent = db.stage_entity(intake_id, "Ada", "person", r#"["Ada"]"#, None).unwrap();
        db.stage_mention(intake_id, staged_ent, staged_mem, "Ada", 1.0, 0, 3, &lb_memory::store::EdgeMeta::default()).unwrap();
        db.record_provenance(intake_id, 0, "{}").unwrap();
        db.record_outcome(intake_id, r#"{"kind":"written","status":"active"}"#).unwrap();

        match pipe.sync_via_extension(&db, ExtensionSource::File(&ext)) {
            Ok(stats) => {
                assert_eq!(stats.via, "extension");
                assert_eq!(store.counts().unwrap(), (1, 1, 1, 0));
                assert_eq!(db.pending_count().unwrap(), 0);
            }
            Err(e) => {
                assert!(format!("{e:#}").contains("_sqlite_rowid"), "unexpected error: {e:#}");
                assert_eq!(store.counts().unwrap(), (0, 0, 0, 0), "failed tick must be atomic");
                // And the embedded tick still converges afterwards.
                let stats = pipe.sync_intake(&db, None).unwrap();
                assert_eq!(stats.applied_intake, 1);
                assert_eq!(store.counts().unwrap().0, 1);
            }
        }
    }

    #[test]
    fn extension_tick_commits_clean_batches() {
        // Nodes-only staging has no edge statements, so the extension tick
        // succeeds: proves the COMMIT path of the per-batch transactions.
        let ext = match std::env::var("LB_SQLITE_EXTENSION") {
            Ok(p) => p,
            Err(_) => {
                eprintln!("skip: LB_SQLITE_EXTENSION unset");
                return;
            }
        };
        use lb_memory::sqlite_sync::ExtensionSource;
        let dir = tempfile::tempdir().unwrap();
        let store = Store::open(dir.path().join("c.lbdb").to_str().unwrap()).unwrap();
        let db = store.intake_db().unwrap();
        let pipe = Pipeline::new(&store, PipelineOptions::default());
        let intake_id = db.append_intake("Ada wrote the algorithm.", "u", "p", "", 5).unwrap();
        db.stage_memory(intake_id, 0, "Ada wrote the algorithm.", "technical", "u", "p", "", 5, "active").unwrap();
        db.record_provenance(intake_id, 0, "{}").unwrap();
        db.record_outcome(intake_id, r#"{"kind":"written","status":"active"}"#).unwrap();

        let stats = pipe.sync_via_extension(&db, ExtensionSource::File(&ext)).unwrap();
        assert_eq!(stats.via, "extension");
        assert_eq!(stats.applied_intake, 1);
        assert_eq!(db.pending_count().unwrap(), 0);
        assert_eq!(store.counts().unwrap().0, 1);
    }

    fn rel(head: &str, tail: &str, kind: &str) -> lb_memory::types::ScoredRelation {
        lb_memory::types::ScoredRelation {
            relation_type: kind.into(),
            head_text: head.into(),
            head_confidence: 0.95,
            tail_text: tail.into(),
            tail_confidence: 0.95,
            score: 0.9,
        }
    }

    #[test]
    fn typed_relations_replace_cooccur() {
        let store = Store::ephemeral().unwrap();
        let pipe = Pipeline::new(&store, PipelineOptions::default());
        let mut ex = FakeExtractor {
            mentions: vec![mention("Steve Jobs", "person", 0.99), mention("Apple", "organization", 0.98)],
            classify_scores: vec![],
            relations: vec![rel("Steve Jobs", "Apple", "founded")],
        };
        pipe.remember(&mut ex, "Steve Jobs founded Apple.", &scope(), None, prov()).unwrap();
        // Exactly one edge, typed — no coOccurs twin.
        assert_eq!(
            store.relates_to_labels().unwrap(),
            vec![("Steve Jobs".to_string(), "Apple".to_string(), "founded by".to_string())]
        );
        // Recall surfaces the relation on the hit.
        let hits = pipe.recall("who founded Apple", &scope(), 5).unwrap();
        assert!(!hits.is_empty());
        assert!(hits[0].relations.iter().any(|r| r.contains("founded")), "{:?}", hits[0].relations);
    }

    #[test]
    fn cooccur_fallback_without_typed() {
        let store = Store::ephemeral().unwrap();
        let pipe = Pipeline::new(&store, PipelineOptions::default());
        let mut ex = FakeExtractor {
            mentions: vec![mention("Tim Cook", "person", 0.99), mention("Apple", "organization", 0.98)],
            classify_scores: vec![],
            relations: vec![],
        };
        pipe.remember(&mut ex, "Tim Cook runs Apple.", &scope(), None, prov()).unwrap();
        let edges = store.relates_to_labels().unwrap();
        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0].2, "coOccurs");
    }

    #[test]
    fn sqlite_typed_relations_replicate_idempotently() {
        let store = Store::ephemeral().unwrap();
        let pipe = Pipeline::new(&store, PipelineOptions::default());
        let db = intake();
        let mut ex = FakeExtractor {
            mentions: vec![mention("Steve Jobs", "person", 0.99), mention("Apple", "organization", 0.98)],
            classify_scores: vec![],
            relations: vec![rel("Steve Jobs", "Apple", "founded")],
        };
        let (reports, stats) =
            pipe.remember_via_sqlite(&mut ex, &db, "Steve Jobs founded Apple.", &scope(), None, prov(), None)
                .unwrap();
        assert_eq!(stats.via, "embedded");
        assert!(reports[0].written.is_some());
        assert_eq!(
            store.relates_to_labels().unwrap(),
            vec![("Steve Jobs".to_string(), "Apple".to_string(), "founded by".to_string())]
        );
        // Re-tick converges with no duplicate edge.
        let second = pipe.sync_intake(&db, None).unwrap();
        assert_eq!(second.applied_intake, 0);
        assert_eq!(store.relates_to_labels().unwrap().len(), 1);
    }

    #[test]
    fn variant_phrases_collapse_to_one_canonical_label() {
        let store = Store::ephemeral().unwrap();
        let pipe = Pipeline::new(&store, PipelineOptions::default());
        // "great ceo of" and "ceo" must land as the same canonical edge.
        let mut ex = FakeExtractor {
            mentions: vec![mention("Steve Jobs", "person", 0.99), mention("Apple", "organization", 0.98)],
            classify_scores: vec![("fact".to_string(), 0.9)],
            relations: vec![scirel("Steve Jobs", "Apple", "great ceo of")],
        };
        pipe.remember(&mut ex, "Steve Jobs was a great CEO of Apple.", &scope(), None, prov()).unwrap();
        ex.mentions = vec![mention("John Ternus", "person", 0.95), mention("Apple", "organization", 0.98)];
        ex.relations = vec![scirel("John Ternus", "Apple", "ceo")];
        pipe.remember(&mut ex, "John Ternus is the CEO of Apple.", &scope(), None, prov()).unwrap();
        let edges = store.relates_to_labels().unwrap();
        let kinds: std::collections::HashSet<String> =
            edges.iter().filter(|(_, _, t)| t != &"coOccurs").map(|(_, _, t)| t.clone()).collect();
        assert_eq!(kinds, ["chief executive officer".to_string()].into_iter().collect(), "{edges:?}");
    }

    #[test]
    fn recall_boosts_typed_match() {
        let store = Store::ephemeral().unwrap();
        let pipe = Pipeline::new(&store, PipelineOptions::default());
        // Lower id, no typed relation.
        let mut ex = FakeExtractor {
            mentions: vec![mention("Apple", "organization", 0.98), mention("Vision Pro", "product", 0.9)],
            classify_scores: vec![],
            relations: vec![],
        };
        pipe.remember(&mut ex, "Apple released Vision Pro.", &scope(), None, prov()).unwrap();
        // Higher id, founder relation — the boost must outrank id order.
        ex.mentions = vec![mention("Steve Jobs", "person", 0.99), mention("Apple", "organization", 0.98)];
        ex.relations = vec![rel("Steve Jobs", "Apple", "founded")];
        let second =
            pipe.remember(&mut ex, "Steve Jobs founded Apple.", &scope(), None, prov()).unwrap();
        let winner = second[0].written.as_ref().unwrap().id;
        let hits = pipe.recall("founder Apple", &scope(), 5).unwrap();
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].entry.id, winner, "typed-relation boost should outrank id order");
        assert!(hits[0].relations.iter().any(|r| r == "founded by via Steve Jobs"), "{:?}", hits[0].relations);
    }

    fn scirel(head: &str, tail: &str, kind: &str) -> lb_memory::types::ScoredRelation {
        lb_memory::types::ScoredRelation {
            relation_type: kind.into(),
            head_text: head.into(),
            head_confidence: 0.9,
            tail_text: tail.into(),
            tail_confidence: 0.9,
            score: 0.9,
        }
    }

    #[test]
    fn pronoun_links_in_call() {
        let store = Store::ephemeral().unwrap();
        let pipe = Pipeline::new(&store, PipelineOptions::default());
        let mut ex = FakeExtractor {
            mentions: vec![
                mention("Steve Jobs", "person", 0.99),
                mention("Apple", "organization", 0.98),
                mention("He", "person", 0.9),
            ],
            classify_scores: vec![("fact".to_string(), 0.9)],
            relations: vec![],
        };
        let reports = pipe
            .remember_auto(&mut ex, "Steve Jobs founded Apple. He was a visionary.", &scope(), prov())
            .unwrap();
        assert_eq!(reports.len(), 2);
        // No "He" node ever exists.
        assert!(store.list_entities().unwrap().iter().all(|e| e.canonical != "He"));
        // The second fact links the antecedent.
        let fact2 = reports[1].written.as_ref().unwrap();
        let ents = store.entities_for_memory(fact2.id).unwrap();
        assert!(ents.contains(&"Steve Jobs".to_string()), "{ents:?}");
    }

    #[test]
    fn pronoun_links_across_calls() {
        let store = Store::ephemeral().unwrap();
        let pipe = Pipeline::new(&store, PipelineOptions::default());
        let mut ex = FakeExtractor {
            mentions: vec![mention("Steve Jobs", "person", 0.99)],
            classify_scores: vec![("fact".to_string(), 0.9)],
            relations: vec![],
        };
        pipe.remember(&mut ex, "Steve Jobs founded Apple.", &scope(), None, prov()).unwrap();
        ex.mentions = vec![mention("He", "person", 0.9)];
        let reports = pipe.remember(&mut ex, "He was a visionary.", &scope(), None, prov()).unwrap();
        assert!(store.list_entities().unwrap().iter().all(|e| e.canonical != "He"));
        let fact2 = reports[0].written.as_ref().unwrap();
        assert!(store.entities_for_memory(fact2.id).unwrap().contains(&"Steve Jobs".to_string()));
    }

    #[test]
    fn unresolvable_pronoun_is_dropped() {
        let store = Store::ephemeral().unwrap();
        let pipe = Pipeline::new(&store, PipelineOptions::default());
        let mut ex = FakeExtractor {
            mentions: vec![mention("He", "person", 0.9)],
            classify_scores: vec![("fact".to_string(), 0.9)],
            relations: vec![],
        };
        let reports = pipe.remember(&mut ex, "He was great.", &scope(), None, prov()).unwrap();
        assert!(matches!(
            reports[0].decision,
            lb_memory::types::FactDecision::LowConfidence { .. }
        ));
        assert!(store.list_entities().unwrap().is_empty());
    }

    #[test]
    fn implicit_org_links_lone_person_with_role() {
        let store = Store::ephemeral().unwrap();
        let pipe = Pipeline::new(&store, PipelineOptions::default());
        let mut ex = FakeExtractor {
            mentions: vec![mention("Apple", "organization", 0.98)],
            classify_scores: vec![("fact".to_string(), 0.9)],
            relations: vec![],
        };
        pipe.remember(&mut ex, "Apple released Vision Pro.", &scope(), None, prov()).unwrap();
        ex.mentions = vec![mention("John Ternus", "person", 0.95)];
        let reports = pipe.remember(&mut ex, "John Ternus is the CEO.", &scope(), None, prov()).unwrap();
        assert!(reports[0].written.is_some());
        let edges = store.relates_to_labels().unwrap();
        assert!(
            edges.iter().any(|(a, b, t)| a == "John Ternus" && b == "Apple" && t == "chief executive officer"),
            "{edges:?}"
        );
    }

    #[test]
    fn time_range_from_explicit_year() {
        let store = Store::ephemeral().unwrap();
        let pipe = Pipeline::new(&store, PipelineOptions::default());
        let mut ex = FakeExtractor {
            mentions: vec![
                mention("Steve Jobs", "person", 0.99),
                mention("Apple", "organization", 0.98),
                mention("1976", "date", 0.9),
            ],
            classify_scores: vec![("fact".to_string(), 0.9)],
            relations: vec![],
        };
        pipe.remember(&mut ex, "Steve Jobs founded Apple in 1976.", &scope(), None, prov()).unwrap();
        let edges = store.relates_to_full().unwrap();
        let sj = edges.iter().find(|(a, b, _, _, _, _, _, _, _)| a == "Steve Jobs" && b == "Apple");
        let (_, _, _, _, vf, vt, _, _, _) = sj.expect("typed or cooccur edge").clone();
        assert_eq!((vf.as_str(), vt.as_str()), ("1976-01-01", "1976-12-31"));
    }

    #[test]
    fn time_range_present_and_past_defaults() {
        let store = Store::ephemeral().unwrap();
        let pipe = Pipeline::new(&store, PipelineOptions::default());
        let mut ex = FakeExtractor {
            mentions: vec![mention("John Ternus", "person", 0.95), mention("Apple", "organization", 0.98)],
            classify_scores: vec![("fact".to_string(), 0.9)],
            relations: vec![],
        };
        pipe.remember(&mut ex, "Now John Ternus is the CEO of Apple.", &scope(), None, prov()).unwrap();
        let today = time::OffsetDateTime::now_utc().date().to_string();
        let edges = store.relates_to_full().unwrap();
        assert!(!edges.is_empty());
        for (_, _, _, _, vf, vt, tense, _, _) in &edges {
            assert_eq!(vf, &today, "{edges:?}");
            assert_eq!(tense, "present");
            let _ = vt;
        }
    }

    #[test]
    fn viewpoint_labels_typed_edges() {
        let store = Store::ephemeral().unwrap();
        let pipe = Pipeline::new(&store, PipelineOptions::default());
        let mut ex = FakeExtractor {
            mentions: vec![mention("Steve Jobs", "person", 0.99), mention("Apple", "organization", 0.98)],
            classify_scores: vec![("opinion".to_string(), 0.85)],
            relations: vec![scirel("Steve Jobs", "Apple", "founded")],
        };
        pipe.remember(&mut ex, "Steve Jobs was a great CEO of Apple.", &scope(), None, prov()).unwrap();
        let edges = store.relates_to_full().unwrap();
        let found = edges.iter().find(|(_, _, kind, _, _, _, _, vp, vpc)| {
            kind == &"founded by" && vp == &"opinion" && (*vpc - 0.85).abs() < 1e-6
        });
        assert!(found.is_some(), "typed opinion edge missing: {edges:?}");
    }

    #[test]
    fn succession_dated_not_conflict() {
        // Year-mismatch cue fires, but the tenures are disjoint:
        // succession, not contradiction.
        let store = Store::ephemeral().unwrap();
        let pipe = Pipeline::new(&store, PipelineOptions::default());
        let mut ex = FakeExtractor {
            mentions: vec![
                mention("Steve Jobs", "person", 0.99),
                mention("Apple", "organization", 0.98),
                mention("2011", "date", 0.9),
            ],
            classify_scores: vec![("fact".to_string(), 0.9)],
            relations: vec![],
        };
        pipe.remember(
            &mut ex,
            "Steve Jobs was CEO of Apple until 2011.",
            &scope(),
            None,
            prov(),
        )
        .unwrap();
        ex.mentions = vec![
            mention("John Ternus", "person", 0.95),
            mention("Apple", "organization", 0.98),
            mention("2025", "date", 0.9),
        ];
        let reports = pipe
            .remember(
                &mut ex,
                "John Ternus became CEO of Apple in 2025.",
                &scope(),
                None,
                prov(),
            )
            .unwrap();
        assert!(
            matches!(reports[0].decision, lb_memory::types::FactDecision::New { .. }),
            "dated succession must not conflict: {:?}",
            reports[0].decision
        );
        assert_eq!(reports[0].written.as_ref().unwrap().status, lb_memory::types::MemoryStatus::Active);
    }

    #[test]
    fn overlap_still_conflicts() {
        // Same entity, antonym cue, unknown time: genuinely contradictory.
        let store = Store::ephemeral().unwrap();
        let pipe = Pipeline::new(&store, PipelineOptions::default());
        let mut ex = FakeExtractor {
            mentions: vec![mention("John", "person", 0.95)],
            classify_scores: vec![("fact".to_string(), 0.9)],
            relations: vec![],
        };
        pipe.remember(&mut ex, "John loves being CEO.", &scope(), None, prov()).unwrap();
        ex.mentions = vec![mention("John", "person", 0.95)];
        let reports = pipe.remember(&mut ex, "John hates being CEO.", &scope(), None, prov()).unwrap();
        assert!(
            matches!(reports[0].decision, lb_memory::types::FactDecision::Conflict { .. }),
            "overlapping antonyms must conflict: {:?}",
            reports[0].decision
        );
        assert_eq!(reports[0].written.as_ref().unwrap().status, lb_memory::types::MemoryStatus::NeedsReview);
    }

    #[test]
    fn sqlite_quarantine_stays_hidden() {
        let store = Store::ephemeral().unwrap();
        let pipe = Pipeline::new(&store, PipelineOptions::default());
        let db = intake();
        let mut ex = FakeExtractor { mentions: vec![], classify_scores: vec![], relations: vec![] };
        let (reports, _) =
            pipe.remember_via_sqlite(&mut ex, &db, "Hmm, vague rambling without names.", &scope(), None, prov(), None)
                .unwrap();
        assert!(matches!(reports[0].decision, FactDecision::LowConfidence { .. }));
        assert_eq!(reports[0].written.as_ref().unwrap().status, MemoryStatus::Quarantined);
        assert!(pipe.recall("rambling", &scope(), 5).unwrap().is_empty());
    }
}
