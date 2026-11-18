//! Parity test: Rust extraction must match the python ONNX baseline.
//!
//! Regenerate the fixture with:
//! ```sh
//! cd ../ladybug-memory && .venv/bin/python examples/onnx_baseline.py \
//!   --threshold 0.5 --dump ../ladybug-memory-rs/crates/gliner-onnx/tests/data/baseline.json
//! ```
//!
//! Needs the model snapshot + ORT lib at test time (`lbm fetch-model`
//! provisions both; see workspace README).

use gliner_onnx::GlinerOnnx;
use serde_json::Value;
use std::collections::HashMap;

fn snapshot_dir() -> String {
    std::env::var("GLINER_SNAPSHOT").unwrap_or_else(|_| {
        format!("{}/.cache/ladybug-memory/models/gliner2.5-base-v1-nico-onnx", std::env::var("HOME").unwrap())
    })
}

#[test]
fn parity_with_python_onnx_baseline() {
    let fixture: Value = serde_json::from_str(include_str!("data/baseline.json")).unwrap();
    let threshold = fixture["threshold"].as_f64().unwrap() as f32;
    let labels: Vec<String> = fixture["labels"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap().to_string())
        .collect();

    let mut model = GlinerOnnx::load(snapshot_dir()).expect("load model");

    for (name, case) in fixture["cases"].as_object().unwrap() {
        let text = case["text"].as_str().unwrap();

        // 1. Prompt parity: exact input_ids, word routing, query positions.
        let built = model.build_inputs(text, &labels).expect("build inputs");
        let expected_ids: Vec<i64> =
            case["input_ids"].as_array().unwrap().iter().map(|v| v.as_i64().unwrap()).collect();
        assert_eq!(built.input_ids, expected_ids, "{name}: input_ids diverge");
        let expected_w: Vec<usize> = case["word_first_positions"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        assert_eq!(built.word_first, expected_w, "{name}: word routing diverges");
        let expected_q: Vec<usize> = case["query_positions"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        assert_eq!(built.query_pos, expected_q, "{name}: query positions diverge");

        // 2. Mention parity: same (label, text, span) set; confidence within 1e-4.
        let out = model.extract(text, &labels, threshold).expect("extract");
        let mut expected: HashMap<(String, String, usize, usize), f64> = HashMap::new();
        for (label, v) in case["entities"].as_object().unwrap() {
            for m in v["mentions"].as_array().unwrap() {
                expected.insert(
                    (
                        label.clone(),
                        m["text"].as_str().unwrap().to_string(),
                        m["start"].as_u64().unwrap() as usize,
                        m["end"].as_u64().unwrap() as usize,
                    ),
                    m["confidence"].as_f64().unwrap(),
                );
            }
        }
        assert_eq!(
            out.mentions.len(),
            expected.len(),
            "{name}: mention count differs (rust={} python={})\nrust: {:?}",
            out.mentions.len(),
            expected.len(),
            out.mentions
        );
        for m in &out.mentions {
            let key = (m.label.clone(), m.text.clone(), m.start, m.end);
            let exp = expected.remove(&key).unwrap_or_else(|| panic!("{name}: unexpected mention {key:?}"));
            assert!(
                (m.confidence as f64 - exp).abs() < 1e-4,
                "{name}: confidence differs for {key:?}: rust={} python={exp}",
                m.confidence
            );
        }
        assert!(expected.is_empty(), "{name}: missing mentions: {expected:?}");
    }
}

#[test]
fn parity_classify_with_python_onnx_baseline() {
    let fixture: Value = serde_json::from_str(include_str!("data/baseline.json")).unwrap();
    let cls = &fixture["classify"];
    let task = cls["task"].as_str().unwrap();
    let labels: Vec<String> = cls["labels"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap().to_string())
        .collect();

    let mut model = GlinerOnnx::load(snapshot_dir()).expect("load model");
    for (name, case) in cls["cases"].as_object().unwrap() {
        let text = case["text"].as_str().unwrap();

        // Prompt parity first.
        let built = model.build_classify_inputs(text, task, &labels).expect("build inputs");
        let expected_ids: Vec<i64> =
            case["input_ids"].as_array().unwrap().iter().map(|v| v.as_i64().unwrap()).collect();
        assert_eq!(built.input_ids, expected_ids, "{name}: classify input_ids diverge");
        let expected_q: Vec<usize> = case["query_positions"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        assert_eq!(built.query_pos, expected_q, "{name}: [L] positions diverge");

        // Score parity per label within 1e-4.
        let out = model.classify(text, task, &labels, false, 0.5).expect("classify");
        let expected: Vec<f64> =
            case["probs"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
        assert_eq!(out.scores.len(), expected.len(), "{name}: label count differs");
        for ((label, got), want) in out.scores.iter().zip(expected.iter()) {
            assert!(
                (*got as f64 - *want).abs() < 1e-4,
                "{name}: {label} differs: rust={got} python={want}"
            );
        }
        // Same argmax verdict as python's single-label decode.
        let best_rust = out.best().unwrap().0.clone();
        let best_py = labels[expected
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0]
            .clone();
        assert_eq!(best_rust, best_py, "{name}: argmax differs");
    }
}
