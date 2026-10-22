//! Native Rust inference for the DanKau `gliner2.5-base-v1-onnx` export.
//!
//! The export carries three fragments (`encoder.onnx`, `boundary.onnx`,
//! `classifier.onnx`) over a `deberta-v3-base` backbone. The boundary head is
//! a **single** graph — no length buckets, no manifest — which makes the
//! pipeline short:
//!
//! 1. word-split the text ([WhitespaceSplitter]), lowercase the words,
//! 2. build the GLiNER2 prompt `( [P] entities ( [E] label … ) ) [SEP_TEXT] …`,
//!    tokenizing each prompt token standalone (as `gliner2`'s
//!    `SchemaTransformer._format_input_with_mapping` does),
//! 3. run `encoder.onnx`, gather `token_states` (first sub-token of each
//!    word) and `query_states` (hidden state at each `[E]` marker),
//! 4. run `boundary.onnx`, decode with `sigmoid(pair_logit) >= threshold`,
//!    the null-logit abstention gate, and per-label `flat` overlap
//!    resolution — the same contract as
//!    `gliner2.models.boundary.engine.BoundaryExtractor` with default
//!    settings (`pair_temperature=1`, `abstention_threshold=0.5`,
//!    `overlap_policy="flat"`, `adaptive_threshold=false`).
//!
//! Word spans are half-open `[start, end)` over words; character offsets are
//! half-open over the (punctuation-normalized) text, converted exactly as
//! `token_boundaries_to_character_offsets` does:
//! `char_start = starts[start]`, `char_end = ends[end - 1]`.
//!
//! What this does **not** do yet: relations (the export has no relation
//! scorer — `gliner2` scores them in torch via `relation_scorer`, so they
//! need a separate export or the LLM fallback) and classification
//! (`classifier.onnx` is loaded but not yet wired to `[L]` prompts).

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result, anyhow};
use ort::environment::Environment;
use ort::value::Tensor;
use ort::session::Session;
use regex::Regex;
use tokenizers::Tokenizer;

/// Special-token ids for this checkpoint (from `gliner2_config.json`).
/// Do not hard-code the multilingual export's ids here — they differ.
pub const SEP_TEXT_ID: u32 = 128002;
pub const P_ID: u32 = 128003;
pub const E_ID: u32 = 128005;

/// One extracted entity mention.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Mention {
    /// Entity label, e.g. `"person"`.
    pub label: String,
    /// Surface text sliced from the normalized input.
    pub text: String,
    /// `sigmoid(pair_logit)`.
    pub confidence: f32,
    /// Half-open char span into [`Extraction::normalized_text`].
    pub start: usize,
    pub end: usize,
    /// Half-open word span.
    pub word_start: usize,
    pub word_end: usize,
}

/// Full result of one extraction call.
#[derive(Debug, Clone)]
pub struct Extraction {
    pub normalized_text: String,
    pub mentions: Vec<Mention>,
    /// Per-label `sigmoid(null_logit)`; queries above 0.5 abstain.
    pub null_probs: Vec<(String, f32)>,
}

/// One word with char offsets into the original text.
#[derive(Debug, Clone)]
pub struct Word {
    /// Lowercased surface (what the model sees).
    pub lowered: String,
    /// Half-open char span into the original text.
    pub start: usize,
    pub end: usize,
}

/// Regex word splitter mirroring `gliner2.processing.word_splitter.
/// WhitespaceTokenSplitter`: URLs, emails, @handles, word runs with interior
/// `-`/`_`, then any single non-space character. Lower-casing applies to the
/// yielded value only — offsets index the original text, since Unicode case
/// folding can change length.
pub struct WhitespaceSplitter {
    re: Regex,
}

impl WhitespaceSplitter {
    pub fn new() -> Result<Self> {
        let re = Regex::new(
            r"(?i)(?:https?://\S+|www\.\S+)|[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}|@[a-z0-9_]+|[\w\p{M}‌‍]+(?:[-_][\w\p{M}‌‍]+)*|\S",
        )?;
        Ok(Self { re })
    }

    /// Split `text`, returning words with char offsets.
    pub fn split(&self, text: &str) -> Vec<Word> {
        // Map byte offsets (regex) to char offsets (python `re` semantics).
        let mut byte_to_char = vec![0usize; text.len() + 1];
        let mut ci = 0;
        for (bi, _) in text.char_indices() {
            byte_to_char[bi] = ci;
            ci += 1;
        }
        byte_to_char[text.len()] = ci;
        // Fill gaps (continuation bytes) with the preceding char index.
        let mut last = 0;
        for slot in byte_to_char.iter_mut() {
            if *slot == 0 {
                *slot = last;
            } else {
                last = *slot;
            }
        }

        self.re
            .find_iter(text)
            .map(|m| {
                let surface = &text[m.start()..m.end()];
                Word {
                    lowered: surface.to_lowercase(),
                    start: byte_to_char[m.start()],
                    end: byte_to_char[m.end()],
                }
            })
            .collect()
    }
}

impl Default for WhitespaceSplitter {
    fn default() -> Self {
        Self::new().expect("word-splitter regex must compile")
    }
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let z = x.exp();
        z / (1.0 + z)
    }
}

/// Prompt tokens for one entity schema group, mirroring
/// `SchemaTransformer._transform_schema("entities", labels, "[E]")`:
/// `( [P] entities ( [E] label … ) )`.
fn entity_schema_tokens(labels: &[String]) -> Vec<String> {
    let mut toks = vec!["(".to_string(), "[P]".to_string(), "entities".to_string(), "(".to_string()];
    for lb in labels {
        toks.push("[E]".to_string());
        toks.push(lb.clone());
    }
    toks.push(")".to_string());
    toks.push(")".to_string());
    toks
}

/// Loaded model: tokenizer + encoder/boundary sessions.
pub struct GlinerOnnx {
    tokenizer: Tokenizer,
    #[allow(dead_code)] // sessions borrow it; keep alive explicitly
    env: Arc<Environment>,
    encoder: Session,
    boundary: Session,
    splitter: WhitespaceSplitter,
    snapshot_dir: PathBuf,
}

impl GlinerOnnx {
    /// Load from a DanKau snapshot directory (or the HF cache snapshot that
    /// holds `tokenizer.json` + `onnx/*.onnx`).
    ///
    /// The ONNX Runtime dylib resolves via `ORT_DYLIB_PATH`, or
    /// `GLINER_ORT_DYLIB` to override without touching the shared variable.
    pub fn load(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref().to_path_buf();
        let dylib = std::env::var("GLINER_ORT_DYLIB")
            .or_else(|_| std::env::var("ORT_DYLIB_PATH"))
            .map_err(|_| anyhow!("set ORT_DYLIB_PATH to libonnxruntime (see README)"))?;
        let env = Arc::new(
            ort::init_from(dylib.as_str())
                .map_err(|e| anyhow!("loading ONNX Runtime dylib: {e}"))?
                .with_name("gliner-onnx")
                .build()?,
        );
        let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json"))
            .map_err(|e| anyhow!("loading tokenizer.json: {e}"))?;
        let encoder = Session::builder(&env)?
            .commit_from_file(dir.join("onnx/encoder.onnx"))
            .with_context(|| format!("loading {}", dir.join("onnx/encoder.onnx").display()))?;
        let boundary = Session::builder(&env)?
            .commit_from_file(dir.join("onnx/boundary.onnx"))
            .with_context(|| format!("loading {}", dir.join("onnx/boundary.onnx").display()))?;
        Ok(Self { tokenizer, env, encoder, boundary, splitter: WhitespaceSplitter::new()?, snapshot_dir: dir })
    }

    pub fn snapshot_dir(&self) -> &Path {
        &self.snapshot_dir
    }

    /// Append sentence punctuation like `SchemaTransformer._normalize_text`.
    fn normalize(text: &str) -> String {
        if text.is_empty() {
            return ".".to_string();
        }
        if text.ends_with(['.', '!', '?']) {
            text.to_string()
        } else {
            format!("{text}.")
        }
    }

    /// Build `input_ids`, word-first subword positions, and `[E]` query
    /// positions. Each prompt token is encoded standalone, exactly like the
    /// python `_format_input_with_mapping` loop over cached `tokenize()`.
    pub fn build_inputs(
        &self,
        text: &str,
        labels: &[String],
    ) -> Result<BuiltInputs> {
        let normalized = Self::normalize(text);
        let words = self.splitter.split(&normalized);

        let mut combined: Vec<String> = entity_schema_tokens(labels);
        combined.push("[SEP_TEXT]".to_string());
        for w in &words {
            combined.push(w.lowered.clone());
        }

        let mut input_ids: Vec<i64> = Vec::new();
        let mut word_first: Vec<usize> = Vec::with_capacity(words.len());
        let mut query_pos: Vec<usize> = Vec::with_capacity(labels.len());
        // Combined-token index of each `[E]` marker: schema tokens are
        // `( [P] entities ( [E] label … ) )`, so `[E]` sits at 4, 6, 8, ….
        let n_schema = 4 + 2 * labels.len() + 2;
        let mut e_seen = 0usize;
        for (tok_idx, tok) in combined.iter().enumerate() {
            let pos = input_ids.len();
            let enc = self
                .tokenizer
                .encode(tok.as_str(), false)
                .map_err(|e| anyhow!("tokenizing {tok:?}: {e}"))?;
            input_ids.extend(enc.get_ids().iter().map(|&id| id as i64));
            if tok_idx < n_schema {
                // Schema region: `[P]` at 1, `[E]` markers at 4, 6, ….
                if tok_idx == 1 {
                    debug_assert_eq!(input_ids[pos] as u32, P_ID, "[P] id mismatch");
                }
                if tok_idx >= 4 && (tok_idx - 4) % 2 == 0 && tok == "[E]" {
                    debug_assert_eq!(input_ids[pos] as u32, E_ID, "[E] id mismatch");
                    query_pos.push(pos);
                    e_seen += 1;
                }
            } else if tok_idx > n_schema {
                // Text region: first subword of each new word.
                word_first.push(pos);
            }
        }
        anyhow::ensure!(e_seen == labels.len(), "found {e_seen} [E] markers for {} labels", labels.len());
        anyhow::ensure!(word_first.len() == words.len(), "word routing mismatch");

        let starts: Vec<usize> = words.iter().map(|w| w.start).collect();
        let ends: Vec<usize> = words.iter().map(|w| w.end).collect();
        Ok(BuiltInputs { normalized, words, input_ids, word_first, query_pos, starts, ends })
    }

    /// Extract entities for `labels` at `threshold` (python default 0.5;
    /// ladybug uses 0.85).
    pub fn extract(&mut self, text: &str, labels: &[String], threshold: f32) -> Result<Extraction> {
        let built = self.build_inputs(text, labels)?;
        let seq_len = built.input_ids.len();
        let n_words = built.words.len();
        let n_labels = labels.len();

        let input_ids = Tensor::from_array((vec![1, seq_len as i64], built.input_ids.clone()))?.upcast();
        let attention_mask = Tensor::from_array((vec![1, seq_len as i64], vec![1i64; seq_len]))?.upcast();
        let outputs = self.encoder.run(ort::inputs![input_ids, attention_mask])?;
        let (hshape, hdata) = outputs["hidden_states"].try_extract_tensor::<f32>()?;
        let hdim: Vec<i64> = hshape.iter().copied().collect();
        anyhow::ensure!(hdim == vec![1, seq_len as i64, 768], "unexpected encoder output shape {hdim:?}");
        // Gather token_states (first sub-token of each word) and query_states
        // (hidden state at each `[E]` marker), mirroring the "first" pooling path.
        let mut token_states = vec![0f32; n_words * 768];
        let mut query_states = vec![0f32; n_labels * 768];
        for (w, &pos) in built.word_first.iter().enumerate() {
            token_states[w * 768..(w + 1) * 768].copy_from_slice(&hdata[pos * 768..(pos + 1) * 768]);
        }
        for (q, &pos) in built.query_pos.iter().enumerate() {
            query_states[q * 768..(q + 1) * 768].copy_from_slice(&hdata[pos * 768..(pos + 1) * 768]);
        }

        let token_states = Tensor::from_array((vec![1, n_words as i64, 768], token_states))?.upcast();
        let text_mask = Tensor::from_array((vec![1, n_words as i64], vec![true; n_words]))?.upcast();
        let query_states = Tensor::from_array((vec![1, n_labels as i64, 768], query_states))?.upcast();
        let query_mask = Tensor::from_array((vec![1, n_labels as i64], vec![true; n_labels]))?.upcast();
        let bouts = self.boundary.run(ort::inputs![
            "token_states" => token_states,
            "text_mask" => text_mask,
            "query_states" => query_states,
            "query_mask" => query_mask
        ])?;
        let (pshape, pair_logits) = bouts["pair_logits"].try_extract_tensor::<f32>()?;
        let pdim: Vec<i64> = pshape.iter().copied().collect();
        anyhow::ensure!(
            pdim.len() == 3 && pdim[0] == 1 && pdim[1] == n_labels as i64,
            "unexpected pair_logits shape {pdim:?}"
        );
        let pool = pdim[2] as usize;
        let (_, cand_indices) = bouts["candidate_indices"].try_extract_tensor::<i64>()?;
        let (_, cand_valid) = bouts["candidate_valid"].try_extract_tensor::<bool>()?;
        let (_, null_logits) = bouts["null_logits"].try_extract_tensor::<f32>()?;
        let mut mentions = Vec::new();
        let mut null_probs = Vec::with_capacity(n_labels);
        for (qi, label) in labels.iter().enumerate() {
            let null_p = sigmoid(null_logits[qi]);
            null_probs.push((label.clone(), null_p));
            if null_p > 0.5 {
                continue; // abstention gate, mirrors BoundaryExtractor
            }
            let mut scored: Vec<(f32, usize, usize)> = Vec::new();
            for c in 0..pool {
                if !cand_valid[qi * pool + c] {
                    continue;
                }
                let p = sigmoid(pair_logits[qi * pool + c]);
                if p < threshold {
                    continue;
                }
                let s = cand_indices[(qi * pool + c) * 2] as usize;
                let e = cand_indices[(qi * pool + c) * 2 + 1] as usize;
                scored.push((p, s, e));
            }
            for (p, s, e) in resolve_flat(scored) {
                if !(s < e && e <= n_words) {
                    continue;
                }
                let cs = built.starts[s];
                let ce = built.ends[e - 1];
                let chars: Vec<char> = built.normalized.chars().collect();
                if ce > chars.len() || cs > ce {
                    continue;
                }
                let surface: String = chars[cs..ce].iter().collect();
                let surface = surface.trim().to_string();
                if surface.is_empty() {
                    continue;
                }
                mentions.push(Mention {
                    label: label.clone(),
                    text: surface,
                    confidence: p,
                    start: cs,
                    end: ce,
                    word_start: s,
                    word_end: e,
                });
            }
        }
        // Deterministic global rank: -conf, start, end, label (mirrors
        // `stable_candidate_sort_key`).
        mentions.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap()
                .then(a.start.cmp(&b.start))
                .then(a.end.cmp(&b.end))
                .then(a.label.cmp(&b.label))
        });
        Ok(Extraction { normalized_text: built.normalized, mentions, null_probs })
    }
}

/// Tokenized prompt plus routing tables.
#[derive(Debug, Clone)]
pub struct BuiltInputs {
    pub normalized: String,
    pub words: Vec<Word>,
    pub input_ids: Vec<i64>,
    /// Subword position of each word's first sub-token.
    pub word_first: Vec<usize>,
    /// Subword positions of the `[E]` query markers, in label order.
    pub query_pos: Vec<usize>,
    pub starts: Vec<usize>,
    pub ends: Vec<usize>,
}

/// `flat` overlap policy from `gliner2.inference.overlap.resolve_overlaps`:
/// maximum-total-score set of non-overlapping half-open spans, with
/// deterministic tie-breaks (higher score → more spans → lexicographic rank).
/// Exact-boundary duplicates collapse to the highest score first.
pub fn resolve_flat(items: Vec<(f32, usize, usize)>) -> Vec<(f32, usize, usize)> {
    if items.is_empty() {
        return items;
    }
    // Rank: (-score, start, end); stable by original index.
    let mut idx: Vec<usize> = (0..items.len()).collect();
    idx.sort_by(|&a, &b| {
        items[b]
            .0
            .partial_cmp(&items[a].0)
            .unwrap()
            .then(items[a].1.cmp(&items[b].1))
            .then(items[a].2.cmp(&items[b].2))
            .then(a.cmp(&b))
    });
    // Collapse exact duplicates, keeping the first (highest-ranked).
    let mut seen = std::collections::HashSet::new();
    let mut distinct: Vec<usize> = Vec::new();
    for &i in &idx {
        if seen.insert((items[i].1, items[i].2)) {
            distinct.push(i);
        }
    }
    // Weighted interval scheduling by end time.
    let mut by_end = distinct.clone();
    by_end.sort_by(|&a, &b| {
        items[a]
            .2
            .cmp(&items[b].2)
            .then(items[a].1.cmp(&items[b].1))
            .then(items[b].0.partial_cmp(&items[a].0).unwrap())
            .then(a.cmp(&b))
    });
    let ends: Vec<usize> = by_end.iter().map(|&i| items[i].2).collect();
    let n = by_end.len();
    // best[k] = best solution using first k items (k in 0..=n).
    let mut best_score = vec![0f64; n + 1];
    let mut best_sel: Vec<Vec<usize>> = vec![Vec::new(); n + 1];
    let rank_of = |sel: &[usize]| -> Vec<(std::cmp::Reverse<u32>, usize, usize, usize)> {
        let mut rows: Vec<usize> = sel.to_vec();
        rows.sort_by(|&a, &b| {
            items[b]
                .0
                .partial_cmp(&items[a].0)
                .unwrap()
                .then(items[a].1.cmp(&items[b].1))
                .then(items[a].2.cmp(&items[b].2))
                .then(a.cmp(&b))
        });
        rows.into_iter()
            .map(|i| (std::cmp::Reverse(items[i].0.to_bits()), items[i].1, items[i].2, i))
            .collect()
    };
    // Recompute predecessors cleanly.
    let pred: Vec<Option<usize>> = (0..n)
        .map(|k| {
            let s = items[by_end[k]].1;
            let j = ends[..k].partition_point(|&e| e <= s);
            if j == 0 { None } else { Some(j - 1) }
        })
        .collect();
    for k in 0..n {
        let i = by_end[k];
        let prev = match pred[k] {
            Some(j) => j + 1,
            None => 0,
        };
        let with_score = best_score[prev] + items[i].0 as f64;
        let with_len = best_sel[prev].len() + 1;
        let without_score = best_score[k];
        let without_len = best_sel[k].len();
        let take = if with_score > without_score {
            true
        } else if with_score < without_score {
            false
        } else if with_len != without_len {
            with_len > without_len
        } else {
            let mut with_sel = best_sel[prev].clone();
            with_sel.push(i);
            rank_of(&with_sel) < rank_of(&best_sel[k])
        };
        if take {
            let mut sel = best_sel[prev].clone();
            sel.push(i);
            best_score[k + 1] = with_score;
            best_sel[k + 1] = sel;
        } else {
            best_score[k + 1] = without_score;
            best_sel[k + 1] = best_sel[k].clone();
        }
    }
    let mut out: Vec<(f32, usize, usize)> =
        best_sel[n].iter().map(|&i| items[i]).collect();
    out.sort_by(|a, b| {
        b.0.partial_cmp(&a.0)
            .unwrap()
            .then(a.1.cmp(&b.1))
            .then(a.2.cmp(&b.2))
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splitter_matches_python_offsets() {
        let sp = WhitespaceSplitter::new().unwrap();
        let words = sp.split("Elon Musk founded SpaceX in 2002.");
        let texts: Vec<&str> = vec!["Elon", "Musk", "founded", "SpaceX", "in", "2002", "."];
        assert_eq!(words.len(), texts.len());
        for (w, t) in words.iter().zip(texts) {
            assert_eq!(&w.lowered, &t.to_lowercase());
        }
        assert_eq!(words[0].start, 0);
        assert_eq!(words[0].end, 4);
    }

    #[test]
    fn flat_overlap_prefers_best_non_overlapping_set() {
        // Overlapping (0,2)@.9 vs (0,1)@.5+(1,2)@.5: total 1.0 wins.
        let out = resolve_flat(vec![(0.9, 0, 2), (0.5, 0, 1), (0.5, 1, 2)]);
        assert_eq!(out, vec![(0.5, 0, 1), (0.5, 1, 2)]);
    }
}
