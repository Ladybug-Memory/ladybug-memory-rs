//! Step-4 decisions: duplicates, conflicts, low-confidence additions.
//!
//! Deliberately LLM-free, so the rules are auditable:
//!
//! - **Duplicate**: token-set Jaccard ≥ 0.8 with an active fact that shares
//!   an entity → skip the write, strengthen the existing fact.
//! - **Conflict**: same entity + same category + low-to-mid similarity, with
//!   at least one contradiction cue (negation flip, preference antonyms,
//!   differing year/number) → store as `needs_review` with a `ConflictsWith`
//!   edge. The agent adjudicates via `lbm review` (it has the LLM; we don't).
//! - **Low-confidence**: no extracted entity (or nothing above threshold) →
//!   store as `quarantined`: kept for provenance, hidden from recall.
//! - Otherwise → **New**.

use std::collections::HashSet;

/// Token-set Jaccard similarity over lowercased alphanumeric tokens.
pub fn jaccard(a: &str, b: &str) -> f32 {
    fn tokens(s: &str) -> HashSet<String> {
        s.to_lowercase()
            .split(|c: char| !c.is_alphanumeric())
            .filter(|w| w.len() > 1)
            .map(str::to_string)
            .collect()
    }
    let (ta, tb) = (tokens(a), tokens(b));
    if ta.is_empty() || tb.is_empty() {
        return 0.0;
    }
    let inter = ta.intersection(&tb).count() as f32;
    let union = ta.union(&tb).count() as f32;
    inter / union
}

const NEGATIONS: &[&str] = &["not", "no", "never", "n't", "cannot", "can't", "won't", "don't", "doesn't", "didn't", "isn't", "aren't", "wasn't", "weren't", "no longer", "stopped"];

const LIKES: &[&str] = &["like", "likes", "love", "loves", "prefer", "prefers", "favourite", "favorite", "enjoy", "enjoys"];
const DISLIKES: &[&str] = &["hate", "hates", "dislike", "dislikes", "avoid", "avoids", "despise", "despises"];

fn word_set(text: &str) -> HashSet<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .map(str::to_string)
        .collect()
}

fn has_any(words: &HashSet<String>, cues: &[&str]) -> bool {
    cues.iter().any(|c| words.contains(*c))
}

fn negation_count(text: &str) -> usize {
    let lower = format!(" {} ", text.to_lowercase());
    NEGATIONS.iter().filter(|n| lower.contains(n.trim())).count()
}

fn years(text: &str) -> HashSet<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|w| w.len() == 4 && w.chars().all(|c| c.is_ascii_digit()))
        .map(str::to_string)
        .collect()
}

/// Contradiction cues between a candidate fact and an existing fact that
/// already shares an entity and category.
pub fn contradiction_cues(candidate: &str, existing: &str) -> Vec<String> {
    let mut cues = Vec::new();
    let (wc, we) = (word_set(candidate), word_set(existing));

    if negation_count(candidate) != negation_count(existing) {
        cues.push("negation flip".to_string());
    }
    let (c_like, c_dislike) = (has_any(&wc, LIKES), has_any(&wc, DISLIKES));
    let (e_like, e_dislike) = (has_any(&we, LIKES), has_any(&we, DISLIKES));
    if (c_like && e_dislike) || (c_dislike && e_like) {
        cues.push("preference antonyms".to_string());
    }
    let (yc, ye) = (years(candidate), years(existing));
    if !yc.is_empty() && !ye.is_empty() && yc != ye {
        cues.push(format!("year mismatch ({} vs {})", join_sorted(&yc), join_sorted(&ye)));
    }
    cues
}

fn join_sorted(set: &HashSet<String>) -> String {
    let mut v: Vec<&String> = set.iter().collect();
    v.sort();
    v.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(",")
}

/// Outcome of comparing one candidate fact to the active facts that share
/// its entities.
#[derive(Debug, Clone)]
pub enum FactVerdict {
    New,
    Duplicate { of: i64, similarity: f32 },
    Conflict { with: Vec<(i64, Vec<String>)> },
}

pub const DUPLICATE_THRESHOLD: f32 = 0.8;

/// Decide, given candidate text + entity overlap + similar active facts
/// (`id`, `content` pairs, most similar first).
pub fn decide(candidate: &str, similar: &[(i64, String)]) -> FactVerdict {
    let mut conflicts: Vec<(i64, Vec<String>)> = Vec::new();
    for (id, content) in similar {
        let sim = jaccard(candidate, content);
        if sim >= DUPLICATE_THRESHOLD {
            return FactVerdict::Duplicate { of: *id, similarity: sim };
        }
        // Only same-ballpark facts can contradict; wholly unrelated text is
        // just a second fact about the same entity.
        if sim >= 0.15 {
            let cues = contradiction_cues(candidate, content);
            if !cues.is_empty() {
                conflicts.push((*id, cues));
            }
        }
    }
    if conflicts.is_empty() {
        FactVerdict::New
    } else {
        FactVerdict::Conflict { with: conflicts }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn duplicates_detected() {
        let v = decide(
            "User prefers Python for scripting",
            &[(1, "User prefers Python for scripting tasks".into())],
        );
        assert!(matches!(v, FactVerdict::Duplicate { .. }));
    }

    #[test]
    fn preference_flip_is_conflict() {
        let v = decide(
            "User hates Python for scripting",
            &[(1, "User prefers Python for scripting".into())],
        );
        match v {
            FactVerdict::Conflict { with, .. } => assert_eq!(with[0].0, 1),
            other => panic!("expected conflict, got {other:?}"),
        }
    }

    #[test]
    fn unrelated_fact_about_same_entity_is_new() {
        let v = decide("Tim Cook visited Japan", &[(1, "Tim Cook runs Apple".into())]);
        assert!(matches!(v, FactVerdict::New));
    }

    #[test]
    fn year_mismatch_is_conflict() {
        let v = decide(
            "The event is scheduled for March 2024",
            &[(1, "The event is scheduled for March 2025".into())],
        );
        assert!(matches!(v, FactVerdict::Conflict { .. }));
    }
}
