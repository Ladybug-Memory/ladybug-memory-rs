//! Canonical relation inventory: Wikidata property labels.
//!
//! Source: a Wikidata edge-label export (`edge_meta.duckdb`, table
//! `edge_types(id, label)` where `id` is the numeric property id).
//! Regenerate with:
//! ```sh
//! duckdb -readonly /tmp/edge_meta.duckdb "COPY (SELECT id, label FROM edge_types
//!   WHERE NOT regexp_matches(label, '(?i)\b(ID|identifier|number|code|handle|URL|website|image|logo|signature|file|hash|DOI|ISBN|ISSN|ORCID|VIAF|rank|symbol|classification|sign|map|logo)\b')
//!   ORDER BY id) TO 'crates/lb-memory/src/wikidata_properties.csv' (HEADER, DELIMITER ',');"
//! ```
//! Identifier-ish properties are excluded at export; matching quality gates
//! (exact > initialism > token-cover > subsequence) do the rest.

use std::collections::HashMap;
use std::sync::OnceLock;

#[derive(Debug, Clone)]
struct Prop {
    id: u32,
    label: String,
    lowered: String,
    words: Vec<String>,
    stems: Vec<String>,
    initials: String,
}

fn inventory() -> &'static Vec<Prop> {
    static INVENTORY: OnceLock<Vec<Prop>> = OnceLock::new();
    INVENTORY.get_or_init(|| {
        let mut props = Vec::new();
        for (lineno, line) in include_str!("wikidata_properties.csv").lines().enumerate() {
            if lineno == 0 || line.trim().is_empty() {
                continue; // header
            }
            // Labels contain commas (quoted CSV): split on the first comma.
            let Some(comma) = line.find(',') else { continue };
            let (id, label) = (&line[..comma], &line[comma + 1..]);
            let Ok(id) = id.trim().parse::<u32>() else { continue };
            let label = label.trim().trim_matches('"').trim_matches('\'').to_string();
            if label.is_empty() {
                continue;
            }
            let lowered = label.to_lowercase();
            let words: Vec<String> = lowered
                .split(|c: char| !c.is_alphanumeric())
                .filter(|w| !w.is_empty())
                .map(str::to_string)
                .collect();
            if words.is_empty() {
                continue;
            }
            let stems: Vec<String> = words.iter().map(|w| stem_word(w)).collect();
            let initials: String = words.iter().filter_map(|w| w.chars().next()).collect();
            props.push(Prop { id, label, lowered, words, stems, initials });
        }
        props
    })
}

fn exact_index() -> &'static HashMap<String, usize> {
    static INDEX: OnceLock<HashMap<String, usize>> = OnceLock::new();
    INDEX.get_or_init(|| {
        inventory()
            .iter()
            .enumerate()
            .map(|(i, p)| (p.lowered.clone(), i))
            .collect()
    })
}

/// Tokens that must never match as property initialisms ("us" is not the
/// United States property; "it" is not information technology).
const INIT_GLUE: &[&str] = &[
    "and", "or", "with", "of", "the", "a", "an", "to", "in", "on", "for", "by", "at", "from",
    "as", "it", "is", "us", "uk", "usa", "eu", "un", "he", "she", "we", "they", "me", "my",
    "him", "her", "his", "hers", "its", "our", "your", "up", "out", "off", "do", "so", "no",
    "yes", "am", "pm", "ai", "tv", "s", "t", "m", "re", "ll", "d", "ve",
];

/// Unmatched label tokens that may be skipped when subsequence-matching:
/// pure glue carries no relation semantics of its own.
const MATCH_GLUE: &[&str] = &[
    "of", "the", "a", "an", "to", "in", "on", "for", "by", "at", "from", "and", "or",
    "with", "as", "de", "la", "le", "des", "du", "der", "die", "das", "el",
];

/// Crude stemmer shared with recall cue matching.
pub fn stem_word(s: &str) -> String {
    let mut w: String = s.to_lowercase().chars().filter(|c| c.is_alphanumeric()).collect();
    for suf in ["ing", "ed", "es", "s"] {
        if w.len() > suf.len() + 2 && w.ends_with(suf) {
            w.truncate(w.len() - suf.len());
            break;
        }
    }
    w
}

/// Canonicalize a mined relation phrase against the Wikidata property
/// inventory: exact label > initialism ("ceo" in "great ceo of" matches
/// "chief executive officer") > full token-cover > contiguous subsequence.
/// Returns `(property id, canonical label)`, or `None` when nothing clears
/// the bar — callers keep the raw phrase then (open schema preserved).
pub fn canonicalize_relation(raw: &str) -> Option<(u32, String)> {
    let norm: String = raw
        .to_lowercase()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .trim()
        .to_string();
    if norm.is_empty() {
        return None;
    }
    if let Some(&i) = exact_index().get(&norm) {
        let p = &inventory()[i];
        return Some((p.id, p.label.clone()));
    }
    let tokens: Vec<&str> = norm.split(' ').collect();
    let stems: Vec<String> = tokens.iter().map(|t| stem_word(t)).collect();
    // (score, explained-tokens, id, label). Explained prefers the label
    // that accounts for more of the surface ("followed by" beats
    // "follows" for "was followed by").
    let mut best: Option<(i32, usize, u32, String)> = None;
    for p in inventory() {
        // Initialism: a phrase token equals the label's initials
        // ("ceo" -> "chief executive officer", "vp" -> "vice president").
        if p.words.len() >= 2
            && p.initials.len() >= 2
            && tokens.iter().any(|t| *t == p.initials.as_str() && !INIT_GLUE.contains(t))
        {
            best = consider(best, (90, 1, p.id, p.label.clone()));
        }
        // Token-cover: every label token appears in the phrase (stemmed).
        if !p.stems.is_empty() && p.stems.iter().all(|s| stems.contains(s)) {
            best = consider(best, (70, p.stems.len(), p.id, p.label.clone()));
        }
        // Prefix-anchored contiguous run: the phrase opens with the
        // label's head ("founded" -> "founded by"). Trailing label tokens
        // must be pure glue, so single shared words ("street" in
        // "street key") do not match.
        let mut run = 0usize;
        for ps in &stems {
            if run < p.stems.len() && *ps == p.stems[run] {
                run += 1;
            } else {
                break;
            }
        }
        if run >= 1
            && p.stems[run.min(p.stems.len())..]
                .iter()
                .all(|s| MATCH_GLUE.contains(&s.as_str()))
        {
            best = consider(best, (60 + 10 * run.min(3) as i32, run, p.id, p.label.clone()));
        }
    }
    best.map(|(_, _, id, label)| (id, label))
}

/// Best-candidate ordering: higher score wins, then more explained
/// surface tokens, then the smaller property id (deterministic).
fn consider(
    best: Option<(i32, usize, u32, String)>,
    cand: (i32, usize, u32, String),
) -> Option<(i32, usize, u32, String)> {
    match best {
        None => Some(cand),
        Some(b) if (cand.0, cand.1, std::cmp::Reverse(cand.2)) > (b.0, b.1, std::cmp::Reverse(b.2)) => {
            Some(cand)
        }
        Some(b) => Some(b),
    }
}

/// Initials of a multi-word string ("chief executive officer" -> "ceo").
fn initials_of(s: &str) -> String {
    s.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| w.len() > 1)
        .filter_map(|w| w.chars().next())
        .collect()
}

/// Whether `query` mentions `relation_type`: stem equality/prefix on the
/// type's parts, or the query word matching the type's initials
/// ("ceo" ~ "chief executive officer").
pub fn relation_cue_match(relation_type: &str, query: &str) -> bool {
    let qw: Vec<String> = query
        .to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| w.len() > 1)
        .map(|w| stem_word(&w))
        .collect();
    if qw.is_empty() {
        return false;
    }
    if qw.iter().any(|w| w == &initials_of(relation_type)) {
        return true;
    }
    relation_type
        .split(|c: char| !c.is_alphanumeric())
        .filter(|p| !p.is_empty())
        .any(|part| {
            let st = stem_word(part);
            if st.len() < 4 {
                return false;
            }
            qw.iter()
                .any(|w| w.len() >= 4 && (w == &st || w.starts_with(&st) || st.starts_with(w)))
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonicalizes_ceo_variants_to_one_property() {
        for phrase in ["ceo", "great ceo of", "ceo since", "ceo of", "the ceo"] {
            let hit = canonicalize_relation(phrase);
            assert_eq!(hit, Some((169, "chief executive officer".to_string())), "{phrase}");
        }
    }

    #[test]
    fn exact_labels_resolve() {
        assert_eq!(
            canonicalize_relation("founded by"),
            Some((112, "founded by".to_string()))
        );
        assert_eq!(canonicalize_relation("employer"), Some((108, "employer".to_string())));
        assert_eq!(canonicalize_relation("member of"), Some((463, "member of".to_string())));
    }

    #[test]
    fn unknown_phrases_stay_open() {
        assert_eq!(canonicalize_relation("runs"), None);
        assert_eq!(canonicalize_relation("down the street"), None);
        assert_eq!(canonicalize_relation(""), None);
    }

    #[test]
    fn passive_voice_maps_to_passive_property() {
        // Cover matching keeps voice: "was followed by" hits P156, not P155.
        assert_eq!(
            canonicalize_relation("was followed by"),
            Some((156, "followed by".to_string()))
        );
        assert_eq!(
            canonicalize_relation("founded"),
            Some((112, "founded by".to_string()))
        );
    }

    #[test]
    fn cue_match_covers_initialisms() {
        assert!(relation_cue_match("chief executive officer", "who is the ceo"));
        assert!(relation_cue_match("founded by", "who founded Apple"));
        assert!(!relation_cue_match("chief executive officer", "who founded Apple"));
    }
}
