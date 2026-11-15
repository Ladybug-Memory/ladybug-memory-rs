//! Domain-driven extraction schemas in SQLite, plus the recall bridge.
//!
//! The inventory lives in `rel_canon.sqlite` (checked in next to this
//! module): `sk_domains` (64 Wikipedia-grounded page types with
//! GLiNER-encoder embeddings), `sk_entity_types` and `sk_relations`
//! (the per-domain extraction schema: entity labels and scorer questions
//! whose text IS the canonical edge label), and `aliases` (colloquial
//! recall bridge). Regenerate with:
//!
//! ```sh
//! ORT_DYLIB_PATH=... LB_MEMORY_SNAPSHOT=... \
//!   cargo run --release -p lb-memory --example build_rel_canon_db -- \
//!   /tmp/rel_canon.new.sqlite crates/lb-memory/src/rel_canon.sqlite
//! ```
//!
//! Retrieval mirrors AutoSchemaKG's conceptualization step without an
//! LLM: the fact text is embedded, the nearest domain is selected by
//! cosine, and that domain's schema is what the pipeline passes to
//! GLiNER for extraction and typed scoring. Anything below the relation
//! bar falls back to co-occurrence (open schema preserved).

use std::collections::{HashMap, HashSet};
use std::sync::OnceLock;

/// Bump when `rel_canon.sqlite` is regenerated; stale spill files are
/// rewritten on version mismatch.
pub const REL_CANON_DB_VERSION: u32 = 2;
/// Max questions per scorer call: the relation prompt must fit the
/// model's 512-token window alongside entities and text.
pub const BATTERY_CAP: usize = 15;
/// Below this cosine the fact keeps the `base` domain instead of the
/// nearest specific one.
pub const DOMAIN_MARGIN: f64 = 0.0;
const EMBED_DIM: usize = 768;
const EMBEDDED_DB: &[u8] = include_bytes!("rel_canon.sqlite");

/// One domain: a Wikipedia-grounded page type with its schema.
#[derive(Debug, Clone)]
pub struct Domain {
    pub name: String,
    pub description: String,
}

/// One scorer question: its text becomes the edge label verbatim.
#[derive(Debug, Clone)]
pub struct BatterySpec {
    pub relation_type: String,
    pub description: String,
    pub head_types: Vec<String>,
    pub tail_types: Vec<String>,
}

struct RelStore {
    aliases: HashMap<String, (u32, String)>,
    domains: Vec<(Domain, Vec<f32>)>,
    exemplars: HashMap<String, Vec<Vec<f32>>>,
    entity_types: HashMap<String, Vec<String>>,
    questions: HashMap<String, Vec<QuestionRow>>,
}

#[derive(Debug, Clone)]
struct QuestionRow {
    relation_type: String,
    description: String,
    head_types: Vec<String>,
    tail_types: Vec<String>,
    priority: i64,
}

fn db_cache_path() -> std::path::PathBuf {
    if let Ok(p) = std::env::var("LB_MEMORY_REL_CANON_DB") {
        return p.into();
    }
    let home = std::env::var("HOME").unwrap_or_else(|_| std::env::temp_dir().display().to_string());
    std::path::Path::new(&home)
        .join(".cache/ladybug-memory")
        .join(format!("rel_canon-v{REL_CANON_DB_VERSION}.sqlite"))
}

/// Spill the checked-in database to the versioned cache path and open
/// it read-only. Freshness is by content (a byte compare — length alone
/// collides across regenerations), so a rebuilt artifact always wins
/// over a stale spill.
fn open_db() -> anyhow::Result<rusqlite::Connection> {
    let path = db_cache_path();
    let stale = std::fs::read(&path)
        .map(|b| b.as_slice() != EMBEDDED_DB)
        .unwrap_or(true);
    if stale {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&path, EMBEDDED_DB)?;
    }
    let conn = rusqlite::Connection::open_with_flags(
        &path,
        rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY,
    )?;
    // Version mismatch (regenerated artifact): respill once and reopen.
    let ver: u32 = conn.query_row("PRAGMA user_version", [], |r| r.get(0)).unwrap_or(0);
    if ver != REL_CANON_DB_VERSION {
        drop(conn);
        std::fs::write(&path, EMBEDDED_DB)?;
        return Ok(rusqlite::Connection::open_with_flags(
            &path,
            rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY,
        )?);
    }
    Ok(conn)
}

fn parse_types(json: &str) -> Vec<String> {
    serde_json::from_str(json).unwrap_or_default()
}

fn decode_emb(blob: &[u8]) -> Option<Vec<f32>> {
    if blob.len() != EMBED_DIM * 4 {
        return None;
    }
    Some(
        blob.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
    )
}

fn load_store() -> RelStore {
    let mut aliases = HashMap::new();
    let mut domains = Vec::new();
    let mut exemplars: HashMap<String, Vec<Vec<f32>>> = HashMap::new();
    let mut entity_types: HashMap<String, Vec<String>> = HashMap::new();
    let mut questions: HashMap<String, Vec<QuestionRow>> = HashMap::new();
    let Ok(conn) = open_db() else {
        return RelStore { aliases, domains, exemplars, entity_types, questions };
    };
    if let Ok(mut stmt) = conn.prepare("SELECT name, description, embedding FROM sk_domains") {
        if let Ok(rows) = stmt.query_map([], |r| {
            Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?, r.get::<_, Vec<u8>>(2)?))
        }) {
            for (name, desc, blob) in rows.flatten() {
                if let Some(emb) = decode_emb(&blob) {
                    domains.push((Domain { name, description: desc }, emb));
                }
            }
        }
    }
    if let Ok(mut stmt) = conn.prepare("SELECT domain, embedding FROM sk_exemplars") {
        if let Ok(rows) =
            stmt.query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, Vec<u8>>(1)?)))
        {
            for (domain, blob) in rows.flatten() {
                if let Some(emb) = decode_emb(&blob) {
                    exemplars.entry(domain).or_default().push(emb);
                }
            }
        }
    }
    if let Ok(mut stmt) =
        conn.prepare("SELECT domain, label FROM sk_entity_types ORDER BY domain, priority")
    {
        if let Ok(rows) =
            stmt.query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?)))
        {
            for (domain, label) in rows.flatten() {
                entity_types.entry(domain).or_default().push(label);
            }
        }
    }
    if let Ok(mut stmt) = conn.prepare(
        "SELECT domain, question, description, head_types, tail_types, priority FROM sk_relations",
    ) {
        if let Ok(rows) = stmt.query_map([], |r| {
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, String>(2)?,
                r.get::<_, String>(3)?,
                r.get::<_, String>(4)?,
                r.get::<_, i64>(5)?,
            ))
        }) {
            for (domain, q, d, heads, tails, prio) in rows.flatten() {
                questions.entry(domain).or_default().push(QuestionRow {
                    relation_type: q,
                    description: d,
                    head_types: parse_types(&heads),
                    tail_types: parse_types(&tails),
                    priority: prio,
                });
            }
        }
    }
    if let Ok(mut stmt) = conn.prepare("SELECT alias, property_id, label FROM aliases") {
        if let Ok(rows) = stmt.query_map([], |r| {
            let alias: String = r.get(0)?;
            let pid: u32 = r.get(1)?;
            let label: String = r.get(2)?;
            Ok((alias, pid, label))
        }) {
            for (alias, pid, label) in rows.flatten() {
                aliases.insert(stem_word(&alias), (pid, label));
            }
        }
    }
    RelStore { aliases, domains, exemplars, entity_types, questions }
}

fn store() -> &'static RelStore {
    static STORE: OnceLock<RelStore> = OnceLock::new();
    STORE.get_or_init(load_store)
}

/// Cosine similarity (both sides L2-normalized).
fn cosine(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Score every domain for an L2-normalized fact embedding: the max over
/// the domain description and its exemplar sentences (same register as
/// queries, so they separate better than abstract descriptions alone).
/// Pure function — unit-testable without the database.
pub fn score_domains(
    emb: &[f32],
    domains: &[(Domain, Vec<f32>)],
    exemplars: &HashMap<String, Vec<Vec<f32>>>,
) -> Vec<(String, f32)> {
    let mut out: Vec<(String, f32)> = Vec::with_capacity(domains.len());
    for (d, e) in domains {
        let mut s = cosine(emb, e);
        if let Some(exs) = exemplars.get(&d.name) {
            for x in exs {
                s = s.max(cosine(emb, x));
            }
        }
        out.push((d.name.clone(), s));
    }
    out.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    out
}

/// Nearest domain for an L2-normalized fact embedding. Falls back to
/// `base` when nothing clears `DOMAIN_MARGIN` or the store is empty.
pub fn nearest_domain<'a>(
    emb: &[f32],
    domains: &'a [(Domain, Vec<f32>)],
    exemplars: &'a HashMap<String, Vec<Vec<f32>>>,
) -> (&'a str, f32) {
    let mut best: (&str, f32) = ("base", f32::MIN);
    for (d, e) in domains {
        let mut s = cosine(emb, e);
        if let Some(exs) = exemplars.get(&d.name) {
            for x in exs {
                s = s.max(cosine(emb, x));
            }
        }
        if s > best.1 {
            best = (&d.name, s);
        }
    }
    if best.1 < DOMAIN_MARGIN as f32 {
        ("base", best.1)
    } else {
        best
    }
}

/// Domain rows for selection (name, embedding); the store behind
/// [`select_domain`].
pub fn all_domains() -> Vec<(Domain, Vec<f32>)> {
    store().domains.clone()
}

/// Select the schema domain for a fact embedding. `None` (no embedder)
/// means `base`.
/// Ranked domains for an L2-normalized fact embedding, best first.
/// `None` (no embedder) or an empty store yields just `base`.
pub fn rank_domains(emb: Option<&[f32]>) -> Vec<String> {
    match emb {
        Some(e) if e.len() == EMBED_DIM => {
            let st = store();
            let scored = score_domains(e, &st.domains, &st.exemplars);
            scored.into_iter().map(|(n, _)| n).collect()
        }
        _ => vec!["base".to_string()],
    }
}

/// Top domain, or `base` without an embedder.
pub fn select_domain(emb: Option<&[f32]>) -> String {
    rank_domains(emb).into_iter().next().unwrap_or_else(|| "base".to_string())
}

/// Entity labels for extraction: the base GLiNER labels plus both
/// domains' extras (deduped).
pub fn domain_labels(primary: &str, secondary: &str) -> Vec<String> {
    let mut out: Vec<String> =
        crate::model::DEFAULT_LABELS.iter().map(|s| s.to_string()).collect();
    for domain in [primary, secondary] {
        if let Some(extra) = store().entity_types.get(domain) {
            for l in extra {
                if !out.contains(l) {
                    out.push(l.clone());
                }
            }
        }
    }
    out
}

/// Scorer battery for the top two domains: the `base` questions plus
/// both domains' extras (deduped), `base` first, then by priority, capped
/// at `BATTERY_CAP` for the 512-token prompt window. Two domains because
/// facts are multi-facet (a graduation sentence is about a person *and*
/// a university); the cap keeps the prompt inside the model window.
/// Labels are canonical by declaration — no canonicalization step.
pub fn battery_for(primary: &str, secondary: &str) -> Vec<BatterySpec> {
    let st = store();
    let mut rows: Vec<(&str, &QuestionRow)> = Vec::new();
    if let Some(base) = st.questions.get("base") {
        rows.extend(base.iter().map(|q| ("base", q)));
    }
    for domain in [primary, secondary] {
        if domain != "base" {
            if let Some(ds) = st.questions.get(domain) {
                rows.extend(ds.iter().map(|q| (domain, q)));
            }
        }
    }
    rows.sort_by(|a, b| {
        let ga = a.0 == "base";
        let gb = b.0 == "base";
        gb.cmp(&ga).then(a.1.priority.cmp(&b.1.priority))
    });
    // One entry per question: the highest-priority occurrence wins.
    let mut seen = std::collections::HashSet::new();
    let mut out = Vec::new();
    for (_, q) in rows {
        if seen.insert(q.relation_type.clone()) {
            out.push(BatterySpec {
                relation_type: q.relation_type.clone(),
                description: q.description.clone(),
                head_types: q.head_types.clone(),
                tail_types: q.tail_types.clone(),
            });
        }
        if out.len() >= BATTERY_CAP {
            break;
        }
    }
    out
}

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

/// Initials of a multi-word string ("chief executive officer" -> "ceo").
fn initials_of(s: &str) -> String {
    s.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| w.len() > 1)
        .filter_map(|w| w.chars().next())
        .collect()
}

/// Whether `query` mentions `relation_type`: stem equality/prefix on the
/// type's parts, the query word matching the type's initials
/// ("ceo" ~ "chief executive officer"), or a curated colloquial alias
/// ("brother" ~ `sibling`, from the SQLite `aliases` table).
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
    if qw.iter().any(|w| {
        store()
            .aliases
            .get(w.as_str())
            .map(|(_, label)| label == relation_type)
            .unwrap_or(false)
    }) {
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

/// All known domain names (for tests and tooling).
pub fn domain_names() -> Vec<String> {
    let mut names: HashSet<String> = store().domains.iter().map(|(d, _)| d.name.clone()).collect();
    names.extend(store().questions.keys().cloned());
    let mut out: Vec<String> = names.into_iter().collect();
    out.sort();
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nearest_domain_picks_exact_match() {
        let domains = vec![
            (Domain { name: "a".into(), description: String::new() }, vec![1.0, 0.0]),
            (Domain { name: "b".into(), description: String::new() }, vec![0.0, 1.0]),
        ];
        let empty: HashMap<String, Vec<Vec<f32>>> = HashMap::new();
        assert_eq!(nearest_domain(&[1.0, 0.0], &domains, &empty).0, "a");
        assert_eq!(nearest_domain(&[0.0, 1.0], &domains, &empty).0, "b");
        // Empty store falls back to base.
        assert_eq!(nearest_domain(&[1.0, 0.0], &[], &empty).0, "base");
    }

    #[test]
    fn exemplars_beat_descriptions() {
        // Exemplar sentences (query register) outrank abstract descriptions.
        let domains = vec![(
            Domain { name: "x".into(), description: String::new() },
            vec![1.0, 0.0],
        )];
        let mut ex: HashMap<String, Vec<Vec<f32>>> = HashMap::new();
        ex.insert("x".into(), vec![vec![0.0, 1.0]]);
        let scored = score_domains(&[0.0, 1.0], &domains, &ex);
        assert_eq!(scored[0], ("x".to_string(), 1.0));
    }

    #[test]
    fn inventory_loads_64_domains() {
        assert_eq!(store().domains.len(), 64, "domain count");
        let names: Vec<&str> = store().domains.iter().map(|(d, _)| d.name.as_str()).collect();
        for want in ["base", "biography", "film", "country", "company", "war", "species"] {
            assert!(names.contains(&want), "{names:?}");
        }
    }

    #[test]
    fn battery_is_base_plus_domains_capped() {
        let b = battery_for("biography", "university");
        assert!(!b.is_empty() && b.len() <= BATTERY_CAP);
        let kinds: Vec<&str> = b.iter().map(|q| q.relation_type.as_str()).collect();
        for want in ["member of", "educated at", "spouse", "sibling", "place of birth"] {
            assert!(kinds.contains(&want), "{kinds:?}");
        }
        // Base domain only: base questions only.
        let base = battery_for("base", "base");
        assert_eq!(base.len(), 8);
    }

    #[test]
    fn battery_unknown_domain_falls_back_to_base() {
        let b = battery_for("no-such-domain", "base");
        let kinds: Vec<&str> = b.iter().map(|q| q.relation_type.as_str()).collect();
        assert!(kinds.contains(&"member of"));
        assert!(!kinds.contains(&"spouse"));
    }

    #[test]
    fn battery_dedupes_repeated_questions() {
        let b = battery_for("war", "war");
        let kinds: Vec<&str> = b.iter().map(|q| q.relation_type.as_str()).collect();
        assert_eq!(kinds.iter().filter(|k| **k == "participant in").count(), 1, "{kinds:?}");
    }

    #[test]
    fn cue_match_covers_initialisms_and_aliases() {
        assert!(relation_cue_match("chief executive officer", "who is the ceo"));
        assert!(relation_cue_match("director", "who directed the film"));
        assert!(relation_cue_match("educated at", "where was she educated"));
        assert!(relation_cue_match("sibling", "is he her brother"));
        assert!(relation_cue_match("spouse", "his wife"));
        assert!(relation_cue_match("founded by", "who founded Apple"));
        assert!(!relation_cue_match("director", "who founded Apple"));
        assert!(!relation_cue_match("sibling", "who founded Apple"));
    }
}
