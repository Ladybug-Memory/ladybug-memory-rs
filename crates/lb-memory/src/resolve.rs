//! Entity normalization + resolution against the existing schema.
//!
//! Step 2 (compare): each mention is normalized and looked up in the stored
//! `Entity` table by canonical name *or* alias — this is the "compare with
//! the existing schema" pass.
//! Step 3 (resolve): hits merge onto the canonical node (new surface forms
//! become aliases); misses become new canonical nodes.

use crate::types::{ComparedEntity, ResolvedEntity, ScoredMention};

/// Closed-class anaphoric surfaces. These never become graph nodes: they
/// bind to a recent entity (in-call resolution order first, then the
/// store's most recently touched entities) or are dropped when nothing
/// binds. GLiNER labels spans, it does not link them, so longer extractor
/// context windows cannot fix coreference — this deterministic pass owns
/// it instead.
const PERSON_PRONOUNS: &[&str] =
    &["he", "him", "his", "himself", "she", "her", "hers", "herself"];
const PLURAL_PRONOUNS: &[&str] = &["they", "them", "their", "theirs", "themselves"];
const NEUTER_PRONOUNS: &[&str] = &["it", "its", "itself", "this", "that", "these", "those"];

/// Anaphoric surface form (case-insensitive, post-normalization).
pub fn is_pronoun(text: &str) -> bool {
    let n = normalize(text);
    let s = n.as_str();
    PERSON_PRONOUNS.contains(&s) || PLURAL_PRONOUNS.contains(&s) || NEUTER_PRONOUNS.contains(&s)
}

/// Antecedent preference class for a pronoun surface.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PronounClass {
    /// he/she-series: the latest `person` entity.
    Person,
    /// they-series: the latest entity of any kind.
    Any,
    /// it/this-series: the latest non-person, else the latest of any kind.
    NonPerson,
}

fn pronoun_class(text: &str) -> Option<PronounClass> {
    let n = normalize(text);
    match n.as_str() {
        s if PERSON_PRONOUNS.contains(&s) => Some(PronounClass::Person),
        s if PLURAL_PRONOUNS.contains(&s) => Some(PronounClass::Any),
        s if NEUTER_PRONOUNS.contains(&s) => Some(PronounClass::NonPerson),
        _ => None,
    }
}

/// A resolved mention surface: what the graph node *is*, independent of
/// the surface form that pointed at it this time ("He" at 0..2).
#[derive(Debug, Clone)]
pub struct Antecedent {
    pub id: i64,
    pub canonical: String,
    pub entity_type: String,
}

/// Bind pronoun mentions to antecedents: in-call resolution order first
/// (freshest discourse context), then store recency. Returns the kept
/// mentions plus `(kept index, antecedent)` links. Unresolvable pronouns
/// are dropped here so they can never fork junk ("He") nodes.
pub fn bind_pronouns(
    mentions: &[ScoredMention],
    known: &[KnownEntity],
    recent_order: &[String],
    store_recents: &[KnownEntity],
) -> (Vec<ScoredMention>, Vec<(usize, Antecedent)>) {
    // Ordered candidate pool: in-call entities (most recent last in
    // `recent_order`, so walk it reversed), then store recency.
    let mut pool: Vec<Antecedent> = Vec::new();
    for norm in recent_order.iter().rev() {
        if let Some(k) = known.iter().find(|k| &k.normalized == norm) {
            if !pool.iter().any(|a: &Antecedent| a.id == k.id) {
                pool.push(Antecedent {
                    id: k.id,
                    canonical: k.canonical.clone(),
                    entity_type: k.entity_type.clone(),
                });
            }
        }
    }
    for k in store_recents {
        if !pool.iter().any(|a: &Antecedent| a.id == k.id)
            && !pool.iter().any(|a: &Antecedent| a.canonical == k.canonical)
        {
            pool.push(Antecedent {
                id: k.id,
                canonical: k.canonical.clone(),
                entity_type: k.entity_type.clone(),
            });
        }
    }
    let pick = |class: PronounClass| -> Option<Antecedent> {
        match class {
            PronounClass::Person => pool.iter().find(|a| a.entity_type == "person").cloned(),
            PronounClass::Any => pool.first().cloned(),
            PronounClass::NonPerson => pool
                .iter()
                .find(|a| a.entity_type != "person")
                .or_else(|| pool.first())
                .cloned(),
        }
    };
    let mut kept = Vec::with_capacity(mentions.len());
    let mut links = Vec::new();
    for m in mentions {
        match pronoun_class(&m.text).and_then(pick) {
            Some(ant) => {
                links.push((kept.len(), ant));
                kept.push(m.clone());
            }
            None if is_pronoun(&m.text) => {
                // Unresolvable pronoun: drop, never a node.
            }
            None => kept.push(m.clone()),
        }
    }
    (kept, links)
}

/// Lowercase, punctuation-stripped identity used to merge spelling variants
/// ("LadybugDB" vs "ladybugdb", "Tim Cook" vs "tim cook").
pub fn normalize(name: &str) -> String {
    let lower = name.trim().to_lowercase().replace(['\'', '`', '’'], "");
    lower
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
}

/// One row of the stored entity schema, as the resolver sees it.
#[derive(Debug, Clone)]
pub struct KnownEntity {
    pub id: i64,
    pub canonical: String,
    pub normalized: String,
    pub entity_type: String,
    pub aliases: Vec<String>,
    pub alias_normalized: Vec<String>,
}

impl KnownEntity {
    /// Does this stored entity already cover `normalized` (canonical or alias)?
    pub fn covers(&self, normalized: &str) -> bool {
        self.normalized == normalized || self.alias_normalized.iter().any(|a| a == normalized)
    }
}

/// Compare mentions against the known schema (step 2).
pub fn compare(mentions: &[ScoredMention], known: &[KnownEntity]) -> Vec<ComparedEntity> {
    mentions
        .iter()
        .map(|m| {
            let normalized = normalize(&m.text);
            let hit = known.iter().find(|k| k.covers(&normalized));
            ComparedEntity {
                mention: m.clone(),
                normalized,
                existing_id: hit.map(|k| k.id),
                existing_type: hit.map(|k| k.entity_type.clone()),
            }
        })
        .collect()
}

/// Resolve compared mentions to canonical entities (step 3).
///
/// Same normalized name always maps to one node. On a type conflict the
/// stored type wins and the surface is recorded as an alias (entity
/// *identity* is the name, not the label).
pub fn resolve(compared: &[ComparedEntity], known: &[KnownEntity]) -> Vec<ResolvedEntity> {
    let mut out: Vec<ResolvedEntity> = Vec::new();
    let mut seen: Vec<String> = Vec::new();
    for c in compared {
        if seen.contains(&c.normalized) {
            continue;
        }
        seen.push(c.normalized.clone());
        // Pre-bound antecedents (pronouns): merge onto the stored node by id.
        // Pronoun surfaces are never recorded as aliases — "he" must not
        // become a global alias of whoever it pointed at this time.
        if let Some(id) = c.existing_id {
            if let Some(k) = known.iter().find(|k| k.id == id) {
                let mut aliases = k.aliases.clone();
                if !is_pronoun(&c.mention.text) && !aliases.iter().any(|a| a == &c.mention.text) {
                    aliases.push(c.mention.text.clone());
                }
                out.push(ResolvedEntity {
                    id: Some(k.id),
                    canonical: k.canonical.clone(),
                    entity_type: k.entity_type.clone(),
                    aliases,
                    created: false,
                });
                continue;
            }
        }
        match known.iter().find(|k| k.covers(&c.normalized)) {
            Some(k) => {
                let mut aliases = k.aliases.clone();
                if !aliases.iter().any(|a| a == &c.mention.text) {
                    aliases.push(c.mention.text.clone());
                }
                out.push(ResolvedEntity {
                    id: Some(k.id),
                    canonical: k.canonical.clone(),
                    entity_type: k.entity_type.clone(),
                    aliases,
                    created: false,
                });
            }
            None => {
                out.push(ResolvedEntity {
                    id: None,
                    canonical: c.mention.text.clone(),
                    entity_type: c.mention.entity_type.clone(),
                    aliases: vec![c.mention.text.clone()],
                    created: true,
                });
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mention(text: &str) -> ScoredMention {
        ScoredMention {
            text: text.into(),
            entity_type: "organization".into(),
            confidence: 0.9,
            start: 0,
            end: text.len(),
            word_start: 0,
            word_end: 1,
        }
    }

    #[test]
    fn spelling_variants_merge() {
        let known = vec![KnownEntity {
            id: 1,
            canonical: "LadybugDB".into(),
            normalized: normalize("LadybugDB"),
            entity_type: "technology".into(),
            aliases: vec!["LadybugDB".into()],
            alias_normalized: vec![normalize("LadybugDB")],
        }];
        let compared = compare(&[mention("ladybugdb")], &known);
        assert_eq!(compared[0].existing_id, Some(1));
        let resolved = resolve(&compared, &known);
        assert_eq!(resolved.len(), 1);
        assert!(!resolved[0].created);
        assert!(resolved[0].aliases.contains(&"ladybugdb".to_string()));
    }

    #[test]
    fn new_names_become_canonical() {
        let compared = compare(&[mention("Acme Corp")], &[]);
        assert_eq!(compared[0].existing_id, None);
        let resolved = resolve(&compared, &[]);
        assert!(resolved[0].created);
        assert_eq!(resolved[0].canonical, "Acme Corp");
    }
}
