//! Entity normalization + resolution against the existing schema.
//!
//! Step 2 (compare): each mention is normalized and looked up in the stored
//! `Entity` table by canonical name *or* alias — this is the "compare with
//! the existing schema" pass.
//! Step 3 (resolve): hits merge onto the canonical node (new surface forms
//! become aliases); misses become new canonical nodes.

use crate::types::{ComparedEntity, ResolvedEntity, ScoredMention};

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
