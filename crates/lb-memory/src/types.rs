//! Distilled-fact memory types.
//!
//! Unlike a session log, this store keeps **facts** (one `Memory` node per
//! remembered item) plus an entity graph. Session message content stays in
//! pi's sqlite; we keep only *references* (`message_refs`) in provenance.

use serde::{Deserialize, Serialize};

/// Memory scope, mirroring the mem0 pi plugin: project defaults to the git
/// root so monorepo subdirectories share one pool.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Scope {
    pub user_id: String,
    pub project: String,
    pub session_id: Option<String>,
}

impl Scope {
    /// `session_id == None` with `global == true` searches everything.
    pub fn matches(&self, _global: bool) -> (Option<String>, Option<String>, Option<String>) {
        (Some(self.user_id.clone()), Some(self.project.clone()), self.session_id.clone())
    }
}

/// Fact categories (mem0's ten, plus `General` as the no-cue fallback).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Category {
    Identity,
    Preferences,
    Goals,
    Projects,
    Decisions,
    Technical,
    Relationships,
    Routines,
    Lessons,
    Work,
    General,
}

impl Category {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Identity => "identity",
            Self::Preferences => "preferences",
            Self::Goals => "goals",
            Self::Projects => "projects",
            Self::Decisions => "decisions",
            Self::Technical => "technical",
            Self::Relationships => "relationships",
            Self::Routines => "routines",
            Self::Lessons => "lessons",
            Self::Work => "work",
            Self::General => "general",
        }
    }

    pub fn parse(s: &str) -> Self {
        match s {
            "identity" => Self::Identity,
            "preferences" => Self::Preferences,
            "goals" => Self::Goals,
            "projects" => Self::Projects,
            "decisions" => Self::Decisions,
            "technical" => Self::Technical,
            "relationships" => Self::Relationships,
            "routines" => Self::Routines,
            "lessons" => Self::Lessons,
            "work" => Self::Work,
            _ => Self::General,
        }
    }
}

/// Lifecycle of a stored fact.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryStatus {
    Active,
    /// Contradiction candidate: kept, but needs agent adjudication.
    NeedsReview,
    /// Nothing solid extracted: kept for provenance, hidden from recall.
    Quarantined,
    /// Replaced by a newer fact (kept for history).
    Superseded,
}

impl MemoryStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Active => "active",
            Self::NeedsReview => "needs_review",
            Self::Quarantined => "quarantined",
            Self::Superseded => "superseded",
        }
    }

    pub fn parse(s: &str) -> Self {
        match s {
            "needs_review" => Self::NeedsReview,
            "quarantined" => Self::Quarantined,
            "superseded" => Self::Superseded,
            _ => Self::Active,
        }
    }
}

/// Where a fact came from. `message_refs` are opaque ids into pi's own
/// session store (sqlite) — never message content.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Provenance {
    #[serde(default)]
    pub session_id: Option<String>,
    #[serde(default)]
    pub message_refs: Vec<String>,
    #[serde(default)]
    pub extractor: String,
    #[serde(default)]
    pub model: String,
    #[serde(default)]
    pub auto_captured: bool,
}

/// One stored fact.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEntry {
    pub id: i64,
    pub content: String,
    pub category: Category,
    pub user_id: String,
    pub project: String,
    pub session_id: Option<String>,
    pub importance: i64,
    pub status: MemoryStatus,
    pub provenance: Provenance,
    pub created_at: String,
    pub updated_at: String,
}

/// One entity mention with its extraction confidence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoredMention {
    pub text: String,
    pub entity_type: String,
    pub confidence: f32,
    pub start: usize,
    pub end: usize,
    pub word_start: usize,
    pub word_end: usize,
}

/// An extracted mention compared against the existing graph schema.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparedEntity {
    pub mention: ScoredMention,
    pub normalized: String,
    /// Existing `Entity.id`, if the name/alias is already in the schema.
    pub existing_id: Option<i64>,
    pub existing_type: Option<String>,
}

/// The five-step pipeline report returned by `remember` (and `--verbose`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RememberReport {
    /// Step 1: what GLiNER2.5 found.
    pub extracted: Vec<ScoredMention>,
    /// Step 2: each mention against the existing schema.
    pub compared: Vec<ComparedEntity>,
    /// Step 3: canonical entities (id, canonical name, whether newly created).
    pub resolved: Vec<ResolvedEntity>,
    /// Step 4: the fact-level decision.
    pub decision: FactDecision,
    /// Step 5: what was written.
    pub written: Option<MemoryEntry>,
}

/// Canonical entity after resolution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResolvedEntity {
    pub id: Option<i64>,
    pub canonical: String,
    pub entity_type: String,
    pub aliases: Vec<String>,
    pub created: bool,
}

/// Step-4 decision for one fact candidate.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum FactDecision {
    /// No sufficiently similar active fact: store as active.
    New { reason: String },
    /// Near-duplicate of an active fact: skip the write, strengthen it.
    Duplicate { of: i64, similarity: f32 },
    /// Contradiction cues vs. an active fact: store as `needs_review`.
    Conflict { with: Vec<i64>, cues: Vec<String> },
    /// Nothing solid extracted: store as `quarantined`, hidden from recall.
    LowConfidence { reason: String },
}

/// Search hit.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchHit {
    pub entry: MemoryEntry,
    pub score: f32,
    pub entities: Vec<String>,
    /// Typed relations adjacent to the hit, e.g. `"founder-of via Tim Cook"`.
    /// Empty for older payloads.
    #[serde(default)]
    pub relations: Vec<String>,
}

/// One typed relation between two resolved entity surfaces, as returned by
/// the extractor. Endpoint identity is the mention *text*: the pipeline
/// resolves it through the same canonical/alias map as entity mentions
/// (no span bookkeeping needed).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoredRelation {
    pub relation_type: String,
    pub head_text: String,
    pub head_confidence: f32,
    pub tail_text: String,
    pub tail_confidence: f32,
    pub score: f32,
}

/// A relation type to score, with head/tail entity-type constraints.
/// Zero-shot: any type name works; `description` carries the natural
/// phrasing the scorer reads ("X was educated at Y").
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelationSpec {
    pub relation_type: String,
    pub head_types: Vec<String>,
    pub tail_types: Vec<String>,
    pub description: Option<String>,
    pub allow_self: bool,
}

impl RelationSpec {
    pub fn new(relation_type: &str, head_types: &[&str], tail_types: &[&str]) -> Self {
        Self {
            relation_type: relation_type.to_string(),
            head_types: head_types.iter().map(|s| s.to_string()).collect(),
            tail_types: tail_types.iter().map(|s| s.to_string()).collect(),
            description: None,
            allow_self: false,
        }
    }

    pub fn with_description(mut self, description: &str) -> Self {
        self.description = Some(description.to_string());
        self
    }
}
