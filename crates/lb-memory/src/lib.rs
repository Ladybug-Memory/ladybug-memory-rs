//! `lb-memory` — distilled-fact memory over LadybugDB.

pub mod chunker;
pub mod classify;
pub mod conflicts;
pub mod model;
pub mod pipeline;
pub mod resolve;
pub mod sqlite_sync;
pub mod store;
pub mod types;

pub use pipeline::{Pipeline, PipelineOptions};
pub use store::{GraphSnapshot, SnapshotLink, SnapshotNode, SnapshotNodeKind, Store};
pub use types::{Category, FactDecision, MemoryEntry, MemoryStatus, Provenance, RememberReport, Scope};
