//! `lb-memory` — distilled-fact memory over LadybugDB.

pub mod chunker;
pub mod classify;
pub mod conflicts;
pub mod model;
pub mod pipeline;
pub mod rel_canon;
pub mod resolve;
pub mod sqlite_sync;
pub mod store;
pub mod types;

pub use pipeline::{Pipeline, PipelineOptions};
pub use store::{GraphSnapshot, SnapshotLink, SnapshotNode, SnapshotNodeKind, Store};
/// Model snapshot fetch (first-run download). Re-exported so the CLI can
/// report/ensure the snapshot without depending on `gliner-onnx` directly.
pub use gliner_onnx::fetch;pub use types::{Category, FactDecision, MemoryEntry, MemoryStatus, Provenance, RememberReport, Scope};
