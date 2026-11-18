//! Background agent: owns the Store + GLiNER extractor on a plain thread.
//!
//! The GPUI thread never blocks on inference or storage: it posts
//! [`ToAgent`] jobs and drains [`FromAgent`] answers in its 33ms layout
//! tick. All channel payloads are plain data.

use std::sync::mpsc::{Receiver, Sender, channel};
use std::thread;

use anyhow::Result;
use lb_memory::model::GlinerExtractor;
use lb_memory::pipeline::{Pipeline, PipelineOptions};
use lb_memory::store::Store;
use lb_memory::types::{FactDecision, Provenance, RememberReport, Scope, SearchHit};

use crate::graph::GraphData;

// ---- protocol ----

#[derive(Debug)]
pub enum ToAgent {
    Remember(String),
    Search(String),
    Reload,
}

#[derive(Debug)]
pub struct ChatReply {
    pub agent_text: String,
    /// Graph node keys to ring-highlight (empty clears).
    pub highlight_keys: Vec<String>,
}

#[derive(Debug)]
pub enum FromAgent {
    Status(String),
    Reply(ChatReply),
    Graph(GraphData),
    Error(String),
}

// ---- worker ----

pub struct AgentHandle {
    pub tx: Sender<ToAgent>,
    pub rx: Receiver<FromAgent>,
}

pub struct AgentConfig {
    pub db_path: String,
    pub snapshot: String,
    pub user: String,
    pub project: String,
}

impl AgentConfig {
    pub fn scope(&self) -> Scope {
        Scope {
            user_id: self.user.clone(),
            project: self.project.clone(),
            session_id: None,
        }
    }
}

pub fn spawn_agent(cfg: AgentConfig) -> AgentHandle {
    let (job_tx, job_rx) = channel::<ToAgent>();
    let (ans_tx, ans_rx) = channel::<FromAgent>();
    thread::spawn(move || {
        if let Err(e) = agent_loop(cfg, job_rx, &ans_tx) {
            let _ = ans_tx.send(FromAgent::Error(format!("agent exited: {e:#}")));
        }
    });
    AgentHandle {
        tx: job_tx,
        rx: ans_rx,
    }
}

fn send_graph(store: &Store, ans: &Sender<FromAgent>) {
    match load_graph(store) {
        Ok(g) => {
            let _ = ans.send(FromAgent::Graph(g));
        }
        Err(e) => {
            let _ = ans.send(FromAgent::Error(format!("graph reload failed: {e:#}")));
        }
    }
}

fn load_graph(store: &Store) -> Result<GraphData> {
    let snap = store.graph_snapshot(500, 2000)?;
    let mut nodes = Vec::with_capacity(snap.nodes.len());
    for n in snap.nodes {
        let key = match n.kind {
            lb_memory::SnapshotNodeKind::Entity => format!("e{}", n.id),
            lb_memory::SnapshotNodeKind::Memory => format!("m{}", n.id),
        };
        nodes.push(crate::graph::GraphNode {
            key,
            name: n.name,
            label: n.label,
            dim: !n.status.is_empty() && n.status != "active",
            aliases: n.aliases,
        });
    }
    let mut links = Vec::with_capacity(snap.links.len());
    for l in snap.links {
        let src = match l.src.1 {
            lb_memory::SnapshotNodeKind::Entity => format!("e{}", l.src.0),
            lb_memory::SnapshotNodeKind::Memory => format!("m{}", l.src.0),
        };
        let dst = match l.dst.1 {
            lb_memory::SnapshotNodeKind::Entity => format!("e{}", l.dst.0),
            lb_memory::SnapshotNodeKind::Memory => format!("m{}", l.dst.0),
        };
        links.push(crate::graph::GraphLink {
            source: src,
            target: dst,
            label: l.label,
            confidence: l.confidence as f32,
            valid_from: l.valid_from,
            valid_to: l.valid_to,
            tense: l.tense,
            viewpoint: l.viewpoint,
            viewpoint_confidence: l.viewpoint_confidence as f32,
        });
    }
    Ok(GraphData { nodes, links })
}

fn agent_loop(cfg: AgentConfig, jobs: Receiver<ToAgent>, ans: &Sender<FromAgent>) -> Result<()> {
    let _ = ans.send(FromAgent::Status("opening memory graph…".into()));
    let store = Store::open(&cfg.db_path)?;
    let pipe_opts = || PipelineOptions {
        extract_threshold: 0.5,
        ..Default::default()
    };
    let mut extractor = GlinerExtractor::new(cfg.snapshot.clone());
    let scope = cfg.scope();
    // Initial graph so the canvas isn't empty.
    send_graph(&store, ans);
    let _ = ans.send(FromAgent::Status("ready".into()));

    for job in jobs {
        match job {
            ToAgent::Reload => send_graph(&store, ans),
            ToAgent::Search(query) => {
                let _ = ans.send(FromAgent::Status("searching…".into()));
                let pipe = Pipeline::new(&store, pipe_opts());
                match pipe.recall(&query, &scope, 8) {
                    Ok(hits) => {
                        let keys = highlight_keys_for(&store, &hits);
                        let _ = ans.send(FromAgent::Reply(ChatReply {
                            agent_text: format_search_reply(&query, &hits),
                            highlight_keys: keys,
                        }));
                    }
                    Err(e) => {
                        let _ = ans.send(FromAgent::Error(format!("search failed: {e:#}")));
                    }
                }
                send_graph(&store, ans);
                let _ = ans.send(FromAgent::Status("ready".into()));
            }
            ToAgent::Remember(text) => {
                let _ = ans.send(FromAgent::Status("thinking…".into()));
                let prov = Provenance {
                    session_id: None,
                    message_refs: vec![],
                    extractor: "gliner2.5".into(),
                    model: "DanKau/gliner2.5-base-v1-onnx".into(),
                    auto_captured: true,
                };
                let pipe = Pipeline::new(&store, pipe_opts());
                match pipe.remember_auto(&mut extractor, &text, &scope, prov) {
                    Ok(reports) => {
                        let agent_text = format_remember_reply(&reports);
                        let _ = ans.send(FromAgent::Reply(ChatReply {
                            agent_text,
                            highlight_keys: vec![],
                        }));
                    }
                    Err(e) => {
                        let _ = ans.send(FromAgent::Error(format!(
                            "remember failed (run `lbm fetch-model` to provision model + ORT lib): {e:#}"
                        )));
                    }
                }
                send_graph(&store, ans);
                let _ = ans.send(FromAgent::Status("ready".into()));
            }
        }
    }
    Ok(())
}

fn highlight_keys_for(store: &Store, hits: &[SearchHit]) -> Vec<String> {
    // Ring the entities linked to matched memories (memory nodes themselves
    // are not displayed).
    let id_of: std::collections::HashMap<String, i64> = store
        .list_entities()
        .unwrap_or_default()
        .into_iter()
        .map(|e| (e.canonical, e.id))
        .collect();
    let mut keys = Vec::new();
    for h in hits {
        for ent in &h.entities {
            if let Some(id) = id_of.get(ent) {
                keys.push(format!("e{id}"));
            }
        }
    }
    keys
}

fn format_search_reply(query: &str, hits: &[SearchHit]) -> String {
    if hits.is_empty() {
        return format!("No memories match “{query}”.");
    }
    let mut out = format!("Top {} for “{query}”:", hits.len());
    for h in hits {
        out.push_str(&format!("\n• #{} [{}] {}", h.entry.id, h.entry.category.as_str(), h.entry.content));
    }
    out
}

fn format_remember_reply(reports: &[RememberReport]) -> String {
    if reports.is_empty() {
        return "Nothing worth keeping there — no entity-bearing sentences.".into();
    }
    let mut parts = Vec::new();
    for r in reports {
        let entities: Vec<String> = r
            .resolved
            .iter()
            .map(|e| {
                if e.created {
                    format!("{}(new)", e.canonical)
                } else {
                    e.canonical.clone()
                }
            })
            .collect();
        let verdict = match &r.decision {
            FactDecision::New { .. } => "stored as a new fact".to_string(),
            FactDecision::Duplicate { of, similarity } => {
                format!("already knew this (#{of}, sim {similarity:.2}) — strengthened")
            }
            FactDecision::Conflict { with, cues } => {
                format!("conflicts with {with:?} [{}] — flagged for review", cues.join("; "))
            }
            FactDecision::LowConfidence { reason } => format!("quarantined ({reason})"),
        };
        let stored = match &r.written {
            Some(w) => format!("#{} [{}|{}]", w.id, w.category.as_str(), w.status.as_str()),
            None => "nothing".into(),
        };
        parts.push(format!("{} → {} → {}", entities.join(", "), verdict, stored));
    }
    parts.join("\n")
}
