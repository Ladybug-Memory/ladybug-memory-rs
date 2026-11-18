//! Headless agent-loop test: drives the demo's worker protocol (the exact
//! messages the GPUI view sends/receives) with the real GLiNER model.
//!
//! Run with:
//!   LB_DEMO_AGENT_TEST=1 cargo test -p demo --release
//! (needs `lbm fetch-model` once for the model snapshot + ORT lib)
//!
//! Pixels aren't covered here (no display in CI); rendering is exercised
//! live via `lbm demo`.

use std::time::{Duration, Instant};

use demo::agent::{FromAgent, ToAgent, spawn_agent};
use demo::graph::GraphData;

fn recv_until(
    rx: &std::sync::mpsc::Receiver<FromAgent>,
    timeout: Duration,
    mut f: impl FnMut(FromAgent) -> Option<FromAgent>,
) -> FromAgent {
    let start = Instant::now();
    loop {
        let left = timeout.saturating_sub(start.elapsed());
        if left.is_zero() {
            panic!("timed out waiting for agent message");
        }
        match rx.recv_timeout(left.min(Duration::from_millis(500))) {
            Ok(m) => {
                if let Some(hit) = f(m) {
                    return hit;
                }
            }
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => continue,
            Err(e) => panic!("agent channel closed: {e}"),
        }
    }
}

fn wait_reply(rx: &std::sync::mpsc::Receiver<FromAgent>) -> demo::agent::ChatReply {
    match recv_until(rx, Duration::from_secs(420), |m| match m {
        FromAgent::Reply(r) => Some(FromAgent::Reply(r)),
        FromAgent::Error(e) => panic!("agent error: {e}"),
        _ => None,
    }) {
        FromAgent::Reply(r) => r,
        _ => unreachable!(),
    }
}

fn wait_graph(rx: &std::sync::mpsc::Receiver<FromAgent>, want_node: &str) -> GraphData {
    match recv_until(rx, Duration::from_secs(60), |m| match m {
        FromAgent::Graph(g) if g.nodes.iter().any(|n| n.name == want_node) => {
            Some(FromAgent::Graph(g))
        }
        FromAgent::Graph(_) => None,
        FromAgent::Error(e) => panic!("agent error: {e}"),
        _ => None,
    }) {
        FromAgent::Graph(g) => g,
        _ => unreachable!(),
    }
}

#[test]
fn agent_remember_and_search_loop() {
    if std::env::var("LB_DEMO_AGENT_TEST").is_err() {
        eprintln!("skip: LB_DEMO_AGENT_TEST unset");
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let db = dir.path().join("demo.lbdb").display().to_string();
    let snapshot = demo::default_snapshot();
    assert!(
        std::path::Path::new(&snapshot).exists(),
        "GLiNER snapshot missing at {snapshot}"
    );
    let handle = spawn_agent(demo::agent::AgentConfig {
        db_path: db,
        snapshot,
        user: "demo".into(),
        project: "demo-chat".into(),
    });

    // Chat: remember a fact through the real model.
    handle.tx.send(ToAgent::Remember("Tim Cook runs Apple.".into())).unwrap();
    let reply = wait_reply(&handle.rx);
    eprintln!("agent: {}", reply.agent_text);
    assert!(
        reply.agent_text.contains("Tim Cook"),
        "expected entities in reply: {}",
        reply.agent_text
    );

    // The graph must contain the entity + memory nodes.
    handle.tx.send(ToAgent::Reload).unwrap();
    let g = wait_graph(&handle.rx, "Tim Cook");
    assert!(g.nodes.iter().any(|n| n.key.starts_with('m')), "memory node missing");
    assert!(!g.links.is_empty(), "expected mention links");

    // Search recalls and highlights.
    handle.tx.send(ToAgent::Search("tim cook".into())).unwrap();
    let reply = wait_reply(&handle.rx);
    eprintln!("search: {}", reply.agent_text);
    assert!(
        reply.agent_text.contains("Tim Cook runs Apple"),
        "expected hit in reply: {}",
        reply.agent_text
    );
    assert!(!reply.highlight_keys.is_empty(), "expected highlight keys");
}
