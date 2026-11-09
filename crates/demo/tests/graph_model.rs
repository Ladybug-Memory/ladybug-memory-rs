use demo::graph::*;
use std::collections::HashMap;

fn node(key: &str, name: &str, label: &str) -> GraphNode {
    GraphNode { key: key.into(), name: name.into(), label: label.into(), dim: false, aliases: vec![] }
}

fn link(source: &str, target: &str, label: &str) -> GraphLink {
    GraphLink {
        source: source.into(),
        target: target.into(),
        label: label.into(),
        confidence: 0.9,
        valid_from: String::new(),
        valid_to: String::new(),
        tense: "present".into(),
        viewpoint: "fact".into(),
        viewpoint_confidence: 0.9,
    }
}

fn sample() -> GraphData {
    GraphData {
        nodes: vec![
            node("e1", "Tim Cook", "person"),
            node("e2", "Apple", "organization"),
            node("m3", "Tim Cook runs Apple", "work"),
        ],
        links: vec![
            link("e1", "m3", "MentionedIn"),
            link("e2", "m3", "MentionedIn"),
            link("e1", "e2", "RelatesTo"),
        ],
    }
}

#[test]
fn layout_stays_finite_and_pickable() {
    let mut model = GraphModel::default();
    model.load(&sample());
    assert_eq!(model.nodes.len(), 3);
    assert_eq!(model.links.len(), 3);
    for _ in 0..60 {
        model.tick();
    }
    for n in &model.nodes {
        assert!(n.pos.x.is_finite() && n.pos.y.is_finite());
    }
    // Pick the first node at its own position.
    let p = model.nodes[0].pos;
    assert_eq!(model.pick(p, 6.0), Some(0));
}

#[test]
fn merge_keeps_survivor_positions() {
    let mut model = GraphModel::default();
    model.load(&sample());
    model.settle();
    let before: HashMap<String, Vec2> =
        model.nodes.iter().map(|n| (n.id.clone(), n.pos)).collect();
    let mut grown = sample();
    grown.nodes.push(node("e4", "iPhone", "product"));
    grown.links.push(link("e2", "e4", "RelatesTo"));
    model.merge(&grown);
    assert_eq!(model.nodes.len(), 4);
    for n in &model.nodes {
        if let Some(p) = before.get(&n.id) {
            assert!((n.pos.x - p.x).abs() < 1e-6 && (n.pos.y - p.y).abs() < 1e-6);
        }
    }
    // Same canonical label keeps its hue across merges.
    let tim = model.nodes.iter().find(|n| n.id == "e1").unwrap();
    assert_eq!(tim.color, 0);
}

#[test]
fn newcomer_settles_near_its_neighbors() {
    use demo::graph::GraphModel;
    // e3 links only to e2: after merge it must start near e2, not at the
    // center spiral — this is what keeps new nodes in the viewport.
    let mut first = GraphData {
        nodes: vec![node("e1", "Tim Cook", "person"), node("e2", "Apple", "organization")],
        links: vec![link("e1", "e2", "RelatesTo")],
    };
    let mut model = GraphModel::default();
    model.load(&first);
    model.settle();
    let apple = model.nodes.iter().find(|n| n.id == "e2").unwrap().pos;
    first.nodes.push(node("e3", "iPhone", "product"));
    first.links.push(link("e2", "e3", "RelatesTo"));
    model.merge(&first);
    let pod = model.nodes.iter().find(|n| n.id == "e3").unwrap().pos;
    let d_new = ((pod.x - apple.x).powi(2) + (pod.y - apple.y).powi(2)).sqrt();
    // Neighbor centroid (plus jitter), not the center spiral (>200 away).
    assert!(d_new < 150.0, "newcomer too far from its neighbor: {d_new}");
}

#[test]
fn edge_pick_hits_nearby_segments() {
    use demo::graph::{GraphModel, Vec2};
    let mut model = GraphModel::default();
    model.load(&sample());
    // Node 0 sits somewhere; pick the e1->e2 edge near its midpoint.
    let a = &model.nodes[0];
    let b = &model.nodes[1];
    let mid = Vec2 { x: (a.pos.x + b.pos.x) / 2.0, y: (a.pos.y + b.pos.y) / 2.0 };
    let hit = model.pick_edge(mid, 30.0);
    assert!(hit.is_some(), "expected an edge near the e1-e2 midpoint");
    // Far away from everything picks nothing.
    assert!(model.pick_edge(Vec2 { x: 1e6, y: 1e6 }, 30.0).is_none());
}

fn snap_link(source: usize, target: usize, label: &str) -> demo::graph::SnapLink {
    demo::graph::SnapLink {
        source,
        target,
        label: label.into(),
        confidence: 0.92,
        valid_from: "1976-01-01".into(),
        valid_to: "1976-12-31".into(),
        tense: "past".into(),
        viewpoint: "fact".into(),
        viewpoint_confidence: 0.98,
    }
}

fn rich_snapshot() -> demo::graph::Snapshot {
    use demo::graph::{SnapLink, SnapNode};
    demo::graph::Snapshot {
        nodes: vec![
            SnapNode {
                x: 0.0,
                y: 0.0,
                name: "Steve Jobs".into(),
                color: 0,
                dim: false,
                degree: 1,
                aliases: vec!["Jobs".into()],
                label: "person".into(),
            },
            SnapNode {
                x: 100.0,
                y: 0.0,
                name: "Apple".into(),
                color: 1,
                dim: false,
                degree: 1,
                aliases: vec![],
                label: "organization".into(),
            },
        ],
        links: vec![snap_link(0, 1, "chief executive officer")],
        camera: Default::default(),
        highlight: Default::default(),
        selected: None,
        hover: None,
        mouse: None,
    }
}

#[test]
fn node_tooltip_lists_incident_relations_with_attributes() {
    use demo::graph::{Hover, tooltip_content};
    let snap = rich_snapshot();
    let (title, lines) = tooltip_content(&snap, Hover::Node(0)).expect("node tooltip");
    assert_eq!(title, "Steve Jobs");
    assert!(lines.iter().any(|l| l.contains("person")), "{lines:?}");
    assert!(lines.iter().any(|l| l.contains("Jobs")), "{lines:?}");
    let rel = lines.iter().find(|l| l.contains("chief executive officer")).expect("incident edge line");
    assert!(rel.contains("Apple"), "{rel}");
    assert!(rel.contains("92%"), "{rel}");
    assert!(rel.contains("1976-01-01"), "{rel}");
    assert!(rel.contains("fact"), "{rel}");
}

#[test]
fn edge_tooltip_shows_full_edge_record() {
    use demo::graph::{Hover, tooltip_content};
    let snap = rich_snapshot();
    let (title, lines) = tooltip_content(&snap, Hover::Edge(0)).expect("edge tooltip");
    assert_eq!(title, "chief executive officer");
    let body = lines.join("\n");
    assert!(body.contains("92%"), "{body}");
    assert!(body.contains("1976-01-01 → 1976-12-31"), "{body}");
    assert!(body.contains("past"), "{body}");
    assert!(body.contains("fact"), "{body}");
}

#[test]
fn node_pick_hits_disc_centers() {
    use demo::graph::{GraphModel, Vec2};
    let mut model = GraphModel::default();
    model.load(&sample());
    for (i, n) in model.nodes.iter().enumerate() {
        let hit = model.pick(Vec2 { x: n.pos.x, y: n.pos.y }, 6.0);
        assert_eq!(hit, Some(i), "node {i} not pickable at its own center");
    }
    assert!(model.pick(Vec2 { x: 1e6, y: -1e6 }, 6.0).is_none());
}

#[test]
fn input_runs_plain_typing() {
    use demo::input_runs;
    // No selection: text, cursor char, rest.
    assert_eq!(
        input_runs("hello", 1, None, true),
        vec![
            ("h".to_string(), false, false),
            ("e".to_string(), false, true),
            ("llo".to_string(), false, false),
        ]
    );
    // Cursor at end gets a placeholder block.
    assert_eq!(
        input_runs("hi", 2, None, true),
        vec![("hi".to_string(), false, false), (" ".to_string(), false, true)]
    );
    // Unfocused: plain text, no cursor.
    assert_eq!(
        input_runs("hi", 1, None, false),
        vec![("hi".to_string(), false, false)]
    );
    // Empty + focused shows the block caret.
    assert_eq!(input_runs("", 0, None, true), vec![("▍".to_string(), false, false)]);
}

#[test]
fn input_runs_selection() {
    use demo::input_runs;
    // Anchor left of cursor: [before][selected][cursor][after].
    assert_eq!(
        input_runs("hello world", 8, Some(6), true),
        vec![
            ("hello ".to_string(), false, false),
            ("wo".to_string(), true, false),
            ("r".to_string(), false, true),
            ("ld".to_string(), false, false),
        ]
    );
    // Anchor right of cursor: the cursor char carries both flags and
    // renders with cursor precedence.
    assert_eq!(
        input_runs("hello world", 6, Some(8), true),
        vec![
            ("hello ".to_string(), false, false),
            ("w".to_string(), true, true),
            ("o".to_string(), true, false),
            ("rld".to_string(), false, false),
        ]
    );
}
