use demo::graph::*;
use std::collections::HashMap;

fn sample() -> GraphData {
    GraphData {
        nodes: vec![
            GraphNode { key: "e1".into(), name: "Tim Cook".into(), label: "person".into(), dim: false },
            GraphNode { key: "e2".into(), name: "Apple".into(), label: "organization".into(), dim: false },
            GraphNode { key: "m3".into(), name: "Tim Cook runs Apple".into(), label: "work".into(), dim: false },
        ],
        links: vec![
            GraphLink { source: "e1".into(), target: "m3".into(), label: "MentionedIn".into() },
            GraphLink { source: "e2".into(), target: "m3".into(), label: "MentionedIn".into() },
            GraphLink { source: "e1".into(), target: "e2".into(), label: "RelatesTo".into() },
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
    grown.nodes.push(GraphNode {
        key: "e4".into(),
        name: "iPhone".into(),
        label: "product".into(),
        dim: false,
    });
    grown.links.push(GraphLink {
        source: "e2".into(),
        target: "e4".into(),
        label: "RelatesTo".into(),
    });
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
