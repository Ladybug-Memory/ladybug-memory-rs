//! Force-directed graph canvas for the demo.
//!
//! Ported from bugscope `src/model.rs` (owned graph container + camera +
//! Fruchterman–Reingold ticks) and the graph paint path of bugscope
//! `src/ui.rs` (discs, rim-trimmed arrow edges, zoom-tiered pills). Only
//! the graph view is ported — no treemap, no sunburst, no schema view.
//!
//! Deliberately dependency-free (plain data in, no `lbug`): the demo feeds
//! it [`GraphData`] built from `lb-memory`'s [`lb_memory::GraphSnapshot`].

use std::collections::{HashMap, HashSet};

use gpui::*;

// ---- data (plain structs; built from lb-memory snapshots) ----

#[derive(Debug, Clone)]
pub struct GraphNode {
    pub key: String,
    pub name: String,
    pub label: String,
    /// Dimmed (e.g. non-active memories) paint smaller/desaturated.
    pub dim: bool,
}

#[derive(Debug, Clone)]
pub struct GraphLink {
    pub source: String,
    pub target: String,
    pub label: String,
}

#[derive(Debug, Clone, Default)]
pub struct GraphData {
    pub nodes: Vec<GraphNode>,
    pub links: Vec<GraphLink>,
}

// ---- simulation model ----

#[derive(Debug, Clone, Copy)]
pub struct Vec2 {
    pub x: f32,
    pub y: f32,
}

#[derive(Debug, Clone)]
pub struct SimNode {
    pub id: String,
    pub name: String,
    pub label: String,
    pub color: usize,
    pub dim: bool,
    pub pos: Vec2,
    pub vel: Vec2,
    pub degree: usize,
}

#[derive(Debug, Clone)]
pub struct SimLink {
    pub source: usize,
    pub target: usize,
    pub label: String,
}

#[derive(Debug, Clone, Default)]
pub struct GraphModel {
    pub nodes: Vec<SimNode>,
    pub links: Vec<SimLink>,
    pub index: HashMap<String, usize>,
}

impl GraphModel {
    pub fn load(&mut self, data: &GraphData) {
        self.nodes.clear();
        self.links.clear();
        self.index.clear();
        let n = data.nodes.len().max(1) as f32;
        let radius = (220.0 * n.sqrt()).clamp(300.0, 4000.0);
        let mut degrees: HashMap<&str, usize> = HashMap::new();
        for l in &data.links {
            *degrees.entry(l.source.as_str()).or_insert(0) += 1;
            *degrees.entry(l.target.as_str()).or_insert(0) += 1;
        }
        let mut color_of: HashMap<&str, usize> = HashMap::new();
        for node in &data.nodes {
            let idx = self.nodes.len();
            self.index.insert(node.key.clone(), idx);
            let color = if let Some(&c) = color_of.get(node.label.as_str()) {
                c
            } else {
                let c = color_of.len();
                color_of.insert(node.label.as_str(), c);
                c
            };
            let angle = (idx as f32) * 2.399963; // golden angle
            let r = radius * (0.2 + 0.8 * (idx as f32 / n).sqrt());
            self.nodes.push(SimNode {
                id: node.key.clone(),
                name: node.name.clone(),
                label: node.label.clone(),
                color,
                dim: node.dim,
                pos: Vec2 {
                    x: angle.cos() * r,
                    y: angle.sin() * r,
                },
                vel: Vec2 { x: 0.0, y: 0.0 },
                degree: *degrees.get(node.key.as_str()).unwrap_or(&0),
            });
        }
        for l in &data.links {
            if let (Some(&s), Some(&t)) = (
                self.index.get(l.source.as_str()),
                self.index.get(l.target.as_str()),
            ) {
                self.links.push(SimLink {
                    source: s,
                    target: t,
                    label: l.label.clone(),
                });
            }
        }
    }

    /// Merge a fresh snapshot into the live layout: keep positions for
    /// surviving nodes so the graph grows in place instead of jumping,
    /// spiral-place newcomers, drop the gone.
    pub fn merge(&mut self, data: &GraphData) {
        let mut keep: HashMap<String, (Vec2, usize)> = HashMap::new();
        for n in &self.nodes {
            keep.insert(n.id.clone(), (n.pos, n.color));
        }
        // Rebuild through load() for degree/color correctness, then restore
        // positions for survivors.
        self.load(data);
        let mut placed = 0usize;
        for n in &mut self.nodes {
            if let Some((pos, color)) = keep.get(&n.id) {
                n.pos = *pos;
                n.color = *color;
                n.vel = Vec2 { x: 0.0, y: 0.0 };
            } else {
                // Newcomers spiral out from the center.
                let angle = (placed as f32) * 2.399963;
                let r = 60.0 + 24.0 * (placed as f32).sqrt();
                n.pos = Vec2 {
                    x: angle.cos() * r,
                    y: angle.sin() * r,
                };
                placed += 1;
            }
        }
        // Reassign surviving colors by label so hues stay stable.
        let mut color_of: HashMap<String, usize> = HashMap::new();
        for n in &self.nodes {
            if let Some((_, color)) = keep.get(&n.id) {
                color_of.entry(n.label.clone()).or_insert(*color);
            }
        }
        for n in &mut self.nodes {
            if keep.contains_key(&n.id) {
                continue;
            }
            let next = color_of.len();
            n.color = *color_of.entry(n.label.clone()).or_insert(next);
        }
    }

    pub fn settle(&mut self) {
        let iters = (self.nodes.len() / 50).clamp(40, 250);
        for _ in 0..iters {
            self.tick();
        }
        for n in &mut self.nodes {
            n.vel = Vec2 { x: 0.0, y: 0.0 };
        }
    }

    pub fn tick(&mut self) {
        let n = self.nodes.len();
        if n == 0 {
            return;
        }
        let repulsion = 9000.0;
        let spring_len = 90.0;
        let spring_k = 0.02;
        let stride = if n > 800 { n / 400 } else { 1 }.max(1);
        for i in (0..n).step_by(stride) {
            for j in (i + 1..n).step_by(stride) {
                let dx = self.nodes[i].pos.x - self.nodes[j].pos.x;
                let dy = self.nodes[i].pos.y - self.nodes[j].pos.y;
                let d2 = (dx * dx + dy * dy).max(100.0);
                let d = d2.sqrt();
                let f = repulsion / d2;
                let fx = f * dx / d;
                let fy = f * dy / d;
                self.nodes[i].vel.x += fx;
                self.nodes[i].vel.y += fy;
                self.nodes[j].vel.x -= fx;
                self.nodes[j].vel.y -= fy;
            }
        }
        for link in self.links.clone() {
            let (s, t) = (link.source, link.target);
            let dx = self.nodes[t].pos.x - self.nodes[s].pos.x;
            let dy = self.nodes[t].pos.y - self.nodes[s].pos.y;
            let d = (dx * dx + dy * dy).sqrt().max(1.0);
            let f = spring_k * (d - spring_len);
            let fx = f * dx / d;
            let fy = f * dy / d;
            self.nodes[s].vel.x += fx;
            self.nodes[s].vel.y += fy;
            self.nodes[t].vel.x -= fx;
            self.nodes[t].vel.y -= fy;
        }
        for node in &mut self.nodes {
            node.vel.x *= 0.85;
            node.vel.y *= 0.85;
            node.vel.x = node.vel.x.clamp(-40.0, 40.0);
            node.vel.y = node.vel.y.clamp(-40.0, 40.0);
            node.pos.x += node.vel.x;
            node.pos.y += node.vel.y;
        }
    }

    pub fn pick(&self, world: Vec2, radius: f32) -> Option<usize> {
        let mut best: Option<(usize, f32)> = None;
        for (i, node) in self.nodes.iter().enumerate() {
            let size = node_size(node.degree) + 4.0;
            let dx = node.pos.x - world.x;
            let dy = node.pos.y - world.y;
            let d = (dx * dx + dy * dy).sqrt() - size;
            if d <= radius && best.map(|(_, b)| d < b).unwrap_or(true) {
                best = Some((i, d));
            }
        }
        best.map(|(i, _)| i)
    }
}

pub fn node_size(degree: usize) -> f32 {
    5.0 + 3.0 * ((degree + 1) as f32).ln()
}

#[derive(Debug, Clone, Copy)]
pub struct Camera {
    pub center_x: f64,
    pub center_y: f64,
    pub zoom: f64,
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            center_x: 0.0,
            center_y: 0.0,
            zoom: 1.0,
        }
    }
}

impl Camera {
    pub fn world_to_screen(&self, wx: f32, wy: f32, vw: f32, vh: f32) -> (f32, f32) {
        (
            ((wx as f64 - self.center_x) * self.zoom + vw as f64 / 2.0) as f32,
            ((wy as f64 - self.center_y) * self.zoom + vh as f64 / 2.0) as f32,
        )
    }

    pub fn screen_to_world(&self, sx: f32, sy: f32, vw: f32, vh: f32) -> Vec2 {
        Vec2 {
            x: ((sx as f64 - vw as f64 / 2.0) / self.zoom + self.center_x) as f32,
            y: ((sy as f64 - vh as f64 / 2.0) / self.zoom + self.center_y) as f32,
        }
    }

    pub fn zoom_by(&mut self, factor: f64, sx: f32, sy: f32, vw: f32, vh: f32) {
        let before = self.screen_to_world(sx, sy, vw, vh);
        self.zoom = (self.zoom * factor).clamp(0.05, 40.0);
        let after = self.screen_to_world(sx, sy, vw, vh);
        self.center_x += (before.x - after.x) as f64;
        self.center_y += (before.y - after.y) as f64;
    }

    pub fn fit(&mut self, model: &GraphModel, vw: f32, vh: f32) {
        if model.nodes.is_empty() {
            return;
        }
        let (mut x0, mut x1) = (f32::MAX, f32::MIN);
        let (mut y0, mut y1) = (f32::MAX, f32::MIN);
        for n in &model.nodes {
            x0 = x0.min(n.pos.x);
            x1 = x1.max(n.pos.x);
            y0 = y0.min(n.pos.y);
            y1 = y1.max(n.pos.y);
        }
        let w = (x1 - x0).max(1.0) as f64;
        let h = (y1 - y0).max(1.0) as f64;
        self.center_x = ((x0 + x1) / 2.0) as f64;
        self.center_y = ((y0 + y1) / 2.0) as f64;
        self.zoom = ((vw as f64 / w).min(vh as f64 / h) * 0.9).clamp(0.05, 40.0);
    }
}

// ---- paint ----

const PALETTE: [u32; 12] = [
    0x5B8FF9, 0x5AD8A6, 0xF6BD16, 0xE8684A, 0x6DC8EC, 0x9270CA, 0xFF9D4D, 0x269A99, 0xFF99C3,
    0xA3B1C6, 0xD9E021, 0x6DD8D8,
];

fn node_color(slot: usize) -> Hsla {
    rgb(PALETTE[slot % PALETTE.len()]).into()
}

fn disc_radius(degree: usize, zoom: f64) -> f32 {
    (node_size(degree) * zoom as f32).clamp(3.0, 26.0)
}

/// Paint-friendly copy of one frame: positions, labels, camera, highlights.
#[derive(Clone, Default)]
pub struct Snapshot {
    pub nodes: Vec<SnapNode>,
    pub links: Vec<SnapLink>,
    pub camera: Camera,
    pub highlight: HashSet<usize>,
    pub selected: Option<usize>,
}

#[derive(Clone)]
pub struct SnapNode {
    pub x: f32,
    pub y: f32,
    pub name: String,
    pub color: usize,
    pub dim: bool,
    pub degree: usize,
}

#[derive(Clone)]
pub struct SnapLink {
    pub source: usize,
    pub target: usize,
    pub label: String,
}

impl Snapshot {
    pub fn capture(model: &GraphModel, camera: Camera, highlight: &HashSet<usize>, selected: Option<usize>) -> Self {
        Self {
            nodes: model
                .nodes
                .iter()
                .map(|n| SnapNode {
                    x: n.pos.x,
                    y: n.pos.y,
                    name: n.name.clone(),
                    color: n.color,
                    dim: n.dim,
                    degree: n.degree,
                })
                .collect(),
            links: model
                .links
                .iter()
                .map(|l| SnapLink {
                    source: l.source,
                    target: l.target,
                    label: l.label.clone(),
                })
                .collect(),
            camera,
            highlight: highlight.clone(),
            selected,
        }
    }
}

const INSET: u32 = 0x24283B;
const FG: u32 = 0xC0CAF5;
const MUTED: u32 = 0x565F89;
const AMBER: u32 = 0xE0AF68;

pub fn paint_graph(snap: &Snapshot, bounds: Bounds<Pixels>, window: &mut Window, cx: &mut App) {
    window.paint_layer(bounds, |window| {
        paint_edges(snap, bounds, window);
        paint_nodes(snap, bounds, window);
    });
    paint_labels(snap, bounds, window, cx);
    paint_edge_labels(snap, bounds, window, cx);
}

fn paint_edges(snap: &Snapshot, bounds: Bounds<Pixels>, window: &mut Window) {
    let vw = f32::from(bounds.size.width);
    let vh = f32::from(bounds.size.height);
    let ox = f32::from(bounds.origin.x);
    let oy = f32::from(bounds.origin.y);
    let zoom = snap.camera.zoom;
    let mut plain = PathBuilder::stroke(px(1.));
    let mut hot = PathBuilder::stroke(px(1.5));
    let mut heads = PathBuilder::fill();
    let draw_heads = snap.links.len() <= 3000 && zoom >= 0.5;
    for l in snap.links.iter() {
        let (s, t) = (l.source, l.target);
        let (Some(a), Some(b)) = (snap.nodes.get(s), snap.nodes.get(t)) else {
            continue;
        };
        let (x0, y0) = snap.camera.world_to_screen(a.x, a.y, vw, vh);
        let (x1, y1) = snap.camera.world_to_screen(b.x, b.y, vw, vh);
        if (x0 < -60.0 || x0 > vw + 60.0) && (x1 < -60.0 || x1 > vw + 60.0) {
            continue;
        }
        let dx = x1 - x0;
        let dy = y1 - y0;
        let len = (dx * dx + dy * dy).sqrt();
        if len < 1.0 {
            continue;
        }
        let ux = dx / len;
        let uy = dy / len;
        let r0 = disc_radius(a.degree, zoom);
        let r1 = disc_radius(b.degree, zoom);
        let sx = ox + x0 + ux * r0;
        let sy = oy + y0 + uy * r0;
        let mut ex = ox + x1 - ux * (r1 + if draw_heads { 7.0 } else { 1.0 });
        let mut ey = oy + y1 - uy * (r1 + if draw_heads { 7.0 } else { 1.0 });
        if (ex - sx).hypot(ey - sy) < 2.0 {
            ex = ox + x1 - ux * r1;
            ey = oy + y1 - uy * r1;
        }
        let from = point(px(sx), px(sy));
        let to = point(px(ex), px(ey));
        let is_hot = Some(s) == snap.selected
            || Some(t) == snap.selected
            || snap.highlight.contains(&s)
            || snap.highlight.contains(&t);
        if is_hot {
            hot.move_to(from);
            hot.line_to(to);
        } else {
            plain.move_to(from);
            plain.line_to(to);
        }
        if draw_heads {
            let tip = point(px(ox + x1 - ux * r1), px(oy + y1 - uy * r1));
            let bx = ox + x1 - ux * (r1 + 7.0);
            let by = oy + y1 - uy * (r1 + 7.0);
            let w = 3.2;
            heads.move_to(tip);
            heads.line_to(point(px(bx - uy * w), px(by + ux * w)));
            heads.line_to(point(px(bx + uy * w), px(by - ux * w)));
            heads.line_to(tip);
        }
    }
    let edge: Hsla = rgb(0x3B4261).into();
    let amber: Hsla = rgb(AMBER).into();
    if let Ok(path) = plain.build() {
        window.paint_path(path, edge);
    }
    if let Ok(path) = hot.build() {
        window.paint_path(path, amber);
    }
    if draw_heads {
        if let Ok(path) = heads.build() {
            window.paint_path(path, edge);
        }
    }
}

fn paint_nodes(snap: &Snapshot, bounds: Bounds<Pixels>, window: &mut Window) {
    let vw = f32::from(bounds.size.width);
    let vh = f32::from(bounds.size.height);
    let ox = f32::from(bounds.origin.x);
    let oy = f32::from(bounds.origin.y);
    for (i, node) in snap.nodes.iter().enumerate() {
        let (sx, sy) = snap.camera.world_to_screen(node.x, node.y, vw, vh);
        if sx < -50.0 || sy < -50.0 || sx > vw + 50.0 || sy > vh + 50.0 {
            continue;
        }
        let r = disc_radius(node.degree, snap.camera.zoom);
        let (ring_width, ring_color): (Pixels, Hsla) =
            if Some(i) == snap.selected || snap.highlight.contains(&i) {
                (px(2.), rgb(AMBER).into())
            } else {
                (px(1.), rgb(INSET).into())
            };
        let mut fill: Hsla = node_color(node.color);
        if node.dim {
            fill = fill.opacity(0.45);
        }
        window.paint_quad(PaintQuad {
            bounds: Bounds::new(
                point(px(ox + sx - r), px(oy + sy - r)),
                size(px(r * 2.0), px(r * 2.0)),
            ),
            corner_radii: Corners::all(px(r)),
            background: fill.into(),
            border_widths: Edges::all(ring_width),
            border_color: ring_color,
            border_style: BorderStyle::Solid,
        });
    }
}

fn paint_labels(snap: &Snapshot, bounds: Bounds<Pixels>, window: &mut Window, cx: &mut App) {
    let zoom = snap.camera.zoom;
    let vw = f32::from(bounds.size.width);
    let vh = f32::from(bounds.size.height);
    let ox = f32::from(bounds.origin.x);
    let oy = f32::from(bounds.origin.y);
    let font_size = px(11.0);
    let mut cands: Vec<(usize, f32, f32, f32)> = Vec::new();
    for (i, node) in snap.nodes.iter().enumerate() {
        let (sx, sy) = snap.camera.world_to_screen(node.x, node.y, vw, vh);
        if sx < -80.0 || sy < -40.0 || sx > vw + 80.0 || sy > vh + 40.0 {
            continue;
        }
        let r = disc_radius(node.degree, zoom);
        cands.push((i, sx, sy, r));
    }
    if cands.is_empty() {
        return;
    }
    cands.sort_by_key(|(i, _, _, _)| std::cmp::Reverse(snap.nodes[*i].degree));
    let cap = if zoom >= 2.0 {
        220
    } else if zoom >= 0.9 {
        80
    } else {
        24
    };
    cands.truncate(cap);
    let bright: Hsla = rgb(FG).into();
    let pill: Hsla = rgb(INSET).into();
    for (i, sx, sy, r) in cands {
        let hot = Some(i) == snap.selected || snap.highlight.contains(&i);
        if !hot && zoom < 0.9 && snap.nodes[i].degree < 2 {
            continue;
        }
        let text: SharedString = snap.nodes[i].name.clone().into();
        let run = TextRun {
            len: text.len(),
            font: font(".SystemUIFont"),
            color: bright,
            background_color: None,
            underline: None,
            strikethrough: None,
        };
        let shaped = window.text_system().shape_line(text, font_size, &[run], None);
        let w = f32::from(shaped.width);
        let h = f32::from(shaped.ascent + shaped.descent);
        let pad_x = 4.0;
        let pad_y = 2.0;
        let lx = (ox + sx + r + 4.0).clamp(2.0, (ox + vw - w - pad_x * 2.0 - 2.0).max(2.0));
        let ly = (oy + sy - h / 2.0 - pad_y).clamp(2.0, (oy + vh - h - pad_y * 2.0 - 2.0).max(2.0));
        window.paint_quad(PaintQuad {
            bounds: Bounds::new(
                point(px(lx), px(ly)),
                size(px(w + pad_x * 2.0), px(h + pad_y * 2.0)),
            ),
            corner_radii: Corners::all(px(3.)),
            background: pill.opacity(0.88).into(),
            border_widths: Edges::default(),
            border_color: transparent_black(),
            border_style: BorderStyle::Solid,
        });
        let _ = shaped.paint(point(px(lx + pad_x), px(ly + pad_y)), px(13.), window, cx);
    }
}

fn paint_edge_labels(snap: &Snapshot, bounds: Bounds<Pixels>, window: &mut Window, cx: &mut App) {
    if snap.links.len() > 2000 {
        return;
    }
    let zoom = snap.camera.zoom;
    let vw = f32::from(bounds.size.width);
    let vh = f32::from(bounds.size.height);
    let ox = f32::from(bounds.origin.x);
    let oy = f32::from(bounds.origin.y);
    let font_size = px(10.0);
    let muted: Hsla = rgb(MUTED).into();
    let pill: Hsla = rgb(INSET).into();
    let mut shown = 0;
    for l in &snap.links {
        if shown >= 120 || l.label.is_empty() {
            continue;
        }
        let hot = snap.highlight.contains(&l.source) || snap.highlight.contains(&l.target);
        if !hot && zoom < 0.9 {
            continue;
        }
        let (Some(a), Some(b)) = (snap.nodes.get(l.source), snap.nodes.get(l.target)) else {
            continue;
        };
        let (x0, y0) = snap.camera.world_to_screen(a.x, a.y, vw, vh);
        let (x1, y1) = snap.camera.world_to_screen(b.x, b.y, vw, vh);
        let mx = (x0 + x1) / 2.0;
        let my = (y0 + y1) / 2.0;
        if (x1 - x0).hypot(y1 - y0) < 48.0 || mx < 0.0 || my < 0.0 || mx > vw || my > vh {
            continue;
        }
        shown += 1;
        let text: SharedString = l.label.clone().into();
        let run = TextRun {
            len: text.len(),
            font: font(".SystemUIFont"),
            color: muted,
            background_color: None,
            underline: None,
            strikethrough: None,
        };
        let shaped = window.text_system().shape_line(text, font_size, &[run], None);
        let w = f32::from(shaped.width);
        let h = f32::from(shaped.ascent + shaped.descent);
        let lx = (ox + mx - w / 2.0 - 3.0).clamp(2.0, (ox + vw - w - 8.0).max(2.0));
        let ly = (oy + my - h / 2.0 - 1.0).clamp(2.0, (oy + vh - h - 4.0).max(2.0));
        window.paint_quad(PaintQuad {
            bounds: Bounds::new(point(px(lx), px(ly)), size(px(w + 6.0), px(h + 2.0))),
            corner_radii: Corners::all(px(3.)),
            background: pill.opacity(0.88).into(),
            border_widths: Edges::default(),
            border_color: transparent_black(),
            border_style: BorderStyle::Solid,
        });
        let _ = shaped.paint(point(px(lx + 3.0), px(ly + 1.0)), px(13.), window, cx);
    }
}
