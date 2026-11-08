//! `lbm demo`: chat with the lb-memory agent, watch the entity graph grow.
//!
//! Left pane: chat log + single-line input (Enter sends, ↑/↓ history).
//! Right pane: the force-directed entity graph (drag nodes, pan, wheel
//! zoom, double-click empty space to refit). Every chat message goes
//! through GLiNER on a worker thread; the graph merges the new snapshot
//! in place when the tick lands.

pub mod agent;
pub mod graph;

use std::cell::Cell;
use std::collections::HashSet;
use std::rc::Rc;

use agent::{AgentConfig, AgentHandle, ChatReply, FromAgent, ToAgent, spawn_agent};
use anyhow::Result;
use gpui::*;
use graph::{Camera, GraphData, GraphModel, Snapshot, paint_graph};

#[derive(Clone)]
pub struct DemoConfig {
    pub db_path: String,
    pub snapshot: String,
}

pub fn default_db() -> String {
    std::env::var("LB_MEMORY_DB").unwrap_or_else(|_| {
        format!(
            "{}/.pi/agent/lb-memory/demo.lbdb",
            std::env::var("HOME").unwrap_or_else(|_| "/tmp".into())
        )
    })
}

pub fn default_snapshot() -> String {
    std::env::var("LB_MEMORY_SNAPSHOT").unwrap_or_else(|_| {
        format!(
            "{}/.cache/ladybug-memory/models/gliner2.5-base-v1-nico-onnx",
            std::env::var("HOME").unwrap_or_else(|_| "/tmp".into())
        )
    })
}

pub fn run(cfg: DemoConfig) -> Result<()> {
    Application::new().run(move |cx: &mut App| {
        cx.open_window(
            WindowOptions {
                window_bounds: Some(WindowBounds::Windowed(Bounds::centered(
                    None,
                    size(px(1280.0), px(860.0)),
                    cx,
                ))),
                titlebar: Some(TitlebarOptions {
                    title: Some("lb-memory demo".into()),
                    ..Default::default()
                }),
                ..Default::default()
            },
            {
                let cfg = cfg.clone();
                move |window, cx| {
                    let view = cx.new(|cx| DemoView::new(cfg, cx));
                    window.focus(&view.read(cx).input_focus_handle());
                    view
                }
            },
        )
        .expect("failed to open demo window");
        cx.activate(true);
    });
    Ok(())
}

// ---- view ----

#[derive(Clone)]
struct ChatMsg {
    role: Role,
    text: String,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Role {
    User,
    Agent,
    Error,
    Info,
}

struct DemoView {
    agent: AgentHandle,
    model: GraphModel,
    camera: Camera,
    running: bool,
    still_ticks: u32,
    messages: Vec<ChatMsg>,
    input: String,
    cursor: usize,
    input_history: Vec<String>,
    hist_idx: Option<usize>,
    input_focus: FocusHandle,
    status: String,
    busy: bool,
    highlight_keys: HashSet<String>,
    highlight: HashSet<usize>,
    selected: Option<usize>,
    drag_node: Option<usize>,
    panning: Option<Point<Pixels>>,
    canvas_origin: Rc<Cell<Point<Pixels>>>,
    canvas_size: Rc<Cell<Size<Pixels>>>,
    counts: (usize, usize, usize),
    fitted_once: bool,
}

impl DemoView {
    fn new(cfg: DemoConfig, cx: &mut Context<Self>) -> Self {
        let agent = spawn_agent(AgentConfig {
            db_path: cfg.db_path,
            snapshot: cfg.snapshot,
            user: "demo".into(),
            project: "demo-chat".into(),
        });
        let mut view = Self {
            agent,
            model: GraphModel::default(),
            camera: Camera::default(),
            running: false,
            still_ticks: 0,
            messages: vec![ChatMsg {
                role: Role::Info,
                text: "Chat with the memory agent. Every message is distilled into facts (GLiNER) and the graph updates live. Try: Tim Cook runs Apple.  Commands: /search <q> · /help".into(),
            }],
            input: String::new(),
            cursor: 0,
            input_history: Vec::new(),
            hist_idx: None,
            input_focus: cx.focus_handle(),
            status: "starting…".into(),
            busy: false,
            highlight_keys: HashSet::new(),
            highlight: HashSet::new(),
            selected: None,
            drag_node: None,
            panning: None,
            canvas_origin: Rc::new(Cell::new(point(px(0.), px(0.)))),
            canvas_size: Rc::new(Cell::new(size(px(900.), px(600.)))),
            counts: (0, 0, 0),
            fitted_once: false,
        };
        view.start_layout_loop(cx);
        view
    }

    fn input_focus_handle(&self) -> FocusHandle {
        self.input_focus.clone()
    }

    fn start_layout_loop(&mut self, cx: &mut Context<Self>) {
        cx.spawn(async move |this, cx| loop {
            cx.background_executor()
                .timer(std::time::Duration::from_millis(33))
                .await;
            let alive = this
                .update(cx, |view, cx| {
                    view.drain_agent(cx);
                    if view.running && !view.model.nodes.is_empty() {
                        view.model.tick();
                        let peak = view
                            .model
                            .nodes
                            .iter()
                            .map(|n| n.vel.x.abs().max(n.vel.y.abs()))
                            .fold(0.0f32, f32::max);
                        if peak < 0.05 {
                            view.still_ticks += 1;
                            if view.still_ticks > 30 {
                                view.running = false;
                            }
                        } else {
                            view.still_ticks = 0;
                        }
                        cx.notify();
                    }
                })
                .is_ok();
            if !alive {
                break;
            }
        })
        .detach();
    }

    fn drain_agent(&mut self, cx: &mut Context<Self>) {
        let mut changed = false;
        while let Ok(msg) = self.agent.rx.try_recv() {
            changed = true;
            match msg {
                FromAgent::Status(s) => {
                    if !self.busy {
                        self.status = s;
                    } else if s == "ready" {
                        self.status = s;
                    } else {
                        self.status = s;
                    }
                }
                FromAgent::Reply(reply) => {
                    self.apply_reply(reply);
                    self.busy = false;
                    self.status = "ready".into();
                }
                FromAgent::Graph(data) => {
                    self.apply_graph(data);
                }
                FromAgent::Error(e) => {
                    self.push(Role::Error, e);
                    self.busy = false;
                    self.status = "error".into();
                }
            }
        }
        if changed {
            cx.notify();
        }
    }

    fn apply_reply(&mut self, reply: ChatReply) {
        self.push(Role::Agent, reply.agent_text);
        self.highlight_keys = reply.highlight_keys.into_iter().collect();
        self.rebuild_highlight();
    }

    fn rebuild_highlight(&mut self) {
        self.highlight.clear();
        for (i, n) in self.model.nodes.iter().enumerate() {
            if self.highlight_keys.contains(&n.id) {
                self.highlight.insert(i);
            }
        }
    }

    fn apply_graph(&mut self, data: GraphData) {
        let was_empty = self.model.nodes.is_empty();
        self.model.merge(&data);
        self.rebuild_highlight();
        let mem = data.nodes.iter().filter(|n| n.key.starts_with('m')).count();
        let ent = data.nodes.len() - mem;
        self.counts = (mem, ent, data.links.len());
        if was_empty && !self.model.nodes.is_empty() {
            // First paint: converge the layout synchronously so the graph
            // opens on formed topology instead of a spiral.
            self.model.settle();
            let (w, h) = self.canvas_size_px();
            self.camera.fit(&self.model, w, h);
            self.fitted_once = true;
        }
        self.running = true;
        self.still_ticks = 0;
    }

    fn push(&mut self, role: Role, text: String) {
        self.messages.push(ChatMsg { role, text });
        if self.messages.len() > 200 {
            let drop = self.messages.len() - 200;
            self.messages.drain(..drop);
        }
    }

    fn submit(&mut self, cx: &mut Context<Self>) {
        let text = self.input.trim().to_string();
        if text.is_empty() {
            return;
        }
        if self.busy {
            self.status = "still thinking…".into();
            cx.notify();
            return;
        }
        self.input.clear();
        self.cursor = 0;
        self.input_history.push(text.clone());
        self.hist_idx = None;
        if text == "/help" {
            self.push(Role::User, text);
            self.push(
                Role::Agent,
                "Type anything to remember it (entity-bearing sentences become facts). /search <q> recalls and rings the matching memory nodes on the graph. /reload re-reads the graph. Drag nodes · drag empty space to pan · wheel to zoom · double-click empty space to refit.".into(),
            );
            cx.notify();
            return;
        }
        self.push(Role::User, text.clone());
        if let Some(q) = text.strip_prefix("/search").map(str::trim) {
            if q.is_empty() {
                self.push(Role::Agent, "Usage: /search <query>".into());
                cx.notify();
                return;
            }
            self.highlight_keys.clear();
            self.highlight.clear();
            let _ = self.agent.tx.send(ToAgent::Search(q.to_string()));
        } else if text == "/reload" {
            let _ = self.agent.tx.send(ToAgent::Reload);
            self.busy = true;
            self.status = "reloading…".into();
            cx.notify();
            return;
        } else if text.starts_with('/') {
            self.push(Role::Agent, "Unknown command — try /search <q>, /reload or /help.".into());
            cx.notify();
            return;
        } else {
            let _ = self.agent.tx.send(ToAgent::Remember(text));
        }
        self.busy = true;
        self.status = "thinking…".into();
        cx.notify();
    }

    fn canvas_size_px(&self) -> (f32, f32) {
        let s = self.canvas_size.get();
        (f32::from(s.width), f32::from(s.height))
    }

    fn to_local(&self, p: Point<Pixels>) -> (f32, f32) {
        let o = self.canvas_origin.get();
        (f32::from(p.x - o.x), f32::from(p.y - o.y))
    }

    // ---- input editing (compact readline) ----

    fn insert_str(&mut self, s: &str) {
        self.input.insert_str(self.cursor, s);
        self.cursor += s.len();
    }

    fn move_left(&mut self) {
        if self.cursor == 0 {
            return;
        }
        let mut c = self.cursor - 1;
        while c > 0 && !self.input.is_char_boundary(c) {
            c -= 1;
        }
        self.cursor = c;
    }

    fn move_right(&mut self) {
        if self.cursor >= self.input.len() {
            return;
        }
        let mut c = self.cursor + 1;
        while c < self.input.len() && !self.input.is_char_boundary(c) {
            c += 1;
        }
        self.cursor = c;
    }

    fn backspace(&mut self) {
        if self.cursor == 0 {
            return;
        }
        let mut c = self.cursor - 1;
        while c > 0 && !self.input.is_char_boundary(c) {
            c -= 1;
        }
        self.input.drain(c..self.cursor);
        self.cursor = c;
    }

    fn delete_at(&mut self) {
        if self.cursor >= self.input.len() {
            return;
        }
        let mut c = self.cursor + 1;
        while c < self.input.len() && !self.input.is_char_boundary(c) {
            c += 1;
        }
        self.input.drain(self.cursor..c);
    }

    fn history_prev(&mut self) {
        if self.input_history.is_empty() {
            return;
        }
        let next = match self.hist_idx {
            None => self.input_history.len() - 1,
            Some(0) => 0,
            Some(i) => i - 1,
        };
        self.hist_idx = Some(next);
        self.input = self.input_history[next].clone();
        self.cursor = self.input.len();
    }

    fn history_next(&mut self) {
        match self.hist_idx {
            None => {}
            Some(i) if i + 1 >= self.input_history.len() => {
                self.hist_idx = None;
                self.input.clear();
                self.cursor = 0;
            }
            Some(i) => {
                self.hist_idx = Some(i + 1);
                self.input = self.input_history[i + 1].clone();
                self.cursor = self.input.len();
            }
        }
    }

    fn on_input_key(&mut self, ev: &KeyDownEvent, cx: &mut Context<Self>) {
        let mods = &ev.keystroke.modifiers;
        let cmd = mods.platform || mods.control;
        match ev.keystroke.key.as_str() {
            "enter" => {
                self.submit(cx);
                return;
            }
            "escape" => {
                self.input.clear();
                self.cursor = 0;
                cx.notify();
                return;
            }
            "backspace" => {
                self.backspace();
                cx.notify();
                return;
            }
            "delete" => {
                self.delete_at();
                cx.notify();
                return;
            }
            "left" => {
                self.move_left();
                cx.notify();
                return;
            }
            "right" => {
                self.move_right();
                cx.notify();
                return;
            }
            "home" => {
                self.cursor = 0;
                cx.notify();
                return;
            }
            "end" => {
                self.cursor = self.input.len();
                cx.notify();
                return;
            }
            "up" if !cmd => {
                self.history_prev();
                cx.notify();
                return;
            }
            "down" if !cmd => {
                self.history_next();
                cx.notify();
                return;
            }
            _ => {}
        }
        if !cmd && !mods.alt {
            if let Some(ch) = ev.keystroke.key_char.clone() {
                self.insert_str(&ch);
                cx.notify();
            }
        }
    }

    // ---- render ----

    fn render_header(&self) -> Div {
        let (mem, ent, links) = self.counts;
        div()
            .flex()
            .flex_row()
            .items_center()
            .gap_3()
            .px_3()
            .py_2()
            .bg(rgb(0x24283B))
            .text_color(rgb(0xC0CAF5))
            .child(div().font_weight(FontWeight::BOLD).child("lb-memory demo"))
            .child(
                div()
                    .text_sm()
                    .text_color(rgb(0x565F89))
                    .child(format!("{mem} memories · {ent} entities · {links} links")),
            )
            .child(div().flex_1())
            .child(
                div()
                    .text_sm()
                    .text_color(rgb(0x565F89))
                    .child(format!("demo-chat · {}", self.status)),
            )
    }

    fn render_chat(&mut self, window: &mut Window, cx: &mut Context<Self>) -> Div {
        let mut log = div()
            .flex()
            .flex_col()
            .gap_2()
            .p_3()
            .flex_1()
            .w_full()
            .id("chat-log")
            .overflow_y_scroll();
        for m in self.messages.clone() {
            let (role, color) = match m.role {
                Role::User => ("you ›".to_string(), rgb(0x7AA2F7)),
                Role::Agent => ("agent ›".to_string(), rgb(0x9ECE6A)),
                Role::Error => ("error ›".to_string(), rgb(0xF7768E)),
                Role::Info => ("· ".to_string(), rgb(0x565F89)),
            };
            let mut col = div().flex().flex_col().gap_0();
            col = col.child(div().text_xs().text_color(color).child(role));
            for line in wrap_text(&m.text, 44) {
                col = col.child(div().text_sm().text_color(rgb(0xC0CAF5)).child(line));
            }
            log = log.child(col);
        }
        let focused = self.input_focus.is_focused(window);
        let (before, cur, after) = caret_parts(&self.input, self.cursor, focused);
        let mut cur_el = div().child(cur);
        if focused && !self.input.is_empty() {
            cur_el = cur_el.bg(rgb(0x7AA2F7)).text_color(rgb(0x1A1B26)).rounded_sm();
        }
        let content = div().flex().flex_row().child(before).child(cur_el).child(after);
        let focused_handle = self.input_focus.clone();
        let input_box = div()
            .id("chat-input")
            .w_full()
            .px_2()
            .py_1()
            .rounded_md()
            .bg(rgb(0x1F2335))
            .border_1()
            .border_color(if focused { rgb(0x7AA2F7) } else { rgb(0x3B4261) })
            .text_color(rgb(0xC0CAF5))
            .text_sm()
            .cursor_text()
            .track_focus(&focused_handle)
            .on_mouse_down(
                MouseButton::Left,
                cx.listener(|view, _, window, cx| {
                    window.focus(&view.input_focus_handle());
                    cx.notify();
                }),
            )
            .on_key_down(cx.listener(|view, ev: &KeyDownEvent, _, cx| {
                view.on_input_key(ev, cx);
            }))
            .child(if self.input.is_empty() && !focused {
                div()
                    .text_color(rgb(0x565F89))
                    .child("chat to remember… (Enter sends)".to_string())
            } else {
                content
            });
        div()
            .flex()
            .flex_col()
            .w(px(380.))
            .h_full()
            .bg(rgb(0x24283B))
            .child(
                div()
                    .px_3()
                    .py_2()
                    .font_weight(FontWeight::BOLD)
                    .text_color(rgb(0xC0CAF5))
                    .child("Memory Chat".to_string()),
            )
            .child(div().flex_1().min_h_0().child(log))
            .child(div().px_3().py_2().child(input_box))
    }

    fn render_canvas(&mut self, cx: &mut Context<Self>) -> Div {
        let snap = Snapshot::capture(&self.model, self.camera, &self.highlight, self.selected);
        let origin_cell = self.canvas_origin.clone();
        let size_cell = self.canvas_size.clone();
        div()
            .flex_1()
            .h_full()
            .bg(rgb(0x1A1B26))
            .overflow_hidden()
            .child(
                canvas(
                    move |bounds, _window, _cx| {
                        origin_cell.set(bounds.origin);
                        size_cell.set(bounds.size);
                        (snap, bounds)
                    },
                    move |_bounds, (snap, bounds), window, cx| {
                        paint_graph(&snap, bounds, window, cx);
                    },
                )
                .flex_1()
                .h_full(),
            )
            .on_mouse_down(
                MouseButton::Left,
                cx.listener(|view, ev: &MouseDownEvent, _, cx| {
                    let (lx, ly) = view.to_local(ev.position);
                    let (vw, vh) = {
                        let s = view.canvas_size.get();
                        (f32::from(s.width), f32::from(s.height))
                    };
                    let world = view.camera.screen_to_world(lx, ly, vw, vh);
                    if ev.click_count >= 2 {
                        if view.to_local(ev.position).0 < 0.0 {
                            return;
                        }
                        // Double-click empty space refits; on a node it selects.
                        if let Some(i) = view.model.pick(world, 6.0) {
                            view.selected = Some(i);
                            let n = &view.model.nodes[i];
                            view.status = format!("{} [{}]", n.name, n.label);
                        } else {
                            view.camera.fit(&view.model, vw, vh);
                            view.selected = None;
                        }
                        cx.notify();
                        return;
                    }
                    view.panning = Some(ev.position);
                    if let Some(i) = view.model.pick(world, 6.0) {
                        view.drag_node = Some(i);
                        view.selected = Some(i);
                        view.panning = None;
                        let n = &view.model.nodes[i];
                        view.status = format!("{} [{}]", n.name, n.label);
                    }
                    cx.notify();
                }),
            )
            .on_mouse_up(
                MouseButton::Left,
                cx.listener(|view, _, _, cx| {
                    view.drag_node = None;
                    view.panning = None;
                    cx.notify();
                }),
            )
            .on_mouse_move(cx.listener(|view, ev: &MouseMoveEvent, _, cx| {
                let (lx, ly) = view.to_local(ev.position);
                let (vw, vh) = {
                    let s = view.canvas_size.get();
                    (f32::from(s.width), f32::from(s.height))
                };
                if let Some(i) = view.drag_node {
                    if let Some(n) = view.model.nodes.get_mut(i) {
                        n.pos = view.camera.screen_to_world(lx, ly, vw, vh);
                        n.vel = graph::Vec2 { x: 0.0, y: 0.0 };
                    }
                    view.running = true;
                    view.still_ticks = 0;
                    cx.notify();
                    return;
                }
                if let Some(last) = view.panning {
                    let dx = ev.position.x - last.x;
                    let dy = ev.position.y - last.y;
                    view.camera.center_x -= f64::from(dx) / view.camera.zoom;
                    view.camera.center_y -= f64::from(dy) / view.camera.zoom;
                    view.panning = Some(ev.position);
                    cx.notify();
                }
            }))
            .on_scroll_wheel(cx.listener(|view, ev: &ScrollWheelEvent, _, cx| {
                let (lx, ly) = view.to_local(ev.position);
                let (vw, vh) = {
                    let s = view.canvas_size.get();
                    (f32::from(s.width), f32::from(s.height))
                };
                let dy: f32 = match ev.delta {
                    ScrollDelta::Pixels(p) => f32::from(p.y),
                    ScrollDelta::Lines(p) => p.y * 20.0,
                };
                let factor = if dy < 0.0 { 1.12 } else { 1.0 / 1.12 };
                view.camera.zoom_by(factor, lx, ly, vw, vh);
                cx.notify();
            }))
    }
}

impl Render for DemoView {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        div()
            .flex()
            .flex_col()
            .size_full()
            .bg(rgb(0x1A1B26))
            .text_color(rgb(0xC0CAF5))
            .child(self.render_header())
            .child(
                div()
                    .flex()
                    .flex_row()
                    .flex_1()
                    .min_h_0()
                    .child(self.render_chat(window, cx))
                    .child(div().flex_1().min_h_0().child(self.render_canvas(cx))),
            )
    }
}

// ---- helpers ----

fn caret_parts(input: &str, cursor: usize, focused: bool) -> (String, String, String) {
    let mut at = cursor.min(input.len());
    if !input.is_char_boundary(at) {
        at = input.len();
    }
    if !focused {
        return (input.to_string(), String::new(), String::new());
    }
    if input.is_empty() {
        return ("▍".to_string(), String::new(), String::new());
    }
    let (b, rest) = input.split_at(at);
    match rest.chars().next() {
        Some(c) => (
            b.to_string(),
            c.to_string(),
            rest[c.len_utf8()..].to_string(),
        ),
        None => (b.to_string(), " ".to_string(), String::new()),
    }
}

fn wrap_text(text: &str, width: usize) -> Vec<String> {
    let mut lines = Vec::new();
    for para in text.split('\n') {
        let mut line = String::new();
        for word in para.split_whitespace() {
            if line.len() + word.len() + 1 > width && !line.is_empty() {
                lines.push(std::mem::take(&mut line));
            }
            if !line.is_empty() {
                line.push(' ');
            }
            line.push_str(word);
        }
        lines.push(line);
    }
    if lines.is_empty() {
        lines.push(String::new());
    }
    lines
}
