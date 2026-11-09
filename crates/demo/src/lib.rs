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
use graph::{Camera, GraphData, GraphModel, Hover, Snapshot, paint_graph};

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
    focused_once: bool,
    hover: Option<Hover>,
    /// Selection anchor (byte offset); the cursor is the active end.
    /// `None` means no selection.
    sel_anchor: Option<usize>,
    /// Mouse-dragging a text selection in the chat input.
    input_drag: bool,
    /// Last mouse position in canvas-local px (for tooltip placement).
    mouse: Option<(f32, f32)>,
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
                text: "Chat with the memory agent. Every message is distilled into facts (GLiNER) and the graph updates live. Try: Tim Cook runs Apple.  ↑↓ history · ⌘V paste · ⌘C/X copy/cut · ⌘A select · ⇧←→ select · /search <q> · /help".into(),
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
            focused_once: false,
            hover: None,
            mouse: None,
            sel_anchor: None,
            input_drag: false,
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
                        if view.mouse.is_some() && view.drag_node.is_none() && view.panning.is_none() {
                            let (lx, ly) = view.mouse.unwrap();
                            let (vw, vh) = view.canvas_size_px();
                            let world = view.camera.screen_to_world(lx, ly, vw, vh);
                            let radius = 6.0 / view.camera.zoom.max(0.2) as f32;
                            view.hover = view
                                .model
                                .pick(world, 6.0)
                                .map(graph::Hover::Node)
                                .or_else(|| view.model.pick_edge(world, radius).map(graph::Hover::Edge));
                        }
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
        if !self.highlight.is_empty() {
            // Frame the hits: search results should land in view.
            let (w, h) = self.canvas_size_px();
            self.camera.fit_subset(&self.model, &self.highlight, w, h);
        }
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
        // Entity graph only: fact (memory) nodes and co-occurrence links are
        // staging detail, not shown. Typed relations between entities are
        // the edges that remain.
        let data = GraphData {
            nodes: data.nodes.into_iter().filter(|n| !n.key.starts_with('m')).collect(),
            links: data
                .links
                .into_iter()
                .filter(|l| {
                    l.label != "coOccurs" && !l.source.starts_with('m') && !l.target.starts_with('m')
                })
                .collect(),
        };
        let had_keys: std::collections::HashSet<&str> =
            self.model.nodes.iter().map(|n| n.id.as_str()).collect();
        let is_new = data.nodes.iter().any(|n| !had_keys.contains(n.key.as_str()));
        let was_empty = self.model.nodes.is_empty();
        self.model.merge(&data);
        self.rebuild_highlight();
        let mem = data.nodes.iter().filter(|n| n.key.starts_with('m')).count();
        let ent = data.nodes.len() - mem;
        self.counts = (mem, ent, data.links.len());
        if !self.model.nodes.is_empty() && (was_empty || is_new) {
            // New topology: converge synchronously (so the camera frames
            // settled positions, not the fling), then fit the view so
            // newcomers land in the viewport instead of drifting out.
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
                "Type anything to remember it (entity-bearing sentences become facts). /search <q> recalls and rings matches on the graph. /reload re-reads. Chat keys: ↑↓ history, ⌘V paste, ⌘C/⌘X copy/cut, ⌘A select all, ⇧←/→ select, ^K/^U kill line. Drag nodes · drag empty space to pan · wheel to zoom · double-click empty space to refit.".into(),
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

    /// Map a window-x position to a byte offset in the chat input.
    /// The left column is a fixed 380px; the input text starts after its
    /// px-3 outer padding (12), 1px border and px-2 inner padding (8).
    /// Approximate by a few px at most — drag selection self-corrects
    /// against the live highlight.
    fn input_hit_index(&self, window: &mut Window, x: Pixels) -> usize {
        const TEXT_DX: f32 = 21.0;
        if self.input.is_empty() {
            return 0;
        }
        let run = TextRun {
            len: self.input.len(),
            font: font(".SystemUIFont"),
            color: rgb(0xC0CAF5).into(),
            background_color: None,
            underline: None,
            strikethrough: None,
        };
        let shaped = window
            .text_system()
            .shape_line(self.input.clone().into(), px(14.), &[run], None);
        let mut idx = shaped.closest_index_for_x(px(f32::from(x) - TEXT_DX));
        idx = idx.min(self.input.len());
        while idx > 0 && !self.input.is_char_boundary(idx) {
            idx -= 1;
        }
        idx
    }

    // ---- input editing (compact readline) ----

    fn insert_str(&mut self, s: &str) {
        self.input.insert_str(self.cursor, s);
        self.cursor += s.len();
    }

    fn move_left(&mut self) {
        self.sel_anchor = None;
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
        self.sel_anchor = None;
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
        if self.delete_selection() {
            return;
        }
        if self.cursor == 0 {
            return;
        }
        let mut c = self.cursor - 1;
        while c > 0 && !self.input.is_char_boundary(c) {
            c -= 1;
        }
        self.input.drain(c..self.cursor);
        self.cursor = c;
        self.hist_idx = None;
    }

    fn delete_at(&mut self) {
        if self.delete_selection() {
            return;
        }
        if self.cursor >= self.input.len() {
            return;
        }
        let mut c = self.cursor + 1;
        while c < self.input.len() && !self.input.is_char_boundary(c) {
            c += 1;
        }
        self.input.drain(self.cursor..c);
        self.hist_idx = None;
    }

    /// Ordered selection bounds, or `None` when there is no selection.
    /// Byte offsets, clamped to char boundaries.
    fn selection(&self) -> Option<(usize, usize)> {
        let anchor = self.sel_anchor?;
        let (mut lo, mut hi) = if anchor <= self.cursor {
            (anchor, self.cursor)
        } else {
            (self.cursor, anchor)
        };
        lo = lo.min(self.input.len());
        hi = hi.min(self.input.len());
        while lo > 0 && !self.input.is_char_boundary(lo) {
            lo -= 1;
        }
        while hi < self.input.len() && !self.input.is_char_boundary(hi) {
            hi += 1;
        }
        if lo == hi {
            None
        } else {
            Some((lo, hi))
        }
    }

    fn clear_selection(&mut self) {
        self.sel_anchor = None;
    }

    /// Delete the selected text, placing the cursor at its start.
    /// Returns true when a selection was consumed.
    fn delete_selection(&mut self) -> bool {
        if let Some((lo, hi)) = self.selection() {
            self.input.drain(lo..hi);
            self.cursor = lo;
            self.sel_anchor = None;
            self.hist_idx = None;
            true
        } else {
            false
        }
    }

    fn select_all(&mut self) {
        if self.input.is_empty() {
            return;
        }
        self.sel_anchor = Some(0);
        self.cursor = self.input.len();
    }

    /// Kill to end of line; returns the killed text.
    fn kill_to_end(&mut self) -> String {
        if self.delete_selection() {
            return String::new();
        }
        let killed = self.input[self.cursor..].to_string();
        self.input.truncate(self.cursor);
        self.hist_idx = None;
        killed
    }

    /// Kill the whole line; returns the killed text.
    fn kill_line(&mut self) -> String {
        let killed = std::mem::take(&mut self.input);
        self.cursor = 0;
        self.sel_anchor = None;
        self.hist_idx = None;
        killed
    }

    /// Kill the word before the cursor (whitespace-skipping, like readline
    /// backward-kill-word); returns the killed text.
    fn kill_word_back(&mut self) -> String {
        if self.delete_selection() {
            return String::new();
        }
        let mut c = self.cursor;
        let bytes = self.input.as_bytes();
        while c > 0 && bytes[c - 1].is_ascii_whitespace() {
            c -= 1;
        }
        while c > 0 && !bytes[c - 1].is_ascii_whitespace() {
            c -= 1;
        }
        while c > 0 && !self.input.is_char_boundary(c) {
            c -= 1;
        }
        let killed = self.input[c..self.cursor].to_string();
        self.input.drain(c..self.cursor);
        self.cursor = c;
        self.hist_idx = None;
        killed
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
        self.sel_anchor = None;
    }

    fn history_next(&mut self) {
        match self.hist_idx {
            None => {}
            Some(i) if i + 1 >= self.input_history.len() => {
                self.hist_idx = None;
                self.input.clear();
                self.cursor = 0;
                self.sel_anchor = None;
            }
            Some(i) => {
                self.hist_idx = Some(i + 1);
                self.input = self.input_history[i + 1].clone();
                self.cursor = self.input.len();
                self.sel_anchor = None;
            }
        }
    }

    fn on_input_key(&mut self, ev: &KeyDownEvent, cx: &mut Context<Self>) {
        let mods = &ev.keystroke.modifiers;
        let cmd = mods.platform || mods.control;
        let only_ctrl = mods.control && !mods.platform && !mods.alt && !mods.shift;
        // Clipboard + select-all (Cmd or Ctrl).
        if cmd {
            match ev.keystroke.key.as_str() {
                "v" => {
                    self.paste(cx);
                    cx.notify();
                    return;
                }
                "c" => {
                    self.copy_selection(cx);
                    return;
                }
                "x" => {
                    self.cut_selection(cx);
                    cx.notify();
                    return;
                }
                "a" if mods.platform => {
                    self.select_all();
                    cx.notify();
                    return;
                }
                _ => {}
            }
        }
        // Readline controls (Ctrl only, so Cmd+A stays select-all).
        if only_ctrl {
            match ev.keystroke.key.as_str() {
                "k" => {
                    let killed = self.kill_to_end();
                    self.copy_text(&killed, cx);
                    cx.notify();
                    return;
                }
                "u" => {
                    let killed = self.kill_line();
                    self.copy_text(&killed, cx);
                    cx.notify();
                    return;
                }
                "w" => {
                    let killed = self.kill_word_back();
                    self.copy_text(&killed, cx);
                    cx.notify();
                    return;
                }
                "a" => {
                    self.cursor = 0;
                    self.clear_selection();
                    cx.notify();
                    return;
                }
                "e" => {
                    self.cursor = self.input.len();
                    self.clear_selection();
                    cx.notify();
                    return;
                }
                _ => {}
            }
        }
        match ev.keystroke.key.as_str() {
            "enter" => {
                self.submit(cx);
                return;
            }
            "escape" => {
                if self.sel_anchor.is_some() {
                    self.clear_selection();
                } else {
                    self.input.clear();
                    self.cursor = 0;
                    self.hist_idx = None;
                }
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
                if mods.shift {
                    self.extend_left();
                } else {
                    self.move_left();
                }
                cx.notify();
                return;
            }
            "right" => {
                if mods.shift {
                    self.extend_right();
                } else {
                    self.move_right();
                }
                cx.notify();
                return;
            }
            "home" => {
                if mods.shift {
                    self.extend_home();
                } else {
                    self.cursor = 0;
                    self.clear_selection();
                }
                cx.notify();
                return;
            }
            "end" => {
                if mods.shift {
                    self.extend_end();
                } else {
                    self.cursor = self.input.len();
                    self.clear_selection();
                }
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

    /// Insert clipboard text at the cursor (single line: newlines become
    /// spaces), replacing any selection.
    fn paste(&mut self, cx: &mut Context<Self>) {
        let text = cx
            .read_from_clipboard()
            .and_then(|item| item.text())
            .unwrap_or_default()
            .replace([char::from(10), char::from(13)], " ");
        if text.is_empty() {
            return;
        }
        self.delete_selection();
        self.input.insert_str(self.cursor, &text);
        self.cursor += text.len();
        self.hist_idx = None;
    }

    fn copy_text(&self, text: &str, cx: &mut Context<Self>) {
        if text.is_empty() {
            return;
        }
        cx.write_to_clipboard(ClipboardItem::new_string(text.to_string()));
    }

    fn copy_selection(&self, cx: &mut Context<Self>) {
        if let Some((lo, hi)) = self.selection() {
            self.copy_text(&self.input[lo..hi], cx);
        }
    }

    fn cut_selection(&mut self, cx: &mut Context<Self>) {
        if let Some((lo, hi)) = self.selection() {
            self.copy_text(&self.input[lo..hi], cx);
            self.input.drain(lo..hi);
            self.cursor = lo;
            self.sel_anchor = None;
            self.hist_idx = None;
        }
    }

    fn extend_left(&mut self) {
        if self.sel_anchor.is_none() {
            self.sel_anchor = Some(self.cursor);
        }
        if self.cursor == 0 {
            return;
        }
        let mut c = self.cursor - 1;
        while c > 0 && !self.input.is_char_boundary(c) {
            c -= 1;
        }
        self.cursor = c;
    }

    fn extend_right(&mut self) {
        if self.sel_anchor.is_none() {
            self.sel_anchor = Some(self.cursor);
        }
        if self.cursor >= self.input.len() {
            return;
        }
        let mut c = self.cursor + 1;
        while c < self.input.len() && !self.input.is_char_boundary(c) {
            c += 1;
        }
        self.cursor = c;
    }

    fn extend_home(&mut self) {
        if self.sel_anchor.is_none() {
            self.sel_anchor = Some(self.cursor);
        }
        self.cursor = 0;
    }

    fn extend_end(&mut self) {
        if self.sel_anchor.is_none() {
            self.sel_anchor = Some(self.cursor);
        }
        self.cursor = self.input.len();
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
        if !self.focused_once {
            self.focused_once = true;
            window.focus(&self.input_focus);
        }
        let mut content = div().flex().flex_row();
        for (text, selected, cursor) in input_runs(&self.input, self.cursor, self.sel_anchor, focused) {
            let mut el = div().child(text);
            if cursor && focused {
                el = el.bg(rgb(0x7AA2F7)).text_color(rgb(0x1A1B26)).rounded_sm();
            } else if selected {
                let hl: Hsla = rgb(0x5B8FF9).into();
                el = el.bg(hl.opacity(0.35)).rounded_sm();
            }
            content = content.child(el);
        }
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
                cx.listener(|view, ev: &MouseDownEvent, window, cx| {
                    window.focus(&view.input_focus_handle());
                    let idx = view.input_hit_index(window, ev.position.x);
                    view.cursor = idx;
                    view.sel_anchor = Some(idx);
                    view.input_drag = true;
                    cx.notify();
                }),
            )
            .on_mouse_move(cx.listener(|view, ev: &MouseMoveEvent, window, cx| {
                if view.input_drag {
                    view.cursor = view.input_hit_index(window, ev.position.x);
                    cx.notify();
                }
            }))
            .on_mouse_up(
                MouseButton::Left,
                cx.listener(|view, _, _, cx| {
                    view.input_drag = false;
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
        let snap = Snapshot::capture(
            &self.model,
            self.camera,
            &self.highlight,
            self.selected,
            self.hover,
            self.mouse,
        );
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
                    view.hover = None;
                    view.input_drag = false;
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
                    return;
                }
                // Hover pick (nodes first, then edges) for tooltips.
                let (lx, ly) = view.to_local(ev.position);
                let (vw, vh) = {
                    let s = view.canvas_size.get();
                    (f32::from(s.width), f32::from(s.height))
                };
                view.mouse = Some((lx, ly));
                let world = view.camera.screen_to_world(lx, ly, vw, vh);
                let radius = 6.0 / view.camera.zoom.max(0.2) as f32;
                view.hover = view
                    .model
                    .pick(world, 6.0)
                    .map(graph::Hover::Node)
                    .or_else(|| view.model.pick_edge(world, radius).map(graph::Hover::Edge));
                cx.notify();
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

/// Render segments for the chat input: (text, is_selected, is_cursor).
/// Pure function so selection rendering is unit-testable.
pub fn input_runs(input: &str, cursor: usize, anchor: Option<usize>, focused: bool) -> Vec<(String, bool, bool)> {
    if !focused {
        return vec![(input.to_string(), false, false)];
    }
    if input.is_empty() {
        return vec![("▍".to_string(), false, false)];
    }
    let mut at = cursor.min(input.len());
    while at > 0 && !input.is_char_boundary(at) {
        at -= 1;
    }
    let mut ac = anchor.unwrap_or(at).min(input.len());
    while ac > 0 && !input.is_char_boundary(ac) {
        ac -= 1;
    }
    let (lo, hi) = if ac <= at { (ac, at) } else { (at, ac) };
    // End offset of the char at the cursor (placeholder when at the end).
    let ce = if at >= input.len() {
        input.len()
    } else {
        let mut c = at + 1;
        while c < input.len() && !input.is_char_boundary(c) {
            c += 1;
        }
        c
    };
    let mut cuts = vec![0, lo, hi, at, ce, input.len()];
    cuts.sort_unstable();
    cuts.dedup();
    let mut runs = Vec::new();
    for w in cuts.windows(2) {
        let (a, b) = (w[0], w[1]);
        if a >= b {
            continue;
        }
        let sel = lo < hi && a >= lo && b <= hi;
        let cur = a == at && b == ce && at < input.len();
        runs.push((input[a..b].to_string(), sel, cur));
    }
    if at >= input.len() {
        runs.push((" ".to_string(), false, true));
    }
    runs
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
