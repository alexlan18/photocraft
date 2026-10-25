//! The Agent panel: a collapsible conversation on the window's far side (Window › Agent). The
//! user asks in words; an agent (`photocraft-automation`'s runner, wired by the desktop app
//! through [`crate::Services::agent_host`]) edits the open document through PhotoCraft's own
//! MCP tools, and the panel shows every tool call, result and screenshot as it happens.
//!
//! The shell owns only the transcript and the input; the conversation with the model lives in
//! the runner thread, reached through [`AgentHandle`]. Settings (model, key, effort) are
//! Preferences › AI Integrations; a session starts with the settings of that moment and Clear
//! starts a new one.

use std::collections::HashMap;

use egui::{Align, CornerRadius, Layout, Rect, RichText, Stroke, TextureHandle, pos2, vec2};
use photocraft_engine::agent::{AgentCommand, AgentEvent, AgentHandle, AgentSettings};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::PhotocraftApp;
use crate::theme::Tokens;
use crate::{icons, widgets};

/// The expanded panel's default and allowed widths (points).
pub const DEFAULT_WIDTH: f32 = 320.0;
pub const MIN_WIDTH: f32 = 240.0;
pub const MAX_WIDTH: f32 = 640.0;
/// The collapsed strip's width.
pub const COLLAPSED_WIDTH: f32 = 28.0;
/// The transcript keeps this many entries; older ones scroll off (the model keeps its own).
const MAX_ENTRIES: usize = 400;
/// A message longer than this is cut in the panel (never in what the model gets).
const MAX_DRAFT_CHARS: usize = 20_000;

/// What the panel remembers between frames and launches (the transcript is per session).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct AgentPanelState {
    /// Only the strip shows; a click expands it.
    pub collapsed: bool,
    /// The message being typed.
    pub draft: String,
    /// The conversation as shown.
    #[serde(skip)]
    pub transcript: Vec<Entry>,
    /// The footer line ("Thinking…", "Running layer_new_layer").
    #[serde(skip)]
    pub status: String,
    /// A turn is running: the input waits, Stop is offered.
    #[serde(skip)]
    pub busy: bool,
    /// Tokens the API reported for this session (input, output).
    #[serde(skip)]
    pub tokens: (u64, u64),
    /// Scroll to the end on the next frame (a new entry arrived).
    #[serde(skip)]
    pub follow: bool,
}

/// One row of the transcript.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "camelCase")]
pub enum Entry {
    User { text: String },
    Assistant { text: String },
    Tool {
        name: String,
        input: String,
        /// `None` while the tool runs.
        result: Option<ToolResultView>,
    },
    /// A problem the user should know about (an API error, a stopped turn).
    Error { text: String },
    Note { text: String },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolResultView {
    pub ok: bool,
    pub summary: String,
    /// A PNG the tool returned, kept for the panel's thumbnail (not serialized).
    #[serde(skip)]
    pub png: Option<Vec<u8>>,
}

/// The live session and the textures of the images it produced: runtime state, never saved.
#[derive(Default)]
pub struct AgentRuntime {
    pub handle: Option<AgentHandle>,
    /// Decoded tool images by transcript index.
    textures: HashMap<usize, Option<TextureHandle>>,
    /// A session's start failed with this message (shown until the next attempt).
    pub start_error: Option<String>,
}

fn rect_id() -> egui::Id {
    egui::Id::new("agent-panel-rect")
}

/// Where the panel was drawn last frame (tests and `ui.inspect`).
pub fn last_rect(ctx: &egui::Context) -> Option<Rect> {
    ctx.data(|d| d.get_temp::<Rect>(rect_id()))
}

/// Whether the desktop app wired an agent (the web build has none).
pub fn available(app: &PhotocraftApp) -> bool {
    app.services.agent_host.is_some()
}

/// `ui.inspect`'s view of the panel.
pub fn inspect(app: &PhotocraftApp) -> Value {
    let s = &app.ui.agent;
    json!({
        "open": app.ui.panels.agent,
        "collapsed": s.collapsed,
        "available": available(app),
        "busy": s.busy,
        "status": s.status,
        "draft": s.draft,
        "tokens": {"input": s.tokens.0, "output": s.tokens.1},
        "session": app.agent.handle.is_some(),
        "transcript": s.transcript,
    })
}

/// Drain the runner's events into the transcript. Called every frame the panel is open, and
/// cheap when nothing arrived.
pub fn poll(app: &mut PhotocraftApp) {
    let Some(handle) = &app.agent.handle else { return };
    let mut events = Vec::new();
    loop {
        match handle.events.try_recv() {
            Ok(e) => events.push(e),
            Err(std::sync::mpsc::TryRecvError::Empty) => break,
            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                events.push(AgentEvent::Error("the agent session ended".into()));
                app.agent.handle = None;
                break;
            }
        }
    }
    for e in events {
        apply(app, e);
    }
}

/// One event into the transcript (also the seam tests drive).
pub fn apply(app: &mut PhotocraftApp, e: AgentEvent) {
    let s = &mut app.ui.agent;
    match e {
        AgentEvent::Status(text) => s.status = text,
        AgentEvent::AssistantText(text) => push(s, Entry::Assistant { text }),
        AgentEvent::ToolCall(call) => push(s, Entry::Tool { name: call.name, input: call.input, result: None }),
        AgentEvent::ToolResult { name, ok, summary, png } => {
            let view = ToolResultView { ok, summary, png };
            match s.transcript.iter_mut().rev().find(|e| matches!(e, Entry::Tool { name: n, result: None, .. } if *n == name)) {
                Some(Entry::Tool { result, .. }) => *result = Some(view),
                _ => push(s, Entry::Tool { name, input: String::new(), result: Some(view) }),
            }
            s.follow = true;
        }
        AgentEvent::TurnDone { input_tokens, output_tokens, stopped } => {
            s.busy = false;
            s.status.clear();
            s.tokens.0 = s.tokens.0.saturating_add(input_tokens);
            s.tokens.1 = s.tokens.1.saturating_add(output_tokens);
            if let Some(why) = stopped.filter(|w| w != "error") {
                push(s, Entry::Note { text: why });
            }
        }
        AgentEvent::Error(text) => {
            s.busy = false;
            s.status.clear();
            push(s, Entry::Error { text });
        }
    }
}

fn push(s: &mut AgentPanelState, e: Entry) {
    s.transcript.push(e);
    s.follow = true;
    if s.transcript.len() > MAX_ENTRIES {
        let extra = s.transcript.len() - MAX_ENTRIES;
        s.transcript.drain(..extra);
    }
}

/// Send the draft (or `text`) to the agent, starting a session if none runs.
pub fn send(app: &mut PhotocraftApp, text: String) -> Result<(), String> {
    let text = text.trim().to_string();
    if text.is_empty() {
        return Err("nothing to send".into());
    }
    if app.ui.agent.busy {
        return Err("the agent is still working; stop it or wait".into());
    }
    if app.agent.handle.is_none() {
        start(app)?;
    }
    let Some(handle) = &app.agent.handle else { return Err("no agent session".into()) };
    if handle.commands.send(AgentCommand::Send(text.clone())).is_err() {
        app.agent.handle = None;
        return Err("the agent session ended; try again".into());
    }
    let s = &mut app.ui.agent;
    s.busy = true;
    s.status = tl!("Thinking…").to_string();
    push(s, Entry::User { text });
    Ok(())
}

/// Start a session with the current Preferences.
fn start(app: &mut PhotocraftApp) -> Result<(), String> {
    let settings = AgentSettings::from_prefs(app.session.prefs());
    let Some(host) = app.services.agent_host.as_mut() else {
        return Err(tl!("The agent needs the desktop app.").to_string());
    };
    match host(settings) {
        Ok(handle) => {
            app.agent.handle = Some(handle);
            app.agent.start_error = None;
            Ok(())
        }
        Err(e) => {
            app.agent.start_error = Some(e.clone());
            Err(e)
        }
    }
}

/// Stop the running turn (the conversation is kept).
pub fn stop(app: &mut PhotocraftApp) {
    if let Some(handle) = &app.agent.handle {
        let _ = handle.commands.send(AgentCommand::Cancel);
        app.ui.agent.status = tl!("Stopping…").to_string();
    }
}

/// Forget the conversation and end the session; the next message starts a new one with the
/// current Preferences.
pub fn clear(app: &mut PhotocraftApp) {
    app.agent.handle = None;
    app.agent.textures.clear();
    app.agent.start_error = None;
    let s = &mut app.ui.agent;
    s.transcript.clear();
    s.status.clear();
    s.busy = false;
    s.tokens = (0, 0);
}

// ----------------------------------------------------------------------------- drawing

/// The side panel, outermost on the dock's side. Draws nothing when Window › Agent is off.
pub fn side_panel(app: &mut PhotocraftApp, ui: &mut egui::Ui) {
    if !app.ui.panels.agent {
        ui.ctx().data_mut(|d| d.remove::<Rect>(rect_id()));
        return;
    }
    poll(app);
    let t = Tokens::get(ui.ctx());
    let collapsed = app.ui.agent.collapsed;
    let mut panel = crate::direction::end_panel("agent-panel").resizable(!collapsed);
    panel = if collapsed { panel.exact_size(COLLAPSED_WIDTH) } else { panel.default_size(DEFAULT_WIDTH).size_range(MIN_WIDTH..=MAX_WIDTH) };
    let margin = if collapsed { egui::Margin::symmetric(2, 6) } else { egui::Margin::same(8) };
    let r = panel.frame(egui::Frame::NONE.fill(t.dock).inner_margin(margin)).show(ui, |ui| {
        let x = if crate::direction::is_rtl() { ui.max_rect().right() + margin.right as f32 } else { ui.max_rect().left() - margin.left as f32 };
        ui.painter().line_segment([pos2(x, ui.max_rect().top() - 8.0), pos2(x, ui.max_rect().bottom() + 8.0)], Stroke::new(1.0, t.separator));
        if collapsed { strip(app, ui, &t) } else { body(app, ui, &t) }
    });
    ui.ctx().data_mut(|d| d.insert_temp(rect_id(), r.response.rect));
}

fn strip(app: &mut PhotocraftApp, ui: &mut egui::Ui, t: &Tokens) {
    ui.vertical_centered(|ui| {
        let on = app.ui.agent.busy;
        if icons::rail_button(ui, "sparkles", 22.0, on, tl!("Agent")).clicked() {
            app.ui.agent.collapsed = false;
        }
        if app.ui.agent.busy {
            ui.add_space(4.0);
            ui.add(egui::Spinner::new().size(12.0).color(t.accent));
        }
    });
}

fn body(app: &mut PhotocraftApp, ui: &mut egui::Ui, t: &Tokens) {
    header(app, ui, t);
    ui.add_space(4.0);
    // The footer is laid out from the bottom edge up, so it always fits whole; the transcript
    // takes what is left above it.
    ui.with_layout(Layout::bottom_up(Align::Min), |ui| {
        footer(app, ui, t);
        ui.add_space(4.0);
        let size = ui.available_size();
        ui.allocate_ui_with_layout(size, Layout::top_down(Align::Min), |ui| transcript(app, ui, t));
    });
}

fn header(app: &mut PhotocraftApp, ui: &mut egui::Ui, t: &Tokens) {
    ui.horizontal(|ui| {
        ui.add(icons::image("sparkles", 14.0, t.accent));
        let title = RichText::new(tl!("Agent")).size(12.5).color(t.text);
        let title = if ui.ctx().fonts(|f| f.families().contains(&egui::FontFamily::Name("medium".into()))) { title.family(egui::FontFamily::Name("medium".into())) } else { title.strong() };
        ui.label(title);
        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
            if icons::rail_button(ui, "x", 22.0, false, tl!("Close")).clicked() {
                app.ui.panels.agent = false;
            }
            if icons::rail_button(ui, "chevrons-right", 22.0, false, tl!("Collapse")).clicked() {
                app.ui.agent.collapsed = true;
            }
            if icons::rail_button(ui, "settings", 22.0, false, tl!("Agent settings (Preferences › AI Integrations)")).clicked() {
                let _ = app.run("edit.preferences.integrations", json!({}));
            }
            let can_clear = !app.ui.agent.transcript.is_empty() || app.agent.handle.is_some();
            if can_clear && icons::rail_button(ui, "trash", 22.0, false, tl!("Clear conversation")).clicked() {
                clear(app);
            }
        });
    });
}

fn transcript(app: &mut PhotocraftApp, ui: &mut egui::Ui, t: &Tokens) {
    let follow = std::mem::take(&mut app.ui.agent.follow);
    let width = ui.available_width();
    egui::ScrollArea::vertical().id_salt("agent-transcript").auto_shrink([false, false]).stick_to_bottom(follow).show(ui, |ui| {
        ui.set_width(width);
        if app.ui.agent.transcript.is_empty() {
            ui.add_space(8.0);
            let hint = if available(app) {
                tl!("Describe an edit, for example “brighten the shadows and add a vignette”. The agent uses PhotoCraft's own commands and shows every step here.")
            } else {
                tl!("The agent needs the desktop app.")
            };
            ui.label(RichText::new(hint).color(t.text_dim).size(12.0));
            if let Some(e) = &app.agent.start_error {
                ui.add_space(6.0);
                ui.label(RichText::new(e).color(t.danger).size(12.0));
            }
        }
        let n = app.ui.agent.transcript.len();
        for i in 0..n {
            entry(app, ui, t, i, width);
            ui.add_space(6.0);
        }
    });
}

fn entry(app: &mut PhotocraftApp, ui: &mut egui::Ui, t: &Tokens, i: usize, width: f32) {
    let Some(e) = app.ui.agent.transcript.get(i).cloned() else { return };
    match e {
        Entry::User { text } => {
            ui.with_layout(Layout::right_to_left(Align::Min), |ui| {
                egui::Frame::NONE.fill(t.accent_soft).corner_radius(CornerRadius::same(t.radius as u8)).inner_margin(egui::Margin::symmetric(8, 6)).show(ui, |ui| {
                    ui.set_max_width(width * 0.85);
                    ui.add(egui::Label::new(RichText::new(text).color(t.text).size(12.5)).wrap());
                });
            });
        }
        Entry::Assistant { text } => {
            ui.add(egui::Label::new(RichText::new(text).color(t.text).size(12.5)).wrap());
        }
        Entry::Error { text } => {
            ui.add(egui::Label::new(RichText::new(text).color(t.danger).size(12.0)).wrap());
        }
        Entry::Note { text } => {
            ui.add(egui::Label::new(RichText::new(text).color(t.text_dim).size(11.5).italics()).wrap());
        }
        Entry::Tool { name, input, result } => {
            let (icon, color) = match &result {
                None => ("ellipsis", t.text_dim),
                Some(r) if r.ok => ("check", t.accent),
                Some(_) => ("triangle-alert", t.danger),
            };
            egui::Frame::NONE.fill(t.card).stroke(Stroke::new(1.0, t.card_border)).corner_radius(CornerRadius::same(t.radius_sm as u8)).inner_margin(egui::Margin::symmetric(6, 4)).show(ui, |ui| {
                ui.set_width(width - 12.0);
                let id = ui.make_persistent_id(("agent-tool", i));
                egui::collapsing_header::CollapsingState::load_with_default_open(ui.ctx(), id, false)
                    .show_header(ui, |ui| {
                        ui.add(icons::image(icon, 12.0, color));
                        ui.label(RichText::new(&name).size(12.0).color(t.text).family(egui::FontFamily::Monospace));
                        if result.is_none() {
                            ui.add(egui::Spinner::new().size(10.0).color(t.accent));
                        }
                    })
                    .body(|ui| {
                        if !input.is_empty() {
                            ui.label(RichText::new(&input).size(11.0).color(t.text_dim).family(egui::FontFamily::Monospace));
                        }
                        if let Some(r) = &result
                            && !r.summary.is_empty()
                        {
                            ui.label(RichText::new(&r.summary).size(11.0).color(if r.ok { t.text } else { t.danger }).family(egui::FontFamily::Monospace));
                        }
                    });
                if let Some(png) = result.as_ref().and_then(|r| r.png.as_deref()) {
                    thumbnail(app, ui, i, png, width - 24.0);
                }
            });
        }
    }
}

/// A tool's PNG, decoded once into a texture and drawn fitting the panel's width.
fn thumbnail(app: &mut PhotocraftApp, ui: &mut egui::Ui, i: usize, png: &[u8], max_w: f32) {
    let ctx = ui.ctx().clone();
    let tex = app.agent.textures.entry(i).or_insert_with(|| decode_png(png).map(|img| ctx.load_texture(format!("agent-img-{i}"), img, egui::TextureOptions::LINEAR)));
    if let Some(tex) = tex {
        let size = tex.size_vec2();
        if size.x > 0.0 && size.y > 0.0 {
            let scale = (max_w / size.x).min(1.0);
            ui.add_space(4.0);
            ui.add(egui::Image::new((tex.id(), size * scale)).corner_radius(CornerRadius::same(3)));
        }
    }
}

fn decode_png(png: &[u8]) -> Option<egui::ColorImage> {
    use photocraft_codecs::{ChannelLayout, SampleType};
    let img = photocraft_codecs::decode(png).ok()?.convert(ChannelLayout::Rgba, SampleType::U8);
    let (w, h) = (usize::try_from(img.width()).ok()?, usize::try_from(img.height()).ok()?);
    let bytes = img.data();
    if w == 0 || h == 0 || bytes.len() != w.checked_mul(h)?.checked_mul(4)? {
        return None;
    }
    Some(egui::ColorImage::from_rgba_unmultiplied([w, h], bytes))
}

/// The input area, drawn bottom-up (the caller's layout): the buttons row, the message box,
/// then the status line above it.
fn footer(app: &mut PhotocraftApp, ui: &mut egui::Ui, t: &Tokens) {
    let enabled = available(app) && !app.ui.agent.busy;
    let mut send_now = false;
    let width = ui.available_width();
    // Buttons row: Send (or Stop) at the trailing edge, the key hint before it.
    ui.allocate_ui_with_layout(vec2(width, 26.0), Layout::right_to_left(Align::Center), |ui| {
        if app.ui.agent.busy {
            if widgets::secondary_button(ui, tl!("Stop"), 64.0).clicked() {
                stop(app);
            }
        } else {
            let can_send = enabled && !app.ui.agent.draft.trim().is_empty();
            if ui.add_enabled_ui(can_send, |ui| widgets::primary_button(ui, tl!("Send"), 64.0)).inner.clicked() {
                send_now = true;
            }
        }
        ui.add_space(4.0);
        ui.add(egui::Label::new(RichText::new(tl!("Enter to send, Shift+Enter for a new line")).size(10.5).color(t.text_faint)).truncate());
    });
    ui.add_space(4.0);
    // The message box.
    let row = ui.text_style_height(&egui::TextStyle::Body);
    let edit_h = row * 3.0 + 12.0;
    let (rect, _) = ui.allocate_exact_size(vec2(width, edit_h), egui::Sense::hover());
    widgets::surface(ui, rect, t.field, false);
    if !t.bevel {
        ui.painter().rect_stroke(rect, t.radius_sm, Stroke::new(1.0, t.field_border), egui::StrokeKind::Inside);
    }
    if app.ui.agent.draft.chars().count() > MAX_DRAFT_CHARS {
        app.ui.agent.draft = app.ui.agent.draft.chars().take(MAX_DRAFT_CHARS).collect();
    }
    let mut child = ui.new_child(egui::UiBuilder::new().max_rect(rect.shrink2(vec2(6.0, 4.0))).layout(Layout::top_down(Align::Min)));
    egui::ScrollArea::vertical().id_salt("agent-draft-scroll").auto_shrink([false, false]).show(&mut child, |ui| {
        let edit = egui::TextEdit::multiline(&mut app.ui.agent.draft)
            .id_salt("agent-draft")
            .frame(egui::Frame::NONE)
            .desired_width(f32::INFINITY)
            .desired_rows(3)
            .font(egui::FontId::proportional(12.5))
            .text_color(t.text)
            .hint_text(RichText::new(tl!("Ask the agent to edit the image…")).color(t.text_faint))
            .return_key(egui::KeyboardShortcut::new(egui::Modifiers::SHIFT, egui::Key::Enter));
        let resp = ui.add_enabled(enabled, edit);
        if resp.has_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter) && !i.modifiers.shift) {
            send_now = true;
        }
    });
    ui.add_space(4.0);
    // Status line: what the agent is doing, else the session's token count.
    let status = app.ui.agent.status.clone();
    let label = if status.is_empty() {
        let (i, o) = app.ui.agent.tokens;
        if i + o > 0 { tl!("Tokens: {in} in, {out} out").replace("{in}", &i.to_string()).replace("{out}", &o.to_string()) } else { String::new() }
    } else {
        status
    };
    ui.allocate_ui_with_layout(vec2(width, 16.0), Layout::left_to_right(Align::Center), |ui| {
        if app.ui.agent.busy {
            ui.add(egui::Spinner::new().size(11.0).color(t.accent));
        }
        ui.add(egui::Label::new(RichText::new(label).size(11.0).color(t.text_dim)).truncate());
    });
    if send_now && enabled {
        let text = std::mem::take(&mut app.ui.agent.draft);
        if let Err(e) = send(app, text.clone()) {
            app.ui.agent.draft = text;
            push(&mut app.ui.agent, Entry::Error { text: e });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Services;
    use photocraft_engine::agent::ToolCallEvent;

    fn app() -> PhotocraftApp {
        PhotocraftApp::new(photocraft_engine::Session::new(), Services::default())
    }

    #[test]
    fn events_build_the_transcript_and_results_find_their_call() {
        let mut app = app();
        apply(&mut app, AgentEvent::Status("Thinking…".into()));
        apply(&mut app, AgentEvent::ToolCall(ToolCallEvent { name: "layer_new_layer".into(), input: "{}".into() }));
        apply(&mut app, AgentEvent::ToolCall(ToolCallEvent { name: "ui_screenshot".into(), input: "{}".into() }));
        apply(&mut app, AgentEvent::ToolResult { name: "ui_screenshot".into(), ok: true, summary: "(image)".into(), png: Some(vec![1, 2, 3]) });
        apply(&mut app, AgentEvent::ToolResult { name: "layer_new_layer".into(), ok: false, summary: "no document".into(), png: None });
        apply(&mut app, AgentEvent::AssistantText("Done.".into()));
        apply(&mut app, AgentEvent::TurnDone { input_tokens: 10, output_tokens: 2, stopped: Some("stopped".into()) });
        let s = &app.ui.agent;
        assert_eq!(s.transcript.len(), 4, "{:?}", s.transcript);
        assert!(matches!(&s.transcript[0], Entry::Tool { name, result: Some(r), .. } if name == "layer_new_layer" && !r.ok));
        assert!(matches!(&s.transcript[1], Entry::Tool { name, result: Some(r), .. } if name == "ui_screenshot" && r.png.is_some()));
        assert!(matches!(&s.transcript[3], Entry::Note { text } if text == "stopped"));
        assert_eq!(s.tokens, (10, 2));
        assert!(!s.busy && s.status.is_empty());
        let v = inspect(&app);
        assert_eq!(v["transcript"].as_array().unwrap().len(), 4);
        assert_eq!(v["available"], false);
        assert!(v["transcript"][1]["result"].get("png").is_none(), "PNG bytes stay out of ui.inspect");
    }

    #[test]
    fn sending_without_a_host_is_an_error_not_a_crash() {
        let mut app = app();
        assert!(send(&mut app, "  ".into()).unwrap_err().contains("nothing"));
        let e = send(&mut app, "hi".into()).unwrap_err();
        assert!(e.contains("desktop"), "{e}");
        assert!(app.ui.agent.transcript.is_empty());
    }

    #[test]
    fn a_host_receives_the_message_and_clear_ends_the_session() {
        let mut app = app();
        let (tx, rx) = std::sync::mpsc::channel::<AgentCommand>();
        let (_etx, erx) = std::sync::mpsc::channel::<AgentEvent>();
        let slots = std::sync::Arc::new(std::sync::Mutex::new(vec![(tx, erx)]));
        let s2 = slots.clone();
        app.services.agent_host = Some(Box::new(move |settings: AgentSettings| {
            assert_eq!(settings.model, photocraft_engine::agent::DEFAULT_MODEL);
            s2.lock().unwrap().pop().map(|(commands, events)| AgentHandle { commands, events }).ok_or_else(|| "no more sessions".to_string())
        }));
        send(&mut app, "blur it".into()).unwrap();
        assert!(app.ui.agent.busy);
        assert!(matches!(rx.try_recv(), Ok(AgentCommand::Send(t)) if t == "blur it"));
        assert!(send(&mut app, "again".into()).unwrap_err().contains("still working"));
        stop(&mut app);
        assert!(matches!(rx.try_recv(), Ok(AgentCommand::Cancel)));
        clear(&mut app);
        assert!(app.agent.handle.is_none() && app.ui.agent.transcript.is_empty() && !app.ui.agent.busy);
        let e = send(&mut app, "new".into()).unwrap_err();
        assert!(e.contains("no more sessions"), "{e}");
        assert_eq!(app.agent.start_error.as_deref(), Some("no more sessions"));
    }

    #[test]
    fn the_transcript_is_bounded() {
        let mut app = app();
        for i in 0..(MAX_ENTRIES + 20) {
            apply(&mut app, AgentEvent::AssistantText(format!("m{i}")));
        }
        assert_eq!(app.ui.agent.transcript.len(), MAX_ENTRIES);
        assert!(matches!(&app.ui.agent.transcript[0], Entry::Assistant { text } if text == "m20"));
    }

    #[test]
    fn a_disconnected_session_reports_once_and_drops_the_handle() {
        let mut app = app();
        let (tx, _rx) = std::sync::mpsc::channel::<AgentCommand>();
        let (etx, erx) = std::sync::mpsc::channel::<AgentEvent>();
        app.agent.handle = Some(AgentHandle { commands: tx, events: erx });
        app.ui.agent.busy = true;
        drop(etx);
        poll(&mut app);
        assert!(app.agent.handle.is_none());
        assert!(matches!(app.ui.agent.transcript.last(), Some(Entry::Error { .. })));
        assert!(!app.ui.agent.busy);
        poll(&mut app);
        assert_eq!(app.ui.agent.transcript.len(), 1);
    }

    #[test]
    fn png_decoding_rejects_garbage() {
        assert!(decode_png(&[0, 1, 2]).is_none());
    }

    #[test]
    fn the_control_channel_opens_and_inspects_the_panel() {
        use crate::control::{ControlRequest, Outcome, handle};
        let mut app = app();
        let ctx = egui::Context::default();
        let call = |app: &mut PhotocraftApp, method: &str, params: Value| -> Value {
            let (req, _rx) = ControlRequest::new(method, params);
            match handle(app, &ctx, &req) {
                Outcome::Done(v) => v,
                _ => panic!("{method}: expected an immediate reply"),
            }
        };
        assert_eq!(call(&mut app, "ui.inspect", json!({}))["result"]["agent"]["open"], false);
        assert_eq!(call(&mut app, "ui.set", json!({"agentPanel": true}))["ok"], true);
        assert!(app.ui.panels.agent);
        let v = call(&mut app, "ui.inspect", json!({}));
        assert_eq!(v["result"]["agent"]["open"], true, "{v}");
        assert_eq!(v["result"]["agent"]["session"], false);
        assert_eq!(call(&mut app, "ui.set", json!({"agentPanel": "yes"}))["ok"], false);
        // The Window menu toggle and its check mark.
        assert_eq!(call(&mut app, "ui.menu.invoke", json!({"id": "window.toggle.agent"}))["ok"], true);
        assert!(!app.ui.panels.agent);
        let items = call(&mut app, "ui.menu.list", json!({}));
        let agent = items["result"].as_array().into_iter().flatten().find(|i| i["id"] == "window.toggle.agent").cloned().unwrap_or_else(|| panic!("no Window › Agent item: {items}"));
        assert_eq!(agent["checked"], false);
        assert_eq!(agent["label"], "Agent");
    }

    /// The panel is drawn on the dock's outer side; collapsed it is a narrow strip.
    #[test]
    fn the_panel_is_drawn_outermost_and_collapses_to_a_strip() {
        use egui_kittest::Harness;
        let mut app = app();
        app.ui.panels.agent = true;
        app.ui.agent.transcript.push(Entry::Assistant { text: "Hello".into() });
        app.ui.agent.transcript.push(Entry::Tool { name: "layer_new_layer".into(), input: "{}".into(), result: Some(ToolResultView { ok: true, summary: "ok".into(), png: None }) });
        let mut h = Harness::builder().with_size(egui::vec2(1200.0, 700.0)).build_ui_state(
            |ui, app: &mut PhotocraftApp| {
                let ctx = ui.ctx().clone();
                if !ctx.fonts(|f| f.families().contains(&egui::FontFamily::Name("medium".into()))) {
                    return;
                }
                crate::panels::right_dock(app, ui);
                egui::CentralPanel::default().show(ui, |_| {});
            },
            app,
        );
        PhotocraftApp::setup_context(&h.ctx, crate::theme::ThemeKind::Pro);
        h.state_mut().ui.theme = crate::theme::ThemeKind::Pro;
        h.run_steps(4);
        let rect = last_rect(&h.ctx).expect("agent panel drawn");
        assert!(rect.right() >= 1200.0 - 16.0, "outermost on the right: {rect:?}");
        assert!(rect.width() >= MIN_WIDTH - 1.0 && rect.width() <= MAX_WIDTH + 1.0, "{rect:?}");
        let dock_right = crate::dock::last_rects(&h.ctx).iter().map(|(_, r)| r.right()).fold(0.0_f32, f32::max);
        assert!(dock_right <= rect.left() + 1.0, "dock {dock_right} is inside the agent panel {rect:?}");

        h.state_mut().ui.agent.collapsed = true;
        h.run_steps(3);
        let rect = last_rect(&h.ctx).expect("strip drawn");
        assert!(rect.width() <= COLLAPSED_WIDTH + 1.0, "{rect:?}");

        h.state_mut().ui.panels.agent = false;
        h.run_steps(3);
        assert!(last_rect(&h.ctx).is_none());
    }
}
