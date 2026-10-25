//! The agent loop: a thread that holds the conversation, asks the model for the next step and
//! runs the tool calls it makes against PhotoCraft's own MCP server (served in process over a
//! duplex pipe, so the agent sees exactly what an external MCP client sees).
//!
//! The 940-odd per-command tools are sent with `defer_loading`, behind the API's tool search:
//! the model keeps the two dozen built-in tools in context and searches for a command tool when
//! it needs one (`layer_new`, `blur`, `levels|curves`), so a turn costs a few thousand tokens of
//! tool definitions rather than a hundred thousand.

use std::sync::Arc;
use std::sync::mpsc::{Receiver, Sender, TryRecvError, channel};

use base64::Engine as _;
use photocraft_engine::agent::{AgentCommand, AgentEvent, AgentHandle, AgentSettings, ToolCallEvent};
use rmcp::model::{CallToolRequestParams, CallToolResult, ClientConfig, Tool};
use rmcp::service::RunningService;
use rmcp::{ClientHandler, RoleClient, ServiceExt};
use serde_json::{Map, Value, json};

use super::llm::{self, Client, Transport};
use crate::PhotocraftMcp;
use crate::budgets::MAX_PNG_BYTES;

/// What the model is told about its job. The tool catalog comes separately, as tools.
pub const SYSTEM_PROMPT: &str = "You are the Agent panel of PhotoCraft, an image editor, working on the documents the user has open in the app beside you. \
You act only through the tools; the app applies each tool call to the live document and the user watches it happen.\n\
\n\
Tools: the built-in tools (session_list, doc_inspect, doc_new, doc_open, doc_save, command_list, command_run, command_batch, jobs_list, ui_inspect, ui_screenshot, ui_set, ui_menu_invoke, control_call) are always loaded. \
Every engine command is also its own tool, named after the command id with underscores (filter_blur_gaussianBlur runs filter.blur.gaussianBlur, layer_new_layer runs layer.new.layer); there are about a thousand of them, so they load on demand: search for them with tool_search_tool_regex, by feature words or id fragments \
(for example \"blur\", \"layer_new\", \"levels|curves\", \"select\", \"crop|canvasSize|imageSize\", \"type_|text\", \"adjust\"), or discover ids and parameter docs with command_list {filter}. \
Each tool's description documents its parameters as {name: type = default} with ? marking optional ones; pass them as the tool's arguments.\n\
\n\
Way of working: look before you edit (doc_inspect for the layer tree, selection and active layer; ui_screenshot {max_side: 1024} to see the window, render_preview {max_side: 1024} for the flattened image). \
Each command is one undoable history step: apply an edit once, then verify with a screenshot or preview rather than repeating it. When a tool reports an error, read the message and adjust the parameters; command_list tells you why a command is disabled. \
Prefer non-destructive edits (adjustment layers, masks, smart filters) when the user doesn't say otherwise. Files can only be opened or saved through tools when the app was launched with automation read/write roots; otherwise ask the user to use File › Open or File › Save.\n\
\n\
Reply in the user's language, briefly: say what you changed and anything they should check. Ask before destructive or ambiguous operations (flattening, cropping, closing documents).";

/// The tools paragraph for an OpenAI-compatible model, which has no server-side tool search: it
/// gets the built-in tools only and reaches every command through `command_run`.
const OPENAI_TOOLS: &str = "Tools: the built-in tools (session_list, doc_inspect, doc_new, doc_open, doc_save, command_list, command_run, command_batch, jobs_list, ui_inspect, ui_screenshot, ui_set, ui_menu_invoke, control_call). \
Every edit is an engine command run with command_run {id, params}; there are about a thousand. Find ids and their parameter docs with command_list {filter} (for example \"blur\", \"layer.new\", \"levels\", \"select\", \"crop\", \"type\", \"adjust\"); \
parameters are documented as {name: type = default} with ? marking optional ones.\n";

/// The system prompt for this provider: the Claude one names the tool search, the
/// OpenAI-compatible one `command_list` + `command_run`.
pub fn system_prompt(settings: &AgentSettings) -> String {
    if !settings.is_openai() {
        return SYSTEM_PROMPT.to_string();
    }
    let start = SYSTEM_PROMPT.find("Tools:").unwrap_or(0);
    let end = SYSTEM_PROMPT.find("\n\nWay of working").map_or(start, |e| e + 1);
    format!("{}{}{}", &SYSTEM_PROMPT[..start], OPENAI_TOOLS, &SYSTEM_PROMPT[end..])
}

/// Tests the model with these settings on a background thread (Preferences › Test Model): the
/// receiver gets one line to show, or why the request failed.
pub fn test_model(settings: AgentSettings) -> Receiver<Result<String, String>> {
    test_model_with(settings, Box::new(llm::Https::new()))
}

/// [`test_model`] with the transport supplied.
pub fn test_model_with(settings: AgentSettings, transport: Box<dyn Transport>) -> Receiver<Result<String, String>> {
    let (tx, rx) = channel();
    let spawned = std::thread::Builder::new().name("photocraft-agent-test".into()).spawn({
        let tx = tx.clone();
        move || {
            let r = settings.check().and_then(|()| Client::new(settings, transport).test());
            let _ = tx.send(r);
        }
    });
    if let Err(e) = spawned {
        let _ = tx.send(Err(format!("could not start the test: {e}")));
    }
    rx
}

/// Tool results longer than this are cut for the model (with a note), so one verbose reply
/// can't fill the context.
const MAX_RESULT_CHARS: usize = 48_000;
/// The panel's copy of a tool result or call.
const MAX_SUMMARY_CHARS: usize = 1_200;
const MAX_INPUT_CHARS: usize = 4_000;

#[derive(Clone, Default)]
struct McpClient;

impl ClientHandler for McpClient {
    fn get_info(&self) -> ClientConfig {
        ClientConfig::default()
    }
}

type Mcp = RunningService<RoleClient, McpClient>;

/// Start a session: the model is reached over HTTPS, the tools over `server` (an in-process
/// MCP server over whichever backend the caller built). `wake` is called after every event, so
/// a UI that sleeps between frames redraws.
pub fn spawn(settings: AgentSettings, server: PhotocraftMcp, wake: Arc<dyn Fn() + Send + Sync>) -> Result<AgentHandle, String> {
    spawn_with(settings, server, Box::new(llm::Https::new()), wake)
}

/// [`spawn`] with the model transport supplied (tests script it).
pub fn spawn_with(settings: AgentSettings, server: PhotocraftMcp, transport: Box<dyn Transport>, wake: Arc<dyn Fn() + Send + Sync>) -> Result<AgentHandle, String> {
    settings.check()?;
    let (command_tx, command_rx) = channel::<AgentCommand>();
    let (event_tx, event_rx) = channel::<AgentEvent>();
    let max_turns = settings.max_turns;
    let client = Client::new(settings, transport);
    std::thread::Builder::new()
        .name("photocraft-agent".into())
        .spawn(move || {
            let system = system_prompt(client.settings());
            let mut runner = Runner { client, events: event_tx, commands: command_rx, wake, max_turns, system, messages: Vec::new(), tools: Vec::new() };
            runner.run(server);
        })
        .map_err(|e| format!("could not start the agent thread: {e}"))?;
    Ok(AgentHandle { commands: command_tx, events: event_rx })
}

struct Runner {
    client: Client,
    events: Sender<AgentEvent>,
    commands: Receiver<AgentCommand>,
    wake: Arc<dyn Fn() + Send + Sync>,
    max_turns: u32,
    /// The system prompt for the provider.
    system: String,
    /// The conversation, append-only, in the Messages API's shape (translated per request for
    /// OpenAI-compatible APIs).
    messages: Vec<Value>,
    /// The `tools` array of every request.
    tools: Vec<Value>,
}

/// What a tool call produced, for the model and for the panel.
struct ToolOutcome {
    result: Value,
    ok: bool,
    summary: String,
    png: Option<Vec<u8>>,
}

impl Runner {
    fn emit(&self, e: AgentEvent) {
        let _ = self.events.send(e);
        (self.wake)();
    }

    fn run(&mut self, server: PhotocraftMcp) {
        let rt = match tokio::runtime::Builder::new_multi_thread().worker_threads(2).enable_all().build() {
            Ok(rt) => rt,
            Err(e) => return self.emit(AgentEvent::Error(format!("could not start the agent runtime: {e}"))),
        };
        let classify = server.clone();
        let mcp = match rt.block_on(connect(server)) {
            Ok(m) => m,
            Err(e) => return self.emit(AgentEvent::Error(e)),
        };
        let listed = match rt.block_on(mcp.list_all_tools()) {
            Ok(t) => t,
            Err(e) => return self.emit(AgentEvent::Error(format!("could not list the app's tools: {e}"))),
        };
        self.tools = api_tools(&listed, |name| classify.is_command_tool(name));
        let ready = if self.client.settings().is_openai() {
            format!("Ready: {} tools ({} model)", listed.iter().filter(|t| !classify.is_command_tool(&t.name)).count(), self.client.settings().model)
        } else {
            format!("Ready: {} tools", listed.len())
        };
        self.emit(AgentEvent::Status(ready));
        let mut queued: Vec<String> = Vec::new();
        loop {
            let cmd = match queued.pop() {
                Some(text) => AgentCommand::Send(text),
                None => match self.commands.recv() {
                    Ok(c) => c,
                    Err(_) => break,
                },
            };
            match cmd {
                AgentCommand::Send(text) => self.turn(&rt, &mcp, text, &mut queued),
                AgentCommand::Cancel => {}
            }
        }
        let _ = rt.block_on(mcp.cancel());
    }

    /// Drain pending commands between steps: a Cancel stops this turn, later messages wait.
    fn cancelled(&self, queued: &mut Vec<String>) -> bool {
        let mut cancel = false;
        loop {
            match self.commands.try_recv() {
                Ok(AgentCommand::Cancel) => cancel = true,
                Ok(AgentCommand::Send(text)) => queued.insert(0, text),
                Err(TryRecvError::Empty) | Err(TryRecvError::Disconnected) => return cancel,
            }
        }
    }

    fn push_user(&mut self, blocks: Vec<Value>) {
        // Two user messages in a row (a turn stopped after its tool results) become one.
        if let Some(last) = self.messages.last_mut()
            && last.get("role").and_then(Value::as_str) == Some("user")
            && let Some(content) = last.get_mut("content").and_then(Value::as_array_mut)
        {
            content.extend(blocks);
            return;
        }
        self.messages.push(json!({"role": "user", "content": blocks}));
    }

    fn turn(&mut self, rt: &tokio::runtime::Runtime, mcp: &Mcp, text: String, queued: &mut Vec<String>) {
        let start = self.messages.len();
        self.push_user(vec![json!({"type": "text", "text": text})]);
        let (mut input_tokens, mut output_tokens) = (0u64, 0u64);
        let mut rounds = 0u32;
        let stopped = loop {
            if self.cancelled(queued) {
                break Some("stopped".to_string());
            }
            if rounds >= self.max_turns {
                break Some(format!("stopped after {} rounds of tool calls; send \"continue\" to go on", self.max_turns));
            }
            rounds += 1;
            self.emit(AgentEvent::Status("Thinking…".into()));
            // Text streams into the panel as it arrives; a Stop is noticed on every event, and a
            // message typed meanwhile waits for its turn.
            let mut stop = false;
            let mut typed: Vec<String> = Vec::new();
            let streamed = self.client.complete_stream(&self.system, &self.tools, &self.messages, &mut |text, new_block| {
                if !text.is_empty() {
                    let _ = self.events.send(AgentEvent::AssistantDelta { text: text.to_string(), new_block });
                    (self.wake)();
                }
                while let Ok(c) = self.commands.try_recv() {
                    match c {
                        AgentCommand::Cancel => stop = true,
                        AgentCommand::Send(t) => typed.push(t),
                    }
                }
                !stop
            });
            for t in typed {
                queued.insert(0, t);
            }
            let reply = match streamed {
                Ok(r) => r,
                // The half-written reply is dropped; what the tools did so far stays.
                Err(e) if e == llm::STOPPED => break Some("stopped".to_string()),
                Err(e) => {
                    // The failed exchange leaves no trace: the user can resend.
                    self.messages.truncate(start);
                    self.emit(AgentEvent::Error(e));
                    self.emit(AgentEvent::TurnDone { input_tokens, output_tokens, stopped: Some("error".into()) });
                    return;
                }
            };
            input_tokens = input_tokens.saturating_add(reply.input_tokens);
            output_tokens = output_tokens.saturating_add(reply.output_tokens);
            // Text has already streamed into the panel; tool searches show in the status line.
            for block in reply.content.iter().filter(|b| b.get("type").and_then(Value::as_str) == Some("server_tool_use")) {
                let what = block.pointer("/input/pattern").or_else(|| block.pointer("/input/query")).and_then(Value::as_str).unwrap_or("");
                self.emit(AgentEvent::Status(format!("Searching tools: {}", bounded(what, 80))));
            }
            // The reply goes back verbatim (thinking, search results and all).
            self.messages.push(json!({"role": "assistant", "content": reply.content}));
            match reply.stop_reason.as_str() {
                "tool_use" => {
                    let calls: Vec<Value> = reply.content.iter().filter(|b| b.get("type").and_then(Value::as_str) == Some("tool_use")).cloned().collect();
                    if calls.is_empty() {
                        break Some("the model asked for a tool call but made none".into());
                    }
                    let mut results = Vec::with_capacity(calls.len());
                    let mut cancelled = false;
                    for call in &calls {
                        let id = call.get("id").and_then(Value::as_str).unwrap_or("").to_string();
                        let name = call.get("name").and_then(Value::as_str).unwrap_or("").to_string();
                        let input = call.get("input").and_then(Value::as_object).cloned().unwrap_or_default();
                        if cancelled || self.cancelled(queued) {
                            cancelled = true;
                            results.push(json!({"type": "tool_result", "tool_use_id": id, "is_error": true, "content": "cancelled by the user"}));
                            continue;
                        }
                        self.emit(AgentEvent::ToolCall(ToolCallEvent { name: name.clone(), input: bounded(&Value::Object(input.clone()).to_string(), MAX_INPUT_CHARS) }));
                        self.emit(AgentEvent::Status(format!("Running {name}")));
                        let outcome = call_tool(rt, mcp, &name, input);
                        self.emit(AgentEvent::ToolResult { name, ok: outcome.ok, summary: outcome.summary, png: outcome.png });
                        results.push(json!({"type": "tool_result", "tool_use_id": id, "is_error": !outcome.ok, "content": outcome.result}));
                    }
                    self.push_user(results);
                    if cancelled {
                        break Some("stopped".to_string());
                    }
                }
                "end_turn" => break None,
                "pause_turn" => {}
                "max_tokens" => break Some("the reply hit the output limit".into()),
                "refusal" => {
                    let category = reply.stop_details.as_ref().and_then(|d| d.get("category")).and_then(Value::as_str).unwrap_or("policy");
                    break Some(format!("the model declined this request ({category})"));
                }
                other => break Some(format!("stop reason {other}")),
            }
        };
        self.emit(AgentEvent::TurnDone { input_tokens, output_tokens, stopped });
    }
}

async fn connect(server: PhotocraftMcp) -> Result<Mcp, String> {
    let (s, c) = tokio::io::duplex(1 << 20);
    tokio::spawn(async move {
        if let Ok(running) = server.serve(s).await {
            let _ = running.waiting().await;
        }
    });
    McpClient.serve(c).await.map_err(|e| format!("could not start the app's MCP server: {e}"))
}

/// The API's `tools` array: the tool search tool, the built-in tools loaded up front, and every
/// command tool deferred behind the search.
pub fn api_tools(listed: &[Tool], is_command: impl Fn(&str) -> bool) -> Vec<Value> {
    let mut tools = Vec::with_capacity(listed.len() + 1);
    tools.push(json!({"type": "tool_search_tool_regex_20251119", "name": "tool_search_tool_regex"}));
    let (deferred, loaded): (Vec<&Tool>, Vec<&Tool>) = listed.iter().partition(|t| is_command(&t.name));
    for t in loaded.into_iter().chain(deferred.iter().copied()) {
        let mut v = json!({
            "name": t.name,
            "description": t.description.as_deref().unwrap_or(""),
            "input_schema": Value::Object((*t.input_schema).clone()),
        });
        if is_command(&t.name) {
            v["defer_loading"] = json!(true);
        }
        tools.push(v);
    }
    tools
}

fn call_tool(rt: &tokio::runtime::Runtime, mcp: &Mcp, name: &str, input: Map<String, Value>) -> ToolOutcome {
    let params = CallToolRequestParams::new(name.to_string()).with_arguments(input);
    match rt.block_on(mcp.call_tool(params)) {
        Ok(result) => tool_outcome(&result),
        Err(e) => {
            let text = format!("tool call failed: {e}");
            ToolOutcome { result: json!(text), ok: false, summary: bounded(&text, MAX_SUMMARY_CHARS), png: None }
        }
    }
}

/// The MCP result as the API's `tool_result` content: text blocks (bounded) and images
/// (base64 PNG, which the model can look at).
fn tool_outcome(result: &CallToolResult) -> ToolOutcome {
    let ok = result.is_error != Some(true);
    let mut blocks = Vec::new();
    let mut texts = Vec::new();
    let mut png = None;
    for c in &result.content {
        if let Some(t) = c.as_text() {
            texts.push(t.text.as_str());
            blocks.push(json!({"type": "text", "text": bounded(&t.text, MAX_RESULT_CHARS)}));
        } else if let Some(img) = c.as_image() {
            if img.mime_type == "image/png" && png.is_none() {
                png = base64::engine::general_purpose::STANDARD.decode(&img.data).ok().filter(|b| b.len() <= MAX_PNG_BYTES);
            }
            blocks.push(json!({"type": "image", "source": {"type": "base64", "media_type": img.mime_type, "data": img.data}}));
        }
    }
    if blocks.is_empty() {
        blocks.push(json!({"type": "text", "text": if ok { "ok" } else { "error" }}));
    }
    let joined = texts.join("\n");
    let summary = if joined.trim().is_empty() && png.is_some() { "(image)".to_string() } else { bounded(&joined, MAX_SUMMARY_CHARS) };
    ToolOutcome { result: Value::Array(blocks), ok, summary, png }
}

/// `s` cut to `max` characters (at a char boundary), with a note when it was cut.
pub fn bounded(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        return s.to_string();
    }
    let kept: String = s.chars().take(max).collect();
    format!("{kept}… [cut: {} more characters]", s.chars().count().saturating_sub(max))
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::{Duration, Instant};

    use super::*;
    use crate::agent::llm::tests::{Scripted, text_reply, tool_reply};

    fn settings() -> AgentSettings {
        AgentSettings { api_key: "test".into(), max_turns: 3, ..Default::default() }
    }

    fn drain(handle: &AgentHandle, until: impl Fn(&AgentEvent) -> bool) -> Vec<AgentEvent> {
        let mut out = Vec::new();
        let deadline = Instant::now() + Duration::from_secs(60);
        while Instant::now() < deadline {
            match handle.events.recv_timeout(Duration::from_millis(200)) {
                Ok(e) => {
                    let done = until(&e);
                    out.push(e);
                    if done {
                        return out;
                    }
                }
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {}
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => return out,
            }
        }
        out
    }

    fn is_done(e: &AgentEvent) -> bool {
        matches!(e, AgentEvent::TurnDone { .. })
    }

    #[test]
    fn a_turn_runs_builtin_and_command_tools_against_the_headless_app() {
        let headless = crate::Headless::new();
        let session = std::sync::Arc::new(std::sync::Mutex::new(headless));
        let server = PhotocraftMcp::with_backend(crate::Backend::Headless(session.clone()));
        let (transport, requests) = Scripted::new(vec![
            Ok((200, tool_reply("t1", "doc_new", json!({"width": 32, "height": 16})))),
            Ok((200, tool_reply("t2", "layer_new_layer", json!({"name": "Agent layer"})))),
            Ok((200, text_reply("Done: a new document with a layer."))),
        ]);
        let wakes = Arc::new(AtomicUsize::new(0));
        let w = wakes.clone();
        let handle = spawn_with(settings(), server, Box::new(transport), Arc::new(move || {
            w.fetch_add(1, Ordering::Relaxed);
        }))
        .unwrap();
        handle.commands.send(AgentCommand::Send("make a document with a layer".into())).unwrap();
        let events = drain(&handle, is_done);
        let calls: Vec<&str> = events.iter().filter_map(|e| if let AgentEvent::ToolCall(c) = e { Some(c.name.as_str()) } else { None }).collect();
        assert_eq!(calls, ["doc_new", "layer_new_layer"], "{events:?}");
        assert!(events.iter().any(|e| matches!(e, AgentEvent::ToolResult { name, ok: true, .. } if name == "layer_new_layer")), "{events:?}");
        assert!(events.iter().any(|e| matches!(e, AgentEvent::AssistantDelta { text, .. } if text.starts_with("Done"))));
        assert!(matches!(events.last(), Some(AgentEvent::TurnDone { stopped: None, input_tokens: 30, output_tokens: 15 })), "{:?}", events.last());
        // Each event is sent, then the UI woken: the last wake may still be on its way.
        assert!(wakes.load(Ordering::Relaxed) + 1 >= events.len());
        // The app really changed.
        let names = session.lock().unwrap().inspect(None).unwrap()["layers"].as_array().unwrap().iter().filter_map(|l| l["name"].as_str().map(str::to_string)).collect::<Vec<_>>();
        assert!(names.iter().any(|n| n == "Agent layer"), "{names:?}");
        // Every request carried the tool search tool, the built-ins loaded and the commands deferred.
        let reqs = requests.lock().unwrap();
        assert_eq!(reqs.len(), 3);
        let tools = reqs[0]["tools"].as_array().unwrap();
        assert_eq!(tools[0]["type"], "tool_search_tool_regex_20251119");
        let deferred = tools.iter().filter(|t| t["defer_loading"] == json!(true)).count();
        let loaded = tools.iter().filter(|t| t.get("defer_loading").is_none() && t.get("type").is_none()).count();
        assert!(deferred > 800 && loaded >= 20, "deferred {deferred} loaded {loaded}");
        assert!(tools.iter().any(|t| t["name"] == "filter_blur_gaussianBlur" && t["defer_loading"] == json!(true)));
        // History is append-only and alternates: user, assistant, user(results), assistant, user(results), assistant.
        let roles: Vec<&str> = reqs[2]["messages"].as_array().unwrap().iter().map(|m| m["role"].as_str().unwrap()).collect();
        assert_eq!(roles, ["user", "assistant", "user", "assistant", "user"]);
        assert_eq!(reqs[2]["messages"][2]["content"][0]["type"], "tool_result");
        assert_eq!(reqs[2]["messages"][2]["content"][0]["tool_use_id"], "t1");
    }

    #[test]
    fn tool_errors_reach_the_model_as_error_results_and_the_turn_limit_stops_the_loop() {
        let server = PhotocraftMcp::headless();
        let (transport, requests) = Scripted::new(vec![
            Ok((200, tool_reply("t1", "layer_new_layer", json!({})))),
            Ok((200, tool_reply("t2", "no_such_tool", json!({})))),
            Ok((200, tool_reply("t3", "command_list", json!({"filter": "blur"})))),
            Ok((200, text_reply("unreachable"))),
        ]);
        let handle = spawn_with(settings(), server, Box::new(transport), Arc::new(|| {})).unwrap();
        handle.commands.send(AgentCommand::Send("go".into())).unwrap();
        let events = drain(&handle, is_done);
        let results: Vec<(String, bool)> = events.iter().filter_map(|e| if let AgentEvent::ToolResult { name, ok, .. } = e { Some((name.clone(), *ok)) } else { None }).collect();
        assert_eq!(results, [("layer_new_layer".to_string(), false), ("no_such_tool".to_string(), false), ("command_list".to_string(), true)], "{events:?}");
        assert!(matches!(events.last(), Some(AgentEvent::TurnDone { stopped: Some(s), .. }) if s.contains("3 rounds")), "{:?}", events.last());
        let reqs = requests.lock().unwrap();
        assert_eq!(reqs.len(), 3, "the fourth reply is never requested");
        assert_eq!(reqs[1]["messages"][2]["content"][0]["is_error"], true);
        let text = reqs[1]["messages"][2]["content"][0]["content"][0]["text"].as_str().unwrap();
        assert!(text.contains("no document"), "{text}");
    }

    #[test]
    fn an_api_error_rolls_the_turn_back_and_the_next_message_still_works() {
        let server = PhotocraftMcp::headless();
        let (transport, requests) = Scripted::new(vec![Ok((401, json!({"error": {"type": "authentication_error", "message": "bad key"}}))), Ok((200, text_reply("hello")))]);
        let handle = spawn_with(settings(), server, Box::new(transport), Arc::new(|| {})).unwrap();
        handle.commands.send(AgentCommand::Send("one".into())).unwrap();
        let events = drain(&handle, is_done);
        assert!(events.iter().any(|e| matches!(e, AgentEvent::Error(m) if m.contains("401"))), "{events:?}");
        handle.commands.send(AgentCommand::Send("two".into())).unwrap();
        let events = drain(&handle, is_done);
        assert!(events.iter().any(|e| matches!(e, AgentEvent::AssistantDelta { text, .. } if text == "hello")), "{events:?}");
        let reqs = requests.lock().unwrap();
        let msgs = reqs[1]["messages"].as_array().unwrap();
        assert_eq!(msgs.len(), 1, "the failed exchange left nothing behind: {msgs:?}");
        assert_eq!(msgs[0]["content"][0]["text"], "two");
    }

    #[test]
    fn cancel_stops_after_the_tool_in_flight_and_answers_the_rest_as_cancelled() {
        let server = PhotocraftMcp::headless();
        let two_calls = json!({"content": [
            {"type": "tool_use", "id": "a", "name": "session_list", "input": {}},
            {"type": "tool_use", "id": "b", "name": "session_list", "input": {}}
        ], "stop_reason": "tool_use", "usage": {"input_tokens": 1, "output_tokens": 1}});
        let (transport, requests) = Scripted::new(vec![Ok((200, two_calls)), Ok((200, text_reply("after")))]);
        // The cancel arrives while the model's first reply is in flight, so it is seen before
        // either tool starts: both are answered as cancelled (a tool already running finishes
        // first), and the model is not asked again this turn.
        let canceller: Arc<std::sync::Mutex<Option<Sender<AgentCommand>>>> = Arc::default();
        let slot = canceller.clone();
        let transport = transport.with_hook(move |n| {
            if n == 1 && let Some(tx) = slot.lock().unwrap().as_ref() {
                tx.send(AgentCommand::Cancel).unwrap();
            }
        });
        let handle = spawn_with(settings(), server, Box::new(transport), Arc::new(|| {})).unwrap();
        *canceller.lock().unwrap() = Some(handle.commands.clone());
        handle.commands.send(AgentCommand::Send("go".into())).unwrap();
        let events = drain(&handle, is_done);
        assert!(matches!(events.last(), Some(AgentEvent::TurnDone { stopped: Some(s), .. }) if s == "stopped"), "{events:?}");
        assert!(!events.iter().any(|e| matches!(e, AgentEvent::ToolCall(_))), "no tool started: {events:?}");
        assert_eq!(requests.lock().unwrap().len(), 1);
        // The next message continues from the merged user turn.
        handle.commands.send(AgentCommand::Send("more".into())).unwrap();
        let events = drain(&handle, is_done);
        assert!(events.iter().any(|e| matches!(e, AgentEvent::AssistantDelta { text, .. } if text == "after")), "{events:?}");
        let reqs = requests.lock().unwrap();
        let last_user = &reqs[1]["messages"][2]["content"];
        assert_eq!(last_user.as_array().unwrap().len(), 3, "two results and the new text in one user message: {last_user}");
        assert_eq!(last_user[2]["text"], "more");
    }

    #[test]
    fn refusals_and_output_limits_end_the_turn_with_a_reason() {
        for (reply, needle) in [
            (json!({"content": [], "stop_reason": "refusal", "stop_details": {"type": "refusal", "category": "cyber"}}), "cyber"),
            (json!({"content": [{"type": "text", "text": "partial"}], "stop_reason": "max_tokens"}), "output limit"),
        ] {
            let server = PhotocraftMcp::headless();
            let (transport, _) = Scripted::new(vec![Ok((200, reply))]);
            let handle = spawn_with(settings(), server, Box::new(transport), Arc::new(|| {})).unwrap();
            handle.commands.send(AgentCommand::Send("x".into())).unwrap();
            let events = drain(&handle, is_done);
            assert!(matches!(events.last(), Some(AgentEvent::TurnDone { stopped: Some(s), .. }) if s.contains(needle)), "{events:?}");
        }
    }

    #[test]
    fn spawn_refuses_settings_without_a_key() {
        let s = AgentSettings { api_key: String::new(), ..Default::default() };
        let (transport, _) = Scripted::new(vec![]);
        let err = spawn_with(s, PhotocraftMcp::headless(), Box::new(transport), Arc::new(|| {})).err().unwrap_or_default();
        assert!(err.contains("API key"), "{err}");
    }

    #[test]
    fn bounded_cuts_on_char_boundaries() {
        assert_eq!(bounded("héllo", 10), "héllo");
        let cut = bounded("日本語テキスト", 3);
        assert!(cut.starts_with("日本語…"), "{cut}");
    }

    #[test]
    fn openai_models_are_told_to_use_command_run_not_the_tool_search() {
        let p = system_prompt(&AgentSettings { provider: "openai".into(), ..Default::default() });
        assert!(p.contains("command_run {id, params}") && !p.contains("tool_search_tool_regex"), "{p}");
        assert!(p.starts_with("You are the Agent panel") && p.contains("Way of working"));
        assert_eq!(system_prompt(&AgentSettings::default()), SYSTEM_PROMPT);
    }

    #[test]
    fn the_model_test_checks_the_settings_first() {
        let (t, requests) = Scripted::new(vec![]);
        let rx = test_model_with(AgentSettings { api_key: String::new(), ..Default::default() }, Box::new(t));
        let e = rx.recv_timeout(Duration::from_secs(10)).unwrap().unwrap_err();
        assert!(e.contains("no API key"), "{e}");
        assert!(requests.lock().unwrap().is_empty());
    }

    #[test]
    fn an_openai_compatible_model_drives_the_tools() {
        let server = PhotocraftMcp::headless();
        let call = json!({"model": "local", "choices": [{"finish_reason": "tool_calls", "message": {"content": null, "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "doc_new", "arguments": "{\"width\":16,\"height\":8}"}}]}}]});
        let done = json!({"model": "local", "choices": [{"finish_reason": "stop", "message": {"content": "Made it."}}]});
        let (transport, requests) = Scripted::new(vec![Ok((200, call)), Ok((200, done))]);
        let s = AgentSettings { provider: "openai".into(), model: "local".into(), base_url: "http://localhost:11434/v1".into(), api_key: String::new(), max_turns: 3, ..Default::default() };
        let handle = spawn_with(s, server, Box::new(transport), Arc::new(|| {})).unwrap();
        drain(&handle, |e| matches!(e, AgentEvent::Status(s) if s.starts_with("Ready")));
        handle.commands.send(AgentCommand::Send("new doc".into())).unwrap();
        let events = drain(&handle, |e| matches!(e, AgentEvent::TurnDone { .. }));
        assert!(events.iter().any(|e| matches!(e, AgentEvent::ToolResult { name, ok: true, .. } if name == "doc_new")), "{events:?}");
        assert!(events.iter().any(|e| matches!(e, AgentEvent::AssistantDelta { text, .. } if text == "Made it.")));
        let r = requests.lock().unwrap();
        // Chat Completions shape: system first, functions only (no deferred tools), and the tool
        // result as a `tool` message after the assistant's call.
        assert_eq!(r[0]["messages"][0]["role"], "system");
        assert!(r[0]["tools"].as_array().unwrap().iter().all(|t| t["type"] == "function"));
        assert!(r[0]["tools"].as_array().unwrap().len() < 64);
        let second = r[1]["messages"].as_array().unwrap();
        assert_eq!(second[2]["tool_calls"][0]["id"], "c1");
        assert_eq!(second[3]["role"], "tool");
        assert_eq!(second[3]["tool_call_id"], "c1");
    }
}
