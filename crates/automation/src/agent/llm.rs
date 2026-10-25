//! The model API over plain HTTPS, non-streaming: Claude's Messages API (`POST /v1/messages`), or
//! the OpenAI Chat Completions format (`POST <base>/chat/completions`) that OpenAI and most other
//! providers and local servers accept. The runner keeps the conversation in the Messages shape;
//! [`openai`] translates each request and reply. The transport is a trait so the runner's loop is
//! tested with scripted replies and no network.

use std::time::Duration;

use photocraft_engine::agent::AgentSettings;
use serde_json::{Value, json};

/// The API version every request names.
pub const API_VERSION: &str = "2026-10-21";
/// Output cap of one reply: enough for a long answer, under the HTTP timeout without streaming.
pub const MAX_TOKENS: u32 = 16_000;
/// A reply larger than this is refused unread (a runaway body, not a model answer).
pub const MAX_REPLY_BYTES: u64 = 16 << 20;
/// One request's wall-clock limit, including the model's thinking.
pub const REQUEST_TIMEOUT: Duration = Duration::from_secs(600);
/// Models that take `fallbacks: "default"` (the beta header goes with it).
const FALLBACK_MODELS: &[&str] = &["claude-fable-5-1", "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5"];
const FALLBACK_BETA: &str = "server-side-fallback-2026-07-01";
/// Retries of a request the API couldn't take (429, 529, 5xx, a dropped connection).
const RETRIES: &[Duration] = &[Duration::from_secs(2), Duration::from_secs(6)];

/// Why [`Client::complete_stream`] returned early: the user stopped the turn mid-reply.
pub const STOPPED: &str = "stopped by the user";

/// What a streaming POST produced.
#[derive(Debug)]
pub enum StreamReply {
    /// The body was server-sent events, handed to the callback line by line; `aborted` when the
    /// callback asked to stop.
    Lines { aborted: bool },
    /// A whole JSON body: an error status, or a server that answered without streaming.
    Whole(u16, Value),
}

/// A POST of JSON to a URL: the status and the parsed body (or the error text).
pub trait Transport: Send {
    fn post(&self, url: &str, headers: &[(&str, &str)], body: &Value) -> Result<(u16, Value), String>;

    /// The same request with a streamed reply: each line of an event stream goes to `on_line`
    /// (return false to stop reading). A transport that can't stream answers whole.
    fn post_stream(&self, url: &str, headers: &[(&str, &str)], body: &Value, on_line: &mut dyn FnMut(&str) -> bool) -> Result<StreamReply, String> {
        let _ = on_line;
        self.post(url, headers, body).map(|(status, v)| StreamReply::Whole(status, v))
    }
}

/// `ureq` with the platform's trust roots, no redirects (an API key never follows one).
pub struct Https {
    agent: ureq::Agent,
}

impl Https {
    pub fn new() -> Self {
        let agent = ureq::Agent::config_builder()
            .tls_config(ureq::tls::TlsConfig::builder().root_certs(ureq::tls::RootCerts::PlatformVerifier).build())
            .timeout_global(Some(REQUEST_TIMEOUT))
            .max_redirects(0)
            .http_status_as_error(false)
            .build()
            .new_agent();
        Self { agent }
    }
}

impl Default for Https {
    fn default() -> Self {
        Self::new()
    }
}

impl Transport for Https {
    fn post(&self, url: &str, headers: &[(&str, &str)], body: &Value) -> Result<(u16, Value), String> {
        let mut req = self.agent.post(url).header("User-Agent", "PhotoCraft-agent");
        for (k, v) in headers {
            req = req.header(*k, *v);
        }
        let bytes = serde_json::to_vec(body).map_err(|e| format!("could not encode the request: {e}"))?;
        let mut response = req.send(&bytes[..]).map_err(|e| format!("could not reach the API: {e}"))?;
        let status = response.status().as_u16();
        let text = response.body_mut().with_config().limit(MAX_REPLY_BYTES).read_to_string().map_err(|e| format!("could not read the API reply: {e}"))?;
        let body: Value = serde_json::from_str(&text).unwrap_or_else(|_| json!({"raw": text.chars().take(2000).collect::<String>()}));
        Ok((status, body))
    }

    fn post_stream(&self, url: &str, headers: &[(&str, &str)], body: &Value, on_line: &mut dyn FnMut(&str) -> bool) -> Result<StreamReply, String> {
        use std::io::BufRead;
        let mut req = self.agent.post(url).header("User-Agent", "PhotoCraft-agent").header("Accept", "text/event-stream");
        for (k, v) in headers {
            req = req.header(*k, *v);
        }
        let bytes = serde_json::to_vec(body).map_err(|e| format!("could not encode the request: {e}"))?;
        let mut response = req.send(&bytes[..]).map_err(|e| format!("could not reach the API: {e}"))?;
        let status = response.status().as_u16();
        let events = response.headers().get("content-type").and_then(|v| v.to_str().ok()).is_some_and(|v| v.contains("text/event-stream"));
        if status != 200 || !events {
            let text = response.body_mut().with_config().limit(MAX_REPLY_BYTES).read_to_string().map_err(|e| format!("could not read the API reply: {e}"))?;
            let body: Value = serde_json::from_str(&text).unwrap_or_else(|_| json!({"raw": text.chars().take(2000).collect::<String>()}));
            return Ok(StreamReply::Whole(status, body));
        }
        let mut reader = std::io::BufReader::new(response.body_mut().with_config().limit(MAX_REPLY_BYTES).reader());
        let mut line = String::new();
        loop {
            line.clear();
            let n = reader.read_line(&mut line).map_err(|e| format!("the API stream broke off: {e}"))?;
            if n == 0 {
                return Ok(StreamReply::Lines { aborted: false });
            }
            if !on_line(line.trim_end_matches(['\r', '\n'])) {
                return Ok(StreamReply::Lines { aborted: true });
            }
        }
    }
}

/// One model reply.
#[derive(Clone, Debug, PartialEq)]
pub struct Reply {
    /// The assistant message's content blocks, verbatim (they go back into the history as is).
    pub content: Vec<Value>,
    pub stop_reason: String,
    /// `stop_details`, set on a refusal.
    pub stop_details: Option<Value>,
    pub input_tokens: u64,
    pub output_tokens: u64,
    /// The model that answered (a fallback may differ from the one asked).
    pub model: String,
}

pub struct Client {
    settings: AgentSettings,
    transport: Box<dyn Transport>,
    /// OpenAI-compatible: the server refused images in a request, so they are left out from now on
    /// (a text-only model; the model still gets every tool result's text).
    no_images: std::sync::atomic::AtomicBool,
    /// OpenAI-compatible: the server refused `stream_options` (usage in the stream), so it is
    /// left out from now on.
    no_stream_options: std::sync::atomic::AtomicBool,
}

impl Client {
    pub fn new(settings: AgentSettings, transport: Box<dyn Transport>) -> Self {
        Self { settings, transport, no_images: std::sync::atomic::AtomicBool::new(false), no_stream_options: std::sync::atomic::AtomicBool::new(false) }
    }

    pub fn settings(&self) -> &AgentSettings {
        &self.settings
    }

    fn fallbacks(&self) -> bool {
        self.settings.refusal_fallbacks && FALLBACK_MODELS.contains(&self.settings.model.as_str())
    }

    /// The request body for `messages`, with `tools` (already shaped for the API) and the system
    /// prompt. The system prompt and the last non-deferred tool carry cache breakpoints, so the
    /// stable prefix is read from cache on every later turn.
    pub fn request_body(&self, system: &str, tools: &[Value], messages: &[Value]) -> Value {
        let mut tools: Vec<Value> = tools.to_vec();
        if let Some(last) = tools.iter_mut().rev().find(|t| t.get("defer_loading") != Some(&json!(true)) && t.get("type").is_none())
            && let Some(o) = last.as_object_mut()
        {
            o.insert("cache_control".into(), json!({"type": "ephemeral"}));
        }
        let mut body = json!({
            "model": self.settings.model,
            "max_tokens": MAX_TOKENS,
            "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            "tools": tools,
            "messages": messages,
            "output_config": {"effort": self.settings.effort},
        });
        if self.fallbacks() {
            body["fallbacks"] = json!("default");
        }
        body
    }

    /// Ask the model for the next assistant message.
    pub fn complete(&self, system: &str, tools: &[Value], messages: &[Value]) -> Result<Reply, String> {
        if self.settings.is_openai() {
            return self.complete_openai(system, tools, messages);
        }
        let body = self.request_body(system, tools, messages);
        let url = format!("{}/v1/messages", self.settings.base_url);
        let mut headers: Vec<(&str, &str)> = vec![("x-api-key", self.settings.api_key.as_str()), ("anthropic-version", API_VERSION), ("content-type", "application/json")];
        if self.fallbacks() {
            headers.push(("anthropic-beta", FALLBACK_BETA));
        }
        let reply = self.post(&url, &headers, &body)?;
        parse_reply(&reply)
    }

    fn complete_openai(&self, system: &str, tools: &[Value], messages: &[Value]) -> Result<Reply, String> {
        use std::sync::atomic::Ordering;
        let url = format!("{}/chat/completions", self.settings.base_url);
        let auth = format!("Bearer {}", self.settings.api_key);
        let mut headers: Vec<(&str, &str)> = vec![("content-type", "application/json")];
        if !self.settings.api_key.is_empty() {
            headers.push(("authorization", auth.as_str()));
        }
        let images = !self.no_images.load(Ordering::Relaxed);
        let body = openai::request_body(&self.settings.model, system, tools, messages, images);
        match self.post(&url, &headers, &body) {
            Ok(reply) => openai::parse_reply(&reply),
            // A text-only model refuses image content with a 400: say it once, then go on without
            // images rather than failing every turn that follows a screenshot.
            Err(e) if images && e.starts_with("API error 400") && openai::has_images(&body) => {
                self.no_images.store(true, Ordering::Relaxed);
                let body = openai::request_body(&self.settings.model, system, tools, messages, false);
                openai::parse_reply(&self.post(&url, &headers, &body)?)
            }
            Err(e) => Err(e),
        }
    }

    /// POST with retries on 429/5xx and dropped connections; a non-200 reply is an error naming
    /// the status and the API's message.
    fn post(&self, url: &str, headers: &[(&str, &str)], body: &Value) -> Result<Value, String> {
        let mut attempt = 0;
        loop {
            let outcome = self.transport.post(url, headers, body);
            let retry = match &outcome {
                Ok((status, _)) => matches!(*status, 429 | 500..=599),
                Err(_) => true,
            };
            if retry && let Some(wait) = RETRIES.get(attempt) {
                std::thread::sleep(*wait);
                attempt += 1;
                continue;
            }
            let (status, reply) = outcome?;
            if status == 200 {
                return Ok(reply);
            }
            return Err(api_error(status, &reply));
        }
    }

    /// Ask the model for the next assistant message, streaming its text: `on_delta(text,
    /// new_block)` gets each piece of text as it arrives (`new_block` on the first piece of
    /// each text block) and, with empty text, a heartbeat on every other event. Returning false
    /// stops the reply; the call then fails with [`STOPPED`]. A request is retried (429, 5xx, a
    /// dropped connection) only while nothing has been shown.
    pub fn complete_stream(&self, system: &str, tools: &[Value], messages: &[Value], on_delta: &mut dyn FnMut(&str, bool) -> bool) -> Result<Reply, String> {
        use std::sync::atomic::Ordering;
        let openai = self.settings.is_openai();
        let auth = format!("Bearer {}", self.settings.api_key);
        let (url, mut headers) = if openai {
            let mut h: Vec<(&str, &str)> = vec![("content-type", "application/json")];
            if !self.settings.api_key.is_empty() {
                h.push(("authorization", auth.as_str()));
            }
            (format!("{}/chat/completions", self.settings.base_url), h)
        } else {
            (format!("{}/v1/messages", self.settings.base_url), vec![("x-api-key", self.settings.api_key.as_str()), ("anthropic-version", API_VERSION), ("content-type", "application/json")])
        };
        if !openai && self.fallbacks() {
            headers.push(("anthropic-beta", FALLBACK_BETA));
        }
        let mut attempt = 0;
        loop {
            let body = if openai {
                let mut b = openai::request_body(&self.settings.model, system, tools, messages, !self.no_images.load(Ordering::Relaxed));
                b["stream"] = json!(true);
                if !self.no_stream_options.load(Ordering::Relaxed) {
                    b["stream_options"] = json!({"include_usage": true});
                }
                b
            } else {
                let mut b = self.request_body(system, tools, messages);
                b["stream"] = json!(true);
                b
            };
            let mut shown = false;
            let mut acc = if openai { Accumulator::OpenAi(OpenAiStream::default()) } else { Accumulator::Anthropic(AnthropicStream::default()) };
            let mut failure: Option<String> = None;
            let outcome = self.transport.post_stream(&url, &headers, &body, &mut |line| {
                let Some(data) = line.strip_prefix("data:") else { return true };
                let data = data.trim();
                if data.is_empty() || data == "[DONE]" {
                    return true;
                }
                let Ok(v) = serde_json::from_str::<Value>(data) else { return true };
                let mut keep = true;
                let fed = acc.feed(&v, &mut |text, new_block| {
                    shown |= !text.is_empty();
                    keep &= on_delta(text, new_block);
                });
                if let Err(e) = fed {
                    failure = Some(e);
                    return false;
                }
                keep && on_delta("", false)
            });
            if let Some(e) = failure {
                return Err(e);
            }
            let retry = match &outcome {
                Ok(StreamReply::Whole(status, _)) => matches!(*status, 429 | 500..=599),
                Ok(StreamReply::Lines { .. }) => false,
                Err(_) => !shown,
            };
            if retry && let Some(wait) = RETRIES.get(attempt) {
                std::thread::sleep(*wait);
                attempt += 1;
                continue;
            }
            match outcome? {
                StreamReply::Lines { aborted: true } => return Err(STOPPED.into()),
                StreamReply::Lines { aborted: false } => return acc.finish(),
                StreamReply::Whole(200, v) => {
                    // A server that ignored `stream`: show the whole text at once.
                    let reply = if openai { openai::parse_reply(&v)? } else { parse_reply(&v)? };
                    for b in &reply.content {
                        if let Some(t) = b.get("text").and_then(Value::as_str).filter(|t| !t.is_empty())
                            && !on_delta(t, true)
                        {
                            return Err(STOPPED.into());
                        }
                    }
                    return Ok(reply);
                }
                StreamReply::Whole(400, v) if openai => {
                    // A text-only model refuses images, an older server `stream_options`: drop
                    // the one it refused and ask again (once each).
                    if openai::has_images(&body) && !self.no_images.swap(true, Ordering::Relaxed) {
                        continue;
                    }
                    if body.get("stream_options").is_some() && !self.no_stream_options.swap(true, Ordering::Relaxed) {
                        continue;
                    }
                    return Err(api_error(400, &v));
                }
                StreamReply::Whole(status, v) => return Err(api_error(status, &v)),
            }
        }
    }

    /// One short request without tools: proves the endpoint, key and model work. Returns a line
    /// for Preferences (the model that answered, its reply, how long it took).
    pub fn test(&self) -> Result<String, String> {
        let started = std::time::Instant::now();
        let messages = [json!({"role": "user", "content": [{"type": "text", "text": "This is a connection test. Reply with the single word OK."}]})];
        let reply = self.complete("You are testing a connection. Answer in one word.", &[], &messages)?;
        let text: String = reply.content.iter().filter_map(|b| b.get("text").and_then(Value::as_str)).collect::<Vec<_>>().join(" ");
        let text: String = text.trim().chars().take(80).collect();
        let model = if reply.model.is_empty() { self.settings.model.clone() } else { reply.model.clone() };
        Ok(format!("{model} answered \"{}\" in {:.1} s", if text.is_empty() { "(no text)" } else { &text }, started.elapsed().as_secs_f64()))
    }
}

/// A non-200 reply as an error naming the status and the API's message. Messages and
/// OpenAI-style errors share `error.message`; some servers send a plain string.
fn api_error(status: u16, reply: &Value) -> String {
    let message = reply
        .pointer("/error/message")
        .and_then(Value::as_str)
        .or_else(|| reply.get("error").and_then(Value::as_str))
        .or_else(|| reply.get("message").and_then(Value::as_str))
        .or_else(|| reply.get("raw").and_then(Value::as_str))
        .unwrap_or("no details");
    let kind = reply.pointer("/error/type").and_then(Value::as_str).unwrap_or("error");
    match status {
        400 => format!("API error 400 {kind}: {message}"),
        401 => format!("the API rejected the key (401 {kind}): {message}"),
        403 => format!("the API refused the request (403 {kind}): {message}"),
        404 => format!("unknown model or endpoint (404 {kind}): {message}"),
        429 => format!("rate limited (429): {message}"),
        529 => format!("the API is overloaded (529): {message}"),
        _ => format!("API error {status} {kind}: {message}"),
    }
}

/// A streamed reply being put back together.
enum Accumulator {
    Anthropic(AnthropicStream),
    OpenAi(OpenAiStream),
}

impl Accumulator {
    fn feed(&mut self, v: &Value, text: &mut dyn FnMut(&str, bool)) -> Result<(), String> {
        match self {
            Accumulator::Anthropic(a) => a.feed(v, text),
            Accumulator::OpenAi(o) => {
                o.feed(v, text);
                Ok(())
            }
        }
    }

    fn finish(self) -> Result<Reply, String> {
        match self {
            Accumulator::Anthropic(a) => parse_reply(&a.finish()?),
            Accumulator::OpenAi(o) => openai::parse_reply(&o.finish()),
        }
    }
}

/// Messages API events (`message_start`, `content_block_start/delta/stop`, `message_delta`,
/// `message_stop`, `error`) back into the message the non-streaming API returns, every block
/// verbatim (tool inputs from their JSON deltas), so it goes back into the history unchanged.
#[derive(Default)]
pub struct AnthropicStream {
    message: Option<Value>,
    blocks: Vec<Value>,
    json: Vec<String>,
    /// Text blocks that have shown their first piece.
    started: Vec<bool>,
}

impl AnthropicStream {
    pub fn feed(&mut self, v: &Value, text: &mut dyn FnMut(&str, bool)) -> Result<(), String> {
        let index = v.get("index").and_then(Value::as_u64).map(|i| i as usize);
        match v.get("type").and_then(Value::as_str).unwrap_or("") {
            "message_start" => self.message = v.get("message").cloned(),
            "content_block_start" => {
                let (Some(i), Some(block)) = (index, v.get("content_block")) else { return Ok(()) };
                // Indexes come in order; a bound keeps a hostile one from allocating.
                if i > self.blocks.len() + 64 {
                    return Err("the API stream named an impossible content block".into());
                }
                while self.blocks.len() <= i {
                    self.blocks.push(Value::Null);
                    self.json.push(String::new());
                    self.started.push(false);
                }
                self.blocks[i] = block.clone();
                if let Some(t) = block.get("text").and_then(Value::as_str).filter(|t| !t.is_empty()) {
                    self.started[i] = true;
                    text(t, true);
                }
            }
            "content_block_delta" => {
                let (Some(i), Some(d)) = (index, v.get("delta")) else { return Ok(()) };
                let Some(block) = self.blocks.get_mut(i) else { return Ok(()) };
                let append = |block: &mut Value, key: &str, piece: &str| {
                    let joined = format!("{}{piece}", block.get(key).and_then(Value::as_str).unwrap_or(""));
                    block[key] = json!(joined);
                };
                match d.get("type").and_then(Value::as_str).unwrap_or("") {
                    "text_delta" => {
                        let piece = d.get("text").and_then(Value::as_str).unwrap_or("");
                        append(block, "text", piece);
                        if !piece.is_empty() {
                            let first = !self.started[i];
                            self.started[i] = true;
                            text(piece, first);
                        }
                    }
                    "input_json_delta" => self.json[i].push_str(d.get("partial_json").and_then(Value::as_str).unwrap_or("")),
                    "thinking_delta" => append(block, "thinking", d.get("thinking").and_then(Value::as_str).unwrap_or("")),
                    "signature_delta" => block["signature"] = d.get("signature").cloned().unwrap_or(Value::Null),
                    "citations_delta" => {
                        if let Some(c) = d.get("citation") {
                            if !block.get("citations").is_some_and(Value::is_array) {
                                block["citations"] = json!([]);
                            }
                            if let Some(list) = block.get_mut("citations").and_then(Value::as_array_mut) {
                                list.push(c.clone());
                            }
                        }
                    }
                    _ => {}
                }
            }
            "content_block_stop" => {
                let Some(i) = index else { return Ok(()) };
                let Some(json) = self.json.get(i).filter(|j| !j.trim().is_empty()) else { return Ok(()) };
                let input: Value = serde_json::from_str(json).map_err(|e| format!("a streamed tool call had invalid JSON input: {e}"))?;
                if let Some(block) = self.blocks.get_mut(i) {
                    block["input"] = input;
                }
            }
            "message_delta" => {
                let Some(m) = self.message.as_mut() else { return Ok(()) };
                if let Some(d) = v.get("delta") {
                    for key in ["stop_reason", "stop_sequence", "stop_details"] {
                        if let Some(x) = d.get(key) {
                            m[key] = x.clone();
                        }
                    }
                }
                if let Some(u) = v.get("usage").and_then(Value::as_object) {
                    for (k, x) in u {
                        m["usage"][k] = x.clone();
                    }
                }
            }
            "error" => {
                let message = v.pointer("/error/message").and_then(Value::as_str).unwrap_or("the API reported an error mid-stream");
                let kind = v.pointer("/error/type").and_then(Value::as_str).unwrap_or("error");
                return Err(format!("API error ({kind}): {message}"));
            }
            _ => {}
        }
        Ok(())
    }

    /// The whole message; an error when the stream ended before it began.
    pub fn finish(self) -> Result<Value, String> {
        let mut m = self.message.ok_or("the API stream ended without a message")?;
        m["content"] = Value::Array(self.blocks.into_iter().filter(|b| !b.is_null()).collect());
        Ok(m)
    }
}

/// Chat Completions chunks back into the non-streaming reply shape [`openai::parse_reply`] reads.
#[derive(Default)]
pub struct OpenAiStream {
    model: String,
    content: String,
    /// (id, name, arguments) per tool call index.
    calls: Vec<(String, String, String)>,
    finish: Option<String>,
    usage: Option<Value>,
}

impl OpenAiStream {
    pub fn feed(&mut self, v: &Value, text: &mut dyn FnMut(&str, bool)) {
        if let Some(m) = v.get("model").and_then(Value::as_str) {
            self.model = m.to_string();
        }
        if let Some(u) = v.get("usage").filter(|u| u.is_object()) {
            self.usage = Some(u.clone());
        }
        let Some(choice) = v.pointer("/choices/0") else { return };
        if let Some(f) = choice.get("finish_reason").and_then(Value::as_str) {
            self.finish = Some(f.to_string());
        }
        let Some(delta) = choice.get("delta") else { return };
        if let Some(piece) = delta.get("content").and_then(Value::as_str).filter(|p| !p.is_empty()) {
            let first = self.content.is_empty();
            self.content.push_str(piece);
            text(piece, first);
        }
        for c in delta.get("tool_calls").and_then(Value::as_array).into_iter().flatten() {
            let i = c.get("index").and_then(Value::as_u64).map_or(self.calls.len().saturating_sub(1), |i| i as usize);
            if i > self.calls.len() + 64 {
                continue;
            }
            while self.calls.len() <= i {
                self.calls.push(Default::default());
            }
            let call = &mut self.calls[i];
            if let Some(id) = c.get("id").and_then(Value::as_str).filter(|s| !s.is_empty()) {
                call.0 = id.to_string();
            }
            if let Some(n) = c.pointer("/function/name").and_then(Value::as_str) {
                call.1.push_str(n);
            }
            if let Some(a) = c.pointer("/function/arguments").and_then(Value::as_str) {
                call.2.push_str(a);
            }
        }
    }

    pub fn finish(self) -> Value {
        let calls: Vec<Value> = self
            .calls
            .into_iter()
            .filter(|c| !c.1.is_empty())
            .map(|(id, name, args)| json!({"id": id, "type": "function", "function": {"name": name, "arguments": args}}))
            .collect();
        let mut message = json!({"role": "assistant", "content": if self.content.is_empty() { Value::Null } else { json!(self.content) }});
        if !calls.is_empty() {
            message["tool_calls"] = json!(calls);
        }
        let mut reply = json!({"model": self.model, "choices": [{"finish_reason": self.finish.unwrap_or_else(|| "stop".into()), "message": message}]});
        if let Some(u) = self.usage {
            reply["usage"] = u;
        }
        reply
    }
}

/// A successful Messages reply, checked for the fields the loop relies on.
pub fn parse_reply(v: &Value) -> Result<Reply, String> {
    let content = v.get("content").and_then(Value::as_array).ok_or("the API reply has no `content` array")?.clone();
    let stop_reason = v.get("stop_reason").and_then(Value::as_str).unwrap_or("end_turn").to_string();
    for block in &content {
        if block.get("type").and_then(Value::as_str) == Some("tool_use") {
            if block.get("id").and_then(Value::as_str).is_none_or(str::is_empty) || block.get("name").and_then(Value::as_str).is_none_or(str::is_empty) {
                return Err("a tool_use block in the API reply has no id or name".into());
            }
            if !block.get("input").is_some_and(Value::is_object) {
                return Err("a tool_use block in the API reply has no object `input`".into());
            }
        }
    }
    Ok(Reply {
        content,
        stop_reason,
        stop_details: v.get("stop_details").filter(|d| !d.is_null()).cloned(),
        input_tokens: v.pointer("/usage/input_tokens").and_then(Value::as_u64).unwrap_or(0),
        output_tokens: v.pointer("/usage/output_tokens").and_then(Value::as_u64).unwrap_or(0),
        model: v.get("model").and_then(Value::as_str).unwrap_or_default().to_string(),
    })
}

/// The OpenAI Chat Completions format, translated to and from the Messages shape the runner keeps.
pub mod openai {
    use serde_json::{Map, Value, json};

    use super::Reply;

    /// The Chat Completions request: the system prompt as a system message, the conversation
    /// translated, and the tools that are loaded up front as functions. The deferred per-command
    /// tools and the server-side tool search are Claude-only, so an OpenAI-compatible model works
    /// through the built-in tools (`command_list` to discover, `command_run` to run any command).
    /// With `images` false, screenshots in tool results are left out.
    pub fn request_body(model: &str, system: &str, tools: &[Value], messages: &[Value], images: bool) -> Value {
        let mut out = vec![json!({"role": "system", "content": system})];
        for m in messages {
            translate_message(m, images, &mut out);
        }
        let functions: Vec<Value> = tools
            .iter()
            .filter(|t| t.get("type").is_none() && t.get("defer_loading") != Some(&json!(true)))
            .filter_map(|t| {
                Some(json!({"type": "function", "function": {
                    "name": t.get("name")?.as_str()?,
                    "description": t.get("description").and_then(Value::as_str).unwrap_or(""),
                    "parameters": t.get("input_schema").cloned().unwrap_or_else(|| json!({"type": "object"})),
                }}))
            })
            .collect();
        let mut body = json!({"model": model, "messages": out});
        if !functions.is_empty() {
            body["tools"] = json!(functions);
        }
        body
    }

    /// Whether a translated request carries image content.
    pub fn has_images(body: &Value) -> bool {
        body.get("messages").and_then(Value::as_array).is_some_and(|ms| {
            ms.iter().any(|m| m.get("content").and_then(Value::as_array).is_some_and(|c| c.iter().any(|p| p.get("type").and_then(Value::as_str) == Some("image_url"))))
        })
    }

    fn text_of(content: &Value) -> String {
        match content {
            Value::String(s) => s.clone(),
            Value::Array(blocks) => blocks.iter().filter_map(|b| b.get("text").and_then(Value::as_str)).collect::<Vec<_>>().join("\n"),
            _ => String::new(),
        }
    }

    fn image_parts(content: &Value) -> Vec<Value> {
        content
            .as_array()
            .into_iter()
            .flatten()
            .filter(|b| b.get("type").and_then(Value::as_str) == Some("image"))
            .filter_map(|b| {
                let media = b.pointer("/source/media_type").and_then(Value::as_str)?;
                let data = b.pointer("/source/data").and_then(Value::as_str)?;
                Some(json!({"type": "image_url", "image_url": {"url": format!("data:{media};base64,{data}")}}))
            })
            .collect()
    }

    /// One Messages-shape message as Chat Completions messages. A user message's tool results
    /// become `tool` messages (they must follow the assistant's tool calls directly); its text and
    /// any screenshots follow as one user message.
    fn translate_message(m: &Value, images: bool, out: &mut Vec<Value>) {
        let role = m.get("role").and_then(Value::as_str).unwrap_or("user");
        let content = m.get("content").cloned().unwrap_or(Value::Null);
        let blocks: Vec<Value> = match &content {
            Value::String(s) => vec![json!({"type": "text", "text": s})],
            Value::Array(a) => a.clone(),
            _ => Vec::new(),
        };
        if role == "assistant" {
            let text: Vec<&str> = blocks.iter().filter(|b| b.get("type").and_then(Value::as_str) == Some("text")).filter_map(|b| b.get("text").and_then(Value::as_str)).collect();
            let calls: Vec<Value> = blocks
                .iter()
                .filter(|b| b.get("type").and_then(Value::as_str) == Some("tool_use"))
                .map(|b| {
                    json!({"id": b.get("id").cloned().unwrap_or(json!("")), "type": "function", "function": {
                        "name": b.get("name").cloned().unwrap_or(json!("")),
                        "arguments": b.get("input").map(Value::to_string).unwrap_or_else(|| "{}".into()),
                    }})
                })
                .collect();
            let mut msg = json!({"role": "assistant", "content": if text.is_empty() { Value::Null } else { json!(text.join("\n")) }});
            if !calls.is_empty() {
                msg["tool_calls"] = json!(calls);
            }
            out.push(msg);
            return;
        }
        let mut parts: Vec<Value> = Vec::new();
        for b in &blocks {
            match b.get("type").and_then(Value::as_str) {
                Some("tool_result") => {
                    let c = b.get("content").cloned().unwrap_or(Value::Null);
                    let mut text = text_of(&c);
                    if b.get("is_error") == Some(&json!(true)) {
                        text = format!("Error: {text}");
                    }
                    let shots = image_parts(&c);
                    if !shots.is_empty() {
                        if images {
                            parts.push(json!({"type": "text", "text": "Image returned by the tool call above:"}));
                            parts.extend(shots);
                        } else if text.trim().is_empty() {
                            text = "(an image; this model can't see images)".into();
                        }
                    }
                    out.push(json!({"role": "tool", "tool_call_id": b.get("tool_use_id").cloned().unwrap_or(json!("")), "content": if text.is_empty() { "ok".to_string() } else { text }}));
                }
                Some("text") => {
                    if let Some(t) = b.get("text").and_then(Value::as_str) {
                        parts.push(json!({"type": "text", "text": t}));
                    }
                }
                Some("image") if images => parts.extend(image_parts(&json!([b]))),
                _ => {}
            }
        }
        if parts.is_empty() {
            return;
        }
        // Plain text as a string: the widest-supported form.
        if parts.iter().all(|p| p.get("type").and_then(Value::as_str) == Some("text")) {
            let text: Vec<&str> = parts.iter().filter_map(|p| p.get("text").and_then(Value::as_str)).collect();
            out.push(json!({"role": "user", "content": text.join("\n")}));
        } else {
            out.push(json!({"role": "user", "content": parts}));
        }
    }

    /// A Chat Completions reply as a Messages-shape [`Reply`]: text and `tool_use` blocks, the
    /// finish reason mapped (`tool_calls` → `tool_use`, `stop` → `end_turn`, `length` →
    /// `max_tokens`, `content_filter` → `refusal`).
    pub fn parse_reply(v: &Value) -> Result<Reply, String> {
        let choice = v.pointer("/choices/0").ok_or("the API reply has no `choices`")?;
        let msg = choice.get("message").ok_or("the API reply's choice has no `message`")?;
        let mut content = Vec::new();
        if let Some(t) = msg.get("content").and_then(Value::as_str).filter(|t| !t.trim().is_empty()) {
            content.push(json!({"type": "text", "text": t}));
        }
        let calls = msg.get("tool_calls").and_then(Value::as_array).cloned().unwrap_or_default();
        for (i, c) in calls.iter().enumerate() {
            let name = c.pointer("/function/name").and_then(Value::as_str).filter(|n| !n.is_empty()).ok_or("a tool call in the API reply has no function name")?;
            let id = c.get("id").and_then(Value::as_str).filter(|s| !s.is_empty()).map_or_else(|| format!("call_{i}"), str::to_string);
            // Arguments arrive as JSON text (some servers send an object). Text that isn't an
            // object goes to the tool as is, whose error then names it for the model.
            let input = match c.pointer("/function/arguments") {
                Some(Value::String(s)) if s.trim().is_empty() => json!({}),
                Some(Value::String(s)) => match serde_json::from_str::<Value>(s) {
                    Ok(Value::Object(o)) => Value::Object(o),
                    _ => json!({"_raw_arguments": s}),
                },
                Some(Value::Object(o)) => Value::Object(o.clone()),
                _ => Value::Object(Map::new()),
            };
            content.push(json!({"type": "tool_use", "id": id, "name": name, "input": input}));
        }
        let finish = choice.get("finish_reason").and_then(Value::as_str).unwrap_or("stop");
        let stop_reason = if !calls.is_empty() {
            "tool_use"
        } else {
            match finish {
                "length" => "max_tokens",
                "content_filter" => "refusal",
                _ => "end_turn",
            }
        };
        Ok(Reply {
            content,
            stop_reason: stop_reason.to_string(),
            stop_details: None,
            input_tokens: v.pointer("/usage/prompt_tokens").and_then(Value::as_u64).unwrap_or(0),
            output_tokens: v.pointer("/usage/completion_tokens").and_then(Value::as_u64).unwrap_or(0),
            model: v.get("model").and_then(Value::as_str).unwrap_or_default().to_string(),
        })
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use std::sync::{Arc, Mutex, PoisonError};

    use super::*;

    /// Scripted replies: each `post` pops the next `(status, body)`; the requests are kept.
    pub(crate) struct Scripted {
        pub replies: Mutex<std::collections::VecDeque<Result<(u16, Value), String>>>,
        pub requests: Arc<Mutex<Vec<Value>>>,
        /// Runs on every request, after it was recorded (tests cancel from here).
        pub hook: Option<Box<dyn Fn(usize) + Send>>,
    }

    impl Scripted {
        pub fn new(replies: Vec<Result<(u16, Value), String>>) -> (Self, Arc<Mutex<Vec<Value>>>) {
            let requests = Arc::new(Mutex::new(Vec::new()));
            (Self { replies: Mutex::new(replies.into()), requests: requests.clone(), hook: None }, requests)
        }

        pub fn with_hook(mut self, hook: impl Fn(usize) + Send + 'static) -> Self {
            self.hook = Some(Box::new(hook));
            self
        }
    }

    impl Transport for Scripted {
        fn post(&self, _url: &str, _headers: &[(&str, &str)], body: &Value) -> Result<(u16, Value), String> {
            let n = {
                let mut r = self.requests.lock().unwrap_or_else(PoisonError::into_inner);
                r.push(body.clone());
                r.len()
            };
            if let Some(h) = &self.hook {
                h(n);
            }
            self.replies.lock().unwrap_or_else(PoisonError::into_inner).pop_front().unwrap_or(Err("script exhausted".into()))
        }
    }

    pub(crate) fn text_reply(text: &str) -> Value {
        json!({"content": [{"type": "text", "text": text}], "stop_reason": "end_turn", "usage": {"input_tokens": 10, "output_tokens": 5}, "model": "claude-opus-5-5"})
    }

    pub(crate) fn tool_reply(id: &str, name: &str, input: Value) -> Value {
        json!({"content": [{"type": "tool_use", "id": id, "name": name, "input": input}], "stop_reason": "tool_use", "usage": {"input_tokens": 10, "output_tokens": 5}})
    }

    fn settings() -> AgentSettings {
        AgentSettings { api_key: "k".into(), ..Default::default() }
    }

    #[test]
    fn the_body_names_model_effort_tools_and_cache_breakpoints() {
        let c = Client::new(settings(), Box::new(Scripted::new(vec![]).0));
        let tools = vec![
            json!({"type": "tool_search_tool_regex_20251119", "name": "tool_search_tool_regex"}),
            json!({"name": "a", "description": "A", "input_schema": {"type": "object"}}),
            json!({"name": "b", "description": "B", "input_schema": {"type": "object"}, "defer_loading": true}),
        ];
        let body = c.request_body("sys", &tools, &[json!({"role": "user", "content": "hi"})]);
        assert_eq!(body["model"], "claude-opus-5-5");
        assert_eq!(body["output_config"]["effort"], "high");
        assert_eq!(body["fallbacks"], "default");
        assert_eq!(body["system"][0]["cache_control"]["type"], "ephemeral");
        assert_eq!(body["tools"][1]["cache_control"]["type"], "ephemeral");
        assert!(body["tools"][2].get("cache_control").is_none(), "deferred tools can't carry a breakpoint");
        assert!(body["tools"][0].get("cache_control").is_none());
        assert!(body.get("thinking").is_none() && body.get("temperature").is_none());
    }

    #[test]
    fn no_fallbacks_for_models_without_them_or_when_switched_off() {
        let s = AgentSettings { model: "claude-haiku-5-5".into(), ..settings() };
        let c = Client::new(s, Box::new(Scripted::new(vec![]).0));
        assert!(c.request_body("s", &[], &[]).get("fallbacks").is_none());
        let s = AgentSettings { refusal_fallbacks: false, ..settings() };
        let c = Client::new(s, Box::new(Scripted::new(vec![]).0));
        assert!(c.request_body("s", &[], &[]).get("fallbacks").is_none());
    }

    #[test]
    fn errors_name_the_status_and_message() {
        let (t, _) = Scripted::new(vec![Ok((401, json!({"error": {"type": "authentication_error", "message": "invalid x-api-key"}})))]);
        let c = Client::new(settings(), Box::new(t));
        let e = c.complete("s", &[], &[]).unwrap_err();
        assert!(e.contains("401") && e.contains("invalid x-api-key"), "{e}");
    }

    #[test]
    fn malformed_replies_are_errors_not_panics() {
        assert!(parse_reply(&json!({})).is_err());
        assert!(parse_reply(&json!({"content": [{"type": "tool_use", "name": "x", "input": {}}]})).is_err());
        assert!(parse_reply(&json!({"content": [{"type": "tool_use", "id": "t", "name": "x", "input": 3}]})).is_err());
        let r = parse_reply(&text_reply("hi")).unwrap();
        assert_eq!(r.stop_reason, "end_turn");
        assert_eq!((r.input_tokens, r.output_tokens), (10, 5));
    }

    fn openai_settings() -> AgentSettings {
        AgentSettings { provider: "openai".into(), model: "deepseek-chat".into(), base_url: "https://api.example.com/v1".into(), api_key: "k".into(), ..Default::default() }
    }

    #[test]
    fn openai_requests_translate_the_conversation_and_the_loaded_tools() {
        let tools = vec![
            json!({"type": "tool_search_tool_regex_20251119", "name": "tool_search_tool_regex"}),
            json!({"name": "doc_inspect", "description": "Inspect", "input_schema": {"type": "object", "properties": {}}}),
            json!({"name": "filter_blur", "description": "Blur", "input_schema": {"type": "object"}, "defer_loading": true}),
        ];
        let messages = vec![
            json!({"role": "user", "content": [{"type": "text", "text": "blur it"}]}),
            json!({"role": "assistant", "content": [{"type": "text", "text": "Looking."}, {"type": "tool_use", "id": "c1", "name": "doc_inspect", "input": {"index": 0}}]}),
            json!({"role": "user", "content": [{"type": "tool_result", "tool_use_id": "c1", "is_error": false, "content": [{"type": "text", "text": "{}"}, {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAA"}}]}]}),
        ];
        let body = openai::request_body("m", "sys", &tools, &messages, true);
        let ms = body["messages"].as_array().unwrap();
        assert_eq!(ms[0], json!({"role": "system", "content": "sys"}));
        assert_eq!(ms[1], json!({"role": "user", "content": "blur it"}));
        assert_eq!(ms[2]["content"], "Looking.");
        assert_eq!(ms[2]["tool_calls"][0]["function"]["name"], "doc_inspect");
        assert_eq!(ms[2]["tool_calls"][0]["function"]["arguments"], "{\"index\":0}");
        assert_eq!(ms[3], json!({"role": "tool", "tool_call_id": "c1", "content": "{}"}));
        assert_eq!(ms[4]["content"][1]["image_url"]["url"], "data:image/png;base64,AAA");
        assert!(openai::has_images(&body));
        // Only the loaded tools, as functions.
        assert_eq!(body["tools"].as_array().unwrap().len(), 1);
        assert_eq!(body["tools"][0]["function"]["name"], "doc_inspect");
        let text_only = openai::request_body("m", "sys", &tools, &messages, false);
        assert!(!openai::has_images(&text_only));
        assert_eq!(text_only["messages"].as_array().unwrap().len(), 4);
    }

    #[test]
    fn openai_replies_become_text_and_tool_use_blocks() {
        let r = openai::parse_reply(&json!({"model": "deepseek-chat", "choices": [{"finish_reason": "tool_calls", "message": {"content": null, "tool_calls": [
            {"id": "call_a", "type": "function", "function": {"name": "command_run", "arguments": "{\"id\":\"edit.undo\"}"}},
            {"type": "function", "function": {"name": "doc_inspect", "arguments": ""}},
            {"id": "x", "type": "function", "function": {"name": "doc_inspect", "arguments": "not json"}}
        ]}}], "usage": {"prompt_tokens": 7, "completion_tokens": 3}})).unwrap();
        assert_eq!(r.stop_reason, "tool_use");
        assert_eq!(r.content[0], json!({"type": "tool_use", "id": "call_a", "name": "command_run", "input": {"id": "edit.undo"}}));
        assert_eq!(r.content[1]["id"], "call_1");
        assert_eq!(r.content[1]["input"], json!({}));
        assert_eq!(r.content[2]["input"]["_raw_arguments"], "not json");
        assert_eq!((r.input_tokens, r.output_tokens), (7, 3));
        let r = openai::parse_reply(&json!({"choices": [{"finish_reason": "length", "message": {"content": "partial"}}]})).unwrap();
        assert_eq!((r.stop_reason.as_str(), r.content.len()), ("max_tokens", 1));
        assert!(openai::parse_reply(&json!({})).is_err());
        assert!(openai::parse_reply(&json!({"choices": [{"message": {"tool_calls": [{"function": {}}]}}]})).is_err());
    }

    #[test]
    fn a_text_only_model_gets_requests_without_images_after_refusing_one() {
        let (t, requests) = Scripted::new(vec![
            Ok((400, json!({"error": {"message": "image input is not supported", "type": "invalid_request_error"}}))),
            Ok((200, json!({"choices": [{"finish_reason": "stop", "message": {"content": "done"}}]}))),
            Ok((200, json!({"choices": [{"finish_reason": "stop", "message": {"content": "again"}}]}))),
        ]);
        let c = Client::new(openai_settings(), Box::new(t));
        let messages = vec![json!({"role": "user", "content": [{"type": "tool_result", "tool_use_id": "c", "content": [{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAA"}}]}]})];
        assert_eq!(c.complete("s", &[], &messages).unwrap().content[0]["text"], "done");
        assert_eq!(c.complete("s", &[], &messages).unwrap().content[0]["text"], "again");
        let r = requests.lock().unwrap();
        assert!(openai::has_images(&r[0]) && !openai::has_images(&r[1]) && !openai::has_images(&r[2]));
    }

    #[test]
    fn the_model_test_reports_who_answered() {
        let (t, requests) = Scripted::new(vec![Ok((200, json!({"model": "deepseek-chat", "choices": [{"finish_reason": "stop", "message": {"content": "OK"}}]})))]);
        let line = Client::new(openai_settings(), Box::new(t)).test().unwrap();
        assert!(line.starts_with("deepseek-chat answered \"OK\""), "{line}");
        assert!(requests.lock().unwrap()[0].get("tools").is_none());
        let (t, _) = Scripted::new(vec![Ok((200, text_reply("OK")))]);
        assert!(Client::new(settings(), Box::new(t)).test().unwrap().contains("answered \"OK\""));
        let (t, _) = Scripted::new(vec![Ok((404, json!({"error": {"message": "model not found"}})))]);
        assert!(Client::new(openai_settings(), Box::new(t)).test().unwrap_err().contains("404"));
    }

    /// Replies as event streams: each `post_stream` feeds the next list of lines (or answers
    /// whole); the requests are kept.
    /// An event stream's lines, or a whole reply with its status.
    type StreamScript = Result<Vec<String>, (u16, Value)>;

    struct Streams {
        replies: Mutex<std::collections::VecDeque<StreamScript>>,
        requests: Arc<Mutex<Vec<Value>>>,
    }

    impl Streams {
        fn new(replies: Vec<StreamScript>) -> (Self, Arc<Mutex<Vec<Value>>>) {
            let requests = Arc::new(Mutex::new(Vec::new()));
            (Self { replies: Mutex::new(replies.into()), requests: requests.clone() }, requests)
        }
    }

    impl Transport for Streams {
        fn post(&self, _: &str, _: &[(&str, &str)], _: &Value) -> Result<(u16, Value), String> {
            Err("not streamed".into())
        }

        fn post_stream(&self, _: &str, _: &[(&str, &str)], body: &Value, on_line: &mut dyn FnMut(&str) -> bool) -> Result<StreamReply, String> {
            self.requests.lock().unwrap().push(body.clone());
            match self.replies.lock().unwrap().pop_front().ok_or("script exhausted")? {
                Ok(lines) => {
                    for l in lines {
                        if !on_line(&l) {
                            return Ok(StreamReply::Lines { aborted: true });
                        }
                    }
                    Ok(StreamReply::Lines { aborted: false })
                }
                Err((status, v)) => Ok(StreamReply::Whole(status, v)),
            }
        }
    }

    fn sse(events: &[Value]) -> Vec<String> {
        events.iter().flat_map(|e| [format!("event: {}", e["type"].as_str().unwrap_or("")), format!("data: {e}"), String::new()]).collect()
    }

    fn collect(c: &Client) -> (Result<Reply, String>, Vec<(String, bool)>) {
        let mut seen = Vec::new();
        let r = c.complete_stream("s", &[], &[json!({"role": "user", "content": "hi"})], &mut |t, new_block| {
            if !t.is_empty() {
                seen.push((t.to_string(), new_block));
            }
            true
        });
        (r, seen)
    }

    #[test]
    fn a_messages_stream_shows_text_as_it_comes_and_rebuilds_the_message() {
        let lines = sse(&[
            json!({"type": "message_start", "message": {"id": "m", "type": "message", "role": "assistant", "model": "claude-opus-5-5", "content": [], "stop_reason": null, "usage": {"input_tokens": 12, "output_tokens": 1}}}),
            json!({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
            json!({"type": "ping"}),
            json!({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Brigh"}}),
            json!({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "tening."}}),
            json!({"type": "content_block_stop", "index": 0}),
            json!({"type": "content_block_start", "index": 1, "content_block": {"type": "tool_use", "id": "t1", "name": "command_run", "input": {}}}),
            json!({"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": "{\"id\": \"edit"}}),
            json!({"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": ".undo\"}"}}),
            json!({"type": "content_block_stop", "index": 1}),
            json!({"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 30}}),
            json!({"type": "message_stop"}),
        ]);
        let (t, requests) = Streams::new(vec![Ok(lines)]);
        let c = Client::new(settings(), Box::new(t));
        let (r, seen) = collect(&c);
        let r = r.unwrap();
        assert_eq!(seen, [("Brigh".to_string(), true), ("tening.".to_string(), false)]);
        assert_eq!(r.content[0], json!({"type": "text", "text": "Brightening."}));
        assert_eq!(r.content[1], json!({"type": "tool_use", "id": "t1", "name": "command_run", "input": {"id": "edit.undo"}}));
        assert_eq!((r.stop_reason.as_str(), r.input_tokens, r.output_tokens), ("tool_use", 12, 30));
        assert_eq!(requests.lock().unwrap()[0]["stream"], true);
    }

    #[test]
    fn a_chat_completions_stream_assembles_text_and_split_tool_calls() {
        let chunk = |delta: Value, finish: Value| format!("data: {}", json!({"model": "deepseek-chat", "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}));
        let lines = vec![
            chunk(json!({"role": "assistant", "content": ""}), Value::Null),
            chunk(json!({"content": "On "}), Value::Null),
            chunk(json!({"content": "it."}), Value::Null),
            chunk(json!({"tool_calls": [{"index": 0, "id": "call_1", "type": "function", "function": {"name": "doc_new", "arguments": ""}}]}), Value::Null),
            chunk(json!({"tool_calls": [{"index": 0, "function": {"arguments": "{\"width\""}}]}), Value::Null),
            chunk(json!({"tool_calls": [{"index": 0, "function": {"arguments": ":8}"}}]}), Value::Null),
            chunk(json!({}), json!("tool_calls")),
            format!("data: {}", json!({"model": "deepseek-chat", "choices": [], "usage": {"prompt_tokens": 9, "completion_tokens": 4}})),
            "data: [DONE]".into(),
        ];
        let (t, requests) = Streams::new(vec![Ok(lines)]);
        let c = Client::new(openai_settings(), Box::new(t));
        let (r, seen) = collect(&c);
        let r = r.unwrap();
        assert_eq!(seen, [("On ".to_string(), true), ("it.".to_string(), false)]);
        assert_eq!(r.content[0]["text"], "On it.");
        assert_eq!(r.content[1], json!({"type": "tool_use", "id": "call_1", "name": "doc_new", "input": {"width": 8}}));
        assert_eq!((r.stop_reason.as_str(), r.input_tokens, r.output_tokens), ("tool_use", 9, 4));
        let req = &requests.lock().unwrap()[0];
        assert_eq!((req["stream"].clone(), req["stream_options"]["include_usage"].clone()), (json!(true), json!(true)));
    }

    #[test]
    fn stopping_mid_stream_and_stream_errors_end_the_reply() {
        let lines = sse(&[
            json!({"type": "message_start", "message": {"content": [], "usage": {}}}),
            json!({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
            json!({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "a"}}),
            json!({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "b"}}),
        ]);
        let (t, _) = Streams::new(vec![Ok(lines)]);
        let c = Client::new(settings(), Box::new(t));
        let mut n = 0;
        let r = c.complete_stream("s", &[], &[], &mut |t, _| {
            n += usize::from(!t.is_empty());
            n < 1
        });
        assert_eq!(r.unwrap_err(), STOPPED);
        let (t, _) = Streams::new(vec![Ok(sse(&[json!({"type": "message_start", "message": {"content": []}}), json!({"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}})]))]);
        let e = collect(&Client::new(settings(), Box::new(t))).0.unwrap_err();
        assert!(e.contains("Overloaded"), "{e}");
        let (t, _) = Streams::new(vec![Ok(vec![])]);
        assert!(collect(&Client::new(settings(), Box::new(t))).0.unwrap_err().contains("without a message"));
        let bad_json = sse(&[
            json!({"type": "message_start", "message": {"content": []}}),
            json!({"type": "content_block_start", "index": 0, "content_block": {"type": "tool_use", "id": "t", "name": "x", "input": {}}}),
            json!({"type": "content_block_delta", "index": 0, "delta": {"type": "input_json_delta", "partial_json": "{\"a\":"}}),
            json!({"type": "content_block_stop", "index": 0}),
        ]);
        let (t, _) = Streams::new(vec![Ok(bad_json)]);
        assert!(collect(&Client::new(settings(), Box::new(t))).0.unwrap_err().contains("invalid JSON"));
        let (t, _) = Streams::new(vec![Ok(sse(&[json!({"type": "message_start", "message": {"content": []}}), json!({"type": "content_block_start", "index": 1_000_000, "content_block": {"type": "text"}})]))]);
        assert!(collect(&Client::new(settings(), Box::new(t))).0.is_err(), "a hostile index is an error, not an allocation");
    }

    #[test]
    fn a_server_that_answers_whole_still_shows_its_text() {
        let (t, _) = Scripted::new(vec![Ok((200, text_reply("whole reply")))]);
        let (r, seen) = collect(&Client::new(settings(), Box::new(t)));
        assert_eq!(r.unwrap().content[0]["text"], "whole reply");
        assert_eq!(seen, [("whole reply".to_string(), true)]);
    }

    #[test]
    fn a_server_without_stream_options_is_asked_again_without_them() {
        let done = vec![format!("data: {}", json!({"choices": [{"delta": {"content": "ok"}, "finish_reason": "stop"}]})), "data: [DONE]".into()];
        let (t, requests) = Streams::new(vec![Err((400, json!({"error": {"message": "unknown field stream_options"}}))), Ok(done)]);
        let c = Client::new(openai_settings(), Box::new(t));
        assert_eq!(collect(&c).0.unwrap().content[0]["text"], "ok");
        let r = requests.lock().unwrap();
        assert!(r[0].get("stream_options").is_some() && r[1].get("stream_options").is_none());
    }
}
