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

/// A POST of JSON to a URL: the status and the parsed body (or the error text).
pub trait Transport: Send {
    fn post(&self, url: &str, headers: &[(&str, &str)], body: &Value) -> Result<(u16, Value), String>;
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
}

impl Client {
    pub fn new(settings: AgentSettings, transport: Box<dyn Transport>) -> Self {
        Self { settings, transport, no_images: std::sync::atomic::AtomicBool::new(false) }
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
            // Messages and OpenAI-style errors share `error.message`; some servers send a string.
            let message = reply
                .pointer("/error/message")
                .and_then(Value::as_str)
                .or_else(|| reply.get("error").and_then(Value::as_str))
                .or_else(|| reply.get("message").and_then(Value::as_str))
                .or_else(|| reply.get("raw").and_then(Value::as_str))
                .unwrap_or("no details");
            let kind = reply.pointer("/error/type").and_then(Value::as_str).unwrap_or("error");
            return Err(match status {
                400 => format!("API error 400 {kind}: {message}"),
                401 => format!("the API rejected the key (401 {kind}): {message}"),
                403 => format!("the API refused the request (403 {kind}): {message}"),
                404 => format!("unknown model or endpoint (404 {kind}): {message}"),
                429 => format!("rate limited (429): {message}"),
                529 => format!("the API is overloaded (529): {message}"),
                _ => format!("API error {status} {kind}: {message}"),
            });
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
}
