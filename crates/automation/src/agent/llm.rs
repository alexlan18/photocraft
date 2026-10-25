//! The Claude Messages API over plain HTTPS (`POST /v1/messages`, non-streaming). The transport
//! is a trait so the runner's loop is tested with scripted replies and no network.

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
}

impl Client {
    pub fn new(settings: AgentSettings, transport: Box<dyn Transport>) -> Self {
        Self { settings, transport }
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
        let body = self.request_body(system, tools, messages);
        let url = format!("{}/v1/messages", self.settings.base_url);
        let mut headers: Vec<(&str, &str)> = vec![("x-api-key", self.settings.api_key.as_str()), ("anthropic-version", API_VERSION), ("content-type", "application/json")];
        if self.fallbacks() {
            headers.push(("anthropic-beta", FALLBACK_BETA));
        }
        let mut attempt = 0;
        loop {
            let outcome = self.transport.post(&url, &headers, &body);
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
            if status != 200 {
                let message = reply.pointer("/error/message").and_then(Value::as_str).unwrap_or("no details");
                let kind = reply.pointer("/error/type").and_then(Value::as_str).unwrap_or("error");
                return Err(match status {
                    401 => format!("the API rejected the key (401 {kind}): {message}"),
                    403 => format!("the API refused the request (403 {kind}): {message}"),
                    404 => format!("unknown model or endpoint (404 {kind}): {message}"),
                    429 => format!("rate limited (429): {message}"),
                    529 => format!("the API is overloaded (529): {message}"),
                    _ => format!("API error {status} {kind}: {message}"),
                });
            }
            return parse_reply(&reply);
        }
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
}
