//! The in-app agent: the data the shell and the agent runner exchange.
//!
//! The runner (`photocraft-automation`, native only) is an LLM client whose tools are PhotoCraft's
//! own MCP tools; the shell (`photocraft-ui-egui`) draws the conversation in a side panel. Neither
//! may depend on the other (both are L6), so the messages between them live here: what the user
//! sends ([`AgentCommand`]), what the runner reports ([`AgentEvent`]), and the settings the runner
//! starts with ([`AgentSettings`], from Edit › Preferences › AI Integrations).

use std::sync::mpsc::{Receiver, Sender};

use serde::{Deserialize, Serialize};

use crate::prefs::Preferences;

/// The environment variable that supplies the API key when the preference is empty.
pub const API_KEY_ENV: &str = "ANTHROPIC_API_KEY";
/// The same for an OpenAI-compatible endpoint.
pub const OPENAI_API_KEY_ENV: &str = "OPENAI_API_KEY";

/// The API formats the panel speaks: Anthropic's Messages API (Claude), or the OpenAI Chat
/// Completions format that OpenAI and many other providers and local servers (DeepSeek, Qwen,
/// Mistral, Ollama, vLLM, LM Studio, …) accept.
pub const PROVIDERS: &[&str] = &["anthropic", "openai"];
pub const PROVIDER_ANTHROPIC: &str = "anthropic";
pub const PROVIDER_OPENAI: &str = "openai";
/// The OpenAI-compatible base URL used when the endpoint preference is empty (`/chat/completions`
/// is appended).
pub const OPENAI_DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";

/// The model the panel uses unless Preferences name another.
pub const DEFAULT_MODEL: &str = "claude-opus-5-5";
/// The default `output_config.effort`: the Opus 5.5 default is `medium`, one level below.
pub const DEFAULT_EFFORT: &str = "high";
/// The Messages API endpoint's origin.
pub const DEFAULT_BASE_URL: &str = "https://api.anthropic.com";
/// Tool-use rounds one user message may take before the runner stops and asks to continue.
pub const DEFAULT_MAX_TURNS: u32 = 40;
/// `effort` values the API accepts.
pub const EFFORT_LEVELS: &[&str] = &["low", "medium", "high", "xhigh", "max"];

/// What the runner starts with. Read from [`Integrations`] when the panel starts a session; a
/// change in Preferences takes effect with the next session (the panel's Clear).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct AgentSettings {
    /// [`PROVIDER_ANTHROPIC`] or [`PROVIDER_OPENAI`].
    pub provider: String,
    pub model: String,
    pub effort: String,
    /// The Messages API origin ([`DEFAULT_BASE_URL`]), or for OpenAI-compatible APIs the base
    /// URL `/chat/completions` is appended to ([`OPENAI_DEFAULT_BASE_URL`]).
    pub base_url: String,
    /// Empty: read [`API_KEY_ENV`] (or [`OPENAI_API_KEY_ENV`]) at session start. An
    /// OpenAI-compatible server may need none (a local Ollama).
    pub api_key: String,
    pub max_turns: u32,
    /// Opt into the server-side refusal fallbacks on the models that offer them.
    pub refusal_fallbacks: bool,
}

impl Default for AgentSettings {
    fn default() -> Self {
        Self {
            provider: PROVIDER_ANTHROPIC.into(),
            model: DEFAULT_MODEL.into(),
            effort: DEFAULT_EFFORT.into(),
            base_url: DEFAULT_BASE_URL.into(),
            api_key: String::new(),
            max_turns: DEFAULT_MAX_TURNS,
            refusal_fallbacks: true,
        }
    }
}

impl AgentSettings {
    /// The settings Preferences › AI Integrations hold, with empty fields filled from the
    /// defaults and [`API_KEY_ENV`].
    pub fn from_prefs(prefs: &Preferences) -> Self {
        let p = &prefs.integrations;
        let d = Self::default();
        let pick = |v: &str, default: &str| if v.trim().is_empty() { default.to_string() } else { v.trim().to_string() };
        let openai = p.agent_provider == PROVIDER_OPENAI;
        let (provider, model, base_url, key_env) = if openai {
            // No default model: every OpenAI-compatible provider names its own.
            (PROVIDER_OPENAI, p.agent_model.trim().to_string(), pick(&p.agent_base_url, OPENAI_DEFAULT_BASE_URL), OPENAI_API_KEY_ENV)
        } else {
            (PROVIDER_ANTHROPIC, pick(&p.agent_model, &d.model), pick(&p.agent_base_url, &d.base_url), API_KEY_ENV)
        };
        Self {
            provider: provider.into(),
            model,
            effort: if EFFORT_LEVELS.contains(&p.agent_effort.as_str()) { p.agent_effort.clone() } else { d.effort },
            base_url: base_url.trim_end_matches('/').to_string(),
            api_key: if p.agent_api_key.trim().is_empty() { std::env::var(key_env).unwrap_or_default().trim().to_string() } else { p.agent_api_key.trim().to_string() },
            max_turns: p.agent_max_turns.clamp(1, 1000),
            refusal_fallbacks: p.agent_refusal_fallbacks,
        }
    }

    /// Talks the OpenAI Chat Completions format.
    pub fn is_openai(&self) -> bool {
        self.provider == PROVIDER_OPENAI
    }

    /// Why a session can't start, if it can't (no key, an unusable endpoint).
    pub fn check(&self) -> Result<(), String> {
        if !PROVIDERS.contains(&self.provider.as_str()) {
            return Err(format!("unknown API format `{}` (anthropic or openai)", self.provider));
        }
        // OpenAI-compatible servers on this computer or the local network often take no key.
        if self.api_key.is_empty() && !(self.is_openai() && local_url(&self.base_url)) {
            let env = if self.is_openai() { OPENAI_API_KEY_ENV } else { API_KEY_ENV };
            return Err(format!("no API key: set one in Edit › Preferences › AI Integrations, or export {env} before launching"));
        }
        if self.api_key.chars().any(|c| c.is_control() || c.is_whitespace()) {
            return Err("the API key contains whitespace or control characters".into());
        }
        if !(self.base_url.starts_with("https://") || (self.base_url.starts_with("http://") && local_url(&self.base_url))) {
            return Err("the API endpoint must be an https:// URL (http:// is accepted for this computer and the local network only)".into());
        }
        if self.model.trim().is_empty() {
            return Err(if self.is_openai() {
                "no model: enter the provider's model id in Edit › Preferences › AI Integrations (for example gpt-4o, deepseek-chat, qwen-plus or llama3.1)".into()
            } else {
                "the model id is empty".into()
            });
        }
        if self.model.chars().any(|c| c.is_whitespace() || c.is_control()) {
            return Err("the model id contains whitespace".into());
        }
        Ok(())
    }
}

/// An `http(s)://` URL whose host is this computer or a private network address (a local model
/// server), where plain HTTP and no key are acceptable.
fn local_url(url: &str) -> bool {
    let Some(rest) = url.strip_prefix("http://").or_else(|| url.strip_prefix("https://")) else { return false };
    let authority = rest.split(['/', '?', '#']).next().unwrap_or("");
    let host = if let Some(v6) = authority.strip_prefix('[') { v6.split(']').next().unwrap_or("") } else { authority.rsplit_once(':').map_or(authority, |(h, _)| h) };
    if host.eq_ignore_ascii_case("localhost") {
        return true;
    }
    match host.parse::<std::net::IpAddr>() {
        Ok(std::net::IpAddr::V4(ip)) => ip.is_loopback() || ip.is_private() || ip.is_link_local(),
        Ok(std::net::IpAddr::V6(ip)) => ip.is_loopback() || (ip.segments()[0] & 0xfe00) == 0xfc00,
        Err(_) => false,
    }
}

/// What the panel asks the runner to do.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum AgentCommand {
    /// A user message; the runner answers it, calling tools as needed, until the model ends its turn.
    Send(String),
    /// Stop the current turn after the tool call in flight (the conversation is kept).
    Cancel,
}

/// A tool call the model made, as the panel shows it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolCallEvent {
    /// The MCP tool name (`layer_new_layer`, `ui_screenshot`, …).
    pub name: String,
    /// The arguments, as JSON text (bounded by the runner).
    pub input: String,
}

/// What the runner reports back. Every variant is safe to draw: text is bounded, images are PNG
/// bytes the shell decodes itself.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum AgentEvent {
    /// A one-line status for the panel's footer ("thinking…", "running layer_new_layer").
    Status(String),
    /// Text the model wrote for the user.
    AssistantText(String),
    /// The model called a tool.
    ToolCall(ToolCallEvent),
    /// A tool finished: `ok` false means the tool reported an error (the model sees it too).
    ToolResult {
        name: String,
        ok: bool,
        /// The result's text, trimmed for display.
        summary: String,
        /// A PNG the tool returned (a screenshot or render), if any.
        #[serde(skip_serializing_if = "Option::is_none")]
        png: Option<Vec<u8>>,
    },
    /// The model ended its turn (or the runner stopped it): the panel accepts the next message.
    TurnDone {
        /// Tokens read and written this turn, if the API reported them.
        input_tokens: u64,
        output_tokens: u64,
        /// The turn stopped early (cancelled, the turn limit, `max_tokens`, a refusal): why.
        #[serde(skip_serializing_if = "Option::is_none")]
        stopped: Option<String>,
    },
    /// The runner failed (a transport or API error); the turn is over.
    Error(String),
}

/// A running agent session: commands go in, events come out. Dropping the handle ends the
/// session once its current tool call returns.
pub struct AgentHandle {
    pub commands: Sender<AgentCommand>,
    pub events: Receiver<AgentEvent>,
}

/// Starts an agent session with these settings (the desktop app wires one; the web build has
/// none). Returns the error to show in the panel when the session can't start.
pub type AgentHostFn = Box<dyn FnMut(AgentSettings) -> Result<AgentHandle, String>>;

/// Tests a model with these settings (Preferences › AI Integrations › Test Model) on a
/// background thread: one short request, answered on the receiver with a line to show, or why
/// it failed.
pub type AgentTestFn = Box<dyn FnMut(AgentSettings) -> Receiver<Result<String, String>>>;

/// The port the MCP server for other programs listens on unless Preferences name another.
pub const DEFAULT_MCP_PORT: u16 = 8765;

/// The MCP server for other programs (Preferences › AI Integrations › MCP Server): the app's
/// tools at `http://127.0.0.1:<port>/mcp`, behind a bearer token.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct McpServerSettings {
    pub port: u16,
    /// Empty: the host generates one, and the shell stores it in Preferences.
    pub token: String,
}

impl McpServerSettings {
    /// The server Preferences ask for, or `None` when it is off.
    pub fn from_prefs(prefs: &Preferences) -> Option<Self> {
        let p = &prefs.integrations;
        p.mcp_server.then(|| Self {
            port: u16::try_from(p.mcp_port).ok().filter(|&port| port > 0).unwrap_or(DEFAULT_MCP_PORT),
            token: p.mcp_token.trim().to_string(),
        })
    }
}

/// A running MCP server, as the host reports it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct McpServerStatus {
    /// The endpoint (`http://127.0.0.1:8765/mcp`).
    pub url: String,
    /// The bearer token clients send (the one asked for, or the one generated).
    pub token: String,
}

/// Starts the MCP server with these settings, replacing a running one, or stops it (`None`).
/// The desktop app wires one; the web build has none.
pub type McpServerHostFn = Box<dyn FnMut(Option<McpServerSettings>) -> Result<Option<McpServerStatus>, String>>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prefs::Integrations;

    #[test]
    fn settings_fill_empty_fields_from_the_defaults() {
        let p = Integrations { agent_api_key: " k ".into(), agent_base_url: "https://proxy.example/".into(), agent_max_turns: 0, ..Default::default() };
        let s = AgentSettings::from_prefs(&Preferences { integrations: p, ..Default::default() });
        assert_eq!(s.model, DEFAULT_MODEL);
        assert_eq!(s.effort, DEFAULT_EFFORT);
        assert_eq!(s.base_url, "https://proxy.example");
        assert_eq!(s.api_key, "k");
        assert_eq!(s.max_turns, 1);
        assert!(s.check().is_ok());
    }

    #[test]
    fn settings_reject_what_the_api_would() {
        let mut s = AgentSettings { api_key: "k".into(), ..Default::default() };
        assert!(s.check().is_ok());
        s.base_url = "http://api.example".into();
        assert!(s.check().unwrap_err().contains("https://"));
        s.base_url = "http://localhost:8080".into();
        assert!(s.check().is_ok());
        s.base_url = DEFAULT_BASE_URL.into();
        s.model = "a b".into();
        assert!(s.check().unwrap_err().contains("model"));
        s.model = DEFAULT_MODEL.into();
        s.api_key = "k\n".into();
        assert!(s.check().unwrap_err().contains("whitespace"));
        s.api_key = String::new();
        assert!(s.check().unwrap_err().contains(API_KEY_ENV));
    }

    #[test]
    fn the_mcp_server_is_off_until_preferences_turn_it_on() {
        assert_eq!(McpServerSettings::from_prefs(&Preferences::default()), None);
        let with = |port: u32| Preferences { integrations: Integrations { mcp_server: true, mcp_port: port, mcp_token: " t ".into(), ..Default::default() }, ..Default::default() };
        assert_eq!(McpServerSettings::from_prefs(&with(9000)), Some(McpServerSettings { port: 9000, token: "t".into() }));
        assert_eq!(McpServerSettings::from_prefs(&with(0)).map(|s| s.port), Some(DEFAULT_MCP_PORT));
        assert_eq!(McpServerSettings::from_prefs(&with(70_000)).map(|s| s.port), Some(DEFAULT_MCP_PORT));
    }

    #[test]
    fn openai_compatible_settings_take_their_own_defaults() {
        let with = |i: Integrations| AgentSettings::from_prefs(&Preferences { integrations: i, ..Default::default() });
        let s = with(Integrations { agent_provider: "openai".into(), agent_api_key: "k".into(), ..Default::default() });
        assert!(s.is_openai());
        assert_eq!(s.base_url, OPENAI_DEFAULT_BASE_URL);
        assert!(s.check().unwrap_err().contains("no model"), "an OpenAI-compatible model must be named");
        let s = with(Integrations { agent_provider: "openai".into(), agent_model: "deepseek-chat".into(), agent_base_url: "https://api.deepseek.com/v1/".into(), agent_api_key: "k".into(), ..Default::default() });
        assert_eq!((s.model.as_str(), s.base_url.as_str()), ("deepseek-chat", "https://api.deepseek.com/v1"));
        assert!(s.check().is_ok());
        // A local server needs no key, and may be plain HTTP on the local network.
        for url in ["http://localhost:11434/v1", "http://127.0.0.1:8000/v1", "http://192.168.1.20:1234/v1", "http://[::1]:8080/v1"] {
            let s = AgentSettings { provider: "openai".into(), model: "llama3.1".into(), base_url: url.into(), api_key: String::new(), ..Default::default() };
            assert!(s.check().is_ok(), "{url}: {:?}", s.check());
        }
        let s = AgentSettings { provider: "openai".into(), model: "m".into(), base_url: "http://api.example.com/v1".into(), api_key: "k".into(), ..Default::default() };
        assert!(s.check().is_err());
        let s = AgentSettings { provider: "openai".into(), model: "m".into(), base_url: OPENAI_DEFAULT_BASE_URL.into(), api_key: String::new(), ..Default::default() };
        assert!(s.check().unwrap_err().contains(OPENAI_API_KEY_ENV));
        assert!(AgentSettings { provider: "gemini".into(), api_key: "k".into(), ..Default::default() }.check().is_err());
    }

    #[test]
    fn an_unknown_effort_falls_back_to_the_default() {
        let with = |effort: &str| Preferences { integrations: Integrations { agent_effort: effort.into(), ..Default::default() }, ..Default::default() };
        assert_eq!(AgentSettings::from_prefs(&with("turbo")).effort, DEFAULT_EFFORT);
        assert_eq!(AgentSettings::from_prefs(&with("low")).effort, "low");
    }
}
