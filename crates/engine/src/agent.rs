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
    pub model: String,
    pub effort: String,
    /// The Messages API origin; empty means [`DEFAULT_BASE_URL`].
    pub base_url: String,
    /// Empty: read [`API_KEY_ENV`] at session start.
    pub api_key: String,
    pub max_turns: u32,
    /// Opt into the server-side refusal fallbacks on the models that offer them.
    pub refusal_fallbacks: bool,
}

impl Default for AgentSettings {
    fn default() -> Self {
        Self {
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
        Self {
            model: pick(&prefs.integrations.agent_model, &d.model),
            effort: if EFFORT_LEVELS.contains(&prefs.integrations.agent_effort.as_str()) { p.agent_effort.clone() } else { d.effort },
            base_url: pick(&prefs.integrations.agent_base_url, &d.base_url).trim_end_matches('/').to_string(),
            api_key: if prefs.integrations.agent_api_key.trim().is_empty() {
                std::env::var(API_KEY_ENV).unwrap_or_default().trim().to_string()
            } else {
                p.agent_api_key.trim().to_string()
            },
            max_turns: prefs.integrations.agent_max_turns.clamp(1, 1000),
            refusal_fallbacks: prefs.integrations.agent_refusal_fallbacks,
        }
    }

    /// Why a session can't start, if it can't (no key, an unusable endpoint).
    pub fn check(&self) -> Result<(), String> {
        if self.api_key.is_empty() {
            return Err(format!("no API key: set one in Edit › Preferences › AI Integrations, or export {API_KEY_ENV} before launching"));
        }
        if self.api_key.chars().any(|c| c.is_control() || c.is_whitespace()) {
            return Err("the API key contains whitespace or control characters".into());
        }
        if !(self.base_url.starts_with("https://") || self.base_url.starts_with("http://localhost") || self.base_url.starts_with("http://127.0.0.1")) {
            return Err("the API endpoint must be an https:// URL (http:// is accepted for localhost only)".into());
        }
        if self.model.trim().is_empty() || self.model.chars().any(|c| c.is_whitespace()) {
            return Err("the model id is empty or contains whitespace".into());
        }
        Ok(())
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
    fn an_unknown_effort_falls_back_to_the_default() {
        let with = |effort: &str| Preferences { integrations: Integrations { agent_effort: effort.into(), ..Default::default() }, ..Default::default() };
        assert_eq!(AgentSettings::from_prefs(&with("turbo")).effort, DEFAULT_EFFORT);
        assert_eq!(AgentSettings::from_prefs(&with("low")).effort, "low");
    }
}
