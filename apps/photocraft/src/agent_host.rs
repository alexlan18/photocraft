//! The Agent panel's session factory: `photocraft-automation`'s runner, talking to this app's
//! own MCP server, whose tools reach the window through the control-request queue (the same
//! path the `--control` socket uses, without a socket). The runner thread queues a request,
//! the frame loop answers it, the runner carries on; the UI thread never waits on the agent.

use std::net::{Ipv4Addr, SocketAddr};
use std::sync::Arc;
use std::sync::mpsc::Sender;
use std::time::{Duration, Instant};

use photocraft_automation::http::HttpServer;
use photocraft_automation::{AutomationError, LocalControl, PhotocraftMcp, security};
use photocraft_engine::agent::{AgentHandle, AgentSettings, McpServerHostFn, McpServerSettings, McpServerStatus};
use photocraft_ui_egui::control::{ControlRequest, REPLY_DEADLINE};
use serde_json::{Value, json};

/// A request past this is abandoned: a filter or Content-Aware Fill the frame loop runs as a
/// background job replies when the job ends, which can take minutes on a large image.
const CALL_LIMIT: Duration = Duration::from_secs(15 * 60);

/// The in-process control endpoint.
pub struct UiControl {
    tx: Sender<ControlRequest>,
    ctx: egui::Context,
}

impl UiControl {
    pub fn new(tx: Sender<ControlRequest>, ctx: egui::Context) -> Self {
        Self { tx, ctx }
    }
}

impl LocalControl for UiControl {
    fn call(&self, method: &str, params: Value) -> Result<Value, AutomationError> {
        let (mut req, rx) = ControlRequest::new(method, params);
        // Queue expiry as for the socket: a request still queued after this is dropped, so a
        // frozen frame loop fails the call instead of applying it much later.
        req.deadline = Some(Instant::now() + REPLY_DEADLINE);
        let dispatch = req.dispatch.clone();
        if self.tx.send(req).is_err() {
            return Err(AutomationError::App("the app is shutting down".into()));
        }
        self.ctx.request_repaint();
        let reply = rx.recv_timeout(CALL_LIMIT).unwrap_or_else(|_| json!({"ok": false, "error": dispatch.give_up()}));
        reply_result(reply)
    }
}

/// The control envelope's `result`, or its `error`.
fn reply_result(reply: Value) -> Result<Value, AutomationError> {
    if reply.get("ok").and_then(Value::as_bool).unwrap_or(false) {
        Ok(reply.get("result").cloned().unwrap_or(Value::Null))
    } else {
        Err(AutomationError::App(reply.get("error").and_then(Value::as_str).unwrap_or("the app returned no result").to_string()))
    }
}

/// The session factory for `Services::agent_host`.
pub fn host(tx: Sender<ControlRequest>, ctx: egui::Context) -> photocraft_engine::agent::AgentHostFn {
    Box::new(move |settings: AgentSettings| -> Result<AgentHandle, String> {
        let control: Arc<dyn LocalControl> = Arc::new(UiControl::new(tx.clone(), ctx.clone()));
        let server = PhotocraftMcp::local(control);
        let wake_ctx = ctx.clone();
        let wake: Arc<dyn Fn() + Send + Sync> = Arc::new(move || wake_ctx.request_repaint());
        photocraft_automation::agent::spawn(settings, server, wake)
    })
}

/// The host for `Services::mcp_server`: the same tools as the Agent panel, served to other
/// programs over HTTP on 127.0.0.1 (Preferences › AI Integrations › MCP server). Each call
/// stops the running server, then starts one with the new settings (or none).
pub fn mcp_server(tx: Sender<ControlRequest>, ctx: egui::Context) -> McpServerHostFn {
    let mut running: Option<HttpServer> = None;
    Box::new(move |settings: Option<McpServerSettings>| -> Result<Option<McpServerStatus>, String> {
        if let Some(server) = running.take() {
            server.stop();
        }
        let Some(settings) = settings else { return Ok(None) };
        let token = if settings.token.is_empty() { security::generate_token().map_err(|e| e.to_string())? } else { settings.token.to_ascii_lowercase() };
        security::validate_token(&token).map_err(|_| "the MCP server token must be 64 hexadecimal digits (clear it to generate one)".to_string())?;
        let control: Arc<dyn LocalControl> = Arc::new(UiControl::new(tx.clone(), ctx.clone()));
        let server = HttpServer::start(PhotocraftMcp::local(control), SocketAddr::from((Ipv4Addr::LOCALHOST, settings.port)), token.clone()).map_err(|e| e.to_string())?;
        let url = server.url();
        running = Some(server);
        Ok(Some(McpServerStatus { url, token }))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn replies_split_into_result_and_error() {
        assert_eq!(reply_result(json!({"ok": true, "result": {"a": 1}})).unwrap(), json!({"a": 1}));
        assert_eq!(reply_result(json!({"ok": true})).unwrap(), Value::Null);
        let e = reply_result(json!({"ok": false, "error": "no document open"})).unwrap_err();
        assert!(e.to_string().contains("no document open"), "{e}");
        assert!(reply_result(json!({})).is_err());
    }

    #[test]
    fn a_dropped_queue_is_an_error_not_a_hang() {
        let (tx, rx) = std::sync::mpsc::channel::<ControlRequest>();
        drop(rx);
        let c = UiControl::new(tx, egui::Context::default());
        let e = c.call("ui.inspect", json!({})).unwrap_err();
        assert!(e.to_string().contains("shutting down"), "{e}");
    }

    #[test]
    fn the_mcp_server_starts_restarts_and_stops() {
        let (tx, _rx) = std::sync::mpsc::channel::<ControlRequest>();
        let mut host = mcp_server(tx, egui::Context::default());
        // Port 0 is not a preference value, but lets the test pick a free port.
        let started = host(Some(McpServerSettings { port: 0, token: String::new() })).unwrap().unwrap();
        assert!(started.url.starts_with("http://127.0.0.1:") && started.url.ends_with("/mcp"), "{}", started.url);
        assert!(security::validate_token(&started.token).is_ok());
        let again = host(Some(McpServerSettings { port: 0, token: started.token.clone() })).unwrap().unwrap();
        assert_eq!(again.token, started.token);
        let e = host(Some(McpServerSettings { port: 0, token: "nope".into() })).unwrap_err();
        assert!(e.contains("64 hexadecimal"), "{e}");
        assert_eq!(host(None).unwrap(), None);
    }

    #[test]
    fn the_frame_loop_answers_through_the_queue() {
        let (tx, rx) = std::sync::mpsc::channel::<ControlRequest>();
        let c = UiControl::new(tx, egui::Context::default());
        let answerer = std::thread::spawn(move || {
            let req = rx.recv().unwrap();
            assert_eq!(req.method, "engine.commands");
            req.reply.send(json!({"ok": true, "result": [1, 2]})).unwrap();
        });
        assert_eq!(c.call("engine.commands", json!({})).unwrap(), json!([1, 2]));
        answerer.join().unwrap();
    }
}
