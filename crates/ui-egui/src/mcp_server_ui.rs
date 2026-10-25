//! Preferences › AI Integrations › MCP server: the app's MCP tools for other programs (Claude
//! Desktop, Claude Code, Cursor, scripts) at `http://127.0.0.1:<port>/mcp`.
//!
//! [`sync`] runs every frame and starts, restarts or stops the server through
//! [`Services::mcp_server`](crate::Services::mcp_server) whenever the preferences ask for
//! something else than what was last applied. A failed start (a port in use) is reported once,
//! not retried every frame; changing the preferences tries again.

use photocraft_engine::agent::{McpServerSettings, McpServerStatus};
use serde_json::{Value, json};

use crate::PhotocraftApp;

/// What the server is doing.
#[derive(Default)]
pub struct McpServerRuntime {
    /// The settings last handed to the host (`Some(None)`: stopped on purpose).
    applied: Option<Option<McpServerSettings>>,
    /// The running server.
    pub status: Option<McpServerStatus>,
    /// Why the last start failed.
    pub error: Option<String>,
}

/// Brings the server in line with Preferences.
pub fn sync(app: &mut PhotocraftApp) {
    let want = McpServerSettings::from_prefs(app.session.prefs());
    if app.mcp_server.applied.as_ref() == Some(&want) {
        return;
    }
    let Some(host) = app.services.mcp_server.as_mut() else {
        app.mcp_server.error = want.as_ref().map(|_| "the MCP server needs the desktop app".to_string());
        app.mcp_server.applied = Some(want);
        return;
    };
    let mut applied = want.clone();
    match host(want) {
        Ok(Some(status)) => {
            // A generated token is kept, so client configurations stay valid across launches.
            if let Some(w) = applied.as_mut()
                && w.token.is_empty()
            {
                w.token = status.token.clone();
                let token = status.token.clone();
                app.session.prefs.edit(|p| p.integrations.mcp_token = token);
            }
            app.mcp_server.status = Some(status);
            app.mcp_server.error = None;
        }
        Ok(None) => {
            app.mcp_server.status = None;
            app.mcp_server.error = None;
        }
        Err(e) => {
            app.mcp_server.status = None;
            app.mcp_server.error = Some(e);
        }
    }
    app.mcp_server.applied = Some(applied);
}

/// The `mcpServers` entry MCP clients take (Claude Desktop, Claude Code `.mcp.json`, Cursor).
pub fn client_config(status: &McpServerStatus) -> String {
    let config = json!({"mcpServers": {"photocraft": {"type": "http", "url": status.url, "headers": {"Authorization": format!("Bearer {}", status.token)}}}});
    serde_json::to_string_pretty(&config).unwrap_or_default()
}

/// For `ui.inspect` and the Preferences dialog (no token: it is shown only where it is copied).
pub fn describe(app: &PhotocraftApp) -> Value {
    json!({
        "running": app.mcp_server.status.is_some(),
        "url": app.mcp_server.status.as_ref().map(|s| s.url.clone()),
        "error": app.mcp_server.error,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;
    use std::rc::Rc;

    fn app_with_host(calls: Rc<RefCell<Vec<Option<McpServerSettings>>>>, fail: bool) -> PhotocraftApp {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        app.services.mcp_server = Some(Box::new(move |s: Option<McpServerSettings>| {
            calls.borrow_mut().push(s.clone());
            if fail {
                return Err("port 8765 is in use".into());
            }
            Ok(s.map(|s| McpServerStatus {
                url: format!("http://127.0.0.1:{}/mcp", s.port),
                token: if s.token.is_empty() { "a".repeat(64) } else { s.token },
            }))
        }));
        app
    }

    #[test]
    fn the_server_follows_preferences_and_keeps_a_generated_token() {
        let calls = Rc::new(RefCell::new(Vec::new()));
        let mut app = app_with_host(calls.clone(), false);
        sync(&mut app);
        assert_eq!(*calls.borrow(), vec![None], "off by default");
        app.session.prefs.edit(|p| p.integrations.mcp_server = true);
        sync(&mut app);
        sync(&mut app);
        assert_eq!(calls.borrow().len(), 2, "started once");
        assert_eq!(app.session.prefs().integrations.mcp_token, "a".repeat(64));
        assert_eq!(describe(&app)["url"], json!("http://127.0.0.1:8765/mcp"));
        assert!(client_config(app.mcp_server.status.as_ref().unwrap()).contains("Bearer aaaa"));
        // Storing the generated token is not a change that restarts the server.
        sync(&mut app);
        assert_eq!(calls.borrow().len(), 2);
        app.session.prefs.edit(|p| p.integrations.mcp_port = 9000);
        sync(&mut app);
        assert_eq!(calls.borrow().last().cloned().flatten().map(|s| s.port), Some(9000));
        app.session.prefs.edit(|p| p.integrations.mcp_server = false);
        sync(&mut app);
        assert_eq!(calls.borrow().last().cloned(), Some(None));
        assert_eq!(describe(&app)["running"], json!(false));
    }

    #[test]
    fn a_failed_start_is_reported_once_not_retried_every_frame() {
        let calls = Rc::new(RefCell::new(Vec::new()));
        let mut app = app_with_host(calls.clone(), true);
        app.session.prefs.edit(|p| p.integrations.mcp_server = true);
        sync(&mut app);
        sync(&mut app);
        assert_eq!(calls.borrow().len(), 1);
        assert_eq!(describe(&app)["error"], json!("port 8765 is in use"));
    }

    #[test]
    fn without_a_host_turning_it_on_says_why() {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        app.session.prefs.edit(|p| p.integrations.mcp_server = true);
        sync(&mut app);
        assert!(describe(&app)["error"].as_str().is_some_and(|e| e.contains("desktop")));
    }
}
