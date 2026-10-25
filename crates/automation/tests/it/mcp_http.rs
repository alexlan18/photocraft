//! MCP over Streamable HTTP, driven the way another program would: plain HTTP requests with a
//! bearer token, JSON-RPC in, JSON or server-sent events out.

use std::net::SocketAddr;

use photocraft_automation::http::HttpServer;
use photocraft_automation::{PhotocraftMcp, security};
use serde_json::{Value, json};

struct Client {
    agent: ureq::Agent,
    url: String,
    token: String,
    session: Option<String>,
}

impl Client {
    fn post(&mut self, body: &Value, token: Option<&str>) -> (u16, Option<Value>) {
        let mut req = self.agent.post(&self.url).header("Accept", "application/json, text/event-stream").header("Content-Type", "application/json");
        if let Some(t) = token {
            req = req.header("Authorization", &format!("Bearer {t}"));
        }
        if let Some(s) = &self.session {
            req = req.header("Mcp-Session-Id", s);
        }
        let mut r = req.send(&serde_json::to_vec(body).unwrap()[..]).unwrap();
        if let Some(s) = r.headers().get("mcp-session-id").and_then(|v| v.to_str().ok()) {
            self.session = Some(s.to_string());
        }
        let status = r.status().as_u16();
        let text = r.body_mut().with_config().limit(64 << 20).read_to_string().unwrap_or_default();
        // A JSON body, or the first `data:` event carrying a JSON-RPC response.
        let value = serde_json::from_str(&text).ok().or_else(|| {
            text.lines().filter_map(|l| l.strip_prefix("data:")).filter_map(|d| serde_json::from_str::<Value>(d.trim()).ok()).find(|v| v.get("id").is_some())
        });
        (status, value)
    }

    fn rpc(&mut self, id: u64, method: &str, params: Value) -> Value {
        let token = self.token.clone();
        let (status, v) = self.post(&json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params}), Some(&token));
        assert_eq!(status, 200, "{method}: {v:?}");
        v.unwrap_or_else(|| panic!("{method}: no JSON-RPC reply"))
    }

    fn call(&mut self, id: u64, tool: &str, args: Value) -> Value {
        let r = self.rpc(id, "tools/call", json!({"name": tool, "arguments": args}));
        let result = &r["result"];
        assert_ne!(result["isError"], json!(true), "{tool}: {r}");
        let text = result["content"][0]["text"].as_str().unwrap_or("null");
        serde_json::from_str(text).unwrap_or(Value::String(text.to_string()))
    }
}

#[test]
fn another_program_traces_an_image_to_svg_over_http() {
    let token = security::generate_token().unwrap();
    let server = HttpServer::start(PhotocraftMcp::headless(), SocketAddr::from(([127, 0, 0, 1], 0)), token.clone()).unwrap();
    assert!(server.url().ends_with("/mcp"), "{}", server.url());
    let agent = ureq::Agent::config_builder().http_status_as_error(false).build().new_agent();
    let mut c = Client { agent, url: server.url(), token: token.clone(), session: None };

    // No token, or the wrong one: refused before any MCP handling.
    let init = json!({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "test", "version": "1"}}});
    assert_eq!(c.post(&init, None).0, 401);
    assert_eq!(c.post(&init, Some(&"0".repeat(64))).0, 401);

    let r = c.rpc(1, "initialize", init["params"].clone());
    assert!(r["result"]["serverInfo"].is_object(), "{r}");
    let (status, _) = c.post(&json!({"jsonrpc": "2.0", "method": "notifications/initialized"}), Some(&token));
    assert!(status == 202 || status == 200, "{status}");

    let tools = c.rpc(2, "tools/list", json!({}));
    let names: Vec<&str> = tools["result"]["tools"].as_array().unwrap().iter().filter_map(|t| t["name"].as_str()).collect();
    assert!(names.contains(&"file_export_svg") && names.contains(&"image_trace"), "{} tools", names.len());

    c.call(3, "file_new", json!({"width": 32, "height": 24}));
    let out = c.call(4, "file_export_svg", json!({"colors": 2}));
    let svg = out["svg"].as_str().unwrap_or_else(|| panic!("{out}"));
    assert!(svg.contains("<svg") && svg.contains("<path"), "{svg}");
    // The built-in tool does the same; asking to write a file without a write root is refused.
    let out = c.call(5, "image_to_svg", json!({"colors": 3, "tolerance": 2.0}));
    assert!(out["svg"].as_str().is_some_and(|s| s.contains("<path")), "{out}");
    let r = c.rpc(6, "tools/call", json!({"name": "image_to_svg", "arguments": {"path": "out.svg"}}));
    assert_eq!(r["result"]["isError"], json!(true), "{r}");
    server.stop();
}

#[test]
fn other_paths_are_not_found() {
    let token = security::generate_token().unwrap();
    let server = HttpServer::start(PhotocraftMcp::headless(), SocketAddr::from(([127, 0, 0, 1], 0)), token.clone()).unwrap();
    let agent = ureq::Agent::config_builder().http_status_as_error(false).build().new_agent();
    let r = agent.get(&format!("http://{}/", server.local_addr())).header("Authorization", &format!("Bearer {token}")).call().unwrap();
    assert_eq!(r.status().as_u16(), 404);
}
