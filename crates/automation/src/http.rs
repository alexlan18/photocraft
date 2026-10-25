//! MCP over Streamable HTTP, for other programs: Claude Desktop, Claude Code, Cursor, VS Code,
//! or a script, connect to `http://127.0.0.1:<port>/mcp` with `Authorization: Bearer <token>`.
//!
//! The same [`PhotocraftMcp`] tools as stdio (`photocraft-cli mcp`), over rmcp's Streamable
//! HTTP service behind a plain HTTP/1 listener. Every request needs the bearer token (the
//! loopback control channel's 256-bit token, compared in constant time). Requests from browser
//! pages are refused (any `Origin` header), and on a loopback bind only loopback `Host` names
//! are accepted, which stops DNS rebinding. Connections are capped at
//! [`security::MAX_CONNECTIONS`].
//!
//! [`HttpServer::start`] runs the listener on its own thread and runtime, so callers without
//! tokio (the desktop app) can start and stop it.

use std::convert::Infallible;
use std::net::SocketAddr;
use std::sync::Arc;

use bytes::Bytes;
use http_body_util::{BodyExt, Full, combinators::BoxBody};
use hyper::body::Incoming;
use hyper::{Request, Response, StatusCode};
use rmcp::transport::streamable_http_server::session::local::LocalSessionManager;
use rmcp::transport::{StreamableHttpServerConfig, StreamableHttpService};
use tokio::sync::watch;

use crate::{AutomationError, PhotocraftMcp, security};

/// The endpoint's path.
pub const MCP_PATH: &str = "/mcp";

type Body = BoxBody<Bytes, Infallible>;

/// A running HTTP MCP endpoint. Dropping it stops the listener and ends open sessions.
pub struct HttpServer {
    local: SocketAddr,
    stop: watch::Sender<bool>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl HttpServer {
    /// Binds `addr` (port 0 picks a free one) and serves `server`'s tools there until stopped.
    /// Binding happens before this returns, so a port in use is an error here, not later.
    pub fn start(server: PhotocraftMcp, addr: SocketAddr, token: String) -> Result<HttpServer, AutomationError> {
        security::validate_token(&token)?;
        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .thread_name("photocraft-mcp-http")
            .enable_all()
            .build()
            .map_err(|e| AutomationError::Io(e.to_string()))?;
        let listener = rt.block_on(tokio::net::TcpListener::bind(addr)).map_err(|e| AutomationError::Io(format!("can't listen on {addr}: {e}")))?;
        let local = listener.local_addr().map_err(|e| AutomationError::Io(e.to_string()))?;
        let (stop, stopped) = watch::channel(false);
        let thread = std::thread::Builder::new()
            .name("photocraft-mcp-http".into())
            .spawn(move || {
                rt.block_on(serve(listener, server, token, stopped));
                // Dropping the runtime ends the open sessions' tasks. Stopping may run on the UI
                // thread (a Preferences change), so a tool call waiting on that thread is not
                // waited for long: its blocking task finishes on its own.
                rt.shutdown_timeout(std::time::Duration::from_millis(500));
            })
            .map_err(|e| AutomationError::Io(e.to_string()))?;
        Ok(HttpServer { local, stop, thread: Some(thread) })
    }

    /// The bound address (with the real port when started on port 0).
    pub fn local_addr(&self) -> SocketAddr {
        self.local
    }

    /// The endpoint URL clients connect to.
    pub fn url(&self) -> String {
        format!("http://{}{MCP_PATH}", self.local)
    }

    /// Stops the listener and waits for its thread.
    pub fn stop(mut self) {
        self.shutdown();
    }

    fn shutdown(&mut self) {
        let _ = self.stop.send(true);
        if let Some(t) = self.thread.take() {
            // A panicked server thread has nothing left to clean up.
            let _ = t.join();
        }
    }
}

impl Drop for HttpServer {
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// rmcp's config for this endpoint: browser origins refused, loopback `Host` names only on a
/// loopback bind (any host on another address, where the token is the only gate).
fn config(local: SocketAddr) -> StreamableHttpServerConfig {
    let config = StreamableHttpServerConfig::default().enforce_origin_validation();
    if local.ip().is_loopback() { config } else { config.disable_allowed_hosts() }
}

/// Accepts connections until `stopped` turns true.
pub async fn serve(listener: tokio::net::TcpListener, server: PhotocraftMcp, token: String, mut stopped: watch::Receiver<bool>) {
    let local = listener.local_addr().unwrap_or_else(|_| SocketAddr::from(([127, 0, 0, 1], 0)));
    let service = StreamableHttpService::new(move || Ok(server.clone()), Arc::new(LocalSessionManager::default()), config(local));
    let limiter = security::ConnectionLimiter::new(security::MAX_CONNECTIONS);
    let token: Arc<str> = token.to_ascii_lowercase().into();
    loop {
        let accepted = tokio::select! {
            a = listener.accept() => a,
            _ = stopped.changed() => return,
        };
        let Ok((stream, _)) = accepted else { continue };
        // Over the cap: close at once rather than queue (the client retries).
        let Some(permit) = limiter.try_acquire() else { continue };
        let (service, token, mut stopped) = (service.clone(), token.clone(), stopped.clone());
        tokio::spawn(async move {
            let _permit = permit;
            let handler = hyper::service::service_fn(move |req: Request<Incoming>| {
                let (service, token) = (service.clone(), token.clone());
                async move { Ok::<_, Infallible>(route(&service, &token, req).await) }
            });
            let conn = hyper::server::conn::http1::Builder::new().serve_connection(hyper_util::rt::TokioIo::new(stream), handler);
            tokio::select! {
                _ = conn => {}
                _ = stopped.changed() => {}
            }
        });
    }
}

async fn route(service: &StreamableHttpService<PhotocraftMcp, LocalSessionManager>, token: &str, req: Request<Incoming>) -> Response<Body> {
    if req.uri().path() != MCP_PATH {
        return plain(StatusCode::NOT_FOUND, "not found: the MCP endpoint is /mcp");
    }
    if !authorized(&req, token) {
        let mut r = plain(StatusCode::UNAUTHORIZED, "missing or wrong bearer token");
        if let Ok(v) = hyper::header::HeaderValue::from_str("Bearer") {
            r.headers_mut().insert(hyper::header::WWW_AUTHENTICATE, v);
        }
        return r;
    }
    service.handle(req).await
}

/// `Authorization: Bearer <token>`, the scheme in any case.
fn authorized<B>(req: &Request<B>, token: &str) -> bool {
    let Some(v) = req.headers().get(hyper::header::AUTHORIZATION).and_then(|v| v.to_str().ok()) else { return false };
    let Some((scheme, supplied)) = v.trim().split_once(' ') else { return false };
    scheme.eq_ignore_ascii_case("bearer") && security::token_matches(token, supplied.trim())
}

fn plain(status: StatusCode, text: &'static str) -> Response<Body> {
    let mut r = Response::new(Full::new(Bytes::from_static(text.as_bytes())).boxed());
    *r.status_mut() = status;
    r
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_bearer_token_is_required() {
        let token = security::generate_token().unwrap();
        let with = |v: &str| Request::builder().header("authorization", v).body(()).unwrap();
        assert!(authorized(&with(&format!("Bearer {token}")), &token));
        assert!(authorized(&with(&format!("bearer  {}", token.to_ascii_uppercase())), &token));
        assert!(!authorized(&with(&format!("Basic {token}")), &token));
        assert!(!authorized(&with("Bearer 00"), &token));
        assert!(!authorized(&Request::builder().body(()).unwrap(), &token));
    }

    #[test]
    fn a_bad_token_is_refused_before_binding() {
        let e = HttpServer::start(PhotocraftMcp::headless(), SocketAddr::from(([127, 0, 0, 1], 0)), "short".into()).err().unwrap();
        assert!(!e.to_string().is_empty());
    }
}
