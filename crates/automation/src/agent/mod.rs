//! The in-app agent (the Agent panel): a Claude client whose tools are PhotoCraft's own MCP
//! tools. The shell sends [`photocraft_engine::agent::AgentCommand`]s and draws the
//! [`photocraft_engine::agent::AgentEvent`]s; this module owns the model exchange and the
//! tool loop. Native only: it needs HTTPS and threads.

pub mod llm;
pub mod runner;

pub use runner::{SYSTEM_PROMPT, api_tools, spawn, spawn_with, system_prompt, test_model, test_model_with};
