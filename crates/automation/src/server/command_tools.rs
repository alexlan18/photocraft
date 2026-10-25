//! One MCP tool per engine command, so every feature has its own interface: the tool
//! `filter_blur_gaussianBlur` runs `filter.blur.gaussianBlur` with its arguments as the
//! command's params (one history step), exactly like `command_run {id, params}` does.
//!
//! Tool names are the command id with dots replaced by underscores (MCP tool-name grammars
//! don't all allow dots). A command whose derived name is also a built-in tool's name
//! (`command.list` → `command_list`) keeps the built-in, which does the same thing with a
//! richer schema. The description carries the label, the menu path, the shortcut and the
//! command's parameter doc; the input schema accepts any object, since the parameter docs are
//! prose, not schemas.

use std::collections::HashMap;
use std::sync::Arc;

use photocraft_engine::commands::{CommandSpec, command_specs, search_keywords};
use rmcp::model::{Tool, ToolAnnotations};
use serde_json::{Map, Value, json};

/// The MCP tool name of engine command `id`.
pub fn tool_name(id: &str) -> String {
    id.replace('.', "_")
}

/// The command tools of this server: everything in the registry except the names a built-in
/// tool already owns.
pub struct CommandTools {
    tools: Vec<Tool>,
    by_name: HashMap<String, &'static CommandSpec>,
}

impl CommandTools {
    /// `taken(name)` says whether a built-in tool owns `name`.
    pub fn new(taken: impl Fn(&str) -> bool) -> Self {
        let mut tools = Vec::new();
        let mut by_name = HashMap::new();
        for c in command_specs() {
            let name = tool_name(c.id);
            if taken(&name) || by_name.contains_key(&name) {
                continue;
            }
            tools.push(tool_for(&name, c));
            by_name.insert(name, c);
        }
        Self { tools, by_name }
    }

    pub fn tools(&self) -> &[Tool] {
        &self.tools
    }

    /// The command a tool runs, if `name` is one of the command tools.
    pub fn command(&self, name: &str) -> Option<&'static CommandSpec> {
        self.by_name.get(name).copied()
    }

    pub fn len(&self) -> usize {
        self.tools.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }
}

/// The tool's description: what a client searches (tool search matches names and
/// descriptions), so the label, the menu path and the parameter doc all go in.
pub fn describe(c: &CommandSpec) -> String {
    let mut d = String::with_capacity(128 + c.params.len());
    d.push_str(c.label);
    if !c.menu.is_empty() {
        d.push_str(" (menu ");
        d.push_str(&c.menu.join(" › "));
        d.push_str(" › ");
        d.push_str(c.label);
        d.push(')');
    }
    d.push_str(". Runs engine command `");
    d.push_str(c.id);
    d.push('`');
    if let Some(sc) = c.shortcut {
        d.push_str(", shortcut ");
        d.push_str(sc);
    }
    if c.params.trim() == "{}" {
        d.push_str(". No parameters.");
    } else {
        d.push_str(". Parameters (name:type=default, ? = optional): ");
        d.push_str(c.params);
    }
    if let Some(k) = search_keywords(c.id) {
        d.push_str(" Also known as: ");
        d.push_str(k);
    }
    d
}

fn schema(c: &CommandSpec) -> Map<String, Value> {
    let mut m = Map::new();
    m.insert("type".into(), json!("object"));
    m.insert("properties".into(), json!({}));
    if c.params.trim() == "{}" {
        m.insert("description".into(), json!("This command takes no parameters."));
        m.insert("additionalProperties".into(), json!(false));
    } else {
        m.insert("description".into(), json!("The command's parameters, as documented in the tool description."));
        m.insert("additionalProperties".into(), json!(true));
    }
    m
}

fn tool_for(name: &str, c: &CommandSpec) -> Tool {
    let read_only = !c.journal;
    let mut tool = Tool::new(name.to_string(), describe(c), Arc::new(schema(c)));
    tool.title = Some(c.label.to_string());
    tool.annotations = Some(ToolAnnotations::from_raw(Some(c.label.to_string()), Some(read_only), Some(!read_only), Some(read_only), Some(false)));
    tool
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_command_gets_a_tool_unless_a_builtin_owns_its_name() {
        let taken = |n: &str| n == "command_list";
        let tools = CommandTools::new(taken);
        assert_eq!(tools.len(), command_specs().len() - 1);
        assert!(tools.command("command_list").is_none());
        assert_eq!(tools.command("layer_new_layer").map(|c| c.id), Some("layer.new.layer"));
        for t in tools.tools() {
            assert!(t.name.len() <= 64 && t.name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_'), "{}", t.name);
            assert!(t.description.as_ref().is_some_and(|d| d.contains("Runs engine command")));
            assert_eq!(t.input_schema.get("type"), Some(&json!("object")));
        }
    }

    #[test]
    fn descriptions_carry_label_menu_shortcut_and_params() {
        let c = command_specs().iter().find(|c| c.id == "file.new").expect("file.new");
        let d = describe(c);
        assert!(d.starts_with("New…"), "{d}");
        assert!(d.contains("menu File › New…"), "{d}");
        assert!(d.contains("Runs engine command `file.new`"), "{d}");
        assert!(d.contains("Parameters"), "{d}");
    }
}
