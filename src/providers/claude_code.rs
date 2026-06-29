//! Claude Code adapter.
//!
//! Storage: `~/.claude/projects/<cwd-slug>/<sessionId>.jsonl`, one JSONL file
//! per session. Each line is a tagged-union event keyed on `type`. The format
//! drifts across versions (new fields, new `type` values), so parsing is
//! defensive: unknown fields are ignored, unknown event types become
//! [`Record::Unknown`], and malformed lines become [`Record::Unparsed`] rather
//! than aborting the file.

use std::fs;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;

use chrono::{DateTime, Utc};
use serde::Deserialize;
use serde_json::Value;

use crate::model::{Block, Message, Provider, Result, Role, Session, SessionRef, TokenTotals};
use crate::providers::{file_size, home, mtime};

pub struct ClaudeCode {
    root: PathBuf,
}

impl ClaudeCode {
    pub fn new() -> ClaudeCode {
        ClaudeCode {
            root: home().join(".claude").join("projects"),
        }
    }
}

impl Provider for ClaudeCode {
    fn id(&self) -> &'static str {
        "claude-code"
    }

    fn detect(&self) -> bool {
        self.root.is_dir()
    }

    fn discover(&self) -> Result<Vec<SessionRef>> {
        let mut refs = Vec::new();
        for project in fs::read_dir(&self.root)? {
            let project = project?;
            if !project.file_type()?.is_dir() {
                continue;
            }
            for file in fs::read_dir(project.path())? {
                let path = file?.path();
                if path.extension().and_then(|e| e.to_str()) == Some("jsonl") {
                    refs.push(SessionRef::File(path));
                }
            }
        }
        Ok(refs)
    }

    fn parse(&self, session_ref: &SessionRef) -> Result<Session> {
        let SessionRef::File(path) = session_ref else {
            return Err("claude-code expects a file session ref".into());
        };

        let id = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_string();
        let mut s = Session::new(self.id(), id);
        s.bytes = file_size(path);
        s.mtime = mtime(path);

        let mut ai_title: Option<String> = None;
        let mut custom_title: Option<String> = None;
        let mut tokens = TokenTotals::default();

        let reader = BufReader::new(fs::File::open(path)?);
        for line in reader.lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            match parse_line(&line) {
                Record::User(e) => {
                    s.observe_ts(e.common.timestamp);
                    if s.workspace.is_none() {
                        s.workspace = e.common.cwd.clone();
                    }
                    if e.common.git_branch.is_some() {
                        s.branch = e.common.git_branch.clone();
                    }
                    if !e.common.is_meta {
                        s.counts.user += 1;
                        let text = e.message.content.as_text();
                        let text = text.trim();
                        if !text.is_empty() {
                            s.text_chars += text.len() as u64;
                            if s.first_prompt.is_none() {
                                s.first_prompt = Some(text.to_string());
                            }
                            s.last_prompt.get_or_insert_with(|| text.to_string());
                        }
                    }
                }
                Record::Assistant(e) => {
                    s.observe_ts(e.common.timestamp);
                    s.counts.assistant += 1;
                    if let Some(m) = &e.message.model {
                        s.model = Some(m.clone());
                    }
                    if let Some(u) = &e.message.usage {
                        tokens.input += u.input_tokens;
                        tokens.output += u.output_tokens;
                        tokens.cache_creation += u.cache_creation_input_tokens;
                        tokens.cache_read += u.cache_read_input_tokens;
                    }
                    for block in &e.message.content {
                        match block {
                            ContentBlock::Text { text } => s.text_chars += text.len() as u64,
                            ContentBlock::ToolUse { .. } => s.counts.tool += 1,
                            _ => {}
                        }
                    }
                }
                Record::System(e) => s.observe_ts(e.common.timestamp),
                Record::Attachment(e) => s.observe_ts(e.common.timestamp),
                Record::AiTitle(e) => ai_title = Some(e.ai_title),
                Record::CustomTitle(e) => custom_title = Some(e.custom_title),
                Record::AgentName(e) => s.agent_name = Some(e.agent_name),
                Record::LastPrompt(e) => {
                    if let Some(t) = e.text() {
                        s.last_prompt = Some(t);
                    }
                }
                Record::QueueOperation(e) => s.observe_ts(e.timestamp),
                Record::Mode(_) | Record::PermissionMode(_) | Record::FileHistorySnapshot(_) => {}
                Record::Unknown { .. } => s.health.unknown += 1,
                Record::Unparsed { .. } => s.health.unparsed += 1,
            }
        }

        // A user-set title beats the AI-generated one for recognition.
        s.title = custom_title.or(ai_title);
        s.tokens = Some(tokens);
        Ok(s)
    }

    fn transcript(&self, session_ref: &SessionRef) -> Result<Vec<Message>> {
        let SessionRef::File(path) = session_ref else {
            return Err("claude-code expects a file session ref".into());
        };

        let mut out = Vec::new();
        let reader = BufReader::new(fs::File::open(path)?);
        for line in reader.lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            match parse_line(&line) {
                Record::User(e) => {
                    if e.common.is_meta {
                        continue; // synthetic/command-output turns, not conversation
                    }
                    let blocks = user_blocks(&e.message.content);
                    if blocks.is_empty() {
                        continue;
                    }
                    // A "user" turn that is purely tool output reads better as Tool.
                    let role = if blocks.iter().all(|b| matches!(b, Block::ToolResult { .. })) {
                        Role::Tool
                    } else {
                        Role::User
                    };
                    out.push(Message { role, timestamp: e.common.timestamp, blocks });
                }
                Record::Assistant(e) => {
                    let blocks = map_blocks(&e.message.content);
                    if blocks.is_empty() {
                        continue;
                    }
                    out.push(Message { role: Role::Assistant, timestamp: e.common.timestamp, blocks });
                }
                _ => {}
            }
        }
        Ok(out)
    }
}

/// Map Claude content blocks to canonical transcript blocks.
fn map_blocks(content: &[ContentBlock]) -> Vec<Block> {
    content
        .iter()
        .filter_map(|b| match b {
            ContentBlock::Text { text } if !text.trim().is_empty() => Some(Block::Text(text.clone())),
            ContentBlock::Thinking { thinking } if !thinking.trim().is_empty() => {
                Some(Block::Thinking(thinking.clone()))
            }
            ContentBlock::ToolUse { name, input, .. } => Some(Block::ToolUse {
                name: name.clone().unwrap_or_else(|| "?".into()),
                input: compact_json(input),
            }),
            ContentBlock::ToolResult { content, is_error, .. } => Some(Block::ToolResult {
                text: value_text(content),
                is_error: *is_error,
            }),
            _ => None,
        })
        .collect()
}

fn user_blocks(content: &MessageContent) -> Vec<Block> {
    match content {
        MessageContent::Text(s) if !s.trim().is_empty() => vec![Block::Text(s.clone())],
        MessageContent::Text(_) => Vec::new(),
        MessageContent::Blocks(bs) => map_blocks(bs),
    }
}

/// One-line JSON for a tool input, for compact display.
fn compact_json(v: &Value) -> String {
    if v.is_null() {
        String::new()
    } else {
        serde_json::to_string(v).unwrap_or_default()
    }
}

/// Pull readable text out of a tool_result `content` (string, or array of
/// `{type, text}` blocks, or arbitrary JSON).
fn value_text(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Null => String::new(),
        Value::Array(items) => items
            .iter()
            .map(|x| match x {
                Value::String(s) => s.clone(),
                _ => x.get("text").and_then(Value::as_str).map(str::to_string).unwrap_or_default(),
            })
            .collect::<Vec<_>>()
            .join("\n"),
        other => other.to_string(),
    }
}

// ---------------------------------------------------------------------------
// Wire model: the on-disk JSONL event types.
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub enum Record {
    User(Box<UserEvent>),
    Assistant(Box<AssistantEvent>),
    System(Box<SystemEvent>),
    Attachment(Box<AttachmentEvent>),
    AiTitle(AiTitleEvent),
    CustomTitle(CustomTitleEvent),
    AgentName(AgentNameEvent),
    LastPrompt(LastPromptEvent),
    Mode(ModeEvent),
    PermissionMode(PermissionModeEvent),
    QueueOperation(QueueOperationEvent),
    FileHistorySnapshot(FileHistorySnapshotEvent),
    Unknown { kind: String, raw: Value },
    Unparsed { error: String, raw: String },
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Common {
    pub uuid: Option<String>,
    pub parent_uuid: Option<String>,
    pub timestamp: Option<DateTime<Utc>>,
    pub session_id: Option<String>,
    pub cwd: Option<String>,
    pub git_branch: Option<String>,
    pub version: Option<String>,
    #[serde(default)]
    pub is_sidechain: bool,
    #[serde(default)]
    pub is_meta: bool,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct UserEvent {
    #[serde(flatten)]
    pub common: Common,
    pub message: UserMessage,
    pub prompt_id: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct UserMessage {
    pub role: Option<String>,
    pub content: MessageContent,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Blocks(Vec<ContentBlock>),
}

impl MessageContent {
    pub fn as_text(&self) -> String {
        match self {
            MessageContent::Text(s) => s.clone(),
            MessageContent::Blocks(bs) => bs
                .iter()
                .filter_map(|b| match b {
                    ContentBlock::Text { text } => Some(text.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("\n"),
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AssistantEvent {
    #[serde(flatten)]
    pub common: Common,
    pub message: AssistantMessage,
    pub request_id: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct AssistantMessage {
    pub id: Option<String>,
    pub model: Option<String>,
    #[serde(default)]
    pub content: Vec<ContentBlock>,
    pub usage: Option<Usage>,
    pub stop_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlock {
    Text {
        #[serde(default)]
        text: String,
    },
    Thinking {
        #[serde(default)]
        thinking: String,
    },
    ToolUse {
        id: Option<String>,
        name: Option<String>,
        #[serde(default)]
        input: Value,
    },
    ToolResult {
        tool_use_id: Option<String>,
        #[serde(default)]
        content: Value,
        #[serde(default)]
        is_error: bool,
    },
    #[serde(other)]
    Other,
}

#[derive(Debug, Default, Deserialize)]
pub struct Usage {
    #[serde(default)]
    pub input_tokens: u64,
    #[serde(default)]
    pub output_tokens: u64,
    #[serde(default)]
    pub cache_creation_input_tokens: u64,
    #[serde(default)]
    pub cache_read_input_tokens: u64,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SystemEvent {
    #[serde(flatten)]
    pub common: Common,
    pub subtype: Option<String>,
    pub content: Option<Value>,
    pub level: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AttachmentEvent {
    #[serde(flatten)]
    pub common: Common,
    pub attachment: Option<Value>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AiTitleEvent {
    pub ai_title: String,
    pub session_id: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LastPromptEvent {
    #[serde(default)]
    pub last_prompt: Option<Value>,
    pub leaf_uuid: Option<String>,
    pub session_id: Option<String>,
}

impl LastPromptEvent {
    pub fn text(&self) -> Option<String> {
        self.last_prompt
            .as_ref()
            .and_then(Value::as_str)
            .map(str::to_string)
    }
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CustomTitleEvent {
    pub custom_title: String,
    pub session_id: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AgentNameEvent {
    pub agent_name: String,
    pub session_id: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModeEvent {
    pub mode: Option<String>,
    pub session_id: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PermissionModeEvent {
    pub permission_mode: Option<String>,
    pub session_id: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct QueueOperationEvent {
    pub operation: Option<String>,
    pub timestamp: Option<DateTime<Utc>>,
    pub session_id: Option<String>,
    pub content: Option<Value>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FileHistorySnapshotEvent {
    pub message_id: Option<String>,
    pub snapshot: Option<Value>,
    #[serde(default)]
    pub is_snapshot_update: bool,
}

/// Parse one JSONL line, never failing: bad JSON and unmodeled types are
/// captured rather than propagated.
pub fn parse_line(line: &str) -> Record {
    let value: Value = match serde_json::from_str(line) {
        Ok(v) => v,
        Err(e) => {
            return Record::Unparsed {
                error: e.to_string(),
                raw: line.to_string(),
            }
        }
    };

    let kind = value
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();

    fn de<T, F>(value: Value, line: &str, wrap: F) -> Record
    where
        T: serde::de::DeserializeOwned,
        F: FnOnce(T) -> Record,
    {
        match serde_json::from_value::<T>(value) {
            Ok(t) => wrap(t),
            Err(e) => {
                if std::env::var_os("SSFT_DEBUG").is_some() {
                    eprintln!("UNPARSED: {e}\n  line: {}", &line[..line.len().min(300)]);
                }
                Record::Unparsed {
                    error: e.to_string(),
                    raw: line.to_string(),
                }
            }
        }
    }

    match kind.as_str() {
        "user" => de(value, line, |e| Record::User(Box::new(e))),
        "assistant" => de(value, line, |e| Record::Assistant(Box::new(e))),
        "system" => de(value, line, |e| Record::System(Box::new(e))),
        "attachment" => de(value, line, |e| Record::Attachment(Box::new(e))),
        "ai-title" => de(value, line, Record::AiTitle),
        "custom-title" => de(value, line, Record::CustomTitle),
        "agent-name" => de(value, line, Record::AgentName),
        "last-prompt" => de(value, line, Record::LastPrompt),
        "mode" => de(value, line, Record::Mode),
        "permission-mode" => de(value, line, Record::PermissionMode),
        "queue-operation" => de(value, line, Record::QueueOperation),
        "file-history-snapshot" => de(value, line, Record::FileHistorySnapshot),
        _ => Record::Unknown { kind, raw: value },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::Block;
    use std::io::Write;

    /// Write a small JSONL fixture and confirm transcript() maps the content
    /// blocks (text, thinking, tool_use, tool_result) and roles correctly.
    #[test]
    fn transcript_maps_blocks_and_roles() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("ssft_cc_{}.jsonl", std::process::id()));
        let mut f = fs::File::create(&path).unwrap();
        writeln!(f, r#"{{"type":"user","timestamp":"2026-06-28T10:00:00Z","message":{{"role":"user","content":"Fix the bug"}}}}"#).unwrap();
        writeln!(f, r#"{{"type":"assistant","timestamp":"2026-06-28T10:00:05Z","message":{{"role":"assistant","model":"claude-x","content":[{{"type":"thinking","thinking":"hmm"}},{{"type":"text","text":"On it."}},{{"type":"tool_use","id":"t1","name":"Bash","input":{{"command":"ls"}}}}]}}}}"#).unwrap();
        writeln!(f, r#"{{"type":"user","timestamp":"2026-06-28T10:00:06Z","message":{{"role":"user","content":[{{"type":"tool_result","tool_use_id":"t1","content":"file.txt","is_error":false}}]}}}}"#).unwrap();
        // A meta user turn that should be excluded from the transcript.
        writeln!(f, r#"{{"type":"user","isMeta":true,"timestamp":"2026-06-28T10:00:07Z","message":{{"role":"user","content":"<local-command-stdout>noise"}}}}"#).unwrap();
        drop(f);

        let cc = ClaudeCode::new();
        let msgs = cc.transcript(&SessionRef::File(path.clone())).unwrap();
        let _ = fs::remove_file(&path);

        assert_eq!(msgs.len(), 3, "meta turn should be excluded");
        assert_eq!(msgs[0].role, Role::User);
        assert!(matches!(msgs[0].blocks[0], Block::Text(ref t) if t == "Fix the bug"));

        assert_eq!(msgs[1].role, Role::Assistant);
        assert!(matches!(msgs[1].blocks[0], Block::Thinking(_)));
        assert!(matches!(msgs[1].blocks[1], Block::Text(_)));
        assert!(matches!(msgs[1].blocks[2], Block::ToolUse { ref name, .. } if name == "Bash"));

        // A user turn that is only a tool_result is relabeled Tool.
        assert_eq!(msgs[2].role, Role::Tool);
        assert!(matches!(msgs[2].blocks[0], Block::ToolResult { ref text, is_error: false } if text == "file.txt"));
    }
}
