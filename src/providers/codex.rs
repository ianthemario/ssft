//! OpenAI Codex CLI adapter.
//!
//! Storage: `~/.codex/sessions/YYYY/MM/DD/rollout-<ts>-<uuid>.jsonl`, one JSONL
//! file per session, date-bucketed (older builds used a flat layout - we walk
//! recursively to cover both). Each line is `{timestamp, type, payload}` where
//! `type` is adjacently tagged. Notable differences from Claude Code that the
//! canonical model absorbs:
//!   - No session title is ever recorded.
//!   - Token usage arrives as out-of-band `event_msg`/`token_count` events
//!     carrying a *cumulative* total, so the session total is the LAST such
//!     event, not a sum.
//!   - Tool calls and outputs are flat lines linked by `call_id`, not nested.

use std::fs;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

use chrono::{DateTime, Utc};
use serde::Deserialize;
use serde_json::Value;

use crate::model::{Block, Message, Provider, Result, Role, Session, SessionRef, TokenTotals};
use crate::providers::{file_size, home, mtime};

pub struct Codex {
    root: PathBuf,
}

impl Codex {
    pub fn new() -> Codex {
        Codex {
            root: home().join(".codex").join("sessions"),
        }
    }
}

/// One rollout line. `payload` stays a `Value` so the many payload shapes
/// (and future ones) are navigated defensively rather than modeled exhaustively.
#[derive(Debug, Deserialize)]
struct Line {
    timestamp: Option<DateTime<Utc>>,
    #[serde(rename = "type")]
    kind: Option<String>,
    payload: Option<Value>,
}

/// Codex's cumulative usage block (`payload.info.total_token_usage`).
#[derive(Debug, Default, Deserialize)]
struct CodexUsage {
    #[serde(default)]
    input_tokens: u64,
    #[serde(default)]
    cached_input_tokens: u64,
    #[serde(default)]
    output_tokens: u64,
    #[serde(default)]
    reasoning_output_tokens: u64,
}

impl Provider for Codex {
    fn id(&self) -> &'static str {
        "codex"
    }

    fn detect(&self) -> bool {
        self.root.is_dir()
    }

    fn discover(&self) -> Result<Vec<SessionRef>> {
        let mut refs = Vec::new();
        walk_jsonl(&self.root, &mut refs)?;
        Ok(refs)
    }

    fn parse(&self, session_ref: &SessionRef) -> Result<Session> {
        let SessionRef::File(path) = session_ref else {
            return Err("codex expects a file session ref".into());
        };

        let mut s = Session::new(self.id(), session_id_from_path(path));
        s.bytes = file_size(path);
        s.mtime = mtime(path);

        // Cumulative usage: keep the latest snapshot, assign once at the end.
        let mut latest_usage: Option<CodexUsage> = None;

        let reader = BufReader::new(fs::File::open(path)?);
        for line in reader.lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            let parsed: Line = match serde_json::from_str(&line) {
                Ok(l) => l,
                Err(_) => {
                    s.health.unparsed += 1;
                    continue;
                }
            };
            s.observe_ts(parsed.timestamp);
            let Some(payload) = parsed.payload else {
                continue;
            };

            match parsed.kind.as_deref() {
                Some("session_meta") => {
                    s.workspace = s.workspace.take().or_else(|| str_field(&payload, "cwd"));
                    if let Some(git) = payload.get("git") {
                        s.branch = s.branch.take().or_else(|| str_field(git, "branch"));
                    }
                    // Provider only; the concrete model shows up in turn_context.
                    s.model = s.model.take().or_else(|| str_field(&payload, "model_provider"));
                }
                Some("turn_context") => {
                    if let Some(m) = str_field(&payload, "model") {
                        s.model = Some(m);
                    }
                    s.workspace = s.workspace.take().or_else(|| str_field(&payload, "cwd"));
                }
                Some("response_item") => match str_field(&payload, "type").as_deref() {
                    Some("message") => {
                        let role = str_field(&payload, "role");
                        let text = extract_text(payload.get("content"));
                        match role.as_deref() {
                            Some("assistant") => s.counts.assistant += 1,
                            _ => {
                                s.counts.user += 1;
                                if !text.trim().is_empty() {
                                    if s.first_prompt.is_none() {
                                        s.first_prompt = Some(text.trim().to_string());
                                    }
                                    s.last_prompt = Some(text.trim().to_string());
                                }
                            }
                        }
                        s.text_chars += text.len() as u64;
                    }
                    Some("function_call") => s.counts.tool += 1,
                    _ => {}
                },
                Some("event_msg") => {
                    if str_field(&payload, "type").as_deref() == Some("token_count") {
                        if let Some(info) = payload.get("info") {
                            if let Some(total) = info.get("total_token_usage") {
                                if let Ok(u) = serde_json::from_value::<CodexUsage>(total.clone()) {
                                    latest_usage = Some(u);
                                }
                            }
                        }
                    }
                }
                _ => {} // compacted, turn diffs, future variants: intentionally ignored
            }
        }

        if let Some(u) = latest_usage {
            s.tokens = Some(TokenTotals {
                input: u.input_tokens,
                output: u.output_tokens,
                cache_creation: 0,
                cache_read: u.cached_input_tokens,
                reasoning: u.reasoning_output_tokens,
            });
        }
        Ok(s)
    }

    fn transcript(&self, session_ref: &SessionRef) -> Result<Vec<Message>> {
        let SessionRef::File(path) = session_ref else {
            return Err("codex expects a file session ref".into());
        };

        let mut out = Vec::new();
        let reader = BufReader::new(fs::File::open(path)?);
        for line in reader.lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            let Ok(parsed) = serde_json::from_str::<Line>(&line) else {
                continue;
            };
            let ts = parsed.timestamp;
            let Some(payload) = parsed.payload else { continue };
            if parsed.kind.as_deref() != Some("response_item") {
                continue; // messages, tool calls, reasoning all live here
            }

            match str_field(&payload, "type").as_deref() {
                Some("message") => {
                    let text = extract_text(payload.get("content"));
                    if text.trim().is_empty() {
                        continue;
                    }
                    let role = match str_field(&payload, "role").as_deref() {
                        Some("assistant") => Role::Assistant,
                        Some("system") => Role::System,
                        _ => Role::User,
                    };
                    out.push(Message { role, timestamp: ts, blocks: vec![Block::Text(text)] });
                }
                Some("reasoning") => {
                    let text = extract_text(payload.get("content")).trim().to_string();
                    let text = if text.is_empty() { str_field(&payload, "text").unwrap_or_default() } else { text };
                    if !text.trim().is_empty() {
                        out.push(Message { role: Role::Assistant, timestamp: ts, blocks: vec![Block::Thinking(text)] });
                    }
                }
                Some("function_call") => {
                    let name = str_field(&payload, "name").unwrap_or_else(|| "?".into());
                    let input = str_field(&payload, "arguments").unwrap_or_default();
                    out.push(Message {
                        role: Role::Assistant,
                        timestamp: ts,
                        blocks: vec![Block::ToolUse { name, input }],
                    });
                }
                Some("function_call_output") => {
                    let text = payload
                        .get("output")
                        .map(|o| o.as_str().map(str::to_string).unwrap_or_else(|| o.to_string()))
                        .unwrap_or_default();
                    out.push(Message {
                        role: Role::Tool,
                        timestamp: ts,
                        blocks: vec![Block::ToolResult { text, is_error: false }],
                    });
                }
                _ => {}
            }
        }
        Ok(out)
    }
}

/// `rollout-2025-01-03T12-00-00-<uuid>.jsonl` → the trailing UUID, or the stem.
fn session_id_from_path(path: &Path) -> String {
    let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("");
    stem.rsplit_once('-')
        .map(|(_, id)| id.to_string())
        .filter(|id| id.len() >= 8)
        .unwrap_or_else(|| stem.to_string())
}

fn str_field(v: &Value, key: &str) -> Option<String> {
    v.get(key).and_then(Value::as_str).map(str::to_string)
}

/// Flatten a Codex message `content` array (`[{type, text}, …]`) to its text.
fn extract_text(content: Option<&Value>) -> String {
    let Some(Value::Array(parts)) = content else {
        return content.and_then(Value::as_str).unwrap_or("").to_string();
    };
    parts
        .iter()
        .filter_map(|p| p.get("text").and_then(Value::as_str))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Recursively collect every `*.jsonl` under `dir`.
fn walk_jsonl(dir: &Path, out: &mut Vec<SessionRef>) -> Result<()> {
    if !dir.is_dir() {
        return Ok(());
    }
    for entry in fs::read_dir(dir)? {
        let path = entry?.path();
        if path.is_dir() {
            walk_jsonl(&path, out)?;
        } else if path.extension().and_then(|e| e.to_str()) == Some("jsonl") {
            out.push(SessionRef::File(path));
        }
    }
    Ok(())
}
