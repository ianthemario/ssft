//! Charmbracelet Crush adapter - the proof that a session need not be a file.
//!
//! Storage: a SQLite database (Crush uses sqlc). A single DB holds many
//! sessions, so [`SessionRef::SqliteRow`] points at `{db, session_id}` and
//! [`parse`] reads two tables:
//!
//! ```sql
//! sessions(id, parent_session_id, title, message_count,
//!          prompt_tokens, completion_tokens, cost, updated_at, created_at)
//! messages(id, session_id, role, parts /*JSON*/, model,
//!          created_at, updated_at, finished_at)
//! ```
//!
//! `created_at`/`updated_at` are integer Unix timestamps. Tokens and a title
//! live on the session row directly, so unlike the JSONL providers no event
//! replay is needed for the headline metrics.

use std::path::{Path, PathBuf};

use chrono::{DateTime, TimeZone, Utc};
use rusqlite::{Connection, OpenFlags};
use serde_json::Value;

use crate::model::{Block, Message, Provider, Result, Role, Session, SessionRef, TokenTotals};
use crate::providers::{file_size, home, mtime};

pub struct Crush {
    /// Candidate database locations, in priority order.
    dbs: Vec<PathBuf>,
}

impl Crush {
    pub fn new() -> Crush {
        let home = home();
        let mut dbs = Vec::new();
        if let Some(p) = std::env::var_os("CRUSH_DB") {
            dbs.push(PathBuf::from(p));
        }
        dbs.push(home.join(".local/share/crush/crush.db"));
        dbs.push(home.join(".config/crush/crush.db"));
        dbs.push(PathBuf::from(".crush/crush.db")); // project-local
        Crush { dbs }
    }

    fn existing_dbs(&self) -> Vec<&PathBuf> {
        self.dbs.iter().filter(|p| p.is_file()).collect()
    }
}

impl Provider for Crush {
    fn id(&self) -> &'static str {
        "crush"
    }

    fn detect(&self) -> bool {
        self.dbs.iter().any(|p| p.is_file())
    }

    fn discover(&self) -> Result<Vec<SessionRef>> {
        let mut refs = Vec::new();
        for db in self.existing_dbs() {
            let conn = open_ro(db)?;
            let mut stmt = conn.prepare("SELECT id FROM sessions")?;
            let ids = stmt.query_map([], |row| row.get::<_, String>(0))?;
            for id in ids {
                refs.push(SessionRef::SqliteRow {
                    db: db.to_path_buf(),
                    key: id?,
                });
            }
        }
        Ok(refs)
    }

    fn parse(&self, session_ref: &SessionRef) -> Result<Session> {
        let SessionRef::SqliteRow { db, key } = session_ref else {
            return Err("crush expects a sqlite-row session ref".into());
        };

        let conn = open_ro(db)?;
        let mut s = Session::new(self.id(), key.clone());
        s.bytes = file_size(db);
        s.mtime = mtime(db);
        s.workspace = workspace_from_db_path(db);

        // Headline metrics straight off the session row.
        let (title, prompt_tokens, completion_tokens, created_at, updated_at, msg_count) = conn
            .query_row(
                "SELECT title, prompt_tokens, completion_tokens, created_at, updated_at, \
                 message_count FROM sessions WHERE id = ?1",
                [key],
                |r| {
                    Ok((
                        r.get::<_, String>(0)?,
                        r.get::<_, i64>(1)?,
                        r.get::<_, i64>(2)?,
                        r.get::<_, i64>(3)?,
                        r.get::<_, i64>(4)?,
                        r.get::<_, i64>(5)?,
                    ))
                },
            )?;

        if !title.trim().is_empty() {
            s.title = Some(title);
        }
        s.tokens = Some(TokenTotals {
            input: prompt_tokens.max(0) as u64,
            output: completion_tokens.max(0) as u64,
            ..TokenTotals::default()
        });
        s.observe_ts(unix_to_utc(created_at));
        s.observe_ts(unix_to_utc(updated_at));
        let _ = msg_count; // authoritative count comes from the message scan below

        // Walk messages for role counts, prompt text, and model.
        let mut stmt = conn.prepare(
            "SELECT role, parts, model FROM messages WHERE session_id = ?1 ORDER BY created_at",
        )?;
        let rows = stmt.query_map([key], |r| {
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, Option<String>>(2)?,
            ))
        })?;

        for row in rows {
            let (role, parts_json, model) = row?;
            if let Some(m) = model {
                if !m.is_empty() {
                    s.model = Some(m);
                }
            }
            let text = text_from_parts(&parts_json);
            s.text_chars += text.len() as u64;
            match role.as_str() {
                "assistant" => s.counts.assistant += 1,
                "user" => {
                    s.counts.user += 1;
                    let text = text.trim();
                    if !text.is_empty() {
                        if s.first_prompt.is_none() {
                            s.first_prompt = Some(text.to_string());
                        }
                        s.last_prompt = Some(text.to_string());
                    }
                }
                "tool" => s.counts.tool += 1,
                _ => {}
            }
        }

        Ok(s)
    }

    fn transcript(&self, session_ref: &SessionRef) -> Result<Vec<Message>> {
        let SessionRef::SqliteRow { db, key } = session_ref else {
            return Err("crush expects a sqlite-row session ref".into());
        };

        let conn = open_ro(db)?;
        let mut stmt = conn.prepare(
            "SELECT role, parts, created_at FROM messages WHERE session_id = ?1 ORDER BY created_at",
        )?;
        let rows = stmt.query_map([key], |r| {
            Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?, r.get::<_, i64>(2)?))
        })?;

        let mut out = Vec::new();
        for row in rows {
            let (role, parts_json, created_at) = row?;
            let blocks = blocks_from_parts(&parts_json);
            if blocks.is_empty() {
                continue;
            }
            let role = match role.as_str() {
                "assistant" => Role::Assistant,
                "tool" => Role::Tool,
                "system" => Role::System,
                _ => Role::User,
            };
            out.push(Message { role, timestamp: unix_to_utc(created_at), blocks });
        }
        Ok(out)
    }
}

fn open_ro(db: &Path) -> Result<Connection> {
    Connection::open_with_flags(db, OpenFlags::SQLITE_OPEN_READ_ONLY).map_err(Into::into)
}

/// Crush's project-local DB lives at `<project>/.crush/crush.db`; recover the
/// project dir from that. Global DBs have no associated workspace.
fn workspace_from_db_path(db: &Path) -> Option<String> {
    let parent = db.parent()?;
    if parent.file_name()?.to_str()? == ".crush" {
        return parent.parent().map(|p| p.display().to_string());
    }
    None
}

/// Accept seconds or milliseconds Unix timestamps (Go backends vary).
fn unix_to_utc(ts: i64) -> Option<DateTime<Utc>> {
    if ts <= 0 {
        return None;
    }
    let secs = if ts > 1_000_000_000_000 { ts / 1000 } else { ts };
    Utc.timestamp_opt(secs, 0).single()
}

/// Crush message `parts` is a JSON array of typed parts. Pull readable text out
/// defensively by collecting every `text` string field, at any depth.
fn text_from_parts(parts_json: &str) -> String {
    let Ok(value) = serde_json::from_str::<Value>(parts_json) else {
        return String::new();
    };
    let mut out = Vec::new();
    collect_text(&value, &mut out);
    out.join("\n")
}

fn collect_text(value: &Value, out: &mut Vec<String>) {
    match value {
        Value::Object(map) => {
            if let Some(Value::String(t)) = map.get("text") {
                if !t.is_empty() {
                    out.push(t.clone());
                }
            }
            for (k, v) in map {
                if k != "text" {
                    collect_text(v, out);
                }
            }
        }
        Value::Array(items) => items.iter().for_each(|v| collect_text(v, out)),
        _ => {}
    }
}

/// Map Crush message parts to canonical transcript blocks. The exact part
/// schema varies, so types and field names are matched defensively and any
/// unrecognized part still contributes its text rather than being dropped.
fn blocks_from_parts(parts_json: &str) -> Vec<Block> {
    let Ok(Value::Array(parts)) = serde_json::from_str::<Value>(parts_json) else {
        let t = text_from_parts(parts_json);
        return if t.trim().is_empty() { Vec::new() } else { vec![Block::Text(t)] };
    };

    let mut out = Vec::new();
    for p in &parts {
        match p.get("type").and_then(Value::as_str).unwrap_or("") {
            "reasoning" | "thinking" => {
                if let Some(t) = part_text(p) {
                    out.push(Block::Thinking(t));
                }
            }
            "tool-call" | "tool_call" | "tool-invocation" => {
                let name = p
                    .get("name")
                    .or_else(|| p.get("tool"))
                    .and_then(Value::as_str)
                    .unwrap_or("?")
                    .to_string();
                let input = p
                    .get("input")
                    .or_else(|| p.get("args"))
                    .or_else(|| p.get("parameters"))
                    .filter(|v| !v.is_null())
                    .map(|v| serde_json::to_string(v).unwrap_or_default())
                    .unwrap_or_default();
                out.push(Block::ToolUse { name, input });
            }
            "tool-result" | "tool_result" => {
                let text = p
                    .get("result")
                    .or_else(|| p.get("content"))
                    .or_else(|| p.get("output"))
                    .map(value_to_text)
                    .unwrap_or_default();
                let is_error = p
                    .get("is_error")
                    .or_else(|| p.get("isError"))
                    .and_then(Value::as_bool)
                    .unwrap_or(false);
                out.push(Block::ToolResult { text, is_error });
            }
            _ => {
                if let Some(t) = part_text(p) {
                    if !t.trim().is_empty() {
                        out.push(Block::Text(t));
                    }
                }
            }
        }
    }
    out
}

fn part_text(p: &Value) -> Option<String> {
    p.get("text").and_then(Value::as_str).map(str::to_string)
}

fn value_to_text(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Null => String::new(),
        other => other.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::Block;

    #[test]
    fn transcript_reads_messages_and_parts() {
        let db = std::env::temp_dir().join(format!("ssft_crush_{}.db", std::process::id()));
        let _ = std::fs::remove_file(&db);
        {
            let conn = Connection::open(&db).unwrap();
            conn.execute_batch(
                "CREATE TABLE sessions (id TEXT PRIMARY KEY, parent_session_id TEXT, title TEXT NOT NULL,
                   message_count INTEGER NOT NULL DEFAULT 0, prompt_tokens INTEGER NOT NULL DEFAULT 0,
                   completion_tokens INTEGER NOT NULL DEFAULT 0, cost REAL NOT NULL DEFAULT 0.0,
                   updated_at INTEGER NOT NULL, created_at INTEGER NOT NULL);
                 CREATE TABLE messages (id TEXT PRIMARY KEY, session_id TEXT NOT NULL, role TEXT NOT NULL,
                   parts TEXT NOT NULL DEFAULT '[]', model TEXT, created_at INTEGER NOT NULL,
                   updated_at INTEGER NOT NULL, finished_at INTEGER);",
            ).unwrap();
            conn.execute("INSERT INTO sessions VALUES ('s1',NULL,'OAuth',2,10,5,0.0,1782302400,1782302400)", []).unwrap();
            conn.execute(
                "INSERT INTO messages VALUES ('m1','s1','user','[{\"type\":\"text\",\"text\":\"Add OAuth\"}]',NULL,1782302400,1782302400,NULL)",
                [],
            ).unwrap();
            conn.execute(
                "INSERT INTO messages VALUES ('m2','s1','assistant','[{\"type\":\"reasoning\",\"text\":\"plan it\"},{\"type\":\"text\",\"text\":\"Done.\"},{\"type\":\"tool-call\",\"name\":\"edit\",\"input\":{\"f\":\"a.rs\"}}]','claude-x',1782302410,1782302410,1782302410)",
                [],
            ).unwrap();
        }

        let crush = Crush::new();
        let msgs = crush
            .transcript(&SessionRef::SqliteRow { db: db.clone(), key: "s1".into() })
            .unwrap();
        let _ = std::fs::remove_file(&db);

        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0].role, Role::User);
        assert!(matches!(msgs[0].blocks[0], Block::Text(ref t) if t == "Add OAuth"));
        assert_eq!(msgs[1].role, Role::Assistant);
        assert!(matches!(msgs[1].blocks[0], Block::Thinking(_)));
        assert!(matches!(msgs[1].blocks[1], Block::Text(_)));
        assert!(matches!(msgs[1].blocks[2], Block::ToolUse { ref name, .. } if name == "edit"));
    }
}
