//! The harness-agnostic core: a canonical [`Session`] summary and the
//! [`Provider`] trait that every harness adapter implements. Nothing in this
//! module knows about any specific harness's on-disk format - providers parse
//! their native storage (JSONL, JSON, SQLite, Markdown, …) down to this model,
//! and the TUI/report layer consumes only this model.

use std::path::PathBuf;

use chrono::{DateTime, Utc};

/// Boxed error so providers can surface I/O, serde, and SQLite failures alike
/// without the core depending on any one error crate.
pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;

/// How a provider points at the bytes of one *logical* session. Deliberately
/// richer than a path: a session can span multiple files (Gemini fuses
/// `logs.json` + `checkpoint-*.json`) or live as a row in a database
/// (Cursor, Crush, Goose) rather than a file at all.
#[derive(Debug, Clone)]
pub enum SessionRef {
    File(PathBuf),
    Files(Vec<PathBuf>),
    SqliteRow { db: PathBuf, key: String },
}

#[derive(Debug, Default, Clone)]
pub struct Counts {
    pub user: usize,
    pub assistant: usize,
    pub tool: usize,
}

impl Counts {
    pub fn messages(&self) -> usize {
        self.user + self.assistant
    }
}

/// Token accounting kept by class, since classes are priced differently and not
/// every harness reports every class. `reasoning` covers Codex/o-series
/// reasoning output; `cache_*` cover Claude-style prompt caching.
#[derive(Debug, Default, Clone)]
pub struct TokenTotals {
    pub input: u64,
    pub output: u64,
    pub cache_creation: u64,
    pub cache_read: u64,
    pub reasoning: u64,
}

impl TokenTotals {
    /// "New material" produced - the honest substance proxy. Excludes input and
    /// cache *reads* (re-reading existing context), which inflate without
    /// reflecting work done.
    pub fn substance(&self) -> u64 {
        self.output + self.cache_creation + self.reasoning
    }
}

/// Parse health for one session. Nonzero means a provider's model missed
/// something - surfaced rather than swallowed, so format drift is visible.
#[derive(Debug, Default, Clone)]
pub struct ParseHealth {
    pub unknown: usize,
    pub unparsed: usize,
}

/// A "size" measurement that degrades gracefully across harnesses and always
/// carries its unit, so the UI never silently compares tokens to char counts.
#[derive(Debug, Clone, Copy)]
pub enum Substance {
    Tokens(u64),
    Chars(u64),
    Messages(u64),
    Bytes(u64),
}

impl Substance {
    pub fn value(&self) -> u64 {
        match *self {
            Substance::Tokens(n)
            | Substance::Chars(n)
            | Substance::Messages(n)
            | Substance::Bytes(n) => n,
        }
    }

    /// Short unit label for display (`tok`, `ch`, `msg`, `B`).
    pub fn unit(&self) -> &'static str {
        match self {
            Substance::Tokens(_) => "tok",
            Substance::Chars(_) => "ch",
            Substance::Messages(_) => "msg",
            Substance::Bytes(_) => "B",
        }
    }
}

/// Who produced a message in a transcript.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Role {
    User,
    Assistant,
    System,
    Tool,
}

impl Role {
    pub fn label(self) -> &'static str {
        match self {
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::System => "system",
            Role::Tool => "tool",
        }
    }
}

/// A piece of a message. The harness-agnostic shape every provider maps its
/// native content into, so the transcript renderer is format-blind.
#[derive(Debug, Clone)]
pub enum Block {
    Text(String),
    /// Model reasoning / thinking, kept separate so the UI can dim or fold it.
    Thinking(String),
    ToolUse { name: String, input: String },
    ToolResult { text: String, is_error: bool },
}

/// One turn in a session transcript. Loaded lazily (see [`Provider::transcript`])
/// rather than held for every session, so browsing stays light.
#[derive(Debug, Clone)]
pub struct Message {
    pub role: Role,
    pub timestamp: Option<DateTime<Utc>>,
    pub blocks: Vec<Block>,
}

/// The canonical, harness-agnostic view of one session. Rich fields are
/// `Option` because most harnesses record fewer signals than Claude Code; the
/// always-available fallbacks (`bytes`, `mtime`) keep triage working regardless.
#[derive(Debug, Clone)]
pub struct Session {
    /// Which harness produced it (e.g. `"claude-code"`, `"codex"`, `"crush"`).
    pub provider: &'static str,
    pub id: String,
    /// Working directory / project the session ran in, if recorded.
    pub workspace: Option<String>,

    /// Recognition signals.
    pub title: Option<String>,
    pub first_prompt: Option<String>,
    pub last_prompt: Option<String>,
    pub branch: Option<String>,
    /// Set when the session is a named subagent rather than a top-level chat.
    pub agent_name: Option<String>,
    pub model: Option<String>,

    /// Activity span from in-band event timestamps, if any.
    pub first_ts: Option<DateTime<Utc>>,
    pub last_ts: Option<DateTime<Utc>>,
    /// Filesystem mtime of the backing storage - the universal recency fallback.
    pub mtime: Option<DateTime<Utc>>,

    pub counts: Counts,
    /// `None` when the harness records no token usage (e.g. Gemini, Aider).
    pub tokens: Option<TokenTotals>,
    /// Total length of human-readable message text - the char-based fallback
    /// for substance when tokens are unavailable.
    pub text_chars: u64,
    /// Size on disk of the backing storage - the last-resort substance fallback.
    pub bytes: u64,

    pub health: ParseHealth,

    /// Where this session came from, so its transcript can be reloaded on
    /// demand. Set by the loader after `parse`; defaults to an empty path.
    pub source: SessionRef,
}

impl Session {
    pub fn new(provider: &'static str, id: impl Into<String>) -> Session {
        Session {
            provider,
            id: id.into(),
            source: SessionRef::File(PathBuf::new()),
            workspace: None,
            title: None,
            first_prompt: None,
            last_prompt: None,
            branch: None,
            agent_name: None,
            model: None,
            first_ts: None,
            last_ts: None,
            mtime: None,
            counts: Counts::default(),
            tokens: None,
            text_chars: 0,
            bytes: 0,
            health: ParseHealth::default(),
        }
    }

    /// Best recency signal available: in-band last event, else file mtime.
    pub fn last_activity(&self) -> Option<DateTime<Utc>> {
        self.last_ts.or(self.mtime)
    }

    /// Best "size" available, degrading tokens → chars → messages → bytes.
    pub fn substance(&self) -> Substance {
        if let Some(t) = &self.tokens {
            if t.substance() > 0 {
                return Substance::Tokens(t.substance());
            }
        }
        if self.text_chars > 0 {
            return Substance::Chars(self.text_chars);
        }
        if self.counts.messages() > 0 {
            return Substance::Messages(self.counts.messages() as u64);
        }
        Substance::Bytes(self.bytes)
    }

    /// Display title, falling back to the first prompt when the harness records
    /// no title (most don't).
    pub fn display_title(&self) -> Option<&str> {
        self.title
            .as_deref()
            .or(self.first_prompt.as_deref())
            .map(str::trim)
            .filter(|t| !t.is_empty())
    }

    /// Fold one observed timestamp into the activity span.
    pub fn observe_ts(&mut self, ts: Option<DateTime<Utc>>) {
        if let Some(ts) = ts {
            self.first_ts = Some(self.first_ts.map_or(ts, |f| f.min(ts)));
            self.last_ts = Some(self.last_ts.map_or(ts, |l| l.max(ts)));
        }
    }
}

/// One harness adapter. Implementors own all knowledge of their native format,
/// including intra-harness version drift (e.g. Goose's JSONL→SQLite switch).
pub trait Provider {
    /// Stable identifier, e.g. `"claude-code"`.
    fn id(&self) -> &'static str;

    /// Whether this harness appears to be present on the machine. Discovery is
    /// skipped when this is false, so a missing harness costs nothing.
    fn detect(&self) -> bool;

    /// Locate every session this harness has stored.
    fn discover(&self) -> Result<Vec<SessionRef>>;

    /// Parse one located session into the canonical model (aggregates only).
    fn parse(&self, session_ref: &SessionRef) -> Result<Session>;

    /// Load the full message-by-message transcript for one session. Called
    /// lazily when the user drills in, so message bodies are never held for
    /// every session at once.
    fn transcript(&self, session_ref: &SessionRef) -> Result<Vec<Message>>;
}
