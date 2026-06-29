//! Harness adapters. Each submodule implements [`Provider`] for one harness;
//! [`all`] is the registry the app iterates over. Adding a harness is a new
//! submodule plus one line here - the core and UI never change.

pub mod claude_code;
pub mod codex;
pub mod crush;

use std::fs;
use std::path::{Path, PathBuf};

use chrono::{DateTime, Utc};

use crate::model::{Provider, Session};

/// Every known harness adapter, in display order.
pub fn all() -> Vec<Box<dyn Provider>> {
    vec![
        Box::new(claude_code::ClaudeCode::new()),
        Box::new(codex::Codex::new()),
        Box::new(crush::Crush::new()),
    ]
}

/// The adapter with the given id, for on-demand work like transcript loading.
pub fn by_id(id: &str) -> Option<Box<dyn Provider>> {
    all().into_iter().find(|p| p.id() == id)
}

/// Detect, discover, and parse every session from every present harness.
/// Returns the sessions plus the ids of the harnesses that were detected.
/// Per-session/per-harness failures are reported to stderr and skipped, so one
/// bad file never sinks the run. Call this before entering the TUI, while
/// stderr is still the real terminal.
pub fn load_sessions() -> (Vec<Session>, Vec<&'static str>) {
    let mut sessions = Vec::new();
    let mut detected = Vec::new();

    for provider in all() {
        if !provider.detect() {
            continue;
        }
        detected.push(provider.id());
        let refs = match provider.discover() {
            Ok(refs) => refs,
            Err(e) => {
                eprintln!("warning: {} discovery failed: {e}", provider.id());
                continue;
            }
        };
        for r in &refs {
            match provider.parse(r) {
                Ok(mut s) => {
                    // Record where it came from so its transcript can be
                    // reloaded on demand without re-discovering everything.
                    s.source = r.clone();
                    sessions.push(s);
                }
                Err(e) => eprintln!("warning: {} parse failed: {e}", provider.id()),
            }
        }
    }

    (sessions, detected)
}

/// The user's home directory (`$HOME`), falling back to the current dir.
pub fn home() -> PathBuf {
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."))
}

/// Filesystem mtime as UTC - the universal recency fallback.
pub fn mtime(path: &Path) -> Option<DateTime<Utc>> {
    let modified = fs::metadata(path).ok()?.modified().ok()?;
    Some(DateTime::<Utc>::from(modified))
}

/// Size on disk in bytes, 0 if unreadable.
pub fn file_size(path: &Path) -> u64 {
    fs::metadata(path).map(|m| m.len()).unwrap_or(0)
}
