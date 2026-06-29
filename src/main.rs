// The model intentionally captures more than the report reads; the TUI uses
// the rest.
#![allow(dead_code)]

//! ssft - sift through agent-harness sessions, ncdu-style.
//!
//! Harness-agnostic: each harness is a [`model::Provider`]; the registry detects
//! which are present and parses their sessions into one canonical model. By
//! default this launches an interactive TUI; `ssft report` (or any non-TTY
//! invocation) prints a static triage table instead.

mod model;
mod providers;
mod tui;
mod util;

use model::Session;
use util::{ellipsize, substance_str};

use ratatui::crossterm::tty::IsTty;

fn main() -> model::Result<()> {
    let want_report = std::env::args().nth(1).as_deref() == Some("report");
    // Load before any terminal setup, so provider warnings reach the real stderr.
    let (sessions, detected) = providers::load_sessions();

    if want_report || !std::io::stdout().is_tty() {
        print_report(&sessions, &detected);
    } else {
        tui::run(tui::App::new(sessions, detected))?;
    }
    Ok(())
}

fn print_report(sessions: &[Session], detected: &[&str]) {
    let mut sessions = sessions.to_vec();
    // Recency is the one signal every harness records, so it's the honest sort.
    sessions.sort_by(|a, b| b.last_activity().cmp(&a.last_activity()));

    println!(
        "{:<11} {:<34} {:>9} {:>6} {:>16} {}",
        "HARNESS", "TITLE", "SUBST", "MSGS", "LAST ACTIVITY", "WORKSPACE"
    );
    println!("{}", "-".repeat(108));

    for s in &sessions {
        let title = match s.display_title() {
            Some(t) => ellipsize(t, 33),
            None => "· (unnamed)".to_string(),
        };
        let last = s
            .last_activity()
            .map(|t| t.format("%Y-%m-%d %H:%M").to_string())
            .unwrap_or_else(|| "?".to_string());

        println!(
            "{:<11} {:<34} {:>9} {:>6} {:>16} {}",
            s.provider,
            title,
            substance_str(s.substance()),
            s.counts.messages(),
            last,
            ellipsize(s.workspace.as_deref().unwrap_or(""), 34),
        );
    }

    let unknown: usize = sessions.iter().map(|s| s.health.unknown).sum();
    let unparsed: usize = sessions.iter().map(|s| s.health.unparsed).sum();
    println!("{}", "-".repeat(108));
    println!(
        "{} sessions across {} harness(es): {}",
        sessions.len(),
        detected.len(),
        if detected.is_empty() { "none detected".to_string() } else { detected.join(", ") },
    );
    println!("parse health: {unknown} unknown event types, {unparsed} unparsed lines");
}
