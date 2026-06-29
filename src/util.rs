//! Small formatting helpers shared by the report and the TUI.

use crate::model::Substance;

/// Truncate to `max` display chars, appending an ellipsis when cut. Also
/// collapses whitespace so multi-line prompts render on one row.
pub fn ellipsize(s: &str, max: usize) -> String {
    let s = s.replace(['\n', '\r', '\t'], " ");
    if s.chars().count() <= max {
        return s;
    }
    let kept: String = s.chars().take(max.saturating_sub(1)).collect();
    format!("{kept}…")
}

/// Human-friendly count: 1234 -> "1.2k", 1_200_000 -> "1.2M".
pub fn human(n: u64) -> String {
    match n {
        0..=999 => n.to_string(),
        1_000..=999_999 => format!("{:.1}k", n as f64 / 1_000.0),
        _ => format!("{:.1}M", n as f64 / 1_000_000.0),
    }
}

/// Substance with its unit, e.g. "9.5M tok" or "188.0k ch" - never unit-blind.
pub fn substance_str(s: Substance) -> String {
    format!("{} {}", human(s.value()), s.unit())
}
