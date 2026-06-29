//! Interactive ncdu-style browser over the loaded sessions.
//!
//! Three views, all driven by [`App`] state with immediate-mode rendering:
//!   - **List** - a flat, globally sorted/filtered stream of every session.
//!   - **Browse** - a directory tree built from workspace `cwd`s, navigated
//!     like ncdu (descend into a dir, ascend with `..`); single-child chains
//!     are compressed (`clouds/aws/`). `Tab` toggles List ↔ Browse.
//!   - **Detail** - one session's full metadata and prompts; returns to
//!     whichever view opened it.

use std::collections::HashMap;

use chrono::{DateTime, Utc};
use ratatui::crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind};
use ratatui::layout::{Constraint, Layout};
use ratatui::style::{Color, Modifier, Style, Stylize};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Cell, Paragraph, Row, Table, TableState, Wrap};
use ratatui::{DefaultTerminal, Frame};

use crate::model::{Block as MsgBlock, Message, Provider, ResumeSpec, Role, Session};
use crate::util::{ellipsize, human, substance_str};

#[derive(Clone, Copy, PartialEq)]
enum SortMode {
    Recency,
    Substance,
    Messages,
}

impl SortMode {
    fn label(self) -> &'static str {
        match self {
            SortMode::Recency => "recency",
            SortMode::Substance => "substance",
            SortMode::Messages => "messages",
        }
    }
    fn next(self) -> SortMode {
        match self {
            SortMode::Recency => SortMode::Substance,
            SortMode::Substance => SortMode::Messages,
            SortMode::Messages => SortMode::Recency,
        }
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Mode {
    List,
    Browse,
    Detail,
    Transcript,
}

// ---------------------------------------------------------------------------
// Workspace path tree
// ---------------------------------------------------------------------------

/// A directory node. Sessions whose `cwd` is exactly this node's path live in
/// `sessions`; deeper workspaces live under `children`. Rollup fields aggregate
/// the whole subtree and are filled post-order at build time.
struct TreeNode {
    name: String,
    full_path: Option<String>,
    parent: Option<usize>,
    children: Vec<usize>,
    sessions: Vec<usize>,
    roll_tokens: u64,
    roll_count: usize,
    roll_msgs: usize,
    roll_last: Option<DateTime<Utc>>,
    /// Whether the subtree contains any session without token accounting, so
    /// the rolled-up token sum is flagged as a lower bound rather than total.
    roll_has_nontoken: bool,
}

impl TreeNode {
    fn new(name: &str, full_path: Option<String>, parent: Option<usize>) -> TreeNode {
        TreeNode {
            name: name.to_string(),
            full_path,
            parent,
            children: Vec::new(),
            sessions: Vec::new(),
            roll_tokens: 0,
            roll_count: 0,
            roll_msgs: 0,
            roll_last: None,
            roll_has_nontoken: false,
        }
    }
    /// A node is a passthrough link when it holds no sessions of its own and has
    /// exactly one child - the case chain-compression collapses.
    fn is_passthrough(&self) -> bool {
        self.sessions.is_empty() && self.children.len() == 1
    }
}

const NO_WORKSPACE_KEY: &str = "\0no-workspace";

/// Build the workspace tree from all sessions. Returns the arena, the synthetic
/// root id, and the start node (root after compressing its leading single-child
/// chain, so browsing opens at the first directory that actually branches).
fn build_tree(all: &[Session]) -> (Vec<TreeNode>, usize, usize) {
    let mut nodes = vec![TreeNode::new("", None, None)];
    let root = 0;
    let mut by_path: HashMap<String, usize> = HashMap::new();

    for (i, s) in all.iter().enumerate() {
        let leaf = match &s.workspace {
            Some(ws) if !ws.trim().is_empty() => intern_path(&mut nodes, &mut by_path, root, ws),
            _ => intern_no_workspace(&mut nodes, &mut by_path, root),
        };
        nodes[leaf].sessions.push(i);
    }

    compute_rollup(&mut nodes, all, root);
    let start = compress_down(&nodes, root);
    (nodes, root, start)
}

fn intern_path(
    nodes: &mut Vec<TreeNode>,
    by_path: &mut HashMap<String, usize>,
    root: usize,
    ws: &str,
) -> usize {
    let mut cur = root;
    let mut acc = String::new();
    for comp in ws.trim_end_matches('/').split('/').filter(|c| !c.is_empty()) {
        acc = format!("{acc}/{comp}");
        cur = match by_path.get(&acc) {
            Some(&id) => id,
            None => {
                let id = nodes.len();
                nodes.push(TreeNode::new(comp, Some(acc.clone()), Some(cur)));
                nodes[cur].children.push(id);
                by_path.insert(acc.clone(), id);
                id
            }
        };
    }
    cur
}

fn intern_no_workspace(
    nodes: &mut Vec<TreeNode>,
    by_path: &mut HashMap<String, usize>,
    root: usize,
) -> usize {
    if let Some(&id) = by_path.get(NO_WORKSPACE_KEY) {
        return id;
    }
    let id = nodes.len();
    nodes.push(TreeNode::new("(no workspace)", None, Some(root)));
    nodes[root].children.push(id);
    by_path.insert(NO_WORKSPACE_KEY.to_string(), id);
    id
}

fn compute_rollup(
    nodes: &mut Vec<TreeNode>,
    all: &[Session],
    id: usize,
) -> (u64, usize, usize, Option<DateTime<Utc>>, bool) {
    let sessions = nodes[id].sessions.clone();
    let children = nodes[id].children.clone();
    let (mut tokens, mut count, mut msgs, mut last, mut nontoken) = (0u64, 0usize, 0usize, None, false);

    for i in sessions {
        let s = &all[i];
        count += 1;
        msgs += s.counts.messages();
        last = max_opt(last, s.last_activity());
        match &s.tokens {
            Some(t) if t.substance() > 0 => tokens += t.substance(),
            _ => nontoken = true,
        }
    }
    for c in children {
        let (t, n, m, l, nt) = compute_rollup(nodes, all, c);
        tokens += t;
        count += n;
        msgs += m;
        last = max_opt(last, l);
        nontoken |= nt;
    }

    let node = &mut nodes[id];
    node.roll_tokens = tokens;
    node.roll_count = count;
    node.roll_msgs = msgs;
    node.roll_last = last;
    node.roll_has_nontoken = nontoken;
    (tokens, count, msgs, last, nontoken)
}

/// Follow a single-child chain downward, returning the first branching node.
fn compress_down(nodes: &[TreeNode], id: usize) -> usize {
    let mut cur = id;
    while nodes[cur].is_passthrough() {
        cur = nodes[cur].children[0];
    }
    cur
}

/// Nearest meaningful ancestor: the parent, skipping passthrough links upward,
/// mirroring `compress_down` so ascend and descend are symmetric.
fn ascend(nodes: &[TreeNode], id: usize) -> Option<usize> {
    let mut cur = nodes[id].parent?;
    while let Some(p) = nodes[cur].parent {
        if nodes[cur].is_passthrough() {
            cur = p;
        } else {
            break;
        }
    }
    Some(cur)
}

fn max_opt(a: Option<DateTime<Utc>>, b: Option<DateTime<Utc>>) -> Option<DateTime<Utc>> {
    match (a, b) {
        (Some(a), Some(b)) => Some(a.max(b)),
        (x, None) | (None, x) => x,
    }
}

/// One row at the current browse level.
enum Entry {
    Up,
    Dir { node: usize, name: String },
    Session(usize),
}

// ---------------------------------------------------------------------------
// App
// ---------------------------------------------------------------------------

pub struct App {
    all: Vec<Session>,
    detected: Vec<&'static str>,

    // Flat list state.
    view: Vec<usize>,
    selected: usize,

    // Browse (tree) state.
    nodes: Vec<TreeNode>,
    root: usize,
    cur: usize,
    bsel: usize,

    // Shared.
    sort: SortMode,
    filter: String,
    editing_filter: bool,
    mode: Mode,

    // Detail state.
    detail_index: usize,
    detail_return: Mode,
    detail_scroll: u16,

    // Transcript state - lazily loaded and cached per session.
    providers: Vec<Box<dyn Provider>>,
    transcript_for: Option<usize>,
    transcript_lines: Vec<Line<'static>>,
    transcript_scroll: u16,
    /// Body height of the transcript view, captured during draw so the key
    /// handler can page and clamp against the real viewport.
    transcript_viewport: u16,

    /// Set when the user asks to open a session in its harness; run() execs it
    /// after the terminal is restored.
    pending_launch: Option<ResumeSpec>,
    /// Transient one-line notice shown in the footer (e.g. "not supported").
    status: Option<String>,

    should_quit: bool,
}

impl App {
    pub fn new(all: Vec<Session>, detected: Vec<&'static str>) -> App {
        let (nodes, root, start) = build_tree(&all);
        let mut app = App {
            all,
            detected,
            view: Vec::new(),
            selected: 0,
            nodes,
            root,
            cur: start,
            bsel: 0,
            sort: SortMode::Recency,
            filter: String::new(),
            editing_filter: false,
            mode: Mode::List,
            detail_index: 0,
            detail_return: Mode::List,
            detail_scroll: 0,
            providers: crate::providers::all(),
            transcript_for: None,
            transcript_lines: Vec::new(),
            transcript_scroll: 0,
            transcript_viewport: 0,
            pending_launch: None,
            status: None,
            should_quit: false,
        };
        app.recompute_view();
        app
    }

    // --- flat list ---

    fn recompute_view(&mut self) {
        let needle = self.filter.to_lowercase();
        let mut idx: Vec<usize> = (0..self.all.len())
            .filter(|&i| needle.is_empty() || matches(&self.all[i], &needle))
            .collect();
        idx.sort_by(|&a, &b| self.cmp_sessions(a, b));
        self.view = idx;
        if self.selected >= self.view.len() {
            self.selected = self.view.len().saturating_sub(1);
        }
    }

    fn cmp_sessions(&self, a: usize, b: usize) -> std::cmp::Ordering {
        let (a, b) = (&self.all[a], &self.all[b]);
        match self.sort {
            SortMode::Recency => b.last_activity().cmp(&a.last_activity()),
            SortMode::Substance => b.substance().value().cmp(&a.substance().value()),
            SortMode::Messages => b.counts.messages().cmp(&a.counts.messages()),
        }
    }

    // --- browse ---

    /// Does the subtree under `node` contain a session matching `needle`?
    fn subtree_matches(&self, node: usize, needle: &str) -> bool {
        if self.nodes[node].sessions.iter().any(|&i| matches(&self.all[i], needle)) {
            return true;
        }
        self.nodes[node]
            .children
            .iter()
            .any(|&c| self.subtree_matches(c, needle))
    }

    /// Entries shown at the current node: `..`, then child dirs and session
    /// leaves, filtered and sorted together (ncdu-style, biggest first).
    fn browse_entries(&self) -> Vec<Entry> {
        let needle = self.filter.to_lowercase();
        let filtering = !needle.is_empty();
        let node = &self.nodes[self.cur];

        let mut items: Vec<Entry> = Vec::new();
        for &child in &node.children {
            if filtering && !self.subtree_matches(child, &needle) {
                continue;
            }
            let (deep, name) = self.compress_child(child);
            items.push(Entry::Dir { node: deep, name });
        }
        for &i in &node.sessions {
            if filtering && !matches(&self.all[i], &needle) {
                continue;
            }
            items.push(Entry::Session(i));
        }

        items.sort_by(|a, b| self.entry_key(b).cmp(&self.entry_key(a)));

        let mut entries = Vec::with_capacity(items.len() + 1);
        if ascend(&self.nodes, self.cur).is_some() || self.nodes[self.cur].parent.is_some() {
            entries.push(Entry::Up);
        }
        entries.extend(items);
        entries
    }

    /// Compress a single-child chain for display: returns the deepest node to
    /// descend into and its joined name (`clouds/aws`).
    fn compress_child(&self, child: usize) -> (usize, String) {
        let mut cur = child;
        let mut name = self.nodes[cur].name.clone();
        while self.nodes[cur].is_passthrough() {
            let next = self.nodes[cur].children[0];
            name = format!("{name}/{}", self.nodes[next].name);
            cur = next;
        }
        (cur, name)
    }

    /// Comparable sort key for an entry under the active sort mode.
    fn entry_key(&self, e: &Entry) -> u64 {
        match *e {
            Entry::Up => u64::MAX, // always pinned to the top after reverse-sort
            Entry::Dir { node, .. } => match self.sort {
                SortMode::Recency => self.nodes[node].roll_last.map(ts_key).unwrap_or(0),
                SortMode::Substance => self.nodes[node].roll_tokens,
                SortMode::Messages => self.nodes[node].roll_count as u64,
            },
            Entry::Session(i) => {
                let s = &self.all[i];
                match self.sort {
                    SortMode::Recency => s.last_activity().map(ts_key).unwrap_or(0),
                    SortMode::Substance => s.substance().value(),
                    SortMode::Messages => s.counts.messages() as u64,
                }
            }
        }
    }

    fn activate_browse_entry(&mut self) {
        let entries = self.browse_entries();
        let Some(entry) = entries.get(self.bsel) else { return };
        match *entry {
            Entry::Up => self.ascend_browse(),
            Entry::Dir { node, .. } => {
                self.cur = node;
                self.bsel = 0;
            }
            Entry::Session(i) => self.open_detail(i, Mode::Browse),
        }
    }

    fn ascend_browse(&mut self) {
        if let Some(parent) = ascend(&self.nodes, self.cur) {
            self.cur = parent;
            self.bsel = 0;
        }
    }

    fn breadcrumb(&self) -> String {
        self.nodes[self.cur]
            .full_path
            .clone()
            .unwrap_or_else(|| "all".to_string())
    }

    // --- detail / shared ---

    fn open_detail(&mut self, index: usize, from: Mode) {
        self.detail_index = index;
        self.detail_return = from;
        self.detail_scroll = 0;
        self.mode = Mode::Detail;
    }

    /// Enter the transcript for the session shown in Detail, loading it lazily
    /// (and caching, so re-entry and scrolling don't re-parse).
    fn open_transcript(&mut self) {
        self.mode = Mode::Transcript;
        self.transcript_scroll = 0;
        if self.transcript_for == Some(self.detail_index) {
            return; // already loaded
        }
        let s = &self.all[self.detail_index];
        let loaded = self
            .providers
            .iter()
            .find(|p| p.id() == s.provider)
            .ok_or_else(|| format!("no provider for '{}'", s.provider).into())
            .and_then(|p| p.transcript(&s.source));

        self.transcript_lines = match loaded {
            Ok(msgs) if msgs.is_empty() => {
                vec![Line::from(Span::styled("(transcript is empty)", Style::default().fg(Color::DarkGray)))]
            }
            Ok(msgs) => render_transcript(&msgs),
            Err(e) => vec![Line::from(Span::styled(
                format!("failed to load transcript: {e}"),
                Style::default().fg(Color::Red),
            ))],
        };
        self.transcript_for = Some(self.detail_index);
    }

    fn move_cursor(&mut self, delta: isize, len: usize) -> usize {
        if len == 0 {
            return 0;
        }
        let cur = match self.mode {
            Mode::Browse => self.bsel,
            _ => self.selected,
        } as isize;
        (cur + delta).clamp(0, len as isize - 1) as usize
    }

    /// Open the session under the cursor in its harness. Determines the target
    /// per view, asks the provider how to resume it, and either queues the
    /// launch (handled after the TUI tears down) or reports it's unsupported.
    fn resume_current(&mut self) {
        let target = match self.mode {
            Mode::List => self.view.get(self.selected).copied(),
            Mode::Browse => match self.browse_entries().get(self.bsel) {
                Some(Entry::Session(i)) => Some(*i),
                _ => None, // a directory row is selected, not a session
            },
            Mode::Detail | Mode::Transcript => Some(self.detail_index),
        };
        let Some(idx) = target else {
            self.status = Some("select a session first".into());
            return;
        };
        let s = &self.all[idx];
        let spec = self.providers.iter().find(|p| p.id() == s.provider).and_then(|p| p.resume_command(s));
        match spec {
            Some(spec) => {
                self.pending_launch = Some(spec);
                self.should_quit = true; // run() execs after restoring the terminal
            }
            None => self.status = Some(format!("opening not supported for '{}' yet", s.provider)),
        }
    }

    fn handle_key(&mut self, key: KeyEvent) {
        if self.editing_filter {
            self.handle_filter_key(key);
            return;
        }
        // A keypress clears any transient notice from the previous one.
        self.status = None;
        match self.mode {
            Mode::List => self.handle_list_key(key),
            Mode::Browse => self.handle_browse_key(key),
            Mode::Detail => self.handle_detail_key(key),
            Mode::Transcript => self.handle_transcript_key(key),
        }
    }

    fn handle_filter_key(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Esc => {
                self.filter.clear();
                self.editing_filter = false;
                self.recompute_view();
            }
            KeyCode::Enter => self.editing_filter = false,
            KeyCode::Backspace => {
                self.filter.pop();
                self.recompute_view();
                self.bsel = 0;
            }
            KeyCode::Char(c) => {
                self.filter.push(c);
                self.recompute_view();
                self.bsel = 0;
            }
            _ => {}
        }
    }

    fn handle_list_key(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Char('q') | KeyCode::Esc => self.should_quit = true,
            KeyCode::Tab => self.mode = Mode::Browse,
            KeyCode::Down | KeyCode::Char('j') => self.selected = self.move_cursor(1, self.view.len()),
            KeyCode::Up | KeyCode::Char('k') => self.selected = self.move_cursor(-1, self.view.len()),
            KeyCode::PageDown => self.selected = self.move_cursor(10, self.view.len()),
            KeyCode::PageUp => self.selected = self.move_cursor(-10, self.view.len()),
            KeyCode::Home | KeyCode::Char('g') => self.selected = 0,
            KeyCode::End | KeyCode::Char('G') => self.selected = self.move_cursor(isize::MAX, self.view.len()),
            KeyCode::Char('s') => {
                self.sort = self.sort.next();
                self.recompute_view();
            }
            KeyCode::Char('/') => self.editing_filter = true,
            KeyCode::Char('o') => self.resume_current(),
            KeyCode::Enter | KeyCode::Char('l') | KeyCode::Right => {
                if let Some(&i) = self.view.get(self.selected) {
                    self.open_detail(i, Mode::List);
                }
            }
            _ => {}
        }
    }

    fn handle_browse_key(&mut self, key: KeyEvent) {
        let len = self.browse_entries().len();
        match key.code {
            KeyCode::Char('q') | KeyCode::Esc => self.should_quit = true,
            KeyCode::Tab => self.mode = Mode::List,
            KeyCode::Down | KeyCode::Char('j') => self.bsel = self.move_cursor(1, len),
            KeyCode::Up | KeyCode::Char('k') => self.bsel = self.move_cursor(-1, len),
            KeyCode::PageDown => self.bsel = self.move_cursor(10, len),
            KeyCode::PageUp => self.bsel = self.move_cursor(-10, len),
            KeyCode::Home | KeyCode::Char('g') => self.bsel = 0,
            KeyCode::End | KeyCode::Char('G') => self.bsel = self.move_cursor(isize::MAX, len),
            KeyCode::Char('s') => self.sort = self.sort.next(),
            KeyCode::Char('/') => self.editing_filter = true,
            KeyCode::Char('o') => self.resume_current(),
            KeyCode::Enter | KeyCode::Char('l') | KeyCode::Right => self.activate_browse_entry(),
            KeyCode::Backspace | KeyCode::Char('h') | KeyCode::Left => self.ascend_browse(),
            _ => {}
        }
    }

    fn handle_detail_key(&mut self, key: KeyEvent) {
        match key.code {
            // Deeper: into the transcript (one more level down, like browse).
            KeyCode::Enter | KeyCode::Char('l') | KeyCode::Char('t') | KeyCode::Right => self.open_transcript(),
            // Back: to whichever view opened the detail.
            KeyCode::Char('q') | KeyCode::Esc | KeyCode::Char('h') | KeyCode::Left => self.mode = self.detail_return,
            KeyCode::Char('o') => self.resume_current(),
            KeyCode::Down | KeyCode::Char('j') => self.detail_scroll = self.detail_scroll.saturating_add(1),
            KeyCode::Up | KeyCode::Char('k') => self.detail_scroll = self.detail_scroll.saturating_sub(1),
            KeyCode::PageDown => self.detail_scroll = self.detail_scroll.saturating_add(15),
            KeyCode::PageUp => self.detail_scroll = self.detail_scroll.saturating_sub(15),
            _ => {}
        }
    }

    fn handle_transcript_key(&mut self, key: KeyEvent) {
        let total = self.transcript_lines.len();
        let viewport = self.transcript_viewport.max(1) as usize;
        // Hard ceiling keeps the last line reachable at the top (so a wrapped
        // final screen can't hide it); `g`/`G` snap to a full first/last screen.
        let hard_max = total.saturating_sub(1) as u16;
        let full_last = total.saturating_sub(viewport) as u16;
        let page = viewport.saturating_sub(2).max(1) as u16;
        match key.code {
            // Back up to Detail (q here is "up a level", not quit).
            KeyCode::Char('q') | KeyCode::Esc | KeyCode::Char('h') | KeyCode::Left => self.mode = Mode::Detail,
            KeyCode::Char('o') => self.resume_current(),
            KeyCode::Down | KeyCode::Char('j') => self.transcript_scroll = self.transcript_scroll.saturating_add(1).min(hard_max),
            KeyCode::Up | KeyCode::Char('k') => self.transcript_scroll = self.transcript_scroll.saturating_sub(1),
            KeyCode::PageDown => self.transcript_scroll = self.transcript_scroll.saturating_add(page).min(hard_max),
            KeyCode::PageUp => self.transcript_scroll = self.transcript_scroll.saturating_sub(page),
            KeyCode::Home | KeyCode::Char('g') => self.transcript_scroll = 0,
            KeyCode::End | KeyCode::Char('G') => self.transcript_scroll = full_last,
            _ => {}
        }
    }
}

fn ts_key(t: DateTime<Utc>) -> u64 {
    t.timestamp().max(0) as u64
}

/// Case-insensitive match of `needle` against a session's recognition fields.
fn matches(s: &Session, needle: &str) -> bool {
    [
        s.title.as_deref(),
        s.first_prompt.as_deref(),
        s.last_prompt.as_deref(),
        s.workspace.as_deref(),
        Some(s.provider),
    ]
    .iter()
    .flatten()
    .any(|f| f.to_lowercase().contains(needle))
}

// ---------------------------------------------------------------------------
// Run loop + rendering
// ---------------------------------------------------------------------------

pub fn run(mut app: App) -> std::io::Result<()> {
    let mut terminal = ratatui::init();
    let result = event_loop(&mut terminal, &mut app);
    ratatui::restore();
    result?;
    // If the user asked to open a session, hand off now that the terminal is
    // back to normal. exec replaces this process and only returns on failure.
    if let Some(spec) = app.pending_launch.take() {
        exec_resume(spec);
    }
    Ok(())
}

fn event_loop(terminal: &mut DefaultTerminal, app: &mut App) -> std::io::Result<()> {
    while !app.should_quit {
        terminal.draw(|frame| draw(frame, app))?;
        if let Event::Key(key) = event::read()? {
            if key.kind == KeyEventKind::Press {
                app.handle_key(key);
            }
        }
    }
    Ok(())
}

/// Replace the ssft process with the harness's resume command. Returns only if
/// exec fails (e.g. the harness binary isn't on PATH).
fn exec_resume(spec: ResumeSpec) {
    use std::os::unix::process::CommandExt;
    let mut cmd = std::process::Command::new(&spec.program);
    cmd.args(&spec.args);
    if let Some(cwd) = &spec.cwd {
        cmd.current_dir(cwd);
    }
    let err = cmd.exec(); // on success, never returns
    eprintln!("ssft: could not launch `{} {}`: {err}", spec.program, spec.args.join(" "));
    std::process::exit(127);
}

const COLOR_ACCENT: Color = Color::Cyan;
const HIGHLIGHT_BG: Color = Color::Rgb(40, 40, 55);

fn draw(frame: &mut Frame, app: &mut App) {
    match app.mode {
        Mode::List => draw_list(frame, app),
        Mode::Browse => draw_browse(frame, app),
        Mode::Detail => draw_detail(frame, app),
        Mode::Transcript => draw_transcript(frame, app),
    }
}

/// Header line shared by list and browse, with the active mode/sort/filter.
fn header_line(app: &App, mode_label: &str, extra: String) -> Line<'static> {
    let mut spans = vec![
        Span::styled(" ssft ", Style::default().bg(COLOR_ACCENT).fg(Color::Black).bold()),
        Span::styled(format!(" {mode_label} "), Style::default().fg(COLOR_ACCENT).bold()),
        Span::raw(extra),
        Span::raw("  ·  sort: "),
        Span::styled(app.sort.label(), Style::default().fg(COLOR_ACCENT).bold()),
    ];
    if !app.filter.is_empty() || app.editing_filter {
        spans.push(Span::raw("  ·  filter: "));
        spans.push(Span::styled(
            format!("{}{}", app.filter, if app.editing_filter { "_" } else { "" }),
            Style::default().fg(Color::Yellow),
        ));
    }
    Line::from(spans)
}

fn footer_line(app: &App, hint: &str) -> Line<'static> {
    if app.editing_filter {
        return Line::from(Span::styled(
            " type to filter  ·  Enter: keep  ·  Esc: clear",
            Style::default().fg(Color::DarkGray),
        ));
    }
    // A transient notice takes the footer over the hint until the next keypress.
    if let Some(status) = &app.status {
        return Line::from(Span::styled(format!(" {status}"), Style::default().fg(Color::Yellow)));
    }
    Line::from(Span::styled(hint.to_string(), Style::default().fg(Color::DarkGray)))
}

fn table_widths() -> [Constraint; 7] {
    [
        Constraint::Length(11),
        Constraint::Min(24),
        Constraint::Length(11),
        Constraint::Length(12),
        Constraint::Length(6),
        Constraint::Length(16),
        Constraint::Length(28),
    ]
}

fn draw_list(frame: &mut Frame, app: &mut App) {
    let [header, body, footer] =
        Layout::vertical([Constraint::Length(1), Constraint::Min(0), Constraint::Length(1)])
            .areas(frame.area());

    let extra = format!(
        "· {} sessions · {} harness(es): {}",
        app.view.len(),
        app.detected.len(),
        if app.detected.is_empty() { "none".into() } else { app.detected.join(", ") },
    );
    frame.render_widget(header_line(app, "flat", extra), header);

    let max_for_unit = |unit: &str| -> u64 {
        app.view
            .iter()
            .map(|&i| app.all[i].substance())
            .filter(|s| s.unit() == unit)
            .map(|s| s.value())
            .max()
            .unwrap_or(0)
    };

    let rows = app.view.iter().map(|&i| {
        let s = &app.all[i];
        let subst = s.substance();
        let title = match s.display_title() {
            Some(t) => ellipsize(t, 40),
            None => "· (unnamed)".to_string(),
        };
        Row::new(vec![
            Cell::from(s.provider).style(Style::default().fg(Color::DarkGray)),
            Cell::from(title),
            Cell::from(Span::styled(substance_str(subst), Style::default().fg(COLOR_ACCENT))),
            Cell::from(Span::styled(bar_str(subst.value(), max_for_unit(subst.unit()), 12), Style::default().fg(COLOR_ACCENT))),
            Cell::from(format!("{:>5}", s.counts.messages())),
            Cell::from(last_activity_str(s.last_activity())),
            Cell::from(ellipsize(s.workspace.as_deref().unwrap_or(""), 28)),
        ])
    });

    let table = Table::new(rows, table_widths())
        .header(header_cells(["HARNESS", "TITLE", "SUBST", "", "MSGS", "LAST ACTIVITY", "WORKSPACE"]))
        .row_highlight_style(Style::default().bg(HIGHLIGHT_BG).add_modifier(Modifier::BOLD))
        .highlight_symbol("▶ ");

    let mut state = TableState::default().with_selected(Some(app.selected));
    frame.render_stateful_widget(table, body, &mut state);

    frame.render_widget(
        footer_line(app, " ↑↓/jk move · Enter: detail · o: resume · s: sort · /: filter · Tab: tree · q: quit"),
        footer,
    );
}

fn draw_browse(frame: &mut Frame, app: &mut App) {
    let [header, body, footer] =
        Layout::vertical([Constraint::Length(1), Constraint::Min(0), Constraint::Length(1)])
            .areas(frame.area());

    frame.render_widget(header_line(app, "tree", format!("▸ {}", app.breadcrumb())), header);

    let entries = app.browse_entries();

    // Bar scale: max token-substance among this level's entries (dirs + token
    // sessions); non-token sessions don't participate.
    let max_tokens = entries
        .iter()
        .map(|e| match *e {
            Entry::Dir { node, .. } => app.nodes[node].roll_tokens,
            Entry::Session(i) => match &app.all[i].tokens {
                Some(t) => t.substance(),
                None => 0,
            },
            Entry::Up => 0,
        })
        .max()
        .unwrap_or(0);

    let rows = entries.iter().map(|e| match e {
        Entry::Up => Row::new(vec![
            Cell::from(""),
            Cell::from(Span::styled("..", Style::default().fg(Color::Gray))),
            Cell::from(""),
            Cell::from(""),
            Cell::from(""),
            Cell::from(""),
        ]),
        Entry::Dir { node, name } => {
            let n = &app.nodes[*node];
            let subst = if n.roll_tokens == 0 {
                "-".to_string() // subtree has no token-measured sessions
            } else {
                format!("{}{} tok", if n.roll_has_nontoken { "≥" } else { "" }, human(n.roll_tokens))
            };
            Row::new(vec![
                Cell::from(Span::styled("dir", Style::default().fg(Color::DarkGray))),
                Cell::from(Span::styled(format!("{}/", ellipsize(name, 38)), Style::default().fg(Color::Blue).bold())),
                Cell::from(Span::styled(subst, Style::default().fg(COLOR_ACCENT))),
                Cell::from(Span::styled(bar_str(n.roll_tokens, max_tokens, 12), Style::default().fg(COLOR_ACCENT))),
                Cell::from(format!("{:>4} s", n.roll_count)),
                Cell::from(last_activity_str(n.roll_last)),
            ])
        }
        Entry::Session(i) => {
            let s = &app.all[*i];
            let subst = s.substance();
            let title = match s.display_title() {
                Some(t) => format!("· {}", ellipsize(t, 38)),
                None => "· (unnamed)".to_string(),
            };
            let bar_val = match &s.tokens {
                Some(t) => t.substance(),
                None => 0,
            };
            Row::new(vec![
                Cell::from(Span::styled(s.provider, Style::default().fg(Color::DarkGray))),
                Cell::from(title),
                Cell::from(Span::styled(substance_str(subst), Style::default().fg(COLOR_ACCENT))),
                Cell::from(Span::styled(bar_str(bar_val, max_tokens, 12), Style::default().fg(COLOR_ACCENT))),
                Cell::from(format!("{:>5}", s.counts.messages())),
                Cell::from(last_activity_str(s.last_activity())),
            ])
        }
    });

    let widths = [
        Constraint::Length(11),
        Constraint::Min(24),
        Constraint::Length(11),
        Constraint::Length(12),
        Constraint::Length(6),
        Constraint::Length(16),
    ];
    let table = Table::new(rows, widths)
        .header(header_cells(["KIND", "NAME", "SUBST", "", "COUNT", "LAST ACTIVITY"]))
        .row_highlight_style(Style::default().bg(HIGHLIGHT_BG).add_modifier(Modifier::BOLD))
        .highlight_symbol("▶ ");

    let mut state = TableState::default().with_selected(Some(app.bsel.min(entries.len().saturating_sub(1))));
    frame.render_stateful_widget(table, body, &mut state);

    frame.render_widget(
        footer_line(app, " ↑↓/jk move · Enter/→: open · o: resume · ←/h/Bksp: up · s: sort · /: filter · Tab: flat · q: quit"),
        footer,
    );
}

fn draw_detail(frame: &mut Frame, app: &mut App) {
    let [body, footer] =
        Layout::vertical([Constraint::Min(0), Constraint::Length(1)]).areas(frame.area());

    let s = &app.all[app.detail_index];
    let title = s.display_title().unwrap_or("(unnamed)").to_string();
    let block = Block::default()
        .borders(Borders::ALL)
        .title(Span::styled(format!(" {} ", ellipsize(&title, 60)), Style::default().fg(COLOR_ACCENT).bold()));
    let para = Paragraph::new(detail_lines(s))
        .block(block)
        .wrap(Wrap { trim: false })
        .scroll((app.detail_scroll, 0));
    frame.render_widget(para, body);

    frame.render_widget(
        footer_line(app, " ↑↓/jk scroll · Enter/→/t: transcript · o: resume · Esc/h/←: back"),
        footer,
    );
}

/// Visible `[start, end)` slice of transcript lines for a given scroll offset,
/// viewport height, and total line count. Windowing the slice keeps per-frame
/// work proportional to the viewport, not the whole (uncapped) transcript.
fn transcript_window(scroll: usize, viewport: usize, total: usize) -> (usize, usize) {
    if total == 0 || viewport == 0 {
        return (0, 0);
    }
    let start = scroll.min(total - 1);
    let end = (start + viewport).min(total);
    (start, end)
}

fn draw_transcript(frame: &mut Frame, app: &mut App) {
    let [body, footer] =
        Layout::vertical([Constraint::Min(0), Constraint::Length(1)]).areas(frame.area());

    // Body height inside the borders is the viewport; record it for paging.
    let viewport = body.height.saturating_sub(2) as usize;
    app.transcript_viewport = viewport as u16;

    let total = app.transcript_lines.len();
    let (start, end) = transcript_window(app.transcript_scroll as usize, viewport, total);
    app.transcript_scroll = start as u16; // reconcile state with the clamp
    // Clone only the visible window, never the whole transcript.
    let visible: Vec<Line> = app.transcript_lines[start..end].to_vec();

    let s = &app.all[app.detail_index];
    let title = s.display_title().unwrap_or("(unnamed)").to_string();
    let pos = if total == 0 {
        "empty".to_string()
    } else {
        format!("{}-{} / {}", start + 1, end, total)
    };
    let block = Block::default()
        .borders(Borders::ALL)
        .title(Span::styled(
            format!(" transcript · {} ", ellipsize(&title, 48)),
            Style::default().fg(COLOR_ACCENT).bold(),
        ))
        .title_bottom(Line::from(Span::styled(format!(" {pos} "), Style::default().fg(Color::DarkGray))).right_aligned());
    let para = Paragraph::new(visible).block(block).wrap(Wrap { trim: false });
    frame.render_widget(para, body);

    frame.render_widget(
        footer_line(app, " ↑↓/jk scroll · g/G top/bottom · o: resume · Esc/h/←: back"),
        footer,
    );
}

/// Build the displayable transcript: a role-headed, block-by-block rendering.
/// Reasoning and tool payloads are abbreviated so the conversation stays
/// skimmable; full prose text is kept and wrapped by the Paragraph. The full
/// set is built once and cached; `draw_transcript` windows it, so there is no
/// length cap.
fn render_transcript(messages: &[Message]) -> Vec<Line<'static>> {
    let mut out: Vec<Line<'static>> = Vec::new();

    for m in messages {
        let (marker, color) = match m.role {
            Role::User => ("▌user", Color::Green),
            Role::Assistant => ("▌assistant", COLOR_ACCENT),
            Role::Tool => ("▌tool", Color::Yellow),
            Role::System => ("▌system", Color::Gray),
        };
        let when = m
            .timestamp
            .map(|t| format!("  {}", t.format("%Y-%m-%d %H:%M")))
            .unwrap_or_default();

        out.push(Line::from(""));
        out.push(Line::from(vec![
            Span::styled(marker.to_string(), Style::default().fg(color).bold()),
            Span::styled(when, Style::default().fg(Color::DarkGray)),
        ]));

        for block in &m.blocks {
            match block {
                MsgBlock::Text(text) => {
                    for line in text.lines() {
                        out.push(Line::from(format!("  {line}")));
                    }
                }
                MsgBlock::Thinking(text) => {
                    out.push(Line::from(Span::styled(
                        format!("  thinking: {}", ellipsize(&first_line(text), 100)),
                        Style::default().fg(Color::DarkGray).add_modifier(Modifier::ITALIC),
                    )));
                }
                MsgBlock::ToolUse { name, input } => {
                    out.push(Line::from(vec![
                        Span::styled(format!("  » {name}"), Style::default().fg(Color::Blue).bold()),
                        Span::styled(format!("  {}", ellipsize(input, 80)), Style::default().fg(Color::DarkGray)),
                    ]));
                }
                MsgBlock::ToolResult { text, is_error } => {
                    let (sym, c) = if *is_error { ("« error", Color::Red) } else { ("«", Color::Magenta) };
                    out.push(Line::from(vec![
                        Span::styled(format!("  {sym} "), Style::default().fg(c)),
                        Span::styled(ellipsize(&first_line(text), 100), Style::default().fg(Color::DarkGray)),
                    ]));
                }
            }
        }
    }
    out
}

fn first_line(s: &str) -> String {
    s.lines().next().unwrap_or("").to_string()
}

fn header_cells<'a, const N: usize>(labels: [&'a str; N]) -> Row<'a> {
    Row::new(labels).style(Style::default().fg(Color::Gray).add_modifier(Modifier::BOLD | Modifier::UNDERLINED))
}

fn last_activity_str(t: Option<DateTime<Utc>>) -> String {
    t.map(|t| t.format("%Y-%m-%d %H:%M").to_string()).unwrap_or_else(|| "?".into())
}

fn detail_lines(s: &Session) -> Vec<Line<'static>> {
    fn field(label: &str, value: String) -> Line<'static> {
        Line::from(vec![
            Span::styled(format!("{label:>12}  "), Style::default().fg(Color::DarkGray)),
            Span::raw(value),
        ])
    }
    fn opt(v: Option<&str>) -> String {
        v.unwrap_or("-").to_string()
    }

    let mut out = vec![
        field("harness", s.provider.to_string()),
        field("id", s.id.clone()),
        field("workspace", opt(s.workspace.as_deref())),
        field("branch", opt(s.branch.as_deref())),
        field("model", opt(s.model.as_deref())),
        field("agent", opt(s.agent_name.as_deref())),
    ];

    let span = match (s.first_ts, s.last_ts) {
        (Some(f), Some(l)) => format!("{} → {}", f.format("%Y-%m-%d %H:%M"), l.format("%Y-%m-%d %H:%M")),
        _ => s.mtime.map(|m| format!("{} (file mtime)", m.format("%Y-%m-%d %H:%M"))).unwrap_or_else(|| "-".into()),
    };
    out.push(field("activity", span));
    out.push(field("messages", format!("{} user / {} assistant / {} tool", s.counts.user, s.counts.assistant, s.counts.tool)));
    out.push(field("substance", substance_str(s.substance())));
    match &s.tokens {
        Some(t) => out.push(field("tokens", format!(
            "in {} · out {} · cache+ {} · cache_r {} · reason {}",
            human(t.input), human(t.output), human(t.cache_creation), human(t.cache_read), human(t.reasoning),
        ))),
        None => out.push(field("tokens", "(not recorded by this harness)".into())),
    }
    out.push(field("size", format!("{} on disk", human(s.bytes))));
    if s.health.unknown > 0 || s.health.unparsed > 0 {
        out.push(field("parse health", format!("{} unknown · {} unparsed", s.health.unknown, s.health.unparsed)));
    }

    push_prompt(&mut out, "first prompt", s.first_prompt.as_deref());
    push_prompt(&mut out, "last prompt", s.last_prompt.as_deref());
    out
}

fn push_prompt(out: &mut Vec<Line<'static>>, label: &str, text: Option<&str>) {
    out.push(Line::from(""));
    out.push(Line::from(Span::styled(format!("── {label} ──"), Style::default().fg(COLOR_ACCENT).add_modifier(Modifier::BOLD))));
    match text {
        Some(t) if !t.trim().is_empty() => {
            for line in t.lines() {
                out.push(Line::from(line.to_string()));
            }
        }
        _ => out.push(Line::from(Span::styled("(none)", Style::default().fg(Color::DarkGray)))),
    }
}

fn bar_str(value: u64, max: u64, width: usize) -> String {
    if max == 0 {
        return " ".repeat(width);
    }
    let filled = (((value as f64 / max as f64) * width as f64).round() as usize).min(width);
    format!("{}{}", "█".repeat(filled), "░".repeat(width - filled))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::TokenTotals;
    use chrono::TimeZone;
    use ratatui::backend::TestBackend;
    use ratatui::buffer::Buffer;
    use ratatui::Terminal;

    #[test]
    fn snapshot_transcript() {
        let msgs = vec![
            Message {
                role: Role::User,
                timestamp: Some(chrono::Utc.with_ymd_and_hms(2026, 6, 26, 17, 3, 0).unwrap()),
                blocks: vec![MsgBlock::Text("The bridge isn't resolving the gateway MAC.\nCan you debug it?".into())],
            },
            Message {
                role: Role::Assistant,
                timestamp: Some(chrono::Utc.with_ymd_and_hms(2026, 6, 26, 17, 3, 30).unwrap()),
                blocks: vec![
                    MsgBlock::Thinking("Let me reproduce the ARP path before guessing.".into()),
                    MsgBlock::Text("I'll check the neighbor table first.".into()),
                    MsgBlock::ToolUse { name: "Bash".into(), input: "{\"command\":\"ip neigh show\"}".into() },
                ],
            },
            Message {
                role: Role::Tool,
                timestamp: Some(chrono::Utc.with_ymd_and_hms(2026, 6, 26, 17, 3, 31).unwrap()),
                blocks: vec![MsgBlock::ToolResult { text: "192.168.1.1 dev eth0 lladdr aa:bb FAILED".into(), is_error: false }],
            },
        ];
        let lines = render_transcript(&msgs);
        let mut term = Terminal::new(TestBackend::new(80, 18)).unwrap();
        let para = Paragraph::new(lines).wrap(Wrap { trim: false });
        term.draw(|f| f.render_widget(para, f.area())).unwrap();
        println!("\n===== TRANSCRIPT VIEW =====\n{}", to_text(term.backend().buffer()));
    }

    #[test]
    fn snapshot_transcript_windowed() {
        // 40 numbered lines, scrolled to 10, in a short viewport: the frame must
        // show only the window and the "11-18 / 40"-style position indicator.
        let mut app = App::new(sample(), vec!["claude-code"]);
        app.mode = Mode::Transcript;
        app.detail_index = 0;
        app.transcript_lines = (0..40).map(|i| Line::from(format!("line {i:02}"))).collect();
        app.transcript_scroll = 10;
        let mut term = Terminal::new(TestBackend::new(60, 10)).unwrap();
        term.draw(|f| draw(f, &mut app)).unwrap();
        let text = to_text(term.backend().buffer());
        println!("\n===== TRANSCRIPT (windowed) =====\n{text}");
        // Only windowed lines are present - line 00 (above) and line 39 (below)
        // are not rendered; the boundary lines and a position indicator are.
        assert!(text.contains("line 10"), "window start missing");
        assert!(!text.contains("line 00"), "line above window leaked in");
        assert!(!text.contains("line 39"), "line below window leaked in");
        assert!(text.contains("/ 40"), "position indicator missing");
    }

    #[test]
    fn transcript_window_bounds() {
        // Normal slice.
        assert_eq!(transcript_window(3, 10, 100), (3, 13));
        // Viewport taller than content.
        assert_eq!(transcript_window(0, 10, 5), (0, 5));
        // Scroll past the end clamps so the last line stays reachable.
        assert_eq!(transcript_window(1000, 10, 50), (49, 50));
        // Degenerate inputs.
        assert_eq!(transcript_window(0, 0, 100), (0, 0));
        assert_eq!(transcript_window(5, 10, 0), (0, 0));
    }

    #[test]
    fn transcript_is_not_capped() {
        // Long session: every message is kept, no truncation notice.
        let msgs: Vec<Message> = (0..3000)
            .map(|_| Message { role: Role::Assistant, timestamp: None, blocks: vec![MsgBlock::Text("x".into())] })
            .collect();
        let lines = render_transcript(&msgs);
        let joined: String = lines.iter().flat_map(|l| l.spans.iter()).map(|s| s.content.as_ref()).collect();
        assert!(!joined.contains("truncated"), "windowing should remove the cap");
        // 3000 messages × (blank + header + one text line) = 9000 lines, all kept.
        assert_eq!(lines.len(), 9000, "got {}", lines.len());
    }

    fn to_text(buf: &Buffer) -> String {
        let mut out = String::new();
        for y in 0..buf.area.height {
            for x in 0..buf.area.width {
                out.push_str(buf[(x, y)].symbol());
            }
            while out.ends_with(' ') {
                out.pop();
            }
            out.push('\n');
        }
        out
    }

    fn sess(provider: &'static str, ws: Option<&str>, title: Option<&str>, out_tok: u64, y: u32, mo: u32, d: u32) -> Session {
        let mut s = Session::new(provider, format!("{provider}-{y}{mo}{d}"));
        s.workspace = ws.map(str::to_string);
        s.title = title.map(str::to_string);
        s.first_prompt = Some(format!("first prompt for {}", title.unwrap_or("untitled")));
        s.counts.user = 5;
        s.counts.assistant = 6;
        if out_tok > 0 {
            s.tokens = Some(TokenTotals { output: out_tok, ..Default::default() });
        } else {
            s.text_chars = 1500;
        }
        s.last_ts = Some(Utc.with_ymd_and_hms(y as i32, mo, d, 12, 0, 0).unwrap());
        s
    }

    fn sample() -> Vec<Session> {
        vec![
            sess("claude-code", Some("/home/ian/swxtch-core"), Some("Fix ARP gateway"), 8_400_000, 2026, 6, 26),
            sess("claude-code", Some("/home/ian/swxtch-core"), Some("Refactor multicast"), 5_000_000, 2026, 6, 25),
            sess("claude-code", Some("/home/ian/clouds/aws"), Some("TCP port forwarding"), 2_500_000, 2026, 6, 25),
            sess("codex", Some("/home/ian/clouds/aws"), Some("Terraform import"), 94_000, 2026, 6, 24),
            sess("claude-code", Some("/home/ian/clouds/oracle"), Some("Two VMs"), 238_000, 2026, 6, 24),
            sess("claude-code", Some("/home/ian"), Some("Set up Obsidian"), 1_800_000, 2026, 6, 29),
            sess("crush", None, None, 0, 2026, 6, 20),
        ]
    }

    #[test]
    fn snapshot_browse_root() {
        let mut app = App::new(sample(), vec!["claude-code", "codex", "crush"]);
        app.mode = Mode::Browse;
        let mut term = Terminal::new(TestBackend::new(110, 14)).unwrap();
        term.draw(|f| draw(f, &mut app)).unwrap();
        println!("\n===== BROWSE: {} =====\n{}", app.breadcrumb(), to_text(term.backend().buffer()));
    }

    #[test]
    fn snapshot_browse_descend() {
        let mut app = App::new(sample(), vec!["claude-code", "codex", "crush"]);
        app.mode = Mode::Browse;
        // Descend into the first directory entry (compressed chain) twice.
        app.activate_browse_entry();
        let crumb1 = app.breadcrumb();
        let mut term = Terminal::new(TestBackend::new(110, 14)).unwrap();
        term.draw(|f| draw(f, &mut app)).unwrap();
        println!("\n===== BROWSE descended → {} =====\n{}", crumb1, to_text(term.backend().buffer()));
    }

    fn dir_names(app: &App) -> Vec<String> {
        app.browse_entries()
            .iter()
            .filter_map(|e| match e {
                Entry::Dir { name, .. } => Some(name.clone()),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn chain_compression_collapses_single_child() {
        let mut app = App::new(sample(), vec![]);
        // A no-workspace session gives root two children, so browsing starts at
        // the synthetic root: the home chain is compressed to one "home/ian"
        // row, alongside the "(no workspace)" bucket.
        let root_dirs = dir_names(&app);
        assert!(root_dirs.iter().any(|n| n == "home/ian"), "got {root_dirs:?}");
        assert!(root_dirs.iter().any(|n| n == "(no workspace)"), "got {root_dirs:?}");

        // Descend into home/ian (recency-sorted to the top) and see its dirs.
        app.activate_browse_entry();
        assert_eq!(app.nodes[app.cur].full_path.as_deref(), Some("/home/ian"));
        let names = dir_names(&app);
        assert!(names.iter().any(|n| n == "swxtch-core"), "got {names:?}");
        assert!(names.iter().any(|n| n == "clouds"), "got {names:?}");
    }

    #[test]
    fn rollup_sums_subtree_tokens() {
        let mut app = App::new(sample(), vec![]);
        app.activate_browse_entry(); // descend into /home/ian
        let ian = app.cur;
        assert_eq!(app.nodes[ian].full_path.as_deref(), Some("/home/ian"));
        // 8.4M + 5M + 2.5M + 0.094M + 0.238M + 1.8M
        assert_eq!(app.nodes[ian].roll_tokens, 8_400_000 + 5_000_000 + 2_500_000 + 94_000 + 238_000 + 1_800_000);
        assert_eq!(app.nodes[ian].roll_count, 6);
    }

    #[test]
    fn filter_in_browse_hides_nonmatching_dirs() {
        let mut app = App::new(sample(), vec![]);
        app.mode = Mode::Browse;
        app.activate_browse_entry(); // descend into /home/ian
        app.filter = "terraform".into();
        let dirs = dir_names(&app);
        // Only clouds (→aws holds the Terraform session) should survive.
        assert!(dirs.iter().any(|n| n.starts_with("clouds")), "got {dirs:?}");
        assert!(!dirs.iter().any(|n| n == "swxtch-core"), "got {dirs:?}");
    }
}
