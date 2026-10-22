//! `lbm` — the CLI pi extensions shell out to.
//!
//! Human-readable text by default; `--json` for machine consumption.
//! Exit code is nonzero only on operational failure (not on empty results).
//!
//! ```sh
//! lbm remember "User prefers Python for scripting" --scope project
//! lbm search "python" --json
//! lbm forget "python"            # lists matches; --apply deletes
//! lbm tour --json
//! lbm status
//! lbm review                     # lists needs_review + quarantined
//! lbm review --accept 3 --supersede 1
//! echo "text" | lbm remember --auto   # auto-capture path (distills facts)
//! ```

use std::collections::HashMap;

use anyhow::{Context, Result, anyhow};
use lb_memory::model::GlinerExtractor;
use lb_memory::pipeline::{Pipeline, PipelineOptions};
use lb_memory::store::Store;
use lb_memory::types::{Provenance, Scope};

const DEFAULT_LABELS_NOTE: &str = "person,organization,location,product,event,date,technology,concept";

#[derive(Default)]
struct Args {
    positional: Vec<String>,
    flags: HashMap<String, String>,
    bools: Vec<String>,
}

fn parse_args(raw: &[String]) -> Args {
    let mut out = Args::default();
    let mut i = 0;
    while i < raw.len() {
        let a = &raw[i];
        if let Some(name) = a.strip_prefix("--") {
            if let Some((k, v)) = name.split_once('=') {
                out.flags.insert(k.to_string(), v.to_string());
            } else if i + 1 < raw.len() && !raw[i + 1].starts_with("--") {
                // --key value, unless the flag is a known boolean.
                if is_bool_flag(name) {
                    out.bools.push(name.to_string());
                } else {
                    out.flags.insert(name.to_string(), raw[i + 1].clone());
                    i += 1;
                }
            } else {
                out.bools.push(name.to_string());
            }
        } else {
            out.positional.push(a.clone());
        }
        i += 1;
    }
    out
}

fn is_bool_flag(name: &str) -> bool {
    // NB: accept/drop take values (--accept <id>); listing them here would
    // swallow the id into positionals.
    matches!(name, "json" | "auto" | "apply" | "verbose" | "include-review")
}

impl Args {
    fn bool(&self, name: &str) -> bool {
        self.bools.iter().any(|b| b == name)
    }
    fn str(&self, name: &str) -> Option<&str> {
        self.flags.get(name).map(|s| s.as_str())
    }
}

fn home() -> String {
    std::env::var("HOME").unwrap_or_else(|_| "/tmp".into())
}

fn default_db() -> String {
    std::env::var("LB_MEMORY_DB")
        .unwrap_or_else(|_| format!("{}/.pi/agent/lb-memory/memory.lbdb", home()))
}

fn default_snapshot() -> String {
    std::env::var("LB_MEMORY_SNAPSHOT").unwrap_or_else(|_| {
        format!(
            "{}/.cache/huggingface/hub/models--DanKau--gliner2.5-base-v1-onnx/snapshots/b4920ab2b4f39b2da60c9761802e3d8cc6ec9993",
            home()
        )
    })
}

fn user_id(args: &Args) -> String {
    args.str("user")
        .map(str::to_string)
        .or_else(|| std::env::var("LB_MEMORY_USER").ok())
        .or_else(|| std::env::var("USER").ok())
        .unwrap_or_else(|| "default".into())
}

/// Project slug: git root basename when inside a repo, else cwd slug.
/// Mirrors the mem0 plugin (monorepo subdirs share one pool).
fn project_slug() -> String {
    if let Ok(root) = git_root() {
        return slug(&root);
    }
    slug(&std::env::current_dir().map(|p| p.display().to_string()).unwrap_or_else(|_| "default".into()))
}

fn git_root() -> Result<String> {
    let out = std::process::Command::new("git").args(["rev-parse", "--show-toplevel"]).output()?;
    if !out.status.success() {
        return Err(anyhow!("not a git repo"));
    }
    Ok(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

fn slug(s: &str) -> String {
    let base = s.rsplit('/').next().unwrap_or(s);
    let slug: String = base
        .chars()
        .map(|c| if c.is_alphanumeric() { c.to_ascii_lowercase() } else { '-' })
        .collect();
    let slug = slug.trim_matches('-').to_string();
    if slug.is_empty() {
        "default".into()
    } else {
        slug
    }
}

fn resolve_scope(args: &Args) -> Result<(Scope, ScopeMode)> {
    let mode = match args.str("scope").unwrap_or("project") {
        "session" => ScopeMode::Session,
        "global" => ScopeMode::Global,
        _ => ScopeMode::Project,
    };
    let session = args
        .str("session")
        .map(str::to_string)
        .or_else(|| std::env::var("LB_MEMORY_SESSION").ok())
        .or_else(|| std::env::var("PI_SESSION_ID").ok());
    if mode == ScopeMode::Session && session.is_none() {
        return Err(anyhow!("--scope session needs --session <id> or PI_SESSION_ID"));
    }
    Ok((
        Scope {
            user_id: user_id(args),
            project: args.str("project").map(str::to_string).unwrap_or_else(project_slug),
            // Session filters recall only in session scope (mem0 semantics:
            // project = user+project). It is still *recorded* in provenance.
            // NB: PI_SESSION_ID is set inside agent shells — never let it
            // leak into project/global recall or everything misses.
            session_id: if mode == ScopeMode::Session { session } else { None },
        },
        mode,
    ))
}

#[derive(PartialEq, Eq)]
enum ScopeMode {
    Project,
    Session,
    Global,
}

fn read_text(args: &Args) -> Result<String> {
    if !args.positional.is_empty() {
        return Ok(args.positional.join(" "));
    }
    // Piped stdin (auto-capture path).
    use std::io::Read;
    let mut buf = String::new();
    std::io::stdin().read_to_string(&mut buf)?;
    Ok(buf.trim().to_string())
}

fn print_json(v: &serde_json::Value) {
    println!("{}", serde_json::to_string_pretty(v).unwrap());
}

fn main() -> Result<()> {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    if raw.is_empty() {
        eprintln!("usage: lbm <remember|search|forget|tour|status|review> [--json] ...");
        std::process::exit(2);
    }
    let cmd = raw[0].clone();
    let args = parse_args(&raw[1..]);
    match cmd.as_str() {
        "remember" => cmd_remember(args),
        "search" => cmd_search(args),
        "forget" => cmd_forget(args),
        "tour" => cmd_tour(args),
        "status" => cmd_status(args),
        "review" => cmd_review(args),
        _ => Err(anyhow!("unknown command {cmd} (see module docs)")),
    }
}

fn open_store(args: &Args) -> Result<Store> {
    let path = args.str("db").map(str::to_string).unwrap_or_else(default_db);
    Store::open(&path).with_context(|| format!("opening {path}"))
}

fn cmd_remember(args: Args) -> Result<()> {
    let text = read_text(&args)?;
    if text.trim().is_empty() {
        return Err(anyhow!("nothing to remember (empty text)"));
    }
    let (scope, _mode) = resolve_scope(&args)?;
    let store = open_store(&args)?;
    let pipe = Pipeline::new(
        &store,
        PipelineOptions {
            extract_threshold: args.str("threshold").and_then(|s| s.parse().ok()).unwrap_or(0.7),
            ..Default::default()
        },
    );
    let mut extractor = GlinerExtractor::new(args.str("snapshot").map(str::to_string).unwrap_or_else(default_snapshot));
    let mut prov = Provenance {
        // Record the session for provenance even when it isn't a recall
        // filter (project/global modes): explicit flag wins, then env.
        session_id: args
            .str("session")
            .map(str::to_string)
            .or_else(|| std::env::var("LB_MEMORY_SESSION").ok())
            .or_else(|| std::env::var("PI_SESSION_ID").ok()),
        message_refs: args
            .str("message-ref")
            .map(|s| s.split(',').map(|x| x.trim().to_string()).filter(|x| !x.is_empty()).collect())
            .unwrap_or_default(),
        extractor: "gliner2.5".into(),
        model: "DanKau/gliner2.5-base-v1-onnx".into(),
        auto_captured: false,
    };
    let importance = args.str("importance").and_then(|s| s.parse().ok());
    let reports = if args.bool("auto") {
        prov.auto_captured = true;
        pipe.remember_auto(&mut extractor, &text, &scope, prov)?
    } else {
        pipe.remember(&mut extractor, &text, &scope, importance, prov)?
    };

    if args.bool("json") {
        print_json(&serde_json::json!({ "reports": reports }));
        return Ok(());
    }
    for r in &reports {
        let step1: Vec<String> =
            r.extracted.iter().map(|m| format!("{}[{:.2}]", m.text, m.confidence)).collect();
        println!("extract: {}", step1.join(", "));
        let step3: Vec<String> = r
            .resolved
            .iter()
            .map(|e| format!("{}{}", e.canonical, if e.created { " (new)" } else { "" }))
            .collect();
        println!("resolve: {}", step3.join(", "));
        match &r.decision {
            lb_memory::types::FactDecision::New { .. } => println!("decision: new"),
            lb_memory::types::FactDecision::Duplicate { of, similarity } => {
                println!("decision: duplicate of #{of} (sim {similarity:.2}) — strengthened")
            }
            lb_memory::types::FactDecision::Conflict { with, cues } => {
                println!("decision: conflict with {with:?} [{}] — needs review", cues.join("; "))
            }
            lb_memory::types::FactDecision::LowConfidence { reason } => {
                println!("decision: low confidence ({reason}) — quarantined")
            }
        }
        match &r.written {
            Some(w) => println!("stored #{} [{}|{}]: {}", w.id, w.category.as_str(), w.status.as_str(), w.content),
            None => println!("stored: nothing"),
        }
    }
    Ok(())
}

fn cmd_search(args: Args) -> Result<()> {
    let query = read_text(&args)?;
    let (scope, mode) = resolve_scope(&args)?;
    let store = open_store(&args)?;
    let limit: usize = args.str("limit").and_then(|s| s.parse().ok()).unwrap_or(10);
    // Global scope searches across projects.
    let hits = if mode == ScopeMode::Global {
        global_recall(&store, &query, &scope.user_id, limit)?
    } else {
        Pipeline::new(&store, PipelineOptions::default()).recall(&query, &scope, limit)?
    };
    if args.bool("json") {
        print_json(&serde_json::json!({ "hits": hits }));
        return Ok(());
    }
    if hits.is_empty() {
        println!("(no matches)");
        return Ok(());
    }
    for h in &hits {
        println!("#{} [{:.2}] [{}] {}", h.entry.id, h.score, h.entry.category.as_str(), h.entry.content);
        if !h.entities.is_empty() {
            println!("    entities: {}", h.entities.join(", "));
        }
    }
    Ok(())
}

/// Global recall: user-scoped, cross-project.
fn global_recall(
    store: &Store,
    query: &str,
    user: &str,
    limit: usize,
) -> Result<Vec<lb_memory::types::SearchHit>> {
    use lb_memory::types::SearchHit;
    use std::collections::HashSet;
    let mut hits: Vec<SearchHit> = Vec::new();
    let mut seen: HashSet<i64> = HashSet::new();
    for (canonical, _, _) in store.entities_matching(query, 5)? {
        // entities_matching is global already (no scope columns on Entity).
        for (entry, conf) in store.memories_for_entity_global(&canonical, user, 3)? {
            if seen.insert(entry.id) {
                hits.push(SearchHit { entry, score: conf as f32 + 0.1, entities: vec![canonical.clone()] });
            }
        }
    }
    for (entry, score) in store.search_content_global(query, user, limit)? {
        if seen.insert(entry.id) {
            let entities = store.entities_for_memory(entry.id).unwrap_or_default();
            hits.push(SearchHit { entry, score: score as f32, entities });
        }
    }
    hits.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
    hits.truncate(limit.max(1));
    Ok(hits)
}

fn cmd_forget(args: Args) -> Result<()> {
    let query = read_text(&args)?;
    let (scope, mode) = resolve_scope(&args)?;
    let store = open_store(&args)?;
    let limit: usize = args.str("limit").and_then(|s| s.parse().ok()).unwrap_or(10);
    let hits = if mode == ScopeMode::Global {
        global_recall(&store, &query, &scope.user_id, limit)?
    } else {
        Pipeline::new(&store, PipelineOptions::default()).recall(&query, &scope, limit)?
    };
    if args.bool("apply") {
        let mut deleted = 0;
        for h in &hits {
            store.delete_memory(h.entry.id)?;
            deleted += 1;
        }
        if args.bool("json") {
            print_json(&serde_json::json!({ "deleted": deleted }));
        } else {
            println!("deleted {deleted} memories");
        }
        return Ok(());
    }
    if args.bool("json") {
        print_json(&serde_json::json!({ "matches": hits, "hint": "re-run with --apply to delete" }));
        return Ok(());
    }
    if hits.is_empty() {
        println!("(no matches)");
        return Ok(());
    }
    for h in &hits {
        println!("#{} [{}] {}", h.entry.id, h.entry.category.as_str(), h.entry.content);
    }
    println!("re-run with --apply to delete these");
    Ok(())
}

fn cmd_tour(args: Args) -> Result<()> {
    let (scope, mode) = resolve_scope(&args)?;
    let store = open_store(&args)?;
    let limit: usize = args.str("limit").and_then(|s| s.parse().ok()).unwrap_or(100);
    let entries = match mode {
        ScopeMode::Global => store.list_global(&scope.user_id, limit)?,
        _ => Pipeline::new(&store, PipelineOptions::default()).tour(&scope, limit)?,
    };
    if args.bool("json") {
        print_json(&serde_json::json!({ "entries": entries }));
        return Ok(());
    }
    let mut current = String::new();
    for e in &entries {
        if e.category.as_str() != current {
            current = e.category.as_str().to_string();
            println!("## {current}");
        }
        println!("#{} [{}] {}", e.id, e.status.as_str(), e.content);
    }
    if entries.is_empty() {
        println!("(no memories)");
    }
    Ok(())
}

fn cmd_status(args: Args) -> Result<()> {
    let store = open_store(&args)?;
    let (mem, ent, men, rev) = store.counts()?;
    let path = args.str("db").map(str::to_string).unwrap_or_else(default_db);
    if args.bool("json") {
        print_json(&serde_json::json!({
            "db": path, "memories": mem, "entities": ent,
            "mentions": men, "needs_review": rev,
            "labels": DEFAULT_LABELS_NOTE,
        }));
        return Ok(());
    }
    println!("db: {path}");
    println!("memories: {mem}, entities: {ent}, mentions: {men}, needs_review: {rev}");
    Ok(())
}

fn cmd_review(args: Args) -> Result<()> {
    let store = open_store(&args)?;
    if let Some(id) = args.str("accept").and_then(|s| s.parse::<i64>().ok()) {
        let supersede = args.str("supersede").and_then(|s| s.parse::<i64>().ok());
        let pipe = Pipeline::new(&store, PipelineOptions::default());
        match pipe.review(id, true, supersede) {
            Ok(entry) => {
                if args.bool("json") {
                    print_json(&serde_json::json!({ "accepted": entry }));
                } else {
                    println!("accepted #{}: {}", entry.id, entry.content);
                }
                return Ok(());
            }
            Err(e) => return Err(e),
        }
    }
    if let Some(id) = args.str("drop").and_then(|s| s.parse::<i64>().ok()) {
        store.delete_memory(id)?;
        if args.bool("json") {
            print_json(&serde_json::json!({ "dropped": id }));
        } else {
            println!("dropped #{id}");
        }
        return Ok(());
    }
    let limit: usize = args.str("limit").and_then(|s| s.parse().ok()).unwrap_or(20);
    let queue = store.needs_review(limit)?;
    if args.bool("json") {
        let items: Vec<serde_json::Value> = queue
            .iter()
            .map(|e| {
                let conflicts = store.conflicts_of(e.id).unwrap_or_default();
                serde_json::json!({ "entry": e, "conflicts": conflicts })
            })
            .collect();
        print_json(&serde_json::json!({ "queue": items }));
        return Ok(());
    }
    if queue.is_empty() {
        println!("review queue empty");
        return Ok(());
    }
    for e in &queue {
        println!("#{} [{}|{}] {}", e.id, e.category.as_str(), e.status.as_str(), e.content);
        for (other, cues) in store.conflicts_of(e.id).unwrap_or_default() {
            println!("    conflicts #{other}: {}", cues.join("; "));
        }
    }
    Ok(())
}
