//! `lbm` — the CLI pi extensions shell out to.
//!
//! Memory-only surface: distilled facts plus the entity graph. (The
//! database shell lives in a separate binary.)
//!
//! Human-readable text by default; `--json` for machine consumption.
//! Exit code is nonzero only on operational failure (not on empty results).
//!
//! ```sh
//! lbm remember "User prefers Python for scripting" --scope project
//! lbm remember "Fact" --via-sqlite    # sqlite-first: intake -> GLiNER -> stage -> sync
//! lbm sync                             # drain any pending sqlite intake (crash recovery)
//! lbm fetch-model                      # first-run setup: download the GLiNER snapshot (~780 MB)
//! lbm demo [--db PATH] [--snapshot PATH]  # GPUI demo: chat left, live graph right
//! lbm search "python" --json
//! lbm forget "python"            # lists matches; --apply deletes
//! lbm tour --json
//! lbm status
//! lbm review                     # lists needs_review + quarantined
//! echo "text" | lbm remember --auto   # auto-capture path (distills facts)
//! ```

use std::io::IsTerminal;

use anyhow::{anyhow, Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use lb_memory::model::GlinerExtractor;
use lb_memory::pipeline::{Pipeline, PipelineOptions};
use lb_memory::store::Store;
use lb_memory::types::{Provenance, Scope};

const DEFAULT_LABELS_NOTE: &str = "person,organization,location,product,event,technology,concept";

/// Pi-agent long-term memory over LadybugDB: distilled facts plus an entity
/// graph. Human text by default, `--json` for machines.
#[derive(Parser)]
#[command(name = "lbm", version, about, arg_required_else_help = true)]
struct Cli {
    /// Path to the memory database. Defaults to LB_MEMORY_DB,
    /// else ~/.pi/agent/lb-memory/memory.lbdb (demo: demo.lbdb).
    #[arg(long, value_name = "PATH", global = true, env = "LB_MEMORY_DB")]
    db: Option<String>,

    /// Machine consumption: one JSON object per command on stdout.
    /// Pretty-printed on a terminal, compact when piped (see --pretty).
    #[arg(long, global = true)]
    json: bool,

    /// Indent JSON output even when stdout is not a terminal.
    #[arg(long, global = true)]
    pretty: bool,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Store a fact (GLiNER extraction, dedup, conflict gating included).
    /// TEXT may be given as positionals or piped on stdin.
    Remember(RememberArgs),
    /// Drain any pending SQLite intake into the graph (crash recovery).
    Sync(SyncArgs),
    /// First-run setup: download the GLiNER snapshot + ONNX Runtime lib.
    /// Also provisioned automatically on the first `remember`/`demo` that
    /// needs them; this subcommand just pre-warms without running inference.
    FetchModel(FetchModelArgs),
    /// Open the GPUI demo: chat on the left, live entity graph on the right.
    Demo(DemoArgs),
    /// Recall matching facts and entities by topic.
    Search(SearchArgs),
    /// List memories matching a query; --apply deletes them.
    Forget(ForgetArgs),
    /// Browse stored memories by category.
    Tour(TourArgs),
    /// Counts, database path, and pending intake.
    Status(StatusArgs),
    /// List the needs_review + quarantined queue, or adjudicate one entry.
    Review(ReviewArgs),
}

#[derive(Clone, Copy, PartialEq, Eq, ValueEnum)]
enum ScopeArg {
    Project,
    Session,
    Global,
}

#[derive(clap::Args)]
struct RememberArgs {
    /// Fact text (joined with spaces). Empty reads stdin instead.
    #[arg(value_name = "TEXT")]
    text: Vec<String>,

    /// Memory scope (default project).
    #[arg(long, value_enum, default_value = "project")]
    scope: ScopeArg,
    /// Session id (session scope needs --session or PI_SESSION_ID).
    #[arg(long, value_name = "ID")]
    session: Option<String>,
    /// User id (default LB_MEMORY_USER, else USER, else "default").
    #[arg(long, value_name = "USER")]
    user: Option<String>,
    /// Project slug (default: git root basename, else cwd slug).
    #[arg(long, value_name = "SLUG")]
    project: Option<String>,
    /// Path to the GLiNER model snapshot (default LB_MEMORY_SNAPSHOT, else cache).
    #[arg(long, value_name = "PATH")]
    snapshot: Option<String>,
    /// Extraction confidence threshold.
    #[arg(long, default_value_t = 0.7)]
    threshold: f32,
    /// Optional importance override for the stored fact.
    #[arg(long, value_name = "INT")]
    importance: Option<i64>,
    /// Provenance refs, comma-separated (opaque "<role>:<hash>" ids).
    #[arg(long, value_name = "REFS")]
    message_ref: Option<String>,
    /// SQLite-first intake: intake -> GLiNER -> stage -> sync tick.
    #[arg(long)]
    via_sqlite: bool,
    /// Intake file override (default: <stem>.intake.sqlite sibling of the db).
    #[arg(long, value_name = "PATH")]
    sqlite_path: Option<String>,
    /// Local extension build override for the sync tick.
    #[arg(long, value_name = "PATH")]
    extension: Option<String>,
    /// Auto-capture path: distill per-sentence facts, drop entity-less ones.
    #[arg(long)]
    auto: bool,
}

#[derive(clap::Args)]
struct SearchArgs {
    /// Query text (joined with spaces). Empty reads stdin instead.
    #[arg(value_name = "QUERY")]
    query: Vec<String>,

    /// Memory scope (default project).
    #[arg(long, value_enum, default_value = "project")]
    scope: ScopeArg,
    /// Session id (session scope needs --session or PI_SESSION_ID).
    #[arg(long, value_name = "ID")]
    session: Option<String>,
    /// User id (default LB_MEMORY_USER, else USER, else "default").
    #[arg(long, value_name = "USER")]
    user: Option<String>,
    /// Project slug (default: git root basename, else cwd slug).
    #[arg(long, value_name = "SLUG")]
    project: Option<String>,
    /// Max hits to print.
    #[arg(long, default_value_t = 10)]
    limit: usize,
}

#[derive(clap::Args)]
struct ForgetArgs {
    /// Query text (joined with spaces). Empty reads stdin instead.
    #[arg(value_name = "QUERY")]
    query: Vec<String>,

    /// Memory scope (default project).
    #[arg(long, value_enum, default_value = "project")]
    scope: ScopeArg,
    /// Session id (session scope needs --session or PI_SESSION_ID).
    #[arg(long, value_name = "ID")]
    session: Option<String>,
    /// User id (default LB_MEMORY_USER, else USER, else "default").
    #[arg(long, value_name = "USER")]
    user: Option<String>,
    /// Project slug (default: git root basename, else cwd slug).
    #[arg(long, value_name = "SLUG")]
    project: Option<String>,
    /// Max matches to consider.
    #[arg(long, default_value_t = 10)]
    limit: usize,
    /// Actually delete the matches (default: list only).
    #[arg(long)]
    apply: bool,
}

#[derive(clap::Args)]
struct TourArgs {
    /// Memory scope (default project).
    #[arg(long, value_enum, default_value = "project")]
    scope: ScopeArg,
    /// Session id (session scope needs --session or PI_SESSION_ID).
    #[arg(long, value_name = "ID")]
    session: Option<String>,
    /// User id (default LB_MEMORY_USER, else USER, else "default").
    #[arg(long, value_name = "USER")]
    user: Option<String>,
    /// Project slug (default: git root basename, else cwd slug).
    #[arg(long, value_name = "SLUG")]
    project: Option<String>,
    /// Max entries per section to list.
    #[arg(long, default_value_t = 100)]
    limit: usize,
}

#[derive(clap::Args)]
struct ReviewArgs {
    /// Accept a queued entry back to active.
    #[arg(long, value_name = "ID")]
    accept: Option<i64>,
    /// When accepting, retire this older entry it supersedes.
    #[arg(long, value_name = "ID")]
    supersede: Option<i64>,
    /// Discard a queued entry.
    #[arg(long, value_name = "ID")]
    drop: Option<i64>,
    #[arg(long, default_value_t = 20)]
    limit: usize,
}

#[derive(clap::Args)]
struct StatusArgs {
    /// Intake file override (default: <stem>.intake.sqlite sibling of the db).
    #[arg(long, value_name = "PATH")]
    sqlite_path: Option<String>,
}

#[derive(clap::Args)]
struct SyncArgs {
    #[arg(long, value_name = "PATH")]
    sqlite_path: Option<String>,
    #[arg(long, value_name = "PATH")]
    extension: Option<String>,
}

#[derive(clap::Args)]
struct FetchModelArgs {
    /// Path to the GLiNER model snapshot (default LB_MEMORY_SNAPSHOT, else cache).
    #[arg(long, value_name = "PATH")]
    snapshot: Option<String>,
}

#[derive(clap::Args)]
struct DemoArgs {
    /// Path to the GLiNER model snapshot (default LB_MEMORY_SNAPSHOT, else cache).
    #[arg(long, value_name = "PATH")]
    snapshot: Option<String>,
}

/// Output flags threaded through every command.
#[derive(Clone, Copy)]
struct Out {
    json: bool,
    pretty: bool,
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
            "{}/.cache/ladybug-memory/models/gliner2.5-base-v1-nico-onnx",
            home()
        )
    })
}

fn user_id(arg: Option<&str>) -> String {
    arg.map(str::to_string)
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
    slug(
        &std::env::current_dir()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|_| "default".into()),
    )
}

fn git_root() -> Result<String> {
    let out = std::process::Command::new("git")
        .args(["rev-parse", "--show-toplevel"])
        .output()?;
    if !out.status.success() {
        return Err(anyhow!("not a git repo"));
    }
    Ok(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

fn slug(s: &str) -> String {
    let base = s.rsplit('/').next().unwrap_or(s);
    let slug: String = base
        .chars()
        .map(|c| {
            if c.is_alphanumeric() {
                c.to_ascii_lowercase()
            } else {
                '-'
            }
        })
        .collect();
    let slug = slug.trim_matches('-').to_string();
    if slug.is_empty() {
        "default".into()
    } else {
        slug
    }
}

fn resolve_scope(
    scope: ScopeArg,
    session: Option<&str>,
    user: Option<&str>,
    project: Option<&str>,
) -> Result<(Scope, ScopeMode)> {
    let mode = match scope {
        ScopeArg::Session => ScopeMode::Session,
        ScopeArg::Global => ScopeMode::Global,
        ScopeArg::Project => ScopeMode::Project,
    };
    let session_id = session
        .map(str::to_string)
        .or_else(|| std::env::var("LB_MEMORY_SESSION").ok())
        .or_else(|| std::env::var("PI_SESSION_ID").ok());
    if mode == ScopeMode::Session && session_id.is_none() {
        return Err(anyhow!(
            "--scope session needs --session <id> or PI_SESSION_ID"
        ));
    }
    Ok((
        Scope {
            user_id: user_id(user),
            project: project.map(str::to_string).unwrap_or_else(project_slug),
            // Session filters recall only in session scope (mem0 semantics:
            // project = user+project). It is still *recorded* in provenance.
            // NB: PI_SESSION_ID is set inside agent shells — never let it
            // leak into project/global recall or everything misses.
            session_id: if mode == ScopeMode::Session {
                session_id
            } else {
                None
            },
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

fn read_text(parts: &[String]) -> Result<String> {
    if !parts.is_empty() {
        return Ok(parts.join(" "));
    }
    // Piped stdin (auto-capture path).
    use std::io::Read;
    let mut buf = String::new();
    std::io::stdin().read_to_string(&mut buf)?;
    Ok(buf.trim().to_string())
}

/// JSON output: indented on a terminal, compact when piped; `--pretty`
/// forces indentation either way.
fn print_json(v: &serde_json::Value, out: Out) {
    if out.pretty || std::io::stdout().is_terminal() {
        println!("{}", serde_json::to_string_pretty(v).unwrap());
    } else {
        println!("{}", serde_json::to_string(v).unwrap());
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let out = Out {
        json: cli.json,
        pretty: cli.pretty,
    };
    // `--db` is global: `lbm --db P <cmd>` and `lbm <cmd> --db P` agree.
    let db = cli.db.clone();
    match cli.command {
        Commands::Remember(a) => cmd_remember(a, db, out),
        Commands::Sync(a) => cmd_sync(a, db, out),
        Commands::FetchModel(a) => cmd_fetch_model(a, out),
        Commands::Demo(a) => cmd_demo(a, db),
        Commands::Search(a) => cmd_search(a, db, out),
        Commands::Forget(a) => cmd_forget(a, db, out),
        Commands::Tour(a) => cmd_tour(a, db, out),
        Commands::Status(a) => cmd_status(a, db, out),
        Commands::Review(a) => cmd_review(a, db, out),
    }
}

fn open_store(db: Option<String>) -> Result<Store> {
    let path = db.unwrap_or_else(default_db);
    Store::open(&path).with_context(|| format!("opening {path}"))
}

/// GPUI demo: chat with the agent on the left, watch the entity graph
/// update on the right (graph view only — no treemap/sunburst).
fn cmd_demo(args: DemoArgs, db: Option<String>) -> Result<()> {
    // The demo keeps its own default database; an explicit --db overrides it.
    let db_path = db.unwrap_or_else(demo::default_db);
    let snapshot = args.snapshot.unwrap_or_else(default_snapshot);
    demo::run(demo::DemoConfig { db_path, snapshot })
}

fn open_intake(
    store: &Store,
    sqlite_path: Option<String>,
) -> Result<lb_memory::sqlite_sync::IntakeDb> {
    if let Some(p) = sqlite_path {
        lb_memory::sqlite_sync::IntakeDb::open(&p).with_context(|| format!("opening {p}"))
    } else {
        store.intake_db()
    }
}

fn cmd_remember(args: RememberArgs, db: Option<String>, out: Out) -> Result<()> {
    let text = read_text(&args.text)?;
    if text.trim().is_empty() {
        return Err(anyhow!("nothing to remember (empty text)"));
    }
    let (scope, _mode) = resolve_scope(
        args.scope,
        args.session.as_deref(),
        args.user.as_deref(),
        args.project.as_deref(),
    )?;
    let store = open_store(db)?;
    let pipe = Pipeline::new(
        &store,
        PipelineOptions {
            extract_threshold: args.threshold,
            ..Default::default()
        },
    );
    let mut extractor = GlinerExtractor::new(args.snapshot.unwrap_or_else(default_snapshot));
    let mut prov = Provenance {
        // Record the session for provenance even when it isn't a recall
        // filter (project/global modes): explicit flag wins, then env.
        session_id: args
            .session
            .clone()
            .or_else(|| std::env::var("LB_MEMORY_SESSION").ok())
            .or_else(|| std::env::var("PI_SESSION_ID").ok()),
        message_refs: args
            .message_ref
            .as_deref()
            .map(|s| {
                s.split(',')
                    .map(|x| x.trim().to_string())
                    .filter(|x| !x.is_empty())
                    .collect()
            })
            .unwrap_or_default(),
        extractor: "gliner2.5".into(),
        model: "DanKau/gliner2.5-base-v1-onnx".into(),
        auto_captured: false,
    };
    // SQLite-first: intake -> GLiNER -> stage -> sync tick. The tick uses
    // the `sqlite` extension via INSTALL + LOAD (`--extension <path>`
    // overrides with a local build), falling back to the embedded applier.
    if args.via_sqlite {
        let intake = open_intake(&store, args.sqlite_path)?;
        let ext = match args.extension.as_deref() {
            Some(p) => Some(lb_memory::sqlite_sync::ExtensionSource::File(p)),
            None => Some(lb_memory::sqlite_sync::ExtensionSource::Repo),
        };
        let (reports, stats) = if args.auto {
            prov.auto_captured = true;
            pipe.remember_auto_via_sqlite(&mut extractor, &intake, &text, &scope, prov, ext)?
        } else {
            pipe.remember_via_sqlite(
                &mut extractor,
                &intake,
                &text,
                &scope,
                args.importance,
                prov,
                ext,
            )?
        };
        if out.json {
            print_json(
                &serde_json::json!({
                    "reports": reports,
                    "sync": { "via": stats.via, "applied_intake": stats.applied_intake,
                              "rows_upserted": stats.rows_upserted, "status": stats.status },
                    "intake": intake.path(),
                }),
                out,
            );
            return Ok(());
        }
        for r in &reports {
            print_report(r);
        }
        println!(
            "sync via {}: applied {} intake, {} rows upserted ({}) [{}]",
            stats.via,
            stats.applied_intake,
            stats.rows_upserted,
            stats.status,
            intake.path()
        );
        return Ok(());
    }
    let reports = if args.auto {
        prov.auto_captured = true;
        pipe.remember_auto(&mut extractor, &text, &scope, prov)?
    } else {
        pipe.remember(&mut extractor, &text, &scope, args.importance, prov)?
    };

    if out.json {
        print_json(&serde_json::json!({ "reports": reports }), out);
        return Ok(());
    }
    for r in &reports {
        print_report(r);
    }
    Ok(())
}

fn print_report(r: &lb_memory::types::RememberReport) {
    let step1: Vec<String> = r
        .extracted
        .iter()
        .map(|m| format!("{}[{:.2}]", m.text, m.confidence))
        .collect();
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
            println!(
                "decision: conflict with {with:?} [{}] — needs review",
                cues.join("; ")
            )
        }
        lb_memory::types::FactDecision::LowConfidence { reason } => {
            println!("decision: low confidence ({reason}) — quarantined")
        }
    }
    match &r.written {
        Some(w) => println!(
            "stored #{} [{}|{}]: {}",
            w.id,
            w.category.as_str(),
            w.status.as_str(),
            w.content
        ),
        None => println!("stored: nothing"),
    }
}

/// First-run setup / pre-warm: download the GLiNER snapshot (~780 MB) and
/// the ONNX Runtime dylib (~35 MB) without running inference. Inference
/// commands (`remember`, `demo`) provision both automatically on first
/// use; this is the explicit, scriptable form. Afterwards no
/// `ORT_DYLIB_PATH` is needed — the dylib is opened from its cache path.
fn cmd_fetch_model(args: FetchModelArgs, out: Out) -> Result<()> {
    let snapshot = args.snapshot.unwrap_or_else(default_snapshot);
    let extractor = GlinerExtractor::new(snapshot.clone());
    extractor.ensure_model()?;
    let ort = extractor.ensure_runtime()?;
    if out.json {
        print_json(&serde_json::json!({ "snapshot": snapshot, "ort": ort, "ready": true }), out);
        return Ok(());
    }
    println!("model ready at {snapshot}");
    println!("ort lib ready at {}", ort.display());
    Ok(())
}

/// Drain any pending SQLite intake into the graph (crash recovery,
/// daemon-less tick). Uses the `sqlite` extension via INSTALL + LOAD
/// (`--extension <path>` overrides with a local build), falling back to the
/// embedded applier on any engine error.
fn cmd_sync(args: SyncArgs, db: Option<String>, out: Out) -> Result<()> {
    let store = open_store(db)?;
    let intake = open_intake(&store, args.sqlite_path)?;
    let pipe = Pipeline::new(&store, PipelineOptions::default());
    let ext = match args.extension.as_deref() {
        Some(p) => Some(lb_memory::sqlite_sync::ExtensionSource::File(p)),
        None => Some(lb_memory::sqlite_sync::ExtensionSource::Repo),
    };
    let stats = pipe.sync_intake(&intake, ext)?;
    if out.json {
        print_json(
            &serde_json::json!({
                "intake": intake.path(),
                "via": stats.via, "applied_intake": stats.applied_intake,
                "rows_upserted": stats.rows_upserted, "status": stats.status,
            }),
            out,
        );
        return Ok(());
    }
    println!("intake: {}", intake.path());
    println!(
        "sync via {}: applied {} intake, {} rows upserted ({})",
        stats.via, stats.applied_intake, stats.rows_upserted, stats.status
    );
    Ok(())
}

fn cmd_search(args: SearchArgs, db: Option<String>, out: Out) -> Result<()> {
    let query = read_text(&args.query)?;
    let (scope, mode) = resolve_scope(
        args.scope,
        args.session.as_deref(),
        args.user.as_deref(),
        args.project.as_deref(),
    )?;
    let store = open_store(db)?;
    let limit = args.limit.max(1);
    // Global scope searches across projects.
    let hits = if mode == ScopeMode::Global {
        global_recall(&store, &query, &scope.user_id, limit)?
    } else {
        Pipeline::new(&store, PipelineOptions::default()).recall(&query, &scope, limit)?
    };
    if out.json {
        print_json(&serde_json::json!({ "hits": hits }), out);
        return Ok(());
    }
    if hits.is_empty() {
        println!("(no matches)");
        return Ok(());
    }
    for h in &hits {
        println!(
            "#{} [{:.2}] [{}] {}",
            h.entry.id,
            h.score,
            h.entry.category.as_str(),
            h.entry.content
        );
        if !h.entities.is_empty() {
            println!("    entities: {}", h.entities.join(", "));
        }
        if !h.relations.is_empty() {
            println!("    relations: {}", h.relations.join("; "));
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
    let pipe = Pipeline::new(store, PipelineOptions::default());
    for (canonical, _, _) in store.entities_matching(query, 5)? {
        // entities_matching is global already (no scope columns on Entity).
        for (entry, conf) in store.memories_for_entity_global(&canonical, user, 3)? {
            if seen.insert(entry.id) {
                let entities = vec![canonical.clone()];
                let (boost, relations) = pipe.relations_for(&entities, query).unwrap_or_default();
                hits.push(SearchHit {
                    entry,
                    score: conf as f32 + 0.1 + boost,
                    entities,
                    relations,
                });
            }
        }
    }
    for (entry, score) in store.search_content_global(query, user, limit)? {
        if seen.insert(entry.id) {
            let entities = store.entities_for_memory(entry.id).unwrap_or_default();
            let (boost, relations) = pipe.relations_for(&entities, query).unwrap_or_default();
            hits.push(SearchHit {
                entry,
                score: score as f32 + boost,
                entities,
                relations,
            });
        }
    }
    hits.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
    hits.truncate(limit.max(1));
    Ok(hits)
}

fn cmd_forget(args: ForgetArgs, db: Option<String>, out: Out) -> Result<()> {
    let query = read_text(&args.query)?;
    let (scope, mode) = resolve_scope(
        args.scope,
        args.session.as_deref(),
        args.user.as_deref(),
        args.project.as_deref(),
    )?;
    let store = open_store(db)?;
    let limit = args.limit.max(1);
    let hits = if mode == ScopeMode::Global {
        global_recall(&store, &query, &scope.user_id, limit)?
    } else {
        Pipeline::new(&store, PipelineOptions::default()).recall(&query, &scope, limit)?
    };
    if args.apply {
        let mut deleted = 0;
        for h in &hits {
            store.delete_memory(h.entry.id)?;
            deleted += 1;
        }
        if out.json {
            print_json(&serde_json::json!({ "deleted": deleted }), out);
        } else {
            println!("deleted {deleted} memories");
        }
        return Ok(());
    }
    if out.json {
        print_json(
            &serde_json::json!({ "matches": hits, "hint": "re-run with --apply to delete" }),
            out,
        );
        return Ok(());
    }
    if hits.is_empty() {
        println!("(no matches)");
        return Ok(());
    }
    for h in &hits {
        println!(
            "#{} [{}] {}",
            h.entry.id,
            h.entry.category.as_str(),
            h.entry.content
        );
    }
    println!("re-run with --apply to delete these");
    Ok(())
}

fn cmd_tour(args: TourArgs, db: Option<String>, out: Out) -> Result<()> {
    let (scope, mode) = resolve_scope(
        args.scope,
        args.session.as_deref(),
        args.user.as_deref(),
        args.project.as_deref(),
    )?;
    let store = open_store(db)?;
    let limit = args.limit.max(1);
    let session = scope.session_id.as_deref();
    // Memories stay scope-filtered exactly as before; entities and every
    // relationship kind are drawn from the same scope so a project tour
    // never leaks another project's graph.
    let (entries, entities, mentions, relations, conflicts, supersedes) = match mode {
        ScopeMode::Global => (
            store.list_global(&scope.user_id, limit)?,
            store.tour_entities_global(&scope.user_id, limit)?,
            store.tour_mentions_global(&scope.user_id, limit)?,
            store.tour_relations_global(&scope.user_id, limit)?,
            store.tour_conflicts_global(&scope.user_id, limit)?,
            store.tour_supersedes_global(&scope.user_id, limit)?,
        ),
        _ => (
            Pipeline::new(&store, PipelineOptions::default()).tour(&scope, limit)?,
            store.tour_entities(&scope.user_id, &scope.project, session, limit)?,
            store.tour_mentions(&scope.user_id, &scope.project, session, limit)?,
            store.tour_relations(&scope.user_id, &scope.project, session, limit)?,
            store.tour_conflicts(&scope.user_id, &scope.project, session, limit)?,
            store.tour_supersedes(&scope.user_id, &scope.project, session, limit)?,
        ),
    };
    if out.json {
        print_json(
            &serde_json::json!({
                "entries": entries,
                "entities": entities,
                "mentions": mentions,
                "relationships": relations,
                "conflicts": conflicts,
                "supersedes": supersedes,
            }),
            out,
        );
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
    println!("\n## entities ({})", entities.len());
    if entities.is_empty() {
        println!("(none)");
    }
    for ent in &entities {
        let aliases = ent
            .aliases
            .iter()
            .filter(|a| *a != &ent.canonical)
            .cloned()
            .collect::<Vec<_>>()
            .join(", ");
        if aliases.is_empty() {
            println!(
                "- {} [{}] (#{}; {} mentions)",
                ent.canonical, ent.entity_type, ent.id, ent.mentions
            );
        } else {
            println!(
                "- {} [{}] (#{}; aliases: {}; {} mentions)",
                ent.canonical, ent.entity_type, ent.id, aliases, ent.mentions
            );
        }
    }
    println!("\n## relationships ({})", relations.len());
    if relations.is_empty() {
        println!("(none)");
    }
    for r in &relations {
        println!(
            "- {} -[{}]-> {} (memory #{}, {:.2})",
            r.src, r.relation_type, r.dst, r.memory_id, r.confidence
        );
    }
    println!("\n## mentions ({})", mentions.len());
    if mentions.is_empty() {
        println!("(none)");
    }
    for m in &mentions {
        println!(
            "- {} mentioned in #{} as \"{}\" [{:.2}]",
            m.entity, m.memory_id, m.mention_text, m.confidence
        );
    }
    if !conflicts.is_empty() {
        println!("\n## conflicts ({})", conflicts.len());
        for c in &conflicts {
            let cues = if c.cues.is_empty() {
                String::new()
            } else {
                format!(" [{}]", c.cues.join("; "))
            };
            println!("- #{} conflicts #{} {}", c.src, c.dst, cues);
        }
    }
    if !supersedes.is_empty() {
        println!("\n## supersedes ({})", supersedes.len());
        for s in &supersedes {
            if s.reason.is_empty() {
                println!("- #{} supersedes #{}", s.src, s.dst);
            } else {
                println!("- #{} supersedes #{} ({})", s.src, s.dst, s.reason);
            }
        }
    }
    Ok(())
}

fn cmd_status(args: StatusArgs, db: Option<String>, out: Out) -> Result<()> {
    let path = db.clone().unwrap_or_else(default_db);
    let store = open_store(db)?;
    let (mem, ent, men, rev) = store.counts()?;
    // SQLite intake (first-class write path): report pending work when the
    // intake file exists; missing file just means the direct path is in use.
    let (intake_path, pending) = match open_intake(&store, args.sqlite_path) {
        Ok(db) => (db.path().to_string(), db.pending_count().unwrap_or(0)),
        Err(_) => ("-".to_string(), 0),
    };
    // Model snapshot + ORT lib: a fresh machine has neither until the
    // first `remember`/`demo` (auto-provision) or an explicit `fetch-model`.
    let model_path = std::env::var("LB_MEMORY_SNAPSHOT").unwrap_or_else(|_| default_snapshot());
    let model_ready = lb_memory::fetch::is_loadable(std::path::Path::new(&model_path));
    let (ort_path, ort_ready) = lb_memory::runtime::status();
    if out.json {
        print_json(
            &serde_json::json!({
                "db": path, "memories": mem, "entities": ent,
                "mentions": men, "needs_review": rev,
                "labels": DEFAULT_LABELS_NOTE,
                "intake": intake_path, "intake_pending": pending,
                "model": model_path, "model_ready": model_ready,
                "ort": ort_path, "ort_ready": ort_ready,
            }),
            out,
        );
        return Ok(());
    }
    println!("db: {path}");
    println!("memories: {mem}, entities: {ent}, mentions: {men}, needs_review: {rev}");
    println!("intake: {intake_path} (pending: {pending})");
    println!(
        "model: {model_path} ({})",
        if model_ready { "ready" } else { "missing — run `lbm fetch-model`" }
    );
    println!(
        "ort: {ort_path} ({})",
        if ort_ready { "ready" } else { "missing — run `lbm fetch-model`" }
    );
    Ok(())
}

fn cmd_review(args: ReviewArgs, db: Option<String>, out: Out) -> Result<()> {
    let store = open_store(db)?;
    if let Some(id) = args.accept {
        let pipe = Pipeline::new(&store, PipelineOptions::default());
        match pipe.review(id, true, args.supersede) {
            Ok(entry) => {
                if out.json {
                    print_json(&serde_json::json!({ "accepted": entry }), out);
                } else {
                    println!("accepted #{}: {}", entry.id, entry.content);
                }
                return Ok(());
            }
            Err(e) => return Err(e),
        }
    }
    if let Some(id) = args.drop {
        store.delete_memory(id)?;
        if out.json {
            print_json(&serde_json::json!({ "dropped": id }), out);
        } else {
            println!("dropped #{id}");
        }
        return Ok(());
    }
    let limit = args.limit.max(1);
    let queue = store.needs_review(limit)?;
    if out.json {
        let items: Vec<serde_json::Value> = queue
            .iter()
            .map(|e| {
                let conflicts = store.conflicts_of(e.id).unwrap_or_default();
                serde_json::json!({ "entry": e, "conflicts": conflicts })
            })
            .collect();
        print_json(&serde_json::json!({ "queue": items }), out);
        return Ok(());
    }
    if queue.is_empty() {
        println!("review queue empty");
        return Ok(());
    }
    for e in &queue {
        println!(
            "#{} [{}|{}] {}",
            e.id,
            e.category.as_str(),
            e.status.as_str(),
            e.content
        );
        for (other, cues) in store.conflicts_of(e.id).unwrap_or_default() {
            println!("    conflicts #{other}: {}", cues.join("; "));
        }
    }
    Ok(())
}
