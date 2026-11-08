# ladybug-memory-rs

Rust port of [ladybug-memory](../ladybug-memory): agent long-term memory on
LadybugDB, with knowledge-graph extraction running **ONNX-only** — no Python,
no torch at inference.

## Crates

| crate | what |
|---|---|
| `gliner-onnx` | GLiNER2.5 entity extraction on the `nicolasembleton/gliner2.5-base-v1-onnx` single-graph export (ORT + `tokenizers`, parity-tested vs the python ONNX baseline; ships the `SparseRelationScorer` graph for typed relations) |
| `lb-memory` | Memory API: 5-step pipeline over a LadybugDB entity graph (embedded, via the `lbug` crate) |
| `lbm` | CLI with JSON I/O (`remember/search/forget/tour/status/review/sync`) — what pi extensions spawn |

Plus `pi-plugin/`: a thin pi extension (`lb_memory` tool, `/lb-*` commands,
auto-capture) mirroring `@mem0/pi-agent-plugin`'s surface.

See `crates/gliner-onnx` README notes below for why `gliner25-rs` wasn't used.

## The pipeline (`lbm remember`)

```
1. extract   GLiNER2.5 per logical chunk (H-GLUE chunking, ephemeral)
2. compare   mentions vs. the stored Entity schema (canonical + aliases)
3. resolve   merge onto canonical nodes (spelling variants become aliases)
4. decide    duplicate (strengthen) / conflict (needs_review + cues)
             / low-confidence (quarantined) / new (active)
5. write     Memory node + MentionedIn + co-occurrence + provenance
```

Only **distilled facts** are stored. Session transcripts stay in pi's sqlite;
provenance keeps `message_refs` (opaque ids), never message text.

## Build / run (always `--release` — debug artifacts are ~0.5 GB)

```sh
export ORT_DYLIB_PATH=/path/to/libonnxruntime.dylib   # 1.23+; the ladybug-memory
                                                      # uv venv ships one at
# .venv/lib/python3.13/site-packages/onnxruntime/capi/

cargo test --release -p lb-memory --features test-support
cargo test --release -p gliner-onnx   # needs ORT_DYLIB_PATH + the model snapshot below
cargo build --release -p lbm
```

The GLiNER snapshot resolves to `~/.cache/ladybug-memory/models/gliner2.5-base-v1-nico-onnx`
(override with `LB_MEMORY_SNAPSHOT` / `--snapshot`): the `nicolasembleton`
single-graph export of `fastino/gliner2.5-base-v1` (`onnx/model.onnx` —
encoder + entity pair path + classifier + text/candidate/role states — plus
`onnx/heads.onnx`, the `SparseRelationScorer`). The loader auto-detects the
layout, so either export works, but the DanKau 3-graph split has no relation
head. `extract_relations` (typed edges via `heads.onnx`, zero-shot relation
types) is implemented and smoke-tested; pipeline wiring (typed `RelatesTo`
instead of `coOccurs`) is next.

The `lbug` crate (crates.io, `0.22.3`) links the **shared** prebuilt
`liblbug.dylib` from a persistent cache (`~/.cache/lbug-shared-0.22.3`,
pinned via `.cargo/config.toml` so `cargo clean` never forces a
re-download). Shared linking is load-bearing: extensions (FTS, sqlite, …)
resolve their symbols from the host's dylib, while a statically linked
binary dead-strips them and `LOAD EXTENSION` fails with missing-symbol
errors. The GLiNER model resolves from the shared HF cache.

## CLI quickstart

```sh
export LB_MEMORY_DB=~/.pi/agent/lb-memory/memory.lbdb   # default location
lbm remember "User prefers Python for scripting" --scope project
lbm remember "Tim Cook runs Apple." --via-sqlite --scope project
lbm sync                                   # drain pending intake (no-op when caught up)
lbm search "python" --json
lbm review                                   # conflicts + quarantined + cues
lbm review --accept 3 --supersede 1          # agent adjudicates, old retires
echo "long message…" | lbm remember --auto   # auto-capture: distills sentences
```

Scopes mirror the mem0 plugin: `project` (git root, default), `session`
(requires `--session`/`PI_SESSION_ID`), `global` (cross-project). Project and
global recall deliberately ignore `PI_SESSION_ID` (it is set inside agent
shells and would otherwise filter everything out); the session is still
*recorded* in provenance.

## SQLite-first ingestion (`--via-sqlite`)

Every `remember` is first a **SQLite row**. The intake database lives next
to the graph (`<stem>.intake.sqlite` sibling of the `.lbdb`, or
`--sqlite-path` to override), and the flow is intake -> GLiNER -> stage ->
sync:

```
remember(text):
  1. append   raw text lands in the SQLite `memory_intake` log (WAL mode)
  2. extract  GLiNER2.5 mentions (same extractor as the direct path)
  3. stage    distilled fact staged into `node_*` / `rel_*` tables
  4. sync     staged rows replicated into the native Ladybug graph
```

The staging schema follows the mapping conventions of the `sqlite`
extension (`~/src/ladybug/extension/sqlite`, `docs/cdc_design.md`):
`node_*` tables carry the single-column PK, `rel_*` tables carry `src_*` /
`dst_*` foreign keys into mapped node tables. Mapped tables mirror the
native `Memory` / `Entity` / `MentionedIn` / `RelatesTo` / `ConflictsWith`
schema exactly (the extension `SET`s every column on upsert, so extra
columns would fail to bind); intake linkage, provenance, outcomes, and the
replication cursor live in unprefixed side tables the extension skips.

Two appliers, same staged rows:

- **embedded** (fallback): `replicate_to_store` applies nodes-before-edges
  through `Store`, idempotently (per-intake cursor + `lb_id` backfill, plus
  content/edge guards that converge partial ticks).
- **extension** (default): `INSTALL sqlite` (official repo, latest 0.22.x,
  cached under `~/.lbdb/extension`, idempotent) + `LOAD sqlite` by name
  (production form; `LOAD EXTENSION '<path>'` is testing-only) + explicit label remapping (`sqlite_register_node/edge` onto
  `Memory`/`Entity`/`MentionedIn`/`RelatesTo`/`ConflictsWith`) + `CALL
  sqlite_sync`. Engine errors fall back to embedded with `via` reporting
  the detour, so a failed INSTALL/LOAD/sync never loses a remember
  (verified live: repo 0.22.0 build loads, nodes replicate, edges hit P1
  below, fallback converges to exactly one memory).

Local build override for extension hacking (`--extension <path>`):

```sh
cmake -B /tmp/lbug-sqlite-ext -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_EXTENSIONS="sqlite" -DBUILD_LBUG=FALSE -DBUILD_SHELL=FALSE -DBUILD_TESTS=FALSE ~/src/ladybug
cmake --build /tmp/lbug-sqlite-ext --config Release -j8
# -> ~/src/ladybug/extension/sqlite/build/libsqlite.lbug_extension (git-ignored)
```

Or drive the repo extension by hand against an intake file:

```sql
INSTALL sqlite;
LOAD sqlite;
CALL sqlite_register_node('/abs/mem.intake.sqlite', 'node_Memory', 'Memory', 'id');
CALL sqlite_register_edge('/abs/mem.intake.sqlite', 'rel_MentionedIn', 'MentionedIn', 'src_id', 'dst_id');
CALL sqlite_sync('/abs/mem.intake.sqlite') RETURN *;
```

Caveats: use **one ingestion mode per DB file** — the extension maps staged
ids via MERGE-by-PK while the direct path uses the native SERIAL sequence,
so mixing modes on one file can collide identities. The embedded applier
always uses the native sequence and is safe to mix.

## Demo (`lbm demo`)

A GPUI window with the memory agent on the left and the live entity graph on
the right (graph view only — no treemap/sunburst):

```sh
lbm demo [--db PATH] [--snapshot PATH]   # defaults to ~/.pi/agent/lb-memory/demo.lbdb
```

Type anything to remember it: each message goes through GLiNER on a worker
thread, distilled facts land in the graph, and the canvas merges the new
snapshot in place (drag nodes, drag empty space to pan, wheel to zoom,
double-click empty space to refit). `/search <q>` recalls and rings the
matching memory nodes; `/reload` re-reads the graph; `/help` lists commands.

## Pi wiring

```
pi --extension ladybug-memory-rs/pi-plugin/index.ts
# needs `lbm` on PATH (or LB_MEMORY_BIN) + ORT_DYLIB_PATH in pi's env
```

Tool `lb_memory` (`remember`/`search`), commands `/lb-remember`, `/lb-search`,
`/lb-forget`, `/lb-tour`, `/lb-status`, plus `message_end` auto-capture
(entity-less sentences are dropped by `--auto`, so transcripts are never
copied) and `skills/lb-memory/SKILL.md`.

## Gotchas (earned the hard way)

- **Link liblbug shared, not static.** A statically linked binary dead-strips
  the GDS/table-function symbols extensions need, so `LOAD EXTENSION FTS`
  fails with `symbol not found ... getPhysicalPlan` even though the repo's
  extension build is fine (verified: the 0.21.2 CLI loads it). `.cargo/config.toml`
  pins shared linking + rpath; keep `LBUG_LIBRARY_DIR` in sync with the
  `lbug` crate version on bump.
- **SQLite intake is WAL-mode + FK-metadata, enforcement off.** The intake
  DB enables WAL (the C++ tailer reads it) and declares endpoint FKs (the
  extension resolves edge endpoints through them), but disables FK
  *enforcement* per connection: conflict edges point at native graph ids
  that don't exist in staging.
- **Storage files are shared safely with the `lbug` CLI.** Verified both
  directions on 0.22.3 binding ↔ bundled CLI: no migration, no lockout.
  (Do not use the old `ladybug-rust` 0.18 checkout — different storage
  version.) Still prefer copies when experimenting on live files.
- **Relations are typed, open-schema.** Each remembered fact goes through
  `extract_relations` (zero-shot: no declared ontology — the scorer judges
  relatedness under a generic query and the phrase is mined from the
  between-span surface, e.g. `founded`, `runs`). Typed pairs become directed
  `RelatesTo` edges; pairs with no typed edge keep the old `coOccurs` fallback.
  Every edge carries `valid_from`/`valid_to` (known-true interval, `""` =
  unknown), `tense`, `viewpoint` (`fact` vs `opinion`, zero-shot classified
  per fact) and confidence. Recall boosts hits whose adjacent typed
  relations match the query ("who founded Apple" upranks the founder memory).
  LLM fallback not needed.
- **Pronouns resolve, they never fork.** "He/she/they/it" bind to the most
  recent compatible entity (in-call order, then the store's most-touched
  rows); unresolvable ones are dropped, never "He" nodes. A lone person
  plus a role noun ("... is the CEO") links the most recent organization.
- Recall is entity-boosted FTS (porter stemmer) with an offline-safe
  substring fallback. No embedding/vector search yet; schema is
  forward-compatible (add an embedding column + index later).

## Roadmap (rest of the python port)

- [x] Entity extraction (graph nodes)
- [x] Classification head (`[L]` tasks: topic/category + argmax decode,
      parity-tested; worthiness judgments do *not* zero-shot — a dedicated
      “worth remembering vs chit-chat” probe scored greetings highest, so
      auto-capture uses a deterministic acknowledgment filter instead)
- [x] LogicalChunker, entity resolution, conflict/low-confidence gating
- [x] LadybugDB storage with provenance, FTS search (porter stemmer)
- [x] Hybrid fact categorization (rule cues, classifier fallback for `General`)
- [x] SQLite-first ingestion (intake log -> GLiNER -> `node_*`/`rel_*` staging -> sync tick)
- [x] `lbug` 0.22.3
- [ ] Extension-driven replication without fallback (blocked on the proposals below)
- [ ] Typed relations (export scorer or LLM fallback)

## Extension change proposals (need review — NOT implemented)

The `sqlite` extension itself is untouched. Two behaviors block a pure
extension-driven replication onto our pre-existing tables; both verified
against the real build above:

- **P1 — edge `_sqlite_rowid` slot on pre-existing rel tables.** Edge
  upserts unconditionally `MATCH ()-[e:Label {_sqlite_rowid: …}]` and
  `CREATE … {_sqlite_rowid: …}` (`sync_cypher.cpp::buildEdgeUpserts`). Our
  native `MentionedIn`/`RelatesTo`/`ConflictsWith` tables don't have that
  property, so the tick fails with `Binder exception: Cannot find property
  _sqlite_rowid for e` after the node statements already committed.
  Implemented as
  [extensions#143](https://github.com/LadybugDB/extensions/pull/143)
  (stacked on #142, option (b)): `ALTER TABLE <label> ADD IF NOT EXISTS
  _sqlite_rowid INT64` per mapped edge table in the DDL phase.
- **P2 — tick atomicity.** Node statements commit before a later edge
  statement fails, leaving a partial tick (observed: 1 staged memory landed
  natively, edges missing). Implemented as
  [extensions#142](https://github.com/LadybugDB/extensions/pull/142): one
  SQLite WAL commit maps to one Ladybug transaction (batch + cursor commit
  atomically, rollback on failure). The embedded fallback already
  converges partial ticks (content/edge guards + cursor), so this only
  matters for the pure-extension path.

Verified working, no change needed: explicit label remapping
(`register_node/edge` onto `Memory`/`Entity`/…), MERGE-by-PK with staged
ids, `TEXT` -> `JSON` coercion for `aliases`/`cues`, `TIMESTAMP` omission
(native defaults apply).
- [ ] Embeddings + vector index (recall upgrade)
- [ ] Dynamic per-type schema tables + schema discovery
