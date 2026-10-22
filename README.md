# ladybug-memory-rs

Rust port of [ladybug-memory](../ladybug-memory): agent long-term memory on
LadybugDB, with knowledge-graph extraction running **ONNX-only** — no Python,
no torch at inference.

## Crates

| crate | what |
|---|---|
| `gliner-onnx` | GLiNER2.5 entity extraction on `DanKau/gliner2.5-base-v1-onnx` (ORT + `tokenizers`, parity-tested vs the python ONNX baseline) |
| `lb-memory` | Memory API: 5-step pipeline over a LadybugDB entity graph (embedded, via the `lbug` crate) |
| `lbm` | CLI with JSON I/O (`remember/search/forget/tour/status/review`) — what pi extensions spawn |

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
cargo test --release -p gliner-onnx   # needs ORT_DYLIB_PATH + HF snapshot
cargo build --release -p lbm
```

The `lbug` crate (crates.io, `0.21.2`) links the **shared** prebuilt
`liblbug.dylib` from a persistent cache (`~/.cache/lbug-shared-0.21.2`,
pinned via `.cargo/config.toml` so `cargo clean` never forces a
re-download). Shared linking is load-bearing: extensions (FTS, …) resolve
their symbols from the host's dylib, while a statically linked binary
dead-strips them and `LOAD EXTENSION` fails with missing-symbol errors.
The GLiNER model resolves from the shared HF cache.

## CLI quickstart

```sh
export LB_MEMORY_DB=~/.pi/agent/lb-memory/memory.lbdb   # default location
lbm remember "User prefers Python for scripting" --scope project
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
- **Storage files are shared safely with the `lbug` CLI.** Verified both
  directions on 0.21.2 binding ↔ 0.20.0 CLI: no migration, no lockout.
  (Do not use the old `ladybug-rust` 0.18 checkout — different storage
  version.) Still prefer copies when experimenting on live files.
- **Relations are co-occurrence** (`RelatesTo{coOccurs}`). The ONNX export has
  no relation scorer (gliner2 scores relations in torch), so typed relations
  need a separate export or the LLM fallback — same caveat as the python
  package's next step.
- Recall is entity-boosted FTS (porter stemmer) with an offline-safe
  substring fallback. No embedding/vector search yet; schema is
  forward-compatible (add an embedding column + index later).

## Roadmap (rest of the python port)

- [x] Entity extraction (graph nodes)
- [x] LogicalChunker, entity resolution, conflict/low-confidence gating
- [x] LadybugDB storage with provenance, FTS→substring search
- [ ] `classifier.onnx` wiring for `[L]` tasks
- [ ] Typed relations (export scorer or LLM fallback)
- [ ] Embeddings + vector index (recall upgrade)
- [ ] Dynamic per-type schema tables + schema discovery
