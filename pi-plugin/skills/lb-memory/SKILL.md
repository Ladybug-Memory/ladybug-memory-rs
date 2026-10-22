# lb-memory skill

Long-term memory for this project, stored as distilled facts plus an entity
graph in LadybugDB (`~/.pi/agent/lb-memory/memory.lbdb`). Session transcripts
stay in pi's sqlite — memory keeps facts and provenance references, never
copies of messages.

## Tool

`lb_memory` with `{action, text, scope?}`:

- `remember` — store a fact. The pipeline runs GLiNER2.5 extraction, compares
  against the stored entity schema, resolves aliases, then decides: **new**
  (stored active), **duplicate** (existing strengthened, nothing written),
  **conflict** (stored as `needs_review` with cues), **low-confidence**
  (quarantined, hidden from recall). Nothing is silently dropped.
- `search` — hybrid recall: entity matches plus content search, project scope
  by default (`session` / `global` available).

## Scopes

- `project` (default): shared across the repo (git root). Cross-directory,
  same pool — like the mem0 plugin.
- `session`: only facts explicitly saved in this session.
- `global`: all projects for this user.

## Conflicts

Contradiction candidates land in the review queue (`lbm review`), never
auto-applied. Adjudicate deliberately:

```sh
lbm review                          # list needs_review + quarantined + cues
lbm review --accept <id> --supersede <old-id>   # new fact wins, old retires
lbm review --drop <id>              # discard
```

Prefer `remember` for durable facts (decisions, preferences, identities,
gotchas). Do not store secrets; do not store whole conversation excerpts.

## Standing safeguards

A `tool_call` handler consults memory on every `bash` invocation: when the
command contains `git rebase` and a recalled fact prescribes an env
safeguard (`GIT_EDITOR=true`), the handler prepends the missing assignment
by mutating the tool input — the hang the memory records cannot happen even
if the model never recalls the fact. Only leading `VAR=value` assignments
are ever added; the command itself is never rewritten; failures run the
command unmodified. Each application notifies with the memory id.
See `safeguard-smoke.mjs` (10 checks, mock pi + real `lbm`).
