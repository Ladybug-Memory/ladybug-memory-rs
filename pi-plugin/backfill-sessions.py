#!/usr/bin/env python3
"""Backfill pi sessions into lb-memory for evaluation.

Walks ~/.pi/agent/sessions/<project>/, takes user+assistant messages from
the last N days, and distills each through `lbm remember --auto` into a
TEST database (default /tmp/lbm-eval/memory.lbdb — never your live store).

Provenance: --session <pi-session-id>, --message-ref <pi-message-id>, so
every fact points back at the exact pi message it came from.

Requirements: `lbm` on PATH (or LB_MEMORY_BIN), ORT_DYLIB_PATH set for the
GLiNER model (same as the lbm README).

Usage:
    python3 backfill-sessions.py --days 7                  # dry run first!
    python3 backfill-sessions.py --days 7 --apply          # ingest
    python3 backfill-sessions.py --days 7 --apply --project ladybug-memory-rs
    LB_MEMORY_DB=/tmp/lbm-eval/memory.lbdb lbm tour        # inspect
"""

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timedelta, timezone

SESSIONS = os.path.expanduser("~/.pi/agent/sessions")
DEFAULT_DB = "/tmp/lbm-eval/memory.lbdb"
MIN_CHARS = 40
SKIP_ROLES = {"system", "toolResult"}


def parse_file_date(name: str):
    """Filenames look like 2026-10-22T18-49-08-985Z_<session-id>.jsonl."""
    m = re.match(r"(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2})", name)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1), "%Y-%m-%dT%H-%M-%S").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def message_text(content) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    return "\n".join(
        b.get("text", "") for b in content
        if isinstance(b, dict) and b.get("type") == "text" and isinstance(b.get("text"), str)
    )


def iter_messages(sessions_dir, projects, since):
    """Yield (project_slug, session_id, msg_id, role, text)."""
    for proj in sorted(os.listdir(sessions_dir)):
        if projects and proj not in projects and not any(p in proj for p in projects):
            continue
        pdir = os.path.join(sessions_dir, proj)
        if not os.path.isdir(pdir):
            continue
        for name in sorted(os.listdir(pdir)):
            if not name.endswith(".jsonl"):
                continue
            date = parse_file_date(name)
            if date is None or date < since:
                continue
            session_id = name[:-len(".jsonl")].split("_", 1)[-1]
            with open(os.path.join(pdir, name)) as f:
                for line in f:
                    try:
                        d = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if d.get("type") != "message":
                        continue
                    msg = d.get("message", {})
                    if msg.get("role") in SKIP_ROLES:
                        continue
                    text = message_text(msg.get("content")).strip()
                    yield proj, session_id, d.get("id", "?"), msg.get("role", "?"), text


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=float, default=7)
    ap.add_argument("--project", action="append", default=[], help="substring match on session dir (repeatable)")
    ap.add_argument("--db", default=os.environ.get("LB_MEMORY_DB", DEFAULT_DB))
    ap.add_argument("--apply", action="store_true", help="actually ingest (default: dry run)")
    ap.add_argument("--min-chars", type=int, default=MIN_CHARS)
    ap.add_argument("--include-tool-results", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="max messages to ingest (0 = all)")
    args = ap.parse_args()

    if args.include_tool_results:
        SKIP_ROLES.discard("toolResult")

    lbm = os.environ.get("LB_MEMORY_BIN", "lbm")
    # No ORT_DYLIB_PATH needed: `lbm fetch-model` provisions the model + ORT
    # lib into cache dirs on first run (also auto-provisioned on first use).

    since = datetime.now(timezone.utc) - timedelta(days=args.days)
    cand = [(p, s, i, r, t) for (p, s, i, r, t) in iter_messages(SESSIONS, args.project, since)]
    kept = [(p, s, i, r, t) for (p, s, i, r, t) in cand if len(t) >= args.min_chars]
    dropped = len(cand) - len(kept)
    if args.limit:
        kept = kept[: args.limit]
    print(f"{len(cand)} text messages in scope, {len(kept)} over {args.min_chars} chars ({dropped} skipped as too short)")

    if not args.apply:
        print("-- dry run (pass --apply to ingest into %s) --" % args.db)
        for p, s, i, r, t in kept[:15]:
            print(f"[{p}][{r}] {t[:120].replace(chr(10), ' ')}")
        if len(kept) > 15:
            print(f"... and {len(kept) - 15} more")
        return 0

    env = dict(os.environ, LB_MEMORY_DB=args.db)
    ok, facts, by_session = 0, 0, {}
    for n, (proj, session, mid, role, text) in enumerate(kept, 1):
        # Project scope derives from lbm's own cwd; run from the project dir
        # when it maps to a real path, else repo root fallback inside lbm.
        r = subprocess.run(
            [lbm, "remember", "--auto", "--json",
             "--session", session, "--message-ref", f"{mid}",
             "--project", project_slug(proj)],
            input=text, capture_output=True, text=True, env=env,
        )
        if r.returncode != 0:
            print(f"[{n}/{len(kept)}] FAIL {session}/{mid}: {r.stderr.strip()[:200]}")
            continue
        try:
            reports = json.loads(r.stdout).get("reports", [])
        except json.JSONDecodeError:
            reports = []
        ok += 1
        facts += len(reports)
        by_session[session] = by_session.get(session, 0) + len(reports)
        print(f"[{n}/{len(kept)}] {role} {mid[:8]} -> {len(reports)} fact(s)")
    print(f"ingested {ok}/{len(kept)} messages -> {facts} facts across {len(by_session)} sessions into {args.db}")
    print(f"next: LB_MEMORY_DB={args.db} lbm tour | lbm status | lbm review")
    return 0


def project_slug(session_dir: str) -> str:
    """Map pi's session-dir slug back to lbm's project slug.

    pi turns /Users/arun/src/ladybug-memory-rs into
    --Users-arun-src-ladybug-memory-rs-- (slashes become dashes), while lbm
    uses the git-root basename. Recover it as the longest trailing slice
    that exists under ~/src, so backfilled facts land in the same scope
    live capture would use.
    """
    name = session_dir.strip("-")
    home_src = os.path.expanduser("~/src")
    parts = name.split("-")
    for i in range(len(parts)):
        cand = "-".join(parts[i:])
        if os.path.isdir(os.path.join(home_src, cand)):
            return cand
    return parts[-1] if parts else "default"


if __name__ == "__main__":
    sys.exit(main())
