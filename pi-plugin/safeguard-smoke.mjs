// Mock-handler test for the git-rebase safeguard (no pi session needed).
// The decision half is pure; the handler half runs against a real `lbm`
// database (LB_MEMORY_DB) with the binary at LB_MEMORY_BIN (or `lbm` on
// PATH). The test cwd must derive (via `git rev-parse`) the same project
// slug the facts were stored under — see backfill-sessions.py.
//
// Run (from a checkout whose slug matches the stored facts, e.g. a test
// repo dir):
//   LB_MEMORY_DB=/tmp/sg/memory.lbdb LB_MEMORY_BIN=/path/to/lbm node safeguard-smoke.mjs
import { createLbMemoryExtension, decideSafeguards } from "./extension.ts";

let failures = 0;
function check(name, cond, extra = "") {
  console.log(`${cond ? "PASS" : "FAIL"} ${name} ${extra}`);
  if (!cond) failures++;
}

// --- pure decision unit ---
{
  const d = decideSafeguards("git rebase --continue", [
    { id: 0, content: "Always set GIT_EDITOR=true when running git rebase." },
  ]);
  check("adds missing safeguard", d.command === "GIT_EDITOR=true git rebase --continue", JSON.stringify(d));
  check("reports memory id", d.applied.length === 1 && d.applied[0].memoryId === 0);
}
{
  const d = decideSafeguards("GIT_EDITOR=vim git rebase -i main", [
    { id: 0, content: "Always set GIT_EDITOR=true when running git rebase." },
  ]);
  check("leaves set var alone", d.command === "GIT_EDITOR=vim git rebase -i main" && d.applied.length === 0);
}
{
  const d = decideSafeguards("ls -la", [{ id: 0, content: "Always set GIT_EDITOR=true." }]);
  check("ignores non-rebase", d.command === "ls -la" && d.applied.length === 0);
}

// --- full handler with mock pi + real lbm ---
const handlers = {};
const notes = [];
const mockPi = {
  on: (name, fn) => { handlers[name] = fn; },
  registerTool: () => {},
  registerCommand: () => {},
};
createLbMemoryExtension(mockPi, {});
const toolCall = handlers["tool_call"];
check("handler registered", typeof toolCall === "function");

const ctx = { cwd: process.env.SMOKE_CWD ?? "/tmp/safeguard-test/proj", ui: { notify: (msg) => notes.push(msg) } };

// 1. rebase without safeguard -> mutated
{
  notes.length = 0;
  const event = { toolName: "bash", input: { command: "git rebase --continue" } };
  await toolCall(event, ctx);
  check("mutates rebase command", event.input.command === "GIT_EDITOR=true git rebase --continue", JSON.stringify(event.input));
  check("notifies with memory ref", notes.length === 1 && notes[0].includes("#"), JSON.stringify(notes));
}
// 2. unrelated command -> untouched, no recall noise
{
  notes.length = 0;
  const event = { toolName: "bash", input: { command: "ls" } };
  await toolCall(event, ctx);
  check("passes through other commands", event.input.command === "ls" && notes.length === 0);
}
// 3. non-bash tool -> untouched
{
  const event = { toolName: "read", input: { path: "x" } };
  await toolCall(event, ctx);
  check("ignores non-bash tools", event.input.path === "x");
}
// 4. lbm failure -> fails open (bogus binary)
{
  const badHandlers = {};
  createLbMemoryExtension({ on: (n, f) => { badHandlers[n] = f; }, registerTool: () => {}, registerCommand: () => {} }, { lbmBin: () => "/nonexistent/lbm" });
  const event = { toolName: "bash", input: { command: "git rebase main" } };
  await badHandlers["tool_call"](event, ctx);
  check("fails open on lbm error", event.input.command === "git rebase main");
}

process.exit(failures ? 1 : 0);
