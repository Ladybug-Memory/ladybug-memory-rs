import { execFile } from "node:child_process";
import type { AgentMessage } from "@earendil-works/pi-agent-core";
import type { ExtensionAPI, ExtensionCommandContext } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";

/**
 * Long-term memory for pi, backed by LadybugDB through the `lbm` binary.
 *
 * Design (mirrors @mem0/pi-agent-plugin's surface, mem0's backend replaced):
 * - `lbm` owns storage: one LadybugDB file, distilled facts + entity graph.
 * - Session transcripts stay in pi's sqlite; facts carry `message_refs`
 *   (opaque `<role>:<hash>` ids), never message text. Nothing is duplicated.
 * - Auto-capture distills per-sentence facts (`lbm remember --auto` drops
 *   entity-less sentences); the agent adjudicates conflicts via `lbm review`.
 * - `lbm` derives the project slug from its own cwd (git root), so the
 *   extension MUST spawn it with `cwd: ctx.cwd`.
 */

export type LbMemoryDeps = {
	/** Resolve the lbm binary. Default: LB_MEMORY_BIN, else `lbm` on PATH. */
	lbmBin?: () => string;
	/** Minimum message length worth auto-capturing. Default 40 chars. */
	minCaptureLength?: number;
};

const ToolParams = Type.Object({
	action: Type.Union([Type.Literal("remember"), Type.Literal("search")], {
		description: "remember stores a fact; search recalls matching facts and entities",
	}),
	text: Type.String({ description: "Fact to store, or query to search for" }),
	scope: Type.Optional(
		Type.Union([Type.Literal("project"), Type.Literal("session"), Type.Literal("global")], {
			description: "Memory scope (default project)",
		}),
	),
});

type RunResult = { stdout: string };

function runLbm(bin: string, args: readonly string[], cwd: string, input?: string): Promise<RunResult> {
	return new Promise((resolve, reject) => {
		const child = execFile(bin, [...args], { cwd, maxBuffer: 8 * 1024 * 1024 }, (error, stdout) => {
			if (error !== null) reject(new Error(`lbm ${args[0] ?? ""} failed: ${stderrOf(error)}${stdout}`));
			else resolve({ stdout });
		});
		if (input !== undefined && child.stdin !== null) {
			child.stdin.write(input);
			child.stdin.end();
		}
	});
}

function stderrOf(error: unknown): string {
	if (typeof error === "object" && error !== null && "message" in error) {
		const message = (error as { readonly message?: unknown }).message;
		if (typeof message === "string") return `${message} `;
	}
	return "";
}

/** Stable opaque message ref (pi-session-local); the text lives in sqlite. */
export function messageRefFor(role: string, text: string): string {
	let hash = 0x811c9dc5;
	for (let index = 0; index < text.length; index++) {
		hash ^= text.charCodeAt(index) ?? 0;
		hash = Math.imul(hash, 0x01000193);
	}
	return `${role}:${hash >>> 0}`;
}

function messageRole(message: AgentMessage): string | undefined {
	if (typeof message === "object" && message !== null && "role" in message) {
		const role = (message as { readonly role?: unknown }).role;
		return typeof role === "string" ? role : undefined;
	}
	return undefined;
}

function messageText(content: unknown): string {
	if (typeof content === "string") return content.slice(0, 8000);
	if (!Array.isArray(content)) return "";
	const parts: string[] = [];
	for (const block of content) {
		if (typeof block === "object" && block !== null && "type" in block && block.type === "text") {
			const text = (block as { readonly text?: unknown }).text;
			if (typeof text === "string") parts.push(text);
		}
	}
	return parts.join("\n").slice(0, 8000);
}

function messageContent(message: AgentMessage): unknown {
	if (typeof message === "object" && message !== null && "content" in message) {
		return (message as { readonly content?: unknown }).content;
	}
	return undefined;
}

type RememberHit = { readonly reports?: readonly unknown[] };

function formatRemember(stdout: string): string {
	try {
		const parsed = JSON.parse(stdout) as RememberHit;
		const reports = parsed.reports ?? [];
		if (reports.length === 0) return "remembered nothing (no distillable facts)";
		return `remembered ${reports.length} fact${reports.length === 1 ? "" : "s"} (see lbm review queue on conflict)`;
	} catch {
		return stdout.trim().slice(0, 2000);
	}
}

/**
 * Standing operational safeguards, enforced at every matching tool call.
 *
 * When the agent runs `git rebase` (or any command a memory prescribes an
 * environment safeguard for), the handler recalls lb-memory, extracts
 * `VAR=value` prescriptions from the hits, and prepends the missing ones
 * by mutating `event.input` in place — so the hang this memory records
 * (waiting for an interactive editor) cannot happen, even if the model
 * never saw the fact. Every application is notified with the memory id.
 *
 * Rules: only leading `VAR=value` assignments are ever added (the command
 * itself is never rewritten); commands that already set the var are left
 * alone; any failure fails open (command runs unmodified) — memory must
 * never break the session. Recall is model-free (no GLiNER load), ~50 ms.
 */
const REBASE_COMMAND = /\bgit\s+rebase\b/;
const ENV_PRESCRIPTION = /\b([A-Z][A-Z0-9_]{2,})=(\S+)/g;

export type SafeguardDecision = {
	readonly command: string;
	readonly applied: readonly { readonly assignment: string; readonly memoryId: unknown }[];
};

function hasEnvAssignment(command: string, name: string): boolean {
	return new RegExp(`(?:^|[\\s;&|])${name}=`).test(command);
}

/** Pure decision step (unit-testable): which safeguards to prepend. */
export function decideSafeguards(command: string, memories: readonly { readonly id?: unknown; readonly content?: unknown }[]): SafeguardDecision {
	if (!REBASE_COMMAND.test(command)) return { command, applied: [] };
	const applied: { assignment: string; memoryId: unknown }[] = [];
	let next = command;
	for (const memory of memories) {
		if (typeof memory.content !== "string") continue;
		ENV_PRESCRIPTION.lastIndex = 0;
		for (const match of memory.content.matchAll(ENV_PRESCRIPTION)) {
			const name = match[1] ?? "";
			const assignment = `${name}=${match[2] ?? ""}`;
			if (hasEnvAssignment(next, name)) continue;
			next = `${assignment} ${next}`;
			applied.push({ assignment, memoryId: memory.id });
		}
	}
	return { command: next, applied };
}

type LbmSearchJson = {
	readonly hits?: readonly { readonly entry?: { readonly id?: unknown; readonly content?: unknown } }[];
};

async function recallSafeguardMemories(bin: string, cwd: string, query: string): Promise<LbmSearchJson["hits"]> {
	const { stdout } = await runLbm(bin, ["search", "--scope", "project", "--json", query], cwd);
	try {
		return (JSON.parse(stdout) as LbmSearchJson).hits ?? [];
	} catch {
		return [];
	}
}

type SearchPayload = {
	readonly hits?: readonly {
		readonly entry?: { readonly id?: unknown; readonly content?: unknown; readonly category?: unknown };
		readonly score?: unknown;
		readonly entities?: readonly unknown[];
	}[];
};

function formatSearch(query: string, stdout: string): string {
	try {
		const parsed = JSON.parse(stdout) as SearchPayload;
		const hits = parsed.hits ?? [];
		if (hits.length === 0) return `No memories match "${query}".`;
		const lines = [`Memories matching "${query}":`];
		for (const hit of hits.slice(0, 10)) {
			const content = typeof hit.entry?.content === "string" ? hit.entry.content.replace(/\s+/g, " ") : "";
			const entities = (hit.entities ?? []).filter((e): e is string => typeof e === "string");
			lines.push(`- [#${String(hit.entry?.id ?? "?")}] ${content}${entities.length > 0 ? ` (entities: ${entities.join(", ")})` : ""}`);
		}
		return lines.join("\n");
	} catch {
		return stdout.trim().slice(0, 2000);
	}
}

export function createLbMemoryExtension(pi: ExtensionAPI, deps: LbMemoryDeps): void {
	const lbmBin = deps.lbmBin ?? ((): string => process.env.LB_MEMORY_BIN ?? "lbm");
	const minCaptureLength = deps.minCaptureLength ?? 40;

	pi.registerTool({
		name: "lb_memory",
		label: "Long-term memory",
		description:
			"Project-scoped long-term memory over a LadybugDB entity graph. remember stores a distilled fact (entity extraction, dedup, conflict detection included); search recalls facts and entities by topic.",
		parameters: ToolParams,
		async execute(_toolCallId, params, _signal, _onUpdate, ctx) {
			const bin = lbmBin();
			const scope = params.scope ?? "project";
			const text =
				params.action === "remember"
					? `remembered: ${formatRemember((await runLbm(bin, ["remember", "--scope", scope, "--json", params.text], ctx.cwd)).stdout)}`
					: formatSearch(params.text, (await runLbm(bin, ["search", "--scope", scope, "--json", params.text], ctx.cwd)).stdout);
			return { content: [{ type: "text", text }], details: {} };
		},
	});

	// Live feed: distill finalized messages into facts. Entity-less sentences
	// are dropped by `lbm remember --auto`, so transcripts are never copied.
	// Standing safeguards: consult memory on every bash call and enforce
	// env-var prescriptions (e.g. GIT_EDITOR=true for git rebase) by
	// mutating the command in place. Fail-open: any error runs unmodified.
	pi.on("tool_call", async (event, ctx) => {
		if (event.toolName !== "bash") return;
		const command = typeof event.input.command === "string" ? event.input.command : "";
		if (!REBASE_COMMAND.test(command)) return;
		let memories: LbmSearchJson["hits"] = [];
		try {
			memories = await recallSafeguardMemories(lbmBin(), ctx.cwd, "git rebase");
		} catch {
			return;
		}
		const flat = (memories ?? []).map((hit) => ({ id: hit.entry?.id, content: hit.entry?.content }));
		const decision = decideSafeguards(command, flat);
		if (decision.applied.length === 0) return;
		event.input.command = decision.command;
		for (const applied of decision.applied) {
			ctx.ui.notify(
				`lb-memory: applied memory #${String(applied.memoryId ?? "?")} — prepended \`${applied.assignment}\` to git rebase`,
				"info",
			);
		}
	});

	pi.on("message_end", async (event, ctx) => {
		const message = (event as { readonly message?: unknown }).message as AgentMessage | undefined;
		if (message === undefined) return;
		const role = messageRole(message);
		if (role !== "user" && role !== "assistant") return;
		const text = messageText(messageContent(message));
		if (text.trim().length < minCaptureLength) return;
		try {
			await runLbm(
				lbmBin(),
				["remember", "--auto", "--scope", "project", "--message-ref", messageRefFor(role, text)],
				ctx.cwd,
				text,
			);
		} catch {
			// Memory must never break the session; failures stay silent here
			// and surface under /lb-status (counts) on next use.
		}
	});

	const command = (
		name: string,
		description: string,
		run: (args: string, ctx: ExtensionCommandContext) => Promise<string>,
	): void => {
		pi.registerCommand(name, {
			description,
			handler: async (args: string, ctx: ExtensionCommandContext) => {
				try {
					ctx.ui.notify(await run(args, ctx), "info");
				} catch (error) {
					ctx.ui.notify(`lb-memory: ${error instanceof Error ? error.message : String(error)}`, "error");
				}
			},
		});
	};

	command("lb-remember", "Store a fact in long-term memory: /lb-remember <text>", async (args, ctx) => {
		if (args.trim().length === 0) return "Usage: /lb-remember <text>";
		const { stdout } = await runLbm(lbmBin(), ["remember", "--json", args], ctx.cwd);
		return formatRemember(stdout);
	});

	command("lb-search", "Search long-term memory: /lb-search <query>", async (args, ctx) => {
		if (args.trim().length === 0) return "Usage: /lb-search <query>";
		const { stdout } = await runLbm(lbmBin(), ["search", "--json", args], ctx.cwd);
		return formatSearch(args, stdout);
	});

	command("lb-forget", "Delete memories matching a query (lists first): /lb-forget <query> [--apply]", async (args, ctx) => {
		if (args.trim().length === 0) return "Usage: /lb-forget <query> [--apply]";
		const apply = args.includes("--apply");
		const query = args.replace("--apply", "").trim();
		const { stdout } = await runLbm(lbmBin(), apply ? ["forget", "--apply", query] : ["forget", query], ctx.cwd);
		return stdout.trim().slice(0, 2000);
	});

	command("lb-tour", "Browse memories by category: /lb-tour", async (_args, ctx) => {
		const { stdout } = await runLbm(lbmBin(), ["tour"], ctx.cwd);
		return stdout.trim().slice(0, 4000);
	});

	command("lb-status", "Memory health: counts and database path", async (_args, ctx) => {
		const { stdout } = await runLbm(lbmBin(), ["status"], ctx.cwd);
		return stdout.trim().slice(0, 1000);
	});
}
