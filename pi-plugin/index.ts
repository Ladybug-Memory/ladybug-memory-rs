import { createLbMemoryExtension } from "./extension.ts";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

export default function register(pi: ExtensionAPI): void {
	createLbMemoryExtension(pi, {});
}
