// EZCodex-0.5B coding tools - CPU-only, zero deps, safe-whitelist.
// Usage:
//   node ezcodex-tools.js --tool list [--arg docs] [--lang ms|en]
//   node ezcodex-tools.js --tool read --arg docs/js-basics.md [--lang ms|en]
//   node ezcodex-tools.js --tool run --arg "console.log([1,2,3].map(x=>x*2))" [--lang ms|en]
import fs from "node:fs";
import path from "node:path";
import { execFile } from "node:child_process";

const BASE_URL = "http://localhost:1234/v1";
const MODEL = "ezcodex-0.5b";
const ROOT = process.cwd();

function safePath(p) {
  const resolved = path.resolve(ROOT, p);
  if (!resolved.startsWith(ROOT)) throw new Error("Blocked: path outside workspace");
  return resolved;
}

function toolList(arg = "docs") {
  const dir = safePath(arg);
  return fs.readdirSync(dir).join("\n");
}

function toolRead(arg) {
  const f = safePath(arg);
  const stat = fs.statSync(f);
  if (stat.size > 20000) throw new Error("File too large (>20KB)");
  return fs.readFileSync(f, "utf8").slice(0, 8000);
}

function toolRun(arg) {
  // safe: run via node -e with 5s timeout, no shell
  return new Promise((resolve, reject) => {
    execFile("node", ["-e", arg], { timeout: 5000, windowsHide: true }, (err, stdout, stderr) => {
      if (err) resolve(`ERROR: ${(stderr || err.message).slice(0, 2000)}`);
      else resolve((stdout || "(no output)").slice(0, 4000));
    });
  });
}

const args = process.argv.slice(2);
let tool = "list", arg = "docs", lang = "en";
for (let i = 0; i < args.length; i++) {
  if (args[i] === "--tool") tool = (args[++i] || "list").toLowerCase();
  else if (args[i] === "--arg") arg = args[++i] || "";
  else if (args[i] === "--lang") lang = (args[++i] || "en").toLowerCase();
}

let result = "";
if (tool === "list") result = toolList(arg || "docs");
else if (tool === "read") result = toolRead(arg);
else if (tool === "run") result = await toolRun(arg);
else { console.error("tool must be list|read|run"); process.exit(1); }

console.log(`--- TOOL:${tool} RESULT ---\n${result}\n--- EXPLANATION (${lang}) ---`);

const system = lang === "ms"
  ? "Anda ialah EZCodex. Terangkan hasil tool di bawah dalam Bahasa Melayu ringkas (2-4 ayat)."
  : "You are EZCodex. Explain the tool result below briefly in English (2-4 sentences).";

const res = await fetch(`${BASE_URL}/chat/completions`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    model: MODEL,
    messages: [
      { role: "system", content: system },
      { role: "user", content: `TOOL: ${tool}\nARG: ${arg}\nRESULT:\n${result}` },
    ],
    temperature: 0.2,
    max_tokens: 300,
  }),
});
const data = await res.json();
console.log(data.choices[0].message.content);
