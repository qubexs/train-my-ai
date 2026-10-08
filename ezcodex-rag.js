// EZCodex-0.5B minimal CPU RAG - zero deps.
// Usage: node ezcodex-rag.js "soalan" [--lang ms|en] [--k 2] [--show-sources]
import fs from "node:fs";
import path from "node:path";

const BASE_URL = "http://localhost:1234/v1";
const MODEL = "ezcodex-0.5b";
const DOCS_DIR = new URL("./docs/", import.meta.url);

function tokenize(s) {
  return s.toLowerCase().replace(/[^a-z0-9\u00C0-\u024F_]+/gi, " ").split(/\s+/).filter(Boolean);
}

function chunkText(text, file, size = 600, overlap = 100) {
  const chunks = [];
  for (let i = 0; i < text.length; i += size - overlap) {
    const c = text.slice(i, i + size).trim();
    if (c) chunks.push({ file, text: c });
    if (i + size >= text.length) break;
  }
  return chunks;
}

function loadChunks() {
  const files = fs.readdirSync(DOCS_DIR).filter((f) => /\.(md|txt|js|ts)$/i.test(f));
  let all = [];
  for (const f of files) {
    const text = fs.readFileSync(path.join(DOCS_DIR.pathname.replace(/^\//, ""), f), "utf8");
    // fix Windows path from file URL
    all.push(...chunkText(text, f));
  }
  // fallback for Windows URL quirk: try direct join
  if (all.length === 0) {
    const dir = path.join(process.cwd(), "docs");
    for (const f of fs.readdirSync(dir)) {
      const text = fs.readFileSync(path.join(dir, f), "utf8");
      all.push(...chunkText(text, f));
    }
  }
  return all;
}

function score(queryTokens, docTokens) {
  const set = new Set(docTokens);
  let s = 0;
  for (const t of queryTokens) if (set.has(t)) s++;
  return s;
}

// --- CLI ---
const args = process.argv.slice(2);
let k = 2, lang = "en", showSources = false, query = "";
for (let i = 0; i < args.length; i++) {
  if (args[i] === "--k") k = parseInt(args[++i] || "2", 10);
  else if (args[i] === "--lang") lang = (args[++i] || "en").toLowerCase();
  else if (args[i] === "--show-sources") showSources = true;
  else query += (query ? " " : "") + args[i];
}
if (!query) {
  console.log('Usage: node ezcodex-rag.js "your question" [--lang ms|en] [--k 2] [--show-sources]');
  process.exit(0);
}

const chunks = loadChunks();
const qTokens = tokenize(query);
const ranked = chunks
  .map((c) => ({ ...c, s: score(qTokens, tokenize(c.text)) }))
  .sort((a, b) => b.s - a.s)
  .slice(0, k);

const context = ranked.map((r, i) => `[${i + 1}:${r.file} score=${r.s}]\n${r.text}`).join("\n\n");
if (showSources) console.log("--- RETRIEVED ---\n" + context + "\n--- ANSWER ---\n");

const system = lang === "ms"
  ? "Anda ialah EZCodex. Jawab dalam Bahasa Melayu ringkas HANYA berdasarkan KONTEKS di bawah. Jika tiada jawapan, katakan tidak tahu."
  : "You are EZCodex. Answer briefly ONLY from CONTEXT below. If not in context, say you don't know.";

const res = await fetch(`${BASE_URL}/chat/completions`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    model: MODEL,
    messages: [
      { role: "system", content: system },
      { role: "user", content: `KONTEKS:\n${context}\n\nSOALAN: ${query}` },
    ],
    temperature: 0.1,
    max_tokens: 400,
  }),
});
if (!res.ok) throw new Error(`HTTP ${res.status}: ${await res.text()}`);
const data = await res.json();
console.log(data.choices[0].message.content);
