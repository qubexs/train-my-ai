// EZCodex Chat CLI - clean branding, CPU-only, TTL 5min.
// Usage: node ezcodex-cli.js [--lang ms|en] [--name "Coder 77"]
import readline from "node:readline";
import { execFile } from "node:child_process";

const BASE_URL = "http://localhost:1234/v1";
const PUBLIC_NAME = "ezcodex-0.5b"; // LM Studio identifier, hidden from chat
const INTERNAL_KEY = "qwen2.5-0.5b-instruct"; // local base weights, hidden
const LMS = "C:\\Users\\testlab\\.lmstudio\\bin\\lms.exe";

function argVal(flag, def) {
  const i = process.argv.indexOf(flag);
  return i >= 0 && process.argv[i + 1] ? process.argv[i + 1] : def;
}
const NAME = argVal("--name", "EZCodex-0.5B");

let lang = (process.argv.includes("--lang") ? process.argv[process.argv.indexOf("--lang") + 1] : "ms").toLowerCase();
if (!["ms", "en"].includes(lang)) lang = "en";

function clean(text) {
  let t = text
    .replace(/qwen2\.5-0\.5b-instruct/gi, NAME)
    .replace(/\bqwen2\.5\b/gi, NAME)
    .replace(/\bqwen\b/gi, NAME)
    .replace(/\banthropic\b/gi, NAME)
    .replace(/\bclaude\b/gi, NAME)
    .replace(/\bmeta ai\b/gi, NAME)
    .replace(/\bllama\b/gi, NAME)
    .replace(/\balibaba cloud\b/gi, NAME)
    .replace(/\balibaba\b/gi, NAME)
    .replace(/\btongyi\b/gi, NAME)
    .replace(/\bqianwen\b/gi, NAME)
    .replace(/\btongyi qianwen\b/gi, NAME);
  // Catch-all: "AI language model created by X" -> "coding assistant, I am NAME"
  t = t.replace(/an?\s+ai\s+language\s+model\s+created\s+by\s+[^.,;\n]+/gi, `coding assistant, I am ${NAME}`);
  t = t.replace(/model\s+bahasa\s+ai\s+yang\s+dibangunkan\s+oleh\s+[^.,;\n]+/gi, `pembantu pengekodan, saya ${NAME}`);
  t = t.replace(/designed\s+by\s+[^.,;\n]+/gi, `built by ${NAME}`);
  t = t.replace(/dibina\s+oleh\s+[^.,;\n]+/gi, `dibina untuk ${NAME}`);
  t = t.replace(/created\s+by\s+[^.,;\n]+/gi, `built by ${NAME}`);
  return t;
}

function systemPrompt() {
  return lang === "ms"
    ? `Anda ialah ${NAME}, pembantu pengekodan CPU kecil 0.5B. Jawab ringkas dalam Bahasa Melayu. Jika ditanya siapa anda / siapa cipta anda, jawab HANYA: Saya ${NAME}, dibina untuk pengekodan. Jangan sebut Qwen, Alibaba, Alibaba Cloud, Tongyi, Anthropic, Claude, Meta, Llama.`
    : `You are ${NAME}, a tiny CPU 0.5B coding assistant. Answer briefly in English. If asked who you are / who created you, answer ONLY: I am ${NAME}, built for coding. Never mention Qwen, Alibaba, Alibaba Cloud, Tongyi, Anthropic, Claude, Meta, Llama.`;
}

async function isLoaded() {
  try {
    const r = await fetch(`${BASE_URL}/models`);
    const d = await r.json();
    return (d.data || []).some((m) => m.id === PUBLIC_NAME);
  } catch { return false; }
}

function loadModel() {
  return new Promise((resolve, reject) => {
    execFile(LMS, ["load", INTERNAL_KEY, "--gpu", "off", "--identifier", PUBLIC_NAME, "--ttl", "300", "-y"],
      { timeout: 120000 }, (err, stdout, stderr) => {
        if (err) reject(new Error("Auto-load gagal. Jalankan: lms load " + INTERNAL_KEY));
        else resolve(true);
      });
  });
}

const history = [];
const rl = readline.createInterface({ input: process.stdin, output: process.stdout, prompt: "anda> " });
console.log(`${NAME} Chat CLI [${lang}] — taip /lang ms|en, /clear, /keluar`);
console.log(`Input kekal di bawah. Jawapan di atas. Kelajuan di kanan bawah.\n`);
rl.prompt();

function clearInputLine() {
  try {
    readline.clearLine(process.stdout, 0);
    readline.cursorTo(process.stdout, 0);
  } catch {}
}
function printAbove(s = "") {
  clearInputLine();
  process.stdout.write(s + "\n");
}
function printSpeedRight(tps, totalTok, secs) {
  const w = process.stdout.columns || 80;
  const s = `${tps.toFixed(1)} T/s (${totalTok}t/${secs.toFixed(1)}s)`;
  const pad = Math.max(0, w - s.length);
  // dim grey, right-aligned at bottom of result block
  printAbove(" ".repeat(pad) + `\x1b[2m${s}\x1b[0m`);
}

rl.on("line", async (line) => {
  if (rl.closed) return;
  const q = line.trim();
  if (!q) { if (!rl.closed) rl.prompt(); return; }
  if (q === "/keluar" || q === "/exit" || q === "/quit") return rl.close();
  if (q.startsWith("/lang")) {
    lang = q.split(/\s+/)[1] === "en" ? "en" : "ms";
    history.length = 0;
    printAbove(`Bahasa: ${lang}`);
    if (!rl.closed) rl.prompt();
    return;
  }
  if (q === "/clear") { history.length = 0; printAbove("Sejarah dipadam."); if (!rl.closed) rl.prompt(); return; }

  try {
    if (rl.closed) return;
    if (!(await isLoaded())) {
      if (rl.closed) return;
      printAbove(`Model idle/unloaded — memuat semula ${NAME} (CPU, ~6s)...`);
      await loadModel();
    }
    if (rl.closed) return;
    history.push({ role: "user", content: q });
    // static input: pause typing, clear bottom line, show thinking above
    try { rl.pause(); } catch {}
    clearInputLine();
    printAbove(`… ${NAME} berfikir (CPU)…`);
    const t0 = Date.now();
    const res = await fetch(`${BASE_URL}/chat/completions`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: PUBLIC_NAME,
        messages: [{ role: "system", content: systemPrompt() }, ...history.slice(-10)],
        temperature: 0.3,
        max_tokens: 400,
      }),
    });
    const data = await res.json();
    const raw = data.choices[0].message.content;
    const out = clean(raw);
    history.push({ role: "assistant", content: out });
    const secs = (Date.now() - t0) / 1000;
    const tok = data.usage?.completion_tokens ?? Math.max(1, Math.round(out.length / 4));
    const tps = tok / Math.max(0.1, secs);
    // result above, speed right-bottom of block
    printAbove(`\n${NAME}> ${out}`);
    printSpeedRight(tps, tok, secs);
    printAbove("");
  } catch (e) {
    printAbove(`Ralat: ${e.message}`);
  }
  try { rl.resume(); } catch {}
  if (!rl.closed) rl.prompt();
});
