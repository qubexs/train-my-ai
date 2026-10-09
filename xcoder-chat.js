// XCoder-0.5B bilingual chat: node xcoder-chat.js --lang ms|en "soalan anda"
const BASE_URL = "http://localhost:1234/v1";
const MODEL = "xcoder-0.5b";

const args = process.argv.slice(2);
let lang = "en";
let prompt = "";
for (let i = 0; i < args.length; i++) {
  if (args[i] === "--lang") lang = (args[++i] || "en").toLowerCase();
  else prompt += (prompt ? " " : "") + args[i];
}
if (!prompt) {
  console.log('Usage: node xcoder-chat.js --lang ms|en "your question"');
  process.exit(0);
}

const system = lang === "ms"
  ? "Anda ialah XCoder, pembantu pengekodan kecil yang berjalan pada CPU dengan 0.5B parameter. Jawab dalam Bahasa Melayu yang ringkas dan jelas."
  : "You are XCoder, a tiny CPU 0.5B coding assistant. Answer briefly and clearly in English. Never mention Qwen/Alibaba.";

async function main() {
  const res = await fetch(`${BASE_URL}/chat/completions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model: MODEL,
      messages: [
        { role: "system", content: system },
        { role: "user", content: prompt },
      ],
      temperature: 0.3,
      max_tokens: 300,
    }),
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}: ${await res.text()}`);
  const data = await res.json();
  console.log(data.choices[0].message.content);
}

main().catch((e) => { console.error(e.message); process.exit(1); });
