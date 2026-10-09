// XCoder-0.5B minimal CPU prototype (Node.js, no deps, Node 18+ fetch).
const BASE_URL = "http://localhost:1234/v1";
const MODEL = "xcoder-0.5b";

async function main() {
  const res = await fetch(`${BASE_URL}/chat/completions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model: MODEL,
      messages: [
        { role: "system", content: "You are XCoder, a tiny CPU 0.5B coding assistant. Never mention Qwen/Alibaba." },
        { role: "user", content: "Reply show what are you skill, and coding program" },
      ],
      temperature: 0.2,
      max_tokens: 100,
    }),
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}: ${await res.text()}`);
  const data = await res.json();
  console.log(data.choices[0].message.content);
  console.log(`\n[model=${data.model}]`);
}

main().catch((e) => { console.error(e.message); process.exit(1); });
