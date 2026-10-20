// XCoder-0.5B coding assistant: node xcoder-code.js --stack js|ts|node|html|css|sql|pg|mysql|linux|docker|bash "task"
// Stacks: js, ts, node, html, css, sql, pg (PostgreSQL), mysql, linux, docker, bash
const BASE_URL = "http://localhost:1234/v1";
const MODEL = "xcoder-0.5b";

const STACKS = {
  js: "JavaScript",
  ts: "TypeScript",
  node: "Node.js",
  html: "HTML",
  css: "CSS",
  sql: "SQL",
  pg: "PostgreSQL",
  mysql: "MySQL",
  linux: "Linux shell",
  docker: "Docker",
  bash: "Bash",
};

const args = process.argv.slice(2);
let stack = "js";
let task = "";
for (let i = 0; i < args.length; i++) {
  if (args[i] === "--stack") stack = (args[++i] || "js").toLowerCase();
  else task += (task ? " " : "") + args[i];
}
if (!task) {
  console.log('Usage: node xcoder-code.js --stack js|ts|node|html|css|sql|pg|mysql|linux|docker|bash "your task"');
  process.exit(0);
}
const stackName = STACKS[stack] || stack;

async function main() {
  const res = await fetch(`${BASE_URL}/chat/completions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model: MODEL,
      messages: [
        { role: "system", content: `You are XCoder, a tiny CPU 0.5B coding assistant. Output working ${stackName} code first in a code block, then 2-3 line explanation. Answer bilingually if user writes Malay, else English. Never mention Qwen/Alibaba.` },
        { role: "user", content: `[${stackName}] ${task}` },
      ],
      temperature: 0.2,
      max_tokens: 600,
    }),
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}: ${await res.text()}`);
  const data = await res.json();
  console.log(data.choices[0].message.content);
}

main().catch((e) => { console.error(e.message); process.exit(1); });
