// Ensure XCoder model is loaded, reload if TTL-unloaded. Run before any call.
// Usage: node xcoder-ensure.js  (exits 0 when ready)
import { execFile } from "node:child_process";

const BASE_URL = "http://localhost:1234/v1";
const MODEL = "xcoder-0.5b";
const LMS = "C:\\Users\\testlab\\.lmstudio\\bin\\lms.exe";

async function loaded() {
  try {
    const r = await fetch(`${BASE_URL}/models`);
    const d = await r.json();
    return (d.data || []).some((m) => m.id === MODEL);
  } catch { return false; }
}

function load() {
  return new Promise((resolve, reject) => {
    execFile(LMS, ["load", "qwen2.5-0.5b-instruct", "--gpu", "off", "--identifier", MODEL, "--ttl", "300", "-y"],
      { timeout: 120000 }, (err, stdout, stderr) => {
        if (err) reject(new Error((stderr || err.message).slice(0, 1000)));
        else resolve(stdout);
      });
  });
}

if (await loaded()) {
  console.log(`${MODEL} already loaded.`);
} else {
  console.log(`${MODEL} not loaded, reloading (CPU, TTL 300s)...`);
  await load();
  console.log(`${MODEL} ready.`);
}
