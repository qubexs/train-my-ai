"""EZCodex-0.5B minimal CPU prototype (Python, stdlib only)."""
import json
import urllib.request

BASE_URL = "http://localhost:1234/v1"
MODEL = "ezcodex-0.5b"

payload = {
    "model": MODEL,
    "messages": [
        {"role": "system", "content": "You are EZCodex, a tiny CPU coding assistant."},
        {"role": "user", "content": 'Reply with exactly: Hello, EZCodex! Then on a new line write one short sentence saying you run on CPU with 0.5B params.'},
    ],
    "temperature": 0.2,
    "max_tokens": 100,
}

req = urllib.request.Request(
    f"{BASE_URL}/chat/completions",
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(req, timeout=120) as resp:
    data = json.loads(resp.read().decode("utf-8"))

print(data["choices"][0]["message"]["content"])
print(f"\n[model={data.get('model')}]")
