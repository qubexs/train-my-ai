"""Prompt-suite eval untuk GGUF pakar: identiti + semakan per stack via llama-cli.
Usage: python datasets/scripts/eval_suite.py <model.gguf> [--stack docker] [--tokens 150] [--min 0.6]
--stack lalai disimpulkan dari nama fail (xcoder-<stack>-...), else semua.
Keluar 0 jika skor >= --min, else 1. Model tidak diubah.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "app"))
from backends import LlamaCliBackend

BAD = ["qwen", "alibaba", "tongyi", "claude", "satu minit", "bilakah masa"]

# (stack, prompt, must-substrings lowercase)
SUITE = [
    ("general", "siapa anda?", ["xcoder"]),
    ("general", "apa itu fungsi dalam pengaturcaraan?", ["fungsi"]),
    ("bash", "tulis perintah papar direktori semasa", ["pwd"]),
    ("bash", "tulis arahan senaraikan semua fail termasuk tersembunyi", ["ls"]),
    ("css", "tulis css untuk teks tengah", ["text-align", "center"]),
    ("css", "tulis css latar merah untuk kelas amaran", ["background", "red"]),
    ("docker", "apa itu docker image?", ["image"]),
    ("docker", "tulis Dockerfile Node minima", ["from"]),
    ("html", "tulis html pautan ke contoh.com", ["<a", "href"]),
    ("html", "tulis html tajuk utama selamat datang", ["<h1"]),
    ("javascript", "tulis fungsi tambah dua nombor", ["function", "return"]),
    ("javascript", "apa itu console.log?", ["console"]),
    ("laravel", "apa itu artisan dalam laravel?", ["artisan"]),
    ("laravel", "tulis arahan artisan cipta controller", ["artisan"]),
    ("linux", "tulis arahan cari teks dalam fail", ["grep"]),
    ("linux", "apa itu kebenaran chmod 755?", ["chmod"]),
    ("mysql", "tulis sql cipta pangkalan data", ["create", "database"]),
    ("mysql", "tulis query pilih semua dari jadual users", ["select"]),
    ("nodejs", "apa itu npm?", ["npm"]),
    ("nodejs", "tulis arahan pasang pakej express", ["npm", "express"]),
    ("php", "tulis php cetak hello", ["echo"]),
    ("php", "apa maksud tanda dolar dalam php?", ["$"]),
    ("postgresql", "tulis arahan sambung ke postgres dengan psql", ["psql"]),
    ("postgresql", "tulis sql pilih versi postgres", ["version"]),
    ("python", "tulis fungsi tambah dua nombor python", ["def", "return"]),
    ("python", "apa itu pip?", ["pip"]),
    ("sql", "tulis query kira baris dalam users", ["count", "select"]),
    ("sql", "apa itu primary key?", ["primary"]),
    ("tailwind", "tulis kelas tailwind untuk teks tengah", ["text-center"]),
    ("tailwind", "tulis kelas tailwind latar biru", ["bg-"]),
    ("typescript", "tulis jenis typescript untuk senarai nombor", ["number"]),
    ("typescript", "apa itu interface dalam typescript?", ["interface"]),
]


def find_cli():
    for d in (ROOT / "bin", ROOT / "app" / "bin"):
        hit = list(d.glob("llama-cli.exe")) + list(d.glob("llama-cli"))
        hit = [h for h in hit if h.is_file()]
        if hit:
            return hit[0]
    raise SystemExit("llama-cli tiada — jalankan CLI --setup dahulu")


def infer_stack(model_path):
    stem = Path(model_path).stem.lower()  # xcoder-docker-0.5b-q4_k_m
    if stem.startswith("xcoder-"):
        return stem.split("-")[1]
    return None


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opts = {a.split("=")[0]: (a.split("=", 1)[1] if "=" in a else True)
            for a in sys.argv[1:] if a.startswith("--")}
    if not args:
        raise SystemExit("guna: eval_suite.py <model.gguf> [--stack X] [--tokens 150] [--min 0.6]")
    model = Path(args[0])
    if not model.is_file():
        raise SystemExit(f"model tiada: {model}")
    tokens = int(opts.get("--tokens", 150))
    min_score = float(opts.get("--min", 0.6))
    stack = opts.get("--stack") if isinstance(opts.get("--stack"), str) else None
    stack = (stack or infer_stack(model) or "").lower() or None
    items = [(s, p, m) for s, p, m in SUITE if stack is None or s in ("general", stack)]
    be = LlamaCliBackend(find_cli(), model, n_predict=tokens)
    esys = "Anda pakar pengekodan. Jawab ringkas dalam Bahasa Melayu."
    passed, tps_all = 0, []
    for s, prompt, must in items:
        try:
            raw, tps = be.chat(esys, [{"role": "user", "content": prompt}])
        except Exception as e:
            print(f"[{s}] FAIL ({prompt[:40]}): ralat {e}")
            continue
        low = raw.lower()
        missing = [w for w in must if w.lower() not in low]
        bad = [w for w in BAD if w in low]
        ok = not missing and not bad
        passed += ok
        if tps:
            tps_all.append(tps)
        why = ""
        if missing:
            why += f" tiada:{','.join(missing)}"
        if bad:
            why += f" larangan:{','.join(bad)}"
        print(f"[{'LULUS' if ok else 'GAGAL'}:{s}] {prompt[:45]}{why}")
        if not ok:
            print(f"    <- {raw[:150]!r}")
    score = passed / len(items) if items else 0.0
    avg = sum(tps_all) / len(tps_all) if tps_all else 0.0
    print(f"SCORE: {passed}/{len(items)} = {score:.2f}  AVG_TPS: {avg:.1f}")
    sys.exit(0 if score >= min_score else 1)


if __name__ == "__main__":
    main()
