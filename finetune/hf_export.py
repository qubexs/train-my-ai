"""Merge master LoRA -> folder HF sedia upload (susunan sama repo Qwen).

Guna:
  python finetune/hf_export.py --master finetune/masters/xcoder-gen1-think \
      --out build/v2

Hasil: <out>/merged/
  config.json, model.safetensors (bf16), generation_config.json,
  tokenizer.json, tokenizer_config.json, merges.txt, vocab.json,
  preprocessor_config.json, video_preprocessor_config.json,
  LICENSE, README.md, .gitattributes

Kemudian:
  hf upload myxcoder/xcoder-gen1-0.8b build\\v2\\merged --repo-type model
"""
import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "finetune"))

MODEL = "Qwen/Qwen3.5-0.8B"

EXPECTED = ("config.json", "tokenizer.json", "tokenizer_config.json",
            "merges.txt", "vocab.json", "preprocessor_config.json",
            "video_preprocessor_config.json", "LICENSE", "README.md")

GITATTR = ("*.safetensors filter=lfs diff=lfs merge=lfs -text\n"
           "*.bin filter=lfs diff=lfs merge=lfs -text\n")


def _card(base, name, master):
    return (f"---\nlicense: apache-2.0\nbase_model: {base}\n"
            f"library_name: transformers\npipeline_tag: image-text-to-text\n"
            f"---\n\n# {name}\n\n"
            f"XCoder gen1 (merged LoRA delta) daripada `{base}`.\n\n"
            f"- Adapter asal: `{master}`\n"
            f"- Dibina: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            f"Model penuh `transformers` (bukan GGUF). GGUF ada dalam "
            f"`gguf/` pada folder eksport latihan.\n")


def _copy_license(base, dest_dir):
    from huggingface_hub import hf_hub_download
    try:
        src = Path(hf_hub_download(base, "LICENSE"))
        shutil.copy2(src, Path(dest_dir) / "LICENSE")
        return True
    except Exception as e:  # noqa: BLE001
        print(f"[hf] amaran: LICENSE tidak disalin ({e})")
        return False


BASE_FILES = ("merges.txt", "vocab.json", "preprocessor_config.json",
              "video_preprocessor_config.json")


def _copy_base_files(base, mdir):
    from huggingface_hub import hf_hub_download
    copied = []
    for fn in BASE_FILES:
        if (mdir / fn).exists():
            continue
        try:
            shutil.copy2(Path(hf_hub_download(base, fn)), (mdir / fn))
            copied.append(fn)
        except Exception as e:  # noqa: BLE001
            print(f"[hf] amaran: {fn} tidak disalin ({e})")
    if copied:
        print(f"[hf] fail base disalin: {', '.join(copied)}")


def finalize_hf_dir(mdir, base=MODEL, name="xcoder-gen1-0.8b", master=""):
    """Tambah processor configs + LICENSE + README + .gitattributes."""
    mdir = Path(mdir)
    _copy_base_files(base, mdir)
    _copy_license(base, mdir)
    (mdir / "README.md").write_text(_card(base, name, master), encoding="utf-8")
    if not (mdir / ".gitattributes").exists():
        (mdir / ".gitattributes").write_text(GITATTR, encoding="utf-8")
    return mdir


def verify_hf_dir(mdir):
    """Semak fail wajib; pulangkan (ok, senarai fail)."""
    mdir = Path(mdir)
    files = sorted(p.name for p in mdir.iterdir() if p.is_file())
    have_weights = any(f.endswith((".safetensors", ".bin")) for f in files)
    missing = [f for f in EXPECTED if f not in files]
    ok = have_weights and not missing
    print(f"[hf] {len(files)} fail: {', '.join(files)}")
    if missing:
        print(f"[hf] PERINGATAN: tiada {missing}")
    if not have_weights:
        print("[hf] Ralat: tiada pemberat safetensors/bin")
    try:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(str(mdir))
        print(f"[hf] config ok: {cfg.model_type}")
    except Exception as e:  # noqa: BLE001
        print(f"[hf] amaran: config tidak dimuat ({e})")
        ok = False
    return ok, files


def swap_into_place(stage, dest):
    """Gantikan folder dest dgn stage (selamat: bina dulu, baru tukar)."""
    stage, dest = Path(stage), Path(dest)
    if dest.exists():
        shutil.rmtree(dest)
    os.replace(stage, dest)
    return dest


def merge_master(master, out_dir, base=MODEL, name="xcoder-gen1-0.8b",
                 device="cpu"):
    """master adapter -> <out>/merged/ (bf16, HF-ready)."""
    import torch
    from peft import PeftModel
    from qwen_load import load_qwen

    master = Path(master)
    if not (master / "adapter_model.safetensors").is_file():
        raise SystemExit(f"master adapter tiada: {master}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stage = out_dir / "_merged.tmp"
    if stage.exists():
        shutil.rmtree(stage)

    model, tok = load_qwen(base, dtype=torch.float32)
    model = PeftModel.from_pretrained(model, str(master))
    merged = model.merge_and_unload().to(torch.bfloat16).to(device)
    merged.save_pretrained(str(stage))
    tok.save_pretrained(str(stage))
    finalize_hf_dir(stage, base=base, name=name, master=str(master))
    dest = swap_into_place(stage, out_dir / "merged")
    print(f"[hf] merged -> {dest}")
    return dest


def main():
    ap = argparse.ArgumentParser(description="Master LoRA -> HF merged (bf16)")
    ap.add_argument("--master", default=str(ROOT / "finetune" / "masters" /
                                            "xcoder-gen1-think"))
    ap.add_argument("--out", default=str(ROOT / "build" / "v2"))
    ap.add_argument("--base", default=MODEL)
    ap.add_argument("--name", default="xcoder-gen1-0.8b")
    ap.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    a = ap.parse_args()

    dest = merge_master(a.master, a.out, a.base, a.name, a.device)
    ok, _ = verify_hf_dir(dest)
    if not ok:
        return 1
    print(f"[hf] upload: hf upload myxcoder/{a.name} {dest} --repo-type model")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
