# Delta training untuk xcoder-gen-0.8b pada GTX 1070 (Windows, fp32) - TANPA Colab.
# Sambung-latih adapter semasa (master, cth salinan v2\checkpoint-18400) dengan
# baris baharu (+ replay), merge, eksport GGUF Q4_K_M ke models/*.gguf.new.
#
# PENTING (jangan ganggu proses lain):
#   - Fail model AKTIF (*.gguf) tidak disentuh. Output ditulis sebagai *.gguf.new.
#   - Penukaran .new -> aktif dibuat oleh CLI (swap_pending_models) bila fail
#     tidak dikunci (model tidak sedang digunakan).
#   - Socket lock 17877: tiada dua latihan (train_all/delta) serentak.
#
# Run (dari root repo):
#   .venv-gpu\Scripts\python.exe finetune\train_delta_qwen35.py ^
#       --data build\train\delta-<ts>.jsonl --master finetune\masters\xcoder-gen-0.8b
# Exit: 0 = GGUF .new sedia | 2 = tiada GPU (data kekal jsonl)
#       3 = latihan lain berjalan (data kekal, cuba lagi nanti) | 1 = ralat.
import argparse
import json
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))

from train_all import ensure_llamacpp, find_quantize  # noqa: E402

_LOCK_SOCK = None


def lock_or_exit(port=17877):
    """Elak dua latihan serentak (sama spt train_all). Sibuk => exit 3 (bukan
    ralat: data kekal dalam training.jsonl dan akan cuba lagi kemudian)."""
    global _LOCK_SOCK
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("127.0.0.1", port))
        s.listen(1)
    except OSError:
        print("Latihan lain sedang berjalan (lock 17877) — langkau kali ini.")
        sys.exit(3)
    _LOCK_SOCK = s

MODEL = "Qwen/Qwen3.5-0.8B"
OUT_NAME = "xcoder-gen-0.8b"
GGUF_NAME = f"{OUT_NAME}-q4_k_m.gguf"
SYSTEM = ("You are XCoder, a tiny CPU coding assistant. "
          "Answer briefly in the user's language (Bahasa Melayu or English). "
          "Never mention Qwen/Alibaba.")
ADAPTER_FILES = ("adapter_model.safetensors", "adapter_config.json",
                 "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
                 "processor_config.json", "special_tokens_map.json",
                 "tokenizer.model")


def promote_master(out_dir, master):
    """Salin adapter terbaik dari out_dir ke master (tmp + replace = selamat)."""
    cands = [p for p in out_dir.glob("checkpoint-*")
             if (p / "adapter_model.safetensors").is_file()]
    if (out_dir / "adapter_model.safetensors").is_file():
        cands.append(out_dir)

    def _step(p):
        try:
            return int(p.name.rsplit("-", 1)[-1])
        except ValueError:
            return 10 ** 9
    if not cands:
        return None
    src = sorted(cands, key=_step)[-1]
    master.mkdir(parents=True, exist_ok=True)
    copied = 0
    for fn in ADAPTER_FILES:
        s = src / fn
        if not s.is_file():
            continue
        tmp = master / (fn + ".tmp")
        shutil.copy2(s, tmp)
        tmp.replace(master / fn)
        copied += 1
    return (src, copied) if copied else None


def export_gguf(model_dir, models_dir, gguf_name, quant="Q4_K_M"):
    """HF dir -> f16.gguf -> models/<gguf_name>.new (quant). Return Path|None."""
    conv = ensure_llamacpp(ROOT / "build" / "train")
    conv_py = next((p for p in (ROOT / ".venv-gguf" / "Scripts" / "python.exe",
                                ROOT / ".venv-gguf" / "bin" / "python")
                    if p.is_file()), None)
    f16 = Path(model_dir).parent / (Path(model_dir).name + "-f16.gguf")
    subprocess.run([str(conv_py or sys.executable), str(conv), str(model_dir),
                    "--outfile", str(f16), "--outtype", "f16"], check=True)
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    dest_new = models_dir / f"{gguf_name}.new"
    if dest_new.exists():
        try:
            dest_new.unlink()
        except OSError:
            print(f"PERINGATAN: {dest_new.name} lama tidak boleh dipadam (dikunci?)")
    subprocess.run([find_quantize(), str(f16), str(dest_new), quant], check=True)
    f16.unlink(missing_ok=True)
    if not dest_new.is_file() or dest_new.stat().st_size < 50_000_000:
        print("Ralat: GGUF output mencurigakan kecil")
        return None
    return dest_new


def main():
    ap = argparse.ArgumentParser(description="Delta LoRA untuk xcoder-gen-0.8b (lokal)")
    ap.add_argument("--data", default="", help="JSONL delta (delta+replay dari CLI)")
    ap.add_argument("--master", default=str(ROOT / "finetune" / "masters" / OUT_NAME),
                    help="adapter semasa (checkpoint v2 yang disalin sekali)")
    ap.add_argument("--out", default="", help="folder output (lalai build/train/<ts>)")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--models-dir", default=str(ROOT / "models"))
    ap.add_argument("--thinking", action="store_true",
                    help="sasaran sertakan medan reasoning (<think>)")
    ap.add_argument("--no-export", action="store_true",
                    help="jangan merge/GGUF — simpan adapter + promote master sahaja")
    ap.add_argument("--export-only", action="store_true",
                    help="merge + GGUF dari master semasa (tiada latihan)")
    ap.add_argument("--quant", default="Q4_K_M")
    ap.add_argument("--gguf-name", default="", help="nama fail GGUF (lalai GGUF_NAME)")
    a = ap.parse_args()

    lock_or_exit()
    try:
        import torch
        cuda = torch.cuda.is_available()
    except Exception:
        cuda = False
    if not cuda:
        print("TIADA GPU (CUDA) - data kekal dalam training.jsonl, latihan dilangkau.")
        return 2

    master = Path(a.master)
    if not (master / "adapter_model.safetensors").is_file():
        print(f"Ralat: master adapter tiada di {master} "
              f"(init master dahulu: train_delta_runner.py --init-master)")
        return 1
    models_dir = Path(a.models_dir)
    gguf_name = a.gguf_name or GGUF_NAME

    from datasets import Dataset
    from peft import PeftModel
    from qwen_load import load_qwen
    from transformers import AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    model, tok = load_qwen(MODEL, dtype=torch.float32)
    if (master / "tokenizer_config.json").is_file():
        try:
            mtok = AutoTokenizer.from_pretrained(str(master))
            if mtok is not None:
                tok = mtok
        except Exception:
            pass
    model.config.use_cache = False
    model = PeftModel.from_pretrained(model, str(master), is_trainable=True)

    if a.export_only:
        out_dir = Path(a.out) if a.out else ROOT / "build" / "train" / (
            f"{OUT_NAME}-merge-{time.strftime('%Y%m%d-%H%M%S')}")
        out_dir.mkdir(parents=True, exist_ok=True)
        merged = model.merge_and_unload()
        merged.save_pretrained(str(out_dir))
        tok.save_pretrained(str(out_dir))
        dest = export_gguf(out_dir, models_dir, gguf_name, a.quant)
        if dest:
            print(f"GGUF sedia: {dest} (aktivasi oleh CLI bila model tidak digunakan)")
            return 0
        return 1

    if not a.data:
        print("Ralat: --data diperlukan (atau guna --export-only)")
        return 1
    data_p = Path(a.data)
    if not data_p.is_file():
        print(f"Ralat: fail data tiada: {data_p}")
        return 1
    rows = []
    for line in data_p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if (r.get("instruction") or "").strip() and (r.get("output") or "").strip():
            rows.append(r)
    if not rows:
        print("Ralat: --data kosong")
        return 1
    out_dir = Path(a.out) if a.out else ROOT / "build" / "train" / (
        f"{OUT_NAME}-delta-{time.strftime('%Y%m%d-%H%M%S')}")
    print(f"GPU ok (fp32) | delta={len(rows)} baris | out={out_dir}"
          + (" | thinking" if a.thinking else ""))

    def to_msgs(r):
        user = r["instruction"].strip()
        inp = (r.get("input") or "").strip()
        if inp:
            user += "\n" + inp
        asst = {"role": "assistant", "content": r["output"].strip()}
        if a.thinking:
            asst["reasoning_content"] = (r.get("reasoning") or "").strip()
        return {"messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user},
            asst,
        ]}

    ds = Dataset.from_list([to_msgs(r) for r in rows])
    out_dir.mkdir(parents=True, exist_ok=True)
    args = SFTConfig(
        output_dir=str(out_dir),
        per_device_train_batch_size=a.batch, gradient_accumulation_steps=a.accum,
        num_train_epochs=a.epochs, learning_rate=a.lr, warmup_steps=5,
        lr_scheduler_type="linear", optim="adamw_torch", weight_decay=0.01,
        logging_steps=10, save_steps=100, save_total_limit=2, seed=77,
        packing=False, completion_only_loss=True, max_length=a.max_len,
        fp16=False, bf16=False, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_pin_memory=False, report_to="none",
    )
    trainer = SFTTrainer(model=model, args=args, train_dataset=ds,
                         processing_class=tok)
    trainer.train()

    if a.no_export:
        trainer.save_model(str(out_dir))
        pm = promote_master(out_dir, master)
        if pm:
            print(f"adapter disimpan; master dikemas kini ({pm[1]} fail) -> {out_dir}")
            return 0
        print(f"PERINGATAN: adapter baharu tidak dijumpai di {out_dir}")
        return 1

    merged = trainer.model.merge_and_unload()
    mdir = out_dir / "merged"
    merged.save_pretrained(str(mdir))
    tok.save_pretrained(str(mdir))
    print(f"merged -> {mdir}")
    dest = export_gguf(mdir, models_dir, gguf_name, a.quant)
    if not dest:
        return 1
    print(f"GGUF sedia: {dest} (aktivasi oleh CLI bila model tidak digunakan)")
    pm = promote_master(out_dir, master)
    if pm:
        print(f"master dikemas kini dari {pm[0]} ({pm[1]} fail)")
    else:
        print("PERINGATAN: adapter baharu tidak dijumpai utk master")
    return 0


if __name__ == "__main__":
    sys.exit(main())
