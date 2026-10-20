"""Muat Qwen3.5-0.8B dgn betul (multimodal/processor-based).

https://huggingface.co/Qwen/Qwen3.5-0.8B ialah model image-text-to-text
(processor + vision encoder), BUKAN text-only. `AutoModelForCausalLM` +
`AutoTokenizer` boleh gagal/salah-muat. Fungsi di sini cuba kelas auto
multimodal dahulu, baru jatuh ke CausalLM sebagai fallback, dan pulangkan
tokenizer yang menyokong `apply_chat_template` (cth guna `enable_thinking`).

Contoh:
    from qwen_load import load_qwen
    model, tok = load_qwen("Qwen/Qwen3.5-0.8B", dtype=torch.float32)
"""
MODEL = "Qwen/Qwen3.5-0.8B"
CLASSES = ("AutoModelForMultimodalLM", "AutoModelForImageTextToText",
           "AutoModelForVision2Seq", "AutoModelForCausalLM")


def _from_pretrained(cls, name, dtype):
    if dtype is None:
        return cls.from_pretrained(name)
    try:
        return cls.from_pretrained(name, dtype=dtype)
    except TypeError:
        return cls.from_pretrained(name, torch_dtype=dtype)


def load_qwen(model_name=MODEL, dtype=None):
    """Returns (model, tok). tok sentiasa ada apply_chat_template."""
    import transformers
    from transformers import AutoTokenizer

    tok = None
    try:
        from transformers import AutoProcessor
        proc = AutoProcessor.from_pretrained(model_name)
        tok = getattr(proc, "tokenizer", None)
    except Exception:
        tok = None
    if tok is None or not hasattr(tok, "apply_chat_template"):
        tok = AutoTokenizer.from_pretrained(model_name)

    last = None
    model = None
    for cls_name in CLASSES:
        cls = getattr(transformers, cls_name, None)
        if cls is None:
            continue
        try:
            model = _from_pretrained(cls, model_name, dtype)
            break
        except Exception as e:  # kelas salah atau checkpoint tak serasi
            last = e
    if model is None:
        raise RuntimeError(f"gagal muat {model_name}: {last}")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return model, tok
