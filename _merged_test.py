import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

mpath = 'build/train/xcoder-bash-merged'
tok = AutoTokenizer.from_pretrained(mpath, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(mpath, dtype=torch.float32).eval()
sys.stdout.write('loaded\n')

prompt = ('<|im_start|>system\nYou are XCoder, a tiny CPU 0.5B coding '
          'assistant. Never mention Qwen/Alibaba.<|im_end|>\n'
          '<|im_start|>user\nWhat does `set -euo pipefail` do?<|im_end|>\n'
          '<|im_start|>assistant\n')
ids = tok(prompt, return_tensors='pt')['input_ids']
with torch.no_grad():
    out = model.generate(ids, max_new_tokens=60, do_sample=False,
                         eos_token_id=tok.eos_token_id)
gen = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
print('MERGED-OUT:', repr(gen[:300]))
