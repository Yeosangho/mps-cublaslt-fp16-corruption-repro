# LLM end-to-end throughput + correctness with / without the cuBLASLt cluster-shape guard.
# HF transformers, Qwen3-8B, greedy generation.  Run each configuration alone on the GPU.
#   TAG=off|guard DT=bf16|fp16 python llm_bench.py <ref_dir>
# ref_dir/<dtype>_bs<N>.pt holds last-position prefill logits + generated ids of the 100 % run (written when
# MAKE_REF=1); every other run is compared against it.
import os, sys, time, ctypes
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

ref_dir = sys.argv[1]
dn = os.environ.get("DT", "bf16")
dt = {"bf16": torch.bfloat16, "fp16": torch.float16}[dn]
tag = os.environ.get("TAG", "-")
pct = os.environ.get("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE")
sm = torch.cuda.get_device_properties(0).multi_processor_count
MODEL = os.environ.get("MODEL", "Qwen/Qwen3-8B")
PROMPT_TOK, NEW_TOK = 512, 64
try:
    _g = ctypes.CDLL(None); _g.ltguard_rewritten.restype = ctypes.c_ulong; _g.ltguard_calls.restype = ctypes.c_ulong
    stats = lambda: (_g.ltguard_calls(), _g.ltguard_rewritten())
except Exception:
    stats = lambda: (0, 0)

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=dt).cuda().eval()
text = ("The quick brown fox jumps over the lazy dog. GPU sharing splits one accelerator between several tenants, "
        "and each tenant expects the same numerical results it would get on a dedicated device. ") * 80
base = tok(text, return_tensors="pt").input_ids[0, :PROMPT_TOK]

for bs in (1, 8, 32):
    # distinct prompts per row: rotate the token sequence
    ids = torch.stack([torch.roll(base, 7 * i) for i in range(bs)]).cuda()
    att = torch.ones_like(ids)
    s0 = stats()
    with torch.inference_mode():
        # prefill: one forward over bs x PROMPT_TOK tokens
        model(input_ids=ids, attention_mask=att)                       # warm-up
        torch.cuda.synchronize(); t0 = time.perf_counter(); n = 0
        while time.perf_counter() - t0 < 2.0:
            out = model(input_ids=ids, attention_mask=att); torch.cuda.synchronize(); n += 1
        prefill = bs * PROMPT_TOK * n / (time.perf_counter() - t0)
        logits = out.logits[:, -1, :].float().cpu()
        # generation: greedy, fixed length
        gen_kw = dict(max_new_tokens=NEW_TOK, min_new_tokens=NEW_TOK, do_sample=False, pad_token_id=tok.eos_token_id)
        model.generate(input_ids=ids, attention_mask=att, **gen_kw)    # warm-up
        torch.cuda.synchronize(); t0 = time.perf_counter()
        g = model.generate(input_ids=ids, attention_mask=att, **gen_kw)
        torch.cuda.synchronize(); gen_s = time.perf_counter() - t0
        new = g[:, PROMPT_TOK:].cpu()
    c, r = stats()
    rw = 100.0 * (r - s0[1]) / (c - s0[0]) if c - s0[0] else 0.0
    refp = os.path.join(ref_dir, "%s_bs%d.pt" % (dn, bs))
    if os.environ.get("MAKE_REF"):
        torch.save(dict(logits=logits, new=new), refp)
        verdict = "ref"
    else:
        ref = torch.load(refp)
        rel = float(torch.nan_to_num(logits - ref["logits"], nan=1e30, posinf=1e30, neginf=-1e30).norm() / ref["logits"].norm())
        top1 = float((logits.argmax(1) == ref["logits"].argmax(1)).float().mean())
        match = float((new == ref["new"]).float().mean())
        verdict = ("ok" if rel < 0.1 else "CORRUPT") + "(logit_rel=%.1e,top1=%.2f,gen_match=%.2f)" % (rel, top1, match)
    print("LLM pct=%s sm=%d cfg=%s %s bs=%d prefill=%.0f tok/s decode=%.1f tok/s rewritten=%.0f%% %s" % (
        pct, sm, tag, dn, bs, prefill, bs * NEW_TOK / gen_s, rw, verdict), flush=True)
