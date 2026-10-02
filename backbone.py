#!/usr/bin/env python
# cuDNN-only check: ResNet backbone (conv/BN/pool, fc removed) in fp16/bf16 under an MPS share, compared with a
# reference produced by the same script on the full GPU.  Also reports the fc layer (cuBLAS GEMM) separately.
#   python backbone.py ref   <ref.npz>          (run once without MPS / at 100%)
#   python backbone.py check <ref.npz>          (prints one line: max relative error of backbone and of fc)
import os, sys, copy
import numpy as np, torch, torchvision

mode, refp = sys.argv[1], sys.argv[2]
dev = torch.device("cuda")
sm = torch.cuda.get_device_properties(0).multi_processor_count
out = {}
for name in ("resnet50", "resnet152"):
    m0 = getattr(torchvision.models, name)(weights="IMAGENET1K_V1").eval()
    for dn, dt in (("fp16", torch.float16), ("bf16", torch.bfloat16)):
        for cl in (0, 1):
            m = copy.deepcopy(m0).to(dev).to(dt)
            if cl:
                m = m.to(memory_format=torch.channels_last)
            fc = m.fc
            m.fc = torch.nn.Identity()
            for bench in (0, 1):
                torch.backends.cudnn.benchmark = bool(bench)
                for bs in (1, 16, 64):
                    x = torch.from_numpy(np.random.RandomState(1234 + bs).standard_normal((bs, 3, 224, 224)).astype(np.float32)).to(dev).to(dt)
                    if cl:
                        x = x.contiguous(memory_format=torch.channels_last)
                    with torch.inference_mode():
                        f = m(x)
                        y = fc(f)
                    k = "%s|%s|cl%d|bench%d|bs%d" % (name, dn, cl, bench, bs)
                    out["feat|" + k] = f.float().cpu().numpy()
                    out["fc|" + k] = y.float().cpu().numpy()
            del m
if mode == "ref":
    np.savez(refp, **out)
    print("BACKBONE ref saved sm=%d keys=%d" % (sm, len(out)))
else:
    ref = np.load(refp)
    worst = {"feat": 0.0, "fc": 0.0}
    nbad = {"feat": 0, "fc": 0}
    for k, v in out.items():
        r = ref[k].astype(np.float64)
        rel = float(np.linalg.norm(np.nan_to_num(v.astype(np.float64) - r, nan=1e30, posinf=1e30, neginf=-1e30)) / np.linalg.norm(r))
        part = k.split("|")[0]
        worst[part] = max(worst[part], rel)
        nbad[part] += rel > (0.2 if "bf16" in k else 0.05)
        if os.environ.get("BB_VERBOSE") and rel > (0.2 if "bf16" in k else 0.05):
            print("  BAD %-44s rel=%.2e nonfinite=%d" % (k, rel, int((~np.isfinite(v)).sum())))
    print("BACKBONE sm=%d pct=%s configs=%d  backbone(cuDNN) max_rel=%.2e bad=%d   fc(cuBLAS) max_rel=%.2e bad=%d" % (
        sm, os.environ.get("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"), len(out) // 2, worst["feat"], nbad["feat"], worst["fc"], nbad["fc"]))
