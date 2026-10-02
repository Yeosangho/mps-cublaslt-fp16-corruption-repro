# Silent wrong results from FP16 / BF16 / FP8 GEMM (and cuDNN 1×1 convolutions) on Hopper under an MPS active-thread-percentage share

**With `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE` set to many values below 100, cuBLASLt `algoId=66` configurations
that use thread-block clusters of 4 or 8 CTAs leave part of the output matrix unwritten. No error is returned:
`CUBLAS_STATUS_SUCCESS`, no Xid, nothing in `dmesg`.**

- Affected: H200 (Hopper, compute capability 9.0); FP16, BF16 and FP8 (E4M3) operands; cuBLAS 12.6.4.1,
  12.8.3.14, 13.1.1.3, 13.4.1.3 and 13.8.1.7 (newest on PyPI when tested). cuBLAS 12.4.5.8 is not affected
  (it never selects `algoId=66`).
- Reached through: `cublas*gemm` / `cublasGemmEx` (PyTorch `matmul`, `Linear`), direct `cublasLtMatmul`, and
  **cuDNN**, which calls `cublasLtMatmul` for half-precision convolutions (so `Conv2d` outputs are wrong too).
- Trigger: percentages that leave the client with 18–28, 36–60 or 68–116 SMs. On a 132-SM H200 that is 66 of
  the 100 possible values (14–22, 28–46, 52–89).
- Not affected in our tests: FP32; 100 % and no MPS; **MIG instances, including the 26-SM and 60-SM profiles
  whose SM counts fail under MPS**; B200 and RTX PRO 6000 Blackwell.
- Visible effect: a PyTorch `half()` / `autocast` ResNet returns garbage logits (top-1 agreement with the
  full-GPU run drops to 0 %), Qwen3-8B produces non-finite logits. A plain `a @ b` reproduces it.
- What differs between a correct and a wrong run of the same algorithm with the same launch is one kernel
  argument: a per-cluster table that should be a permutation of the cluster indices and instead has trailing
  zeros (section 5.2).

All measurements: 2026-10-02.

---

## 1. Environment

| | Hopper (affected) | Blackwell (not affected) | Blackwell (not affected) |
|---|---|---|---|
| GPU | 1 × NVIDIA H200, 132 SMs, CC 9.0 | NVIDIA B200, 148 SMs, CC 10.0 (1 of 8 used) | NVIDIA RTX PRO 6000 Blackwell Server Edition, 188 SMs, CC 12.0 (1 of 4 used) |
| Driver | 580.178.04 (open kernel module) | 580.126.20 (open kernel module) | 610.57.04 |
| OS / kernel | Ubuntu 26.04.1, 7.0.0-31-generic | Ubuntu 24.04.4, 6.8.0-106-generic | Ubuntu 22.04.5, 5.15.0-191-generic |
| Container runtime | Docker 29.8.1, NVIDIA Container Toolkit 1.20.0 | Docker 28.5.2 | Docker 28.5.2 |
| MPS | `nvidia-cuda-mps-control -d` on the host, default settings, compute mode Default | same, restricted to one GPU | same, restricted to one GPU |

Containers are the stock Docker Hub `pytorch/pytorch` images; nothing is installed into them unless stated.

| Image | torch | cuBLAS | cuDNN |
|---|---|---|---|
| `pytorch/pytorch:2.14.1-cuda13.0-cudnn9-runtime` (main test image) | 2.14.1+cu130 | 13.1.1.3 | 9.24.0.43 |
| `pytorch/pytorch:2.14.1-cuda13.2-cudnn9-runtime` | 2.14.1+cu132 | 13.4.1.3 | 9.24.0.43 |
| same + `pip install nvidia-cublas==13.8.1.7` | 2.14.1+cu132 | 13.8.1.7 | 9.24.0.43 |
| `pytorch/pytorch:2.14.1-cuda12.6-cudnn9-runtime` | 2.14.1+cu126 | 12.6.4.1 | 9.10.2.21 |
| `pytorch/pytorch:2.9.1-cuda13.0-cudnn9-devel` | 2.9.1+cu130 | 13.0.0.19 | 9.13.0.50 |
| `pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime` | 2.7.1+cu128 | 12.8.3.14 | 9.7.1.26 |
| `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime` | 2.6.0+cu124 | 12.4.5.8 | 9.1.0.70 |

The C programs (`lt_test`, `lt_np`, `cluster_probe`) run on the H200 host against the CUDA 13.2 toolkit
(cuBLAS 13.4.1.3).

---

## 2. Minimal reproduction (PyTorch, one file)

`repro_min.py` multiplies a 16×2048 matrix by a 2048×1000 matrix (the shape of ResNet-50's last layer at
batch 16) in FP32, FP16 and BF16 and compares the GPU result with a CPU float64 product of the same
(already quantized) inputs.

```bash
# host: start MPS (default pipe directory /tmp/nvidia-mps)
nvidia-cuda-mps-control -d

# good: 12 % -> 14 SMs
docker run --rm --gpus all --ipc=host -v /tmp/nvidia-mps:/tmp/nvidia-mps -v $PWD:/w \
  -e CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=12 \
  pytorch/pytorch:2.14.1-cuda13.0-cudnn9-runtime python /w/repro_min.py

# bad: 14 % -> 18 SMs
docker run --rm --gpus all --ipc=host -v /tmp/nvidia-mps:/tmp/nvidia-mps -v $PWD:/w \
  -e CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=14 \
  pytorch/pytorch:2.14.1-cuda13.0-cudnn9-runtime python /w/repro_min.py
```

Output (H200, driver 580.178.04, cuBLAS 13.1.1.3):

```
SM=14 pct=12 torch.float32  MxKxN=16x2048x1000 bad=0/16000 bad_cols=- nonfinite=0 zeros=0 max_abs_err=1.39e-06
SM=14 pct=12 torch.float16  MxKxN=16x2048x1000 bad=0/16000 bad_cols=- nonfinite=0 zeros=0 max_abs_err=0.00103
SM=14 pct=12 torch.bfloat16 MxKxN=16x2048x1000 bad=0/16000 bad_cols=- nonfinite=0 zeros=0 max_abs_err=0.00874

SM=18 pct=14 torch.float32  MxKxN=16x2048x1000 bad=0/16000 bad_cols=- nonfinite=0 zeros=0 max_abs_err=1.39e-06
SM=18 pct=14 torch.float16  MxKxN=16x2048x1000 bad=11597/16000 bad_cols=[256..999] nonfinite=97 zeros=4113 max_abs_err=1e+30
SM=18 pct=14 torch.bfloat16 MxKxN=16x2048x1000 bad=11569/16000 bad_cols=[256..999] nonfinite=15 zeros=4113 max_abs_err=3.3e+38
```

- Expected: `bad=0` on every line.
- Actual at 14 %: only output columns 0..255 are correct; columns 256..999 hold whatever was in the buffer
  (zeros, stale values, NaN). The result is identical on every repetition and with a single MPS client.
- Nothing raises, `torch.cuda.synchronize()` succeeds, `dmesg` shows no Xid.
- The same happens with no container at all (C program below), so Docker is not part of the problem.

---

## 3. Reproduction without PyTorch (C, `cublasLtMatmul`)

`lt_test.c` calls `cublasLtMatmulAlgoGetHeuristic` + `cublasLtMatmul` for FP16
`C[1000×16] = A[1000×2048] · B[2048×16]` (the call PyTorch ends up making), pre-fills `C` with a NaN bit
pattern (`0x7C7C`) so that unwritten elements can be told apart from wrong ones, and compares with a CPU
reference. Arguments after `--` are values for `CUBLASLT_MATMUL_DESC_SM_COUNT_TARGET` (`0` = leave default).

```bash
gcc -O2 -o lt_test lt_test.c -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcublasLt -lcublas -lcudart -lm
export LD_LIBRARY_PATH=/usr/local/cuda/lib64          # CUDA 13.2 toolkit, cuBLAS 13.4.1.3

CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=12 ./lt_test -- 0
CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=14 ./lt_test -- 0
```

```
device=NVIDIA H200 physical/visible SMs=14  MPS pct=12  m=1000 n=16 k=2048  cublasLt=130401
target=0   #0  algo[id=66 tile=448 stages=35 custom=3 cluster=6] status=0 sync=0  ok      bad=0/16000 unwritten=0 bad_rows=[-1..-1]

device=NVIDIA H200 physical/visible SMs=18  MPS pct=14  m=1000 n=16 k=2048  cublasLt=130401
target=0   #0  algo[id=66 tile=312 stages=35 custom=1 cluster=4] status=0 sync=0  CORRUPT bad=11904/16000 unwritten=11904 bad_rows=[256..999]
```

`unwritten == bad`: every wrong element still holds the sentinel, i.e. the kernel never wrote rows 256..999.
256 rows is exactly one cluster's worth of tiles (`tile=64x16`, cluster shape `4x1x1` → 4 × 64 rows).
`status=0` is `CUBLAS_STATUS_SUCCESS`, `sync=0` is `cudaDeviceSynchronize() == cudaSuccess`.
`cluster=` is the `cublasLtClusterShape_t` value (4 = `4x1x1`, 6 = `2x2x1`).

A mainstream shape, same program: `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=40 ./lt_test 2048 2048 2048 -- 0`
→ `algo[id=66 tile=29 stages=35 custom=1 cluster=4] ... CORRUPT bad=491520/4194304 unwritten=491520`.

---

## 4. Scope

### 4.1 Which MPS shares (H200, torch 2.14.1 / cuBLAS 13.1.1.3)

Every `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE` from 1 to 100 was tested with three harnesses:

| Harness | What it runs per share | Shares with wrong results |
|---|---|---|
| `scan.py` | 112 GEMM shapes × {FP32, FP16, BF16}, CPU float64 reference, cuBLASLt algorithm recorded per call | 60: 14–22, 28, 31–46, 55–65, 67–89 |
| `backbone.py` | ResNet-50 / ResNet-152 without the last layer, FP16 / BF16, NCHW / channels-last, `cudnn.benchmark` off / on, batch 1 / 16 / 64 (48 configs), reference = same config at 100 % | 66: 14–22, 28–46, 52–89 |
| `xscan.py` | 1,990 cases: 314 GEMM shapes × {`matmul`, `linear` with bias} × {FP16, BF16}, 7 batched products, 180 convolutions (1×1 and 3×3) × {NCHW, channels-last} × {FP16, BF16}; reference = the same operation in FP32 | 66: 14–22, 28–46, 52–89 |

- GEMM scan: 2,906 of 22,400 half-precision products (13.0 %) were wrong and 0 of 11,200 FP32 products.
  86 of the 112 shapes were wrong at one share or another.
- Extended scan, wrong / total over all 100 shares: `matmul` 7,678 / 62,800, `linear` 9,500 / 62,800,
  `bmm` 286 / 1,400, 1×1 convolution NCHW 3,326 / 28,800, 1×1 convolution channels-last 10,813 / 28,800,
  3×3 convolution 0 / 14,400.
- 100 % is always correct, and so is the full GPU without MPS.

MPS maps percentages to SM counts in steps of 2 (1–7 % → 8, 8–9 % → 10, 10 % → 12, 11–12 % → 14, 13 % → 16,
14–15 % → 18, … 100 % → 132). Expressed in SMs, wrong results occurred at 18–28, 36–60 and 68–116, and never at
8–16, 30–34, 62–66 or 118–132. The clean bands sit around 1/8, 1/4, 1/2 and 1/1 of the GPU's 132 SMs
(the SM counts of the 1g, 2g, 4g and 7g MIG profiles are 16, 32, 64 and 132).

| Requested share | SMs | Wrong results | Largest clean share ≤ requested | SMs | SMs given up |
|---|---|---|---|---|---|
| 1–13 % | 8–16 | no | as requested | 8–16 | 0 |
| 14–22 % | 18–28 | **yes** | 13 % | 16 | 2–12 (11–43 %) |
| 23–27 % | 30–34 | no | as requested | 30–34 | 0 |
| 28–46 % | 36–60 | **yes** | 27 % | 34 | 2–26 (6–43 %) |
| 47–51 % | 62–66 | no | as requested | 62–66 | 0 |
| 52–89 % | 68–116 | **yes** | 51 % | 66 | 2–50 (3–43 %) |
| 90–100 % | 118–132 | no | as requested | 118–132 | 0 |

"Clean" means no wrong result in any of the three harnesses; it is an empirical statement about these
workloads on this GPU and library version, not a guarantee.

### 4.2 Which algorithms

All FP16 and BF16 products in the scan were run with `algoId=66`; FP32 used `algoId` 10, 11, 13, 14, 20 or 55
and was never wrong. Broken down by the cluster shape cuBLASLt selected (FP16 + BF16, all 100 shares):

| `clusterShape` | CTAs per cluster | correct | wrong |
|---|---|---|---|
| `1x1x1` | 1 | 1,908 | 0 |
| `2x1x1` | 2 | 1,064 | 0 |
| `1x2x1` | 2 | 1,482 | 0 |
| `4x1x1` | 4 | 5,617 | 884 |
| `1x4x1` | 4 | 3,080 | 928 |
| `2x2x1` | 4 | 831 | 156 |
| `2x4x1` | 8 | 3,113 | 584 |
| `4x2x1` | 8 | 2,399 | 354 |

- The selection changes with the SM count, so different shares fail with different configurations: 74 of the
  181 distinct (tile, stages, customOption, clusterShape) configurations used were wrong at one SM count or
  more, and 62 of those 74 were correct at some other SM count.
- Only shapes with 4 or 8 CTAs per cluster ever failed. Shapes with 1 or 2 CTAs never did (4,454 runs).

### 4.3 Library versions (H200, driver 580.178.04, same 100-share GEMM scan)

| cuBLAS | torch image | FP16 algorithm ids used | shares with wrong results | wrong FP16+BF16 products | other |
|---|---|---|---|---|---|
| 12.4.5.8 | 2.6.0+cu124 | 13, 14, 21, 35, 39 | **0** | 0 / 22,400 | – |
| 12.6.4.1 | 2.14.1+cu126 | 66 | 66 (14–22, 28–46, 52–89) | 2,934 / 19,712 (14.9 %) | shares 1–12 (8–14 SMs) crash, see 6 |
| 12.8.3.14 | 2.7.1+cu128 | 66 | 64 (14–22, 28–46, 54–89) | 3,387 / 19,712 (17.2 %) | shares 1–12 (8–14 SMs) die with SIGFPE, see 6 |
| 13.1.1.3 | 2.14.1+cu130 | 66 | 60 (14–22, 28, 31–46, 55–65, 67–89) | 2,906 / 22,400 (13.0 %) | – |
| 13.4.1.3 | 2.14.1+cu132 | 66 | 60 (same shares) | 2,826 / 22,400 (12.6 %) | – |
| 13.8.1.7 | 2.14.1+cu132 + pip | 66 | 60 (same shares) | 2,822 / 22,400 (12.6 %) | – |

The CUDA release notes describe two other `algoId=66` defects (split-K, large leading dimensions) and date
both to cuBLAS 12.6.3 (CUDA 12.6 Update 2), which fits the boundary above. We found no release-note entry
that mentions MPS, active thread percentage or SM count together with wrong results.

### 4.4 Other GPUs

Same harnesses, MPS shares 1–100 %:

| GPU | cuBLAS | GEMM scan | ResNet backbone sweep | Extended scan |
|---|---|---|---|---|
| B200 (CC 10.0, 148 SMs), driver 580.126.20 | 13.1.1.3 (same image as the H200) | 0 / 22,400 wrong | 0 of 100 shares wrong | 0 / 199,000 wrong |
| B200 | 13.0.0.19 | 0 / 22,400 wrong | – | – |
| B200 | 12.8.3.14 | 0 / 20,160 wrong; shares 1–10 die with SIGFPE | – | – |
| RTX PRO 6000 Blackwell Server Edition (CC 12.0, 188 SMs), driver 610.57.04 | 13.1.1.3 (same image) | 0 / 22,400 wrong | 0 of 100 shares wrong | 0 / 199,000 wrong |
| RTX PRO 6000 | 13.0.0.19 | 0 / 22,400 wrong | 0 of 100 shares wrong | 0 / 199,000 wrong |

- On the B200 cuBLASLt also selects `algoId=66` with the same cluster shapes (`4x1x1`, `1x4x1`, `2x2x1`, `2x4x1`,
  `4x2x1` in 14,986 of the 22,400 half-precision products of the cuBLAS 13.0.0.19 scan) over the same range of
  SM counts (8–148), and every result is correct.
- On the RTX PRO 6000 the half-precision products do not use cluster kernels at all (`algoId` 11, 13, 21, 23,
  no cluster shape).

H200 MIG instances (no MPS) are correct for all six profiles, see 5.5.

Not tested: H100, and MPS static SM partitioning (`nvidia-cuda-mps-control -S`, not available in driver
580.178.04). Static partitions are multiples of 8 SMs on Hopper; sizes such as 24, 40, 48 or 56 SMs fall into
the SM ranges that fail under percentage shares, but whether they fail depends on the cluster occupancy the
driver reports for a static partition (section 5.2), which we could not check.

### 4.5 Other precisions (FP8, FP4)

`lt_np.c` calls `cublasLtMatmul` directly with narrow-precision operands (8 shapes, every share from 1 to 100 %)
and compares with a CPU reference computed from the decoded operands. `A` is stored k×m with
`CUBLASLT_MATMUL_DESC_TRANSA = CUBLAS_OP_T`; FP4 uses `CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3` with all
scale factors equal to 1.

| GPU | Operands → output | Algorithm ids | Products run | Wrong | Shares with wrong results |
|---|---|---|---|---|---|
| H200 (cuBLAS 13.4.1.3) | FP8 E4M3 → FP16 | 66 | 800 | **202** | 14–22, 28, 31–46, 55–65, 67–86, 88–89 |
| H200 | FP8 E4M3 → BF16 | 66 | 800 | **202** | same |
| H200 | FP8 E4M3 → FP32 | 66 | 800 | **200** | same except 57 |
| H200 | FP16 → FP16 (same program, for comparison) | 66 | 800 | **179** | 14–22, 28, 31–46, 55–65, 67–86, 88–89 |
| H200 | FP4 E2M1 (NVFP4), FP8 E5M2 | – | – | – | not supported by the library on this GPU |
| B200 (cuBLAS 13.0.0.19) | FP8 E4M3 → FP16 / BF16 | 66 | 800 each | 0 | – |
| B200 | FP4 E2M1 (NVFP4) → BF16 / FP32 | 70, 71 | 142 / 152 of 800 | 0 | – (but see below: rejected at most shares) |
| RTX PRO 6000 (cuBLAS 13.0.0.19) | FP8 E4M3 → FP16 / BF16 | 35 | 800 each | 0 | – |
| RTX PRO 6000 | FP4 E2M1 (NVFP4) → BF16 / FP32 | 70 | 83 / 88 of 800 | 0 | – (rejected at most shares) |

- On Hopper FP8 goes through the same `algoId=66` cluster kernels and fails at the same shares. As with FP16,
  only cluster shapes with 4 or 8 CTAs were wrong (`1x4x1` 56/188, `4x1x1` 51/113, `2x2x1` 11/22, `2x4x1` 35/204,
  `4x2x1` 49/157 for FP8 → FP16), never `2x1x1` (0/59) or `1x2x1` (0/57).
- On Blackwell FP8 was correct at every share, including with 4- and 8-CTA cluster shapes on the B200.
- On Blackwell NVFP4 never produced a wrong result, but under most MPS shares it does not run at all:
  `cublasLtMatmulAlgoGetHeuristic` returns `CUBLAS_STATUS_NOT_SUPPORTED` (15) for every one of the 8 shapes.
  All 8 shapes ran only at 1–8, 15–17, 91–93 and 96–100 % on the B200 (8–10, 22–24, 134–136, 142–148 SMs) and
  at 1–8, 12, 96 and 100 % on the RTX PRO 6000; no shape ran at any other share. On the full GPU without MPS
  the 4 shapes we probed were accepted. The same shapes are accepted for FP8 and FP16 at every share, so the
  rejection is specific to the FP4 path. We tested cuBLAS 13.0.0.19 only for FP4, and only one operand layout
  (`A` transposed, unit scale factors); `CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0` was rejected with
  `CUBLAS_STATUS_INVALID_VALUE` in our setup and is not covered.

### 4.6 End-to-end effect on ResNet inference

`mpscorrupt.py` runs torchvision ResNet-50 and ResNet-152 (ImageNet weights, fixed CPU-generated input) in
FP32 / FP16 / BF16, with `model.half()`-style casting and with `torch.autocast`, NCHW and channels-last,
`cudnn.benchmark` off and on, batch 1 / 16 / 64, and compares the logits with the same configuration run on
the full GPU without MPS (120 model configurations + 279 GEMM checks per share).

| Share | SMs | cuBLAS 13.1.1.3 | cuBLAS 12.8.3.14 |
|---|---|---|---|
| no MPS, 100 % | 132 | clean | clean |
| 5–9 % | 8–10 | clean | SIGFPE |
| 10 % | 12 | clean | SIGFPE |
| 11–13 % | 14–16 | clean | clean |
| 14–20 %, 22 % | 18–28 | **corrupt** (FP16/BF16 logits: NaN or relative error ≈ 1, top-1 agreement 0 %) | **corrupt** |
| 25 % | 32 | clean | clean |
| 30 % | 38 | **corrupt** | **corrupt** |
| 50 % | 66 | clean | clean |

FP32 logits differ from the full-GPU run by at most 2.3e-4 (relative) at every share tested.

---

## 5. What we could establish about the cause

### 5.1 The algorithm cuBLASLt selects changes with the SM count

`CUBLASLT_LOG_LEVEL=5`, FP16 16×2048×1000 from `repro_min.py`:

```
pct 12 (14 SMs): Adesc=[type=R_16F]  smCountTarget=14 algo=[algoId=66 tile=MATMUL_TILE_192x8 stages=MATMUL_STAGES_64xAUTO customOption=3 clusterShape=CLUSTER_SHAPE_2x2x1]   -> correct
pct 14 (18 SMs): Adesc=[type=R_16F]  smCountTarget=18 algo=[algoId=66 tile=MATMUL_TILE_64x16 stages=MATMUL_STAGES_64xAUTO customOption=1 clusterShape=CLUSTER_SHAPE_4x1x1]   -> corrupt
both           : Adesc=[type=R_32F]  algo=[algoId=55 tile=MATMUL_TILE_64x32 stages=MATMUL_STAGES_8x3 ... numSplitsK=8]                                                      -> correct
```

### 5.2 Same algorithm, same launch, one kernel argument differs: a per-cluster table that is not filled in

`clog.c` is an `LD_PRELOAD` logger for `cuLaunchKernelEx` and `cuOccupancyMaxActiveClusters`; with
`CLOG_PARAMS=1` it also prints the packed kernel argument buffer. For each product below, the MPS and MIG
contexts use the same algorithm, the same launch geometry and the same 12 argument words in front of a byte
table (argument words 12–13, one byte per cluster); only that table differs between the correct and the
corrupt runs (the full-GPU row has different leading words because of the SM count target):

| Product (FP16) | Context | SMs | Launch | Table | Result |
|---|---|---|---|---|---|
| 1000×16×2048, tile 312, `4x1x1` | full GPU, `SM_COUNT_TARGET=18` | 132 | grid 4x4x1 = 4 clusters | `0 1 2 3` | correct |
| | MPS 20 % | 26 | same | `0 1 2 3` | correct |
| | MIG 1g.35gb | 26 | same | `0 2 1 3` | correct |
| | **MPS 14 %** | 18 | same | `0 0 0 0` | **corrupt**, rows 256..999 unwritten (3 of 4 clusters' share) |
| 2048×64×512, tile 27, `4x1x1` | MIG 1g.35gb | 26 | grid 4x6x1 = 6 clusters | `0 2 1 3 4 5` | correct |
| | **MPS 20 %** | 26 | same | `0 1 2 3 4 0` | **corrupt**, 1/16 of the output unwritten |
| 256×1024×1024, tile 318, `4x2x1` | MIG 3g.71gb | 60 | grid 8x7x1 = 7 clusters | `0 1 3 5 2 4 6` | correct |
| | **MPS 46 %** | 60 | same | `0 1 2 3 4 5 0` | **corrupt**, 1/16 of the output unwritten |

In every correct case the table is a permutation of 0 … (clusters − 1). In every corrupt case trailing entries
are 0, so more than one cluster is given entry 0 and part of the output is never produced — which matches the
"unwritten" pattern exactly. The raw dumps are in `results/h200_kernel_argument_dump.txt`.

cuBLASLt computes this table on the host. The inputs it has are the device SM count and the answers of
`cuOccupancyMaxActiveClusters`, which it queries for cluster sizes 3 … 32 with a probe kernel:

```
cluster size:      3  4  5  6  7  8  9 10 11 12 13 14 15 16
MIG 1g.35gb (26):  7  6  4  3  3  3  1  1  1  1  1  1  1  1
MPS 20 %   (26):   7  6  4  3  3  3  2  2  1  1  1  1  1  1
MIG 3g.71gb (60): 17 14 10  7  7  7  3  3  3  3  3  3  3  3
MPS 46 %   (60):  18 15 10  7  7  7  4  4  4  4  3  3  3  3
```

Under MPS the answers for the same SM count differ from those of a MIG instance (and for an 18-SM share the
driver reports one cluster for every size from 9 up to 18).

### 5.3 Swapping the inputs

For diagnosis only, `smspoof.c` (an `LD_PRELOAD` shim) overrides the value returned by
`cuDeviceGetAttribute(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)` and/or `cuOccupancyMaxActiveClusters`
(`lt_test`, FP16 1000×16×2048):

| Real context | Reported SM count | Reported cluster occupancy | Result |
|---|---|---|---|
| full GPU (132 SMs), **no MPS** | **18** | real (full GPU) | **corrupt**, same pattern as MPS 14 % |
| full GPU, no MPS | 18 | as under MPS 14 % | corrupt |
| full GPU, no MPS | real (132) | as under MPS 14 % | correct |
| MPS 14 % (18 SMs) | **14** | real | **correct** |
| MPS 14 % | **132** | real | **correct** |
| MPS 14 % | real (18) | as on the full GPU | corrupt |

- MPS is not needed to reproduce the problem: an un-partitioned GPU gives the same wrong output once the
  reported SM count is 18. So the wrong table is a function of the values cuBLASLt is given, not of how MPS
  executes the kernel.
- The SM count alone does not decide it either: MIG instances with 26 and 60 SMs are correct (section 5.5)
  while MPS shares with 26 and 60 SMs are not.
- `CUBLASLT_MATMUL_DESC_SM_COUNT_TARGET` does not help: on the full GPU a target of 18 selects the same
  configuration and is correct; under MPS 14 % every target we tried (8 … 132) is corrupt
  (`lt_test -- 0 8 10 … 132`).
- Repeating the full-GPU experiment for every reported SM count from 1 to 132 (`spoof_sweep.sh`, 4 shapes
  only) gives corruption at 18–28, 36, 44–57, 60–61, 76–85, 88–102, 104–113, 115.

### 5.4 The driver executes such clustered launches correctly under the same MPS share

`cluster_probe.cu` launches a trivial cluster kernel with the same geometry (grid 4x4, cluster 4x1,
384 threads, 180692 bytes of dynamic shared memory, plus a few others), has every CTA record that it ran and
read the other CTAs' shared memory through `cluster_group::map_shared_rank`, and repeats 20 times.
Under MPS 12 / 14 / 16 / 20 / 40 % and on the full GPU every CTA ran and every DSMEM read was correct
(`ran=16/16 dsmem_bad=0`, 0 failing repetitions in all 30 context × geometry combinations). Kernels see SM
ids 0 … N−1 both under an N-SM MPS share and inside an N-SM MIG instance.

### 5.5 MIG instances are correct, including SM counts that fail under MPS

All six H200 MIG profiles were tested without MPS (`mig_test.sh`: minimal reproduction, extended scan with
1,990 cases, ResNet backbone sweep with 48 configurations):

| Profile | SMs | Result | Same SM count under MPS |
|---|---|---|---|
| 1g.18gb | 16 | correct | 13 %: correct |
| 1g.35gb | 26 | correct | 20–21 %: **wrong** |
| 2g.35gb | 32 | correct | 25 %: correct |
| 3g.71gb | 60 | correct | 46 %: **wrong** |
| 4g.71gb | 64 | correct | 49 %: correct |
| 7g.141gb | 132 | correct | 100 %: correct |

Inside the MIG instances cuBLASLt uses the same 4- and 8-CTA cluster configurations (1,292 of 2,698
`cublasLtMatmul` calls of the extended scan in the 60-SM instance) and they are correct.

### 5.6 Changing only the cluster shape of the selected algorithm makes the result correct

With `LT_CLUSTER=<id>` `lt_test` takes the algorithm returned by the heuristic, replaces only
`CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID` through `cublasLtMatmulAlgoConfigSetAttribute`, and runs it
(`cublasLtMatmulAlgoCheck` accepts the modified algorithm):

| Share | Selected algorithm | cluster shape as selected | forced to `1x1x1` / `2x1x1` / `1x2x1` |
|---|---|---|---|
| 14 % (18 SMs) | id 66, tile 312, custom 1 | `4x1x1`: corrupt | correct / correct / correct |
| 40 % (52 SMs) | id 66, tile 10, custom 3 | `4x2x1`: corrupt | correct / correct / correct |

Taken together: the kernels themselves and the driver's cluster scheduling work; what goes wrong is the
per-cluster table that cuBLASLt derives for `algoId=66` configurations with 4 or 8 CTAs per cluster when the
SM count and cluster occupancy come from an MPS active-thread-percentage share.

### 5.7 cuDNN convolutions are wrong for the same reason

cuDNN 9 does not link against cuBLASLt; `libcudnn_engines_precompiled.so.9` opens `libcublasLt.so.13` at run
time and calls `cublasLtMatmulAlgoGetHeuristic` / `cublasLtMatmul` for half-precision convolutions.
The cuBLASLt log of one `backbone.py` run at 14 % contains 3,144 `cublasLtMatmul` calls with convolution
dimensions, e.g.

```
R_16F 256x1024x196  algoId=66 tile=64x200 stages=64xAUTO customOption=1 clusterShape=4x1x1     (1x1 conv 1024->256 at 14x14, batch 1)
```

`convdiag.py` recomputes every `Conv2d` of ResNet-50 in FP32 from the same half-precision input and weights:
at 40 % the first wrong layer is `layer2.0.conv3` (1×1, 128→512, 28×28) in NCHW and `layer1.0.conv1` (1×1,
64→64, 56×56) in channels-last. With the cluster shape of those `cublasLtMatmul` calls limited to 2 CTAs
(section 7) all 53 convolutions are correct, in NCHW and channels-last, at batch 1, 16 and 64 — so the wrong
convolution outputs come from the same cuBLASLt path, not from cuDNN's own kernels. In the extended scan only
1×1 convolutions were wrong (14,139 of 57,600), never 3×3 ones (0 of 14,400).

---

## 6. Related: SIGFPE with cuBLAS 12.6.4.1 / 12.8.3.14 at small shares

With cuBLAS 12.6.4.1 (`pytorch/pytorch:2.14.1-cuda12.6-cudnn9-runtime`) and 12.8.3.14
(`pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime`) the FP16 product in `repro_min.py` kills the process with
SIGFPE (exit code 136) at 5 % (8 SMs). With 12.8.3.14 this happens on both the H200 and the B200; in the
100-share scan shares 1–12 (H200) and 1–10 (B200) did not complete. It does not happen with cuBLAS 13.0.0.19 or
later, nor with 12.4.5.8. We mention it because the release notes list a fix for `cublasLtMatmul` /
`cublasLtMatmulAlgoGetHeuristic` FPEs on Hopper in cuBLAS 12.4.5.8, and this looks like the same class of
problem in later 12.x builds.

---

## 7. Workarounds

### 7.1 Keeping cuBLASLt from running the failing kernels (`ltguard.c`)

`ltguard.so` is an `LD_PRELOAD` library that works purely at the cuBLASLt API level. It does not intercept any
CUDA driver function and does not change what the driver reports (SM count, cluster occupancy).

- It wraps the cuBLASLt execution entry points: `cublasLtMatmul` (public API, used by cuDNN) and the typed
  `cublasLt{SSS,HSH,HHH,HSS,BSS,BII,TSS,TST,ACC,CCC,DDD,ZZZ}Matmul` that libcublas calls for `cublas*gemm` /
  `cublasGemmEx`. Because cuDNN resolves cuBLASLt with `dlsym()`, the library also wraps `dlsym` so that those
  lookups receive the same wrappers.
- For every call it reads the cluster shape of the algorithm about to run
  (`cublasLtMatmulAlgoConfigGetAttribute`, `CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID`). Shapes with at most
  2 CTAs pass through untouched.
- A shape with more CTAs is replaced by `2x1x1` / `1x2x1` (or `1x1x1`) on a copy of the algorithm
  (`cublasLtMatmulAlgoConfigSetAttribute`); algorithm id, tile, stages and custom option stay as selected.
  The replacement is used only after `cublasLtMatmulAlgoCheck` accepted it. Accepted decisions are cached
  per algorithm; if the library ever rejects a cached replacement the call is redone through the checked
  path.

```bash
gcc -O2 -shared -fPIC -I/usr/local/cuda/include -o ltguard.so ltguard.c -ldl
docker run ... -v $PWD:/w -e LD_PRELOAD=/w/ltguard.so -e LTGUARD_LOG=1 ...
#   [LTGUARD] pid=1 matmul calls=3144 cluster shape rewritten=248 (cache hits 184) left unsafe=0
```

**Where the guard is needed: Hopper under an MPS percentage share, and nowhere else we tested.**

| Configuration | Wrong results without the guard | Enable the guard? |
|---|---|---|
| Hopper (H200, CC 9.0), `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE` < 100, cuBLAS ≥ 12.6.4.1, FP16 / BF16 / FP8 | yes (66 of 100 shares) | **yes** |
| Hopper, 100 % or no MPS | no | not needed |
| Hopper, MIG instances (all six H200 profiles) | no | not needed |
| Hopper, cuBLAS 12.4.5.8 | no (`algoId=66` is never selected) | not needed |
| Hopper, FP32 | no | not needed |
| Blackwell: B200 (CC 10.0), RTX PRO 6000 (CC 12.0) | no | **no** |

- On Blackwell there is nothing to guard against, and the guard has **not been run there at all**: every B200 and
  RTX PRO 6000 result in this repository is unguarded. Enabling it would rewrite cluster shapes that are
  correct (4- and 8-CTA shapes on the B200, including the FP4 algorithms 70 / 71), with unverified results and
  a throughput cost.
- `ltguard.c` does not check the GPU architecture. Whoever injects it has to decide per node, e.g. set
  `LD_PRELOAD` only on Hopper nodes or set `LTGUARD_DISABLE=1` elsewhere.
- H100 was not tested. It has the same architecture and SM count as the H200, so the same behaviour is
  expected but not verified.

Correctness, H200, cuBLAS 13.1.1.3, every share from 1 to 100 %:

| Harness | without the guard | with the guard |
|---|---|---|
| GEMM scan, 22,400 half-precision products, CPU reference | 2,906 wrong (60 shares) | 0 wrong |
| Extended scan, 199,000 GEMM / convolution cases | 31,603 wrong (66 shares) | 0 wrong |
| ResNet-50/152 backbone sweep, 48 configurations per share | 66 shares wrong | 0 shares wrong |
| Qwen3-8B, HF transformers `generate`, BF16 and FP16, batch 1 / 8 / 32, 27 share × dtype × batch configurations (`llm_bench.py`) | 11 wrong: 40 % at every batch size and 70 % at batch 1 and 8 (logits non-finite); 14 % at every batch size (prefill logits correct, generated tokens match the reference in only 2 % of positions) | 0 wrong |

In all of these runs `left unsafe` was 0, i.e. every call that carried a larger cluster shape was rewritten.

Cost. Guard-on throughput relative to guard-off, each configuration alone on the GPU. Only cases whose
guard-off result is correct are compared (a kernel that skips most of its output is not a baseline).

GEMM by size and vision models (`bench2.py`; H200, cuBLAS 13.1.1.3; min–max over the cases of a row and up to
three repetitions; "(n excl.)" = n pairs dropped because the guard-off result was wrong):

| Guard-on speed, % of guard-off | 100 % | 70 % | 50 % | 40 % | 25 % | 14 % | 12 % |
|---|---|---|---|---|---|---|---|
| square GEMM N×N×N, N = 128 … 8192 | 98–101 | 99–100 (2 excl.) | 98–109 | – (22 excl.) | 98–103 | 100–102 | 100–102 |
| GEMM m×4096×4096, m = 256, 1024 | 93–101 | – (2 excl.) | 99–101 | – (4 excl.) | 99–100 | 100 | 100 |
| GEMM m×4096×4096, m = 1, 16, 64 | 95–101 | 77–84 | 93–107 | 69–74 (2 excl.) | 76–84 | – (3 excl.) | 100 |
| GEMM 16×2048×1000, 64×2048×1000, 1024×4096×128, 32×4096×32000 | 96–101 | 90–98 | 98–115 | – (8 excl.) | 85–98 | 100 (2 excl.) | 82–100 |
| ResNet-50 FP16 inference, batch 16 / 64 | 95–102 | 100 (2 excl.) | 99–103 | – (8 excl.) | 98–101 | 100 (2 excl.) | 100 |
| ResNet-50 FP16 inference, batch 1 | 96–100 | – (2 excl.) | 90–94 | – (4 excl.) | 97–102 | – (2 excl.) | 99–100 |
| ViT-B/16 FP16 inference, batch 16 / 64 | 99–100 | – (2 excl.) | 100–101 | – (4 excl.) | 100–101 | 100 (1 excl.) | 100 |
| ViT-B/16 FP16 inference, batch 1 | 98–99 | 95 | 87 | – (2 excl.) | 99–101 | – (1 excl.) | 96–99 |

- Large and square products are unaffected. The cost is concentrated in "skinny" products (few rows against a
  4096×4096 operand), up to about 30 % slower at 25 %, 40 % and 70 %.
- End-to-end vision inference at batch 16 or more is within ±5 %; batch 1 ranges from 87 to 102 %.
- With the guard, every output in this benchmark was correct at every share.

LLM end to end (`llm_bench.py`: Qwen3-8B, 512-token prompts, 64 greedy tokens; "prefill" = one forward pass over
batch × 512 tokens, "generate" = new tokens per second of `generate`). Guard-on / guard-off, batch 1 / 8 / 32;
"–" where the guard-off output is wrong:

| dtype | share | SMs | prefill | generate | GEMM calls rewritten |
|---|---|---|---|---|---|
| BF16 | 100 % | 132 | 99 / 99 / 100 % | 97 / 97 / 100 % | 44–62 % |
| BF16 | 70 % | 92 | – / – / 99 % | – / – / 99 % | 100 % |
| BF16 | 50 % | 66 | 100 / 100 / 100 % | 99 / 100 / 100 % | 97–100 % |
| BF16 | 40 % | 52 | – | – | 100 % |
| BF16 | 25 % | 32 | 100 / 100 / 100 % | 99 / 99 / 99 % | 100 % |
| BF16 | 14 % | 18 | 100 / 100 / 100 % | – | 64–69 % |
| FP16 | 100 % | 132 | 97 / 100 / 100 % | 97 / 93 / 99 % | 54–87 % |
| FP16 | 40 % | 52 | – | – | 100 % |
| FP16 | 25 % | 32 | 100 / 100 / 100 % | 98 / 98 / 100 % | 100 % |

Absolute numbers (BF16, 100 %, guard off): prefill 19.6k / 30.6k / 31.2k tokens/s, generate 39 / 297 / 965
tokens/s. With the guard the logits of 23 of the 27 configurations are bit-identical to the 100 % reference run,
the other 4 differ by 0.7–8.7 % in relative norm with identical top-1 and identical generated tokens (two
guard-off runs of the same configuration can differ as much: FP16, 100 %, batch 8 is 0.67 % off its own
reference).
HF `generate` spends a large part of each step outside GPU kernels, so a serving stack that is more GPU-bound
may show a larger share of the GEMM-level differences in the table above; we did not measure one.

Other library versions (extended scan, 1,990 cases per share, shares 14 / 20 / 40 / 60 / 70 / 88 %, the same
`ltguard.so` binary built against the CUDA 13.2 headers):

| cuBLAS | wrong cases without the guard | with the guard |
|---|---|---|
| 12.6.4.1 | 758 / 412 / 858 / 336 / 244 / 216 | 0 at every share |
| 12.8.3.14 | 1,002 / 502 / 970 / 406 / 246 / 258 | 0 at every share |
| 13.1.1.3 | 598 / 138 / 964 / 328 / 238 / 232 | 0 at every share |
| 13.8.1.7 | 598 / 128 / 970 / 316 / 248 / 238 | 0 at every share |

FP8 (`lt_np`, E4M3 → FP16 and → BF16, 7 shapes, shares 14 / 20 / 40 / 60 / 70 / 88 %): without the guard
4 / 1 / 7 / 1 / 2 / 1 of 7 products wrong, with the guard 0 at every share.

Limits of this approach:

- It relies on the observation that cluster shapes with at most 2 CTAs never failed (section 4.2), not on a
  statement from NVIDIA; it should be re-validated for every new cuBLAS release and GPU.
- It does not address the SIGFPE of the 12.x libraries at 8–14 SMs (section 6).
- Statically linked cuBLASLt, or a framework that calls cuBLASLt through a path other than the wrapped entry
  points, is not covered.
- It wraps `dlsym`. If another preloaded library also wraps `dlsym`, only the first one in `LD_PRELOAD` sees
  the lookups; with `clog.so` listed before `ltguard.so` the guard did not take effect in our test.
- It also rewrites cluster shapes where the unguarded library is correct (100 %, MIG instances), which is
  where the costs in the tables above come from.

### 7.2 Restricting shares to clean SM counts

Without any library hook, the alternative is to give a tenant only SM counts from the clean bands in 4.1
(8–16, 30–34, 62–66, 118–132 on a 132-SM Hopper GPU), i.e. round a requested percentage down to
13 %, 27 % or 51 %. This costs up to 43 % of the requested SMs and rests on the same empirical table.
MIG instances need no restriction (section 5.5).

### 7.3 Not recommended

Overriding the SM count the driver reports (`cuDeviceGetAttribute`) to a clean value also avoids the wrong
results (section 5.3), but it changes what every other library and the application see.

---

## 8. Files

| File | Purpose |
|---|---|
| `repro_min.py` | Minimal PyTorch reproduction (section 2) |
| `lt_test.c` | Minimal C / cuBLASLt reproduction; SM-count-target sweep, candidate enumeration (`LT_ENUM=N`), cluster-shape override (`LT_CLUSTER=<id>`) |
| `lt_np.c`, `np_sweep.sh`, `nptab.py` | FP8 / FP4 (and FP16 / BF16) `cublasLtMatmul` against a CPU reference, share sweep, table |
| `mps.sh` | Start / stop the MPS control daemon |
| `scan.py`, `scan_campaign.sh`, `agg.py`, `summarize.py` | Per-share GEMM scan with algorithm capture; aggregation |
| `xscan.py`, `mksafe.py` | Extended per-share GEMM + convolution scan (GPU FP32 reference); clean-share table |
| `backbone.py` | ResNet backbone (convolutions) vs last layer, per share |
| `convdiag.py` | Per-layer check of every convolution of ResNet-50; single-convolution mode |
| `mpscorrupt.py`, `run_cell.sh`, `campaign.sh`, `cellverdict.py` | ResNet-50/152 + GEMM harness with per-cell verdicts |
| `diag.py` | Corruption pattern, shape scan and backbone/last-layer split for one share |
| `ltguard.c` | `LD_PRELOAD` guard: limits the cluster shape of cuBLASLt algorithms to 2 CTAs (section 7) |
| `clog.c`, `argdump.sh`, `ctxcmp.sh` | `LD_PRELOAD` logger: clustered launches, kernel argument buffer, `cuOccupancyMaxActiveClusters` answers; wrappers used for section 5.2 |
| `smspoof.c` | `LD_PRELOAD` diagnostic shim: override reported SM count / cluster occupancy (section 5.3 only) |
| `cluster_probe.cu` | Standalone cluster-launch + DSMEM check, prints the SM ids used |
| `mig_test.sh` | Runs the reproductions inside every MIG profile (section 5.5) |
| `spoof_sweep.sh` | Full-GPU sweep over the reported SM count (no MPS) |
| `evidence.sh` | Regenerates the C-level output quoted in sections 3 and 5 |
| `bench2.py`, `benchtab.py`, `benchsum.py` | GEMM-by-size and vision throughput with / without the guard |
| `llm_bench.py`, `llmtab.py` | Qwen3-8B prefill / generate throughput and correctness with / without the guard |
| `results/` | Aggregated output of every campaign quoted above |

Scripts assume this directory is `/root/mpscorrupt` on the host and is mounted at `/w` in the container.

```bash
gcc -O2 -shared -fPIC -I/usr/local/cuda/include -o ltguard.so ltguard.c -ldl
gcc -O2 -shared -fPIC -I/usr/local/cuda/include -o clog.so clog.c -ldl
gcc -O2 -shared -fPIC -I/usr/local/cuda/include -o smspoof.so smspoof.c -ldl
gcc -O2 -o lt_test lt_test.c -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcublasLt -lcublas -lcudart -lm
gcc -O2 -o lt_np lt_np.c -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcublasLt -lcublas -lcudart -lm
nvcc -O2 -arch=sm_90 -o cluster_probe cluster_probe.cu
./mps.sh start
J=4 ./scan_campaign.sh pytorch/pytorch:2.14.1-cuda13.0-cudnn9-runtime h200 "$(seq -s ' ' 1 100)"
python3 agg.py out/scan/h200 -a
./evidence.sh
```

---

## 9. Questions for NVIDIA

1. For `algoId=66` configurations with 4 or 8 CTAs per cluster, cuBLASLt passes the kernel a per-cluster table
   that is a permutation of the cluster indices in every correct run and has trailing zeros in every wrong
   run (section 5.2). Is this table derived from `cuOccupancyMaxActiveClusters`, and are the values that call
   returns under `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE` meant to be a valid input for it? The MPS documentation
   states that the percentage is reflected through `cudaDevAttrMultiProcessorCount` and does not restrict the
   use of thread-block clusters.
2. MIG instances with 26 and 60 SMs are correct while MPS shares with 26 and 60 SMs are not. Is an MPS
   percentage share expected to be usable by cluster kernels at all, or only MPS static SM partitions?
3. Is there a supported way to keep cuBLASLt (and therefore cuDNN) from selecting cluster shapes with more
   than 2 CTAs, other than rewriting the algorithm with `cublasLtMatmulAlgoConfigSetAttribute` as `ltguard`
   does? `CUBLASLT_MATMUL_DESC_SM_COUNT_TARGET` does not have that effect.
4. Is there a fixed cuBLAS release, or a planned one?
