// cluster_probe: launch a thread-block-cluster kernel with a given grid / cluster / block / shared memory and
// check that EVERY CTA ran and that distributed shared memory (DSMEM) reads inside each cluster are correct.
// Purpose: tell apart "driver drops/misplaces clustered CTAs under an MPS SM share" from "library logic".
// usage: cluster_probe gx gy cx cy threads smem_bytes [reps]
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

__global__ void probe(unsigned *out, int smem_words) {
  extern __shared__ unsigned sm[];
  cg::cluster_group cl = cg::this_cluster();
  unsigned b = blockIdx.x + blockIdx.y * gridDim.x;
  if (threadIdx.x == 0) { sm[0] = 1000u + b; sm[smem_words - 1] = 7000u + b; }
  cl.sync();
  if (threadIdx.x == 0) {
    unsigned smid; asm("mov.u32 %0, %%smid;" : "=r"(smid));
    unsigned sum = 0, n = cl.num_blocks();
    for (unsigned r = 0; r < n; r++) {
      unsigned *o = cl.map_shared_rank(sm, r);
      sum += o[0] + o[smem_words - 1];
    }
    out[b * 4 + 0] = 0xC0DE0000u | b;
    out[b * 4 + 1] = smid;
    out[b * 4 + 2] = cl.block_rank();
    out[b * 4 + 3] = sum;
  }
  cl.sync();
}

int main(int argc, char **argv) {
  if (argc < 7) { printf("usage: %s gx gy cx cy threads smem_bytes [reps]\n", argv[0]); return 1; }
  int gx = atoi(argv[1]), gy = atoi(argv[2]), cx = atoi(argv[3]), cy = atoi(argv[4]), th = atoi(argv[5]), smem = atoi(argv[6]);
  int reps = argc > 7 ? atoi(argv[7]) : 1;
  cudaDeviceProp p; cudaGetDeviceProperties(&p, 0);
  int nb = gx * gy;
  unsigned *d, *h = (unsigned *)malloc(nb * 16);
  cudaMalloc(&d, nb * 16);
  if (smem > 48 * 1024) cudaFuncSetAttribute(probe, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  cudaLaunchConfig_t cfg = {0};
  cfg.gridDim = dim3(gx, gy, 1); cfg.blockDim = dim3(th, 1, 1); cfg.dynamicSmemBytes = smem;
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeClusterDimension;
  at[0].val.clusterDim.x = cx; at[0].val.clusterDim.y = cy; at[0].val.clusterDim.z = 1;
  cfg.attrs = at; cfg.numAttrs = 1;
  int occ = -1; cudaError_t oe = cudaOccupancyMaxActiveClusters(&occ, probe, &cfg);
  int bad_total = 0;
  for (int rep = 0; rep < reps; rep++) {
    cudaMemset(d, 0xEE, nb * 16);
    cudaError_t le = cudaLaunchKernelEx(&cfg, probe, d, smem / 4);
    cudaError_t se = cudaDeviceSynchronize();
    cudaMemcpy(h, d, nb * 16, cudaMemcpyDeviceToHost);
    int ran = 0, dsm_bad = 0, sms[512]; memset(sms, 0, sizeof sms);
    for (int b = 0; b < nb; b++) {
      if (h[b * 4] != (0xC0DE0000u | b)) continue;
      ran++; sms[h[b * 4 + 1] & 511] = 1;
      // expected DSMEM sum over the blocks of this cluster
      int bx = b % gx, by = b / gx, x0 = bx / cx * cx, y0 = by / cy * cy; unsigned exp = 0;
      for (int y = y0; y < y0 + cy; y++) for (int x = x0; x < x0 + cx; x++) exp += 1000u + (x + y * gx) + 7000u + (x + y * gx);
      if (h[b * 4 + 3] != exp) dsm_bad++;
    }
    int nsm = 0, smin = 511, smax = 0;
    for (int i = 0; i < 512; i++) if (sms[i]) { nsm++; if (i < smin) smin = i; if (i > smax) smax = i; }
    if (rep == 0 || ran != nb || dsm_bad)
      printf("SMs=%d pct=%s grid=%dx%d cluster=%dx%d threads=%d smem=%d occ=%d(rc=%d) launch=%d sync=%d  ran=%d/%d dsmem_bad=%d distinct_smid=%d smid_range=[%d..%d] %s\n",
             p.multiProcessorCount, getenv("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE") ? getenv("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE") : "-",
             gx, gy, cx, cy, th, smem, occ, (int)oe, (int)le, (int)se, ran, nb, dsm_bad, nsm, smin, smax, (ran == nb && !dsm_bad) ? "OK" : "FAIL");
    bad_total += (ran != nb) || dsm_bad;
    if (le || se) break;
  }
  if (reps > 1) printf("  reps=%d failing_reps=%d\n", reps, bad_total);
  return bad_total ? 2 : 0;
}
