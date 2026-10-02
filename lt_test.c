// lt_test: FP16 cublasLtMatmul (C[m x n] = A[m x k] * B[k x n], column-major) with an explicit
// CUBLASLT_MATMUL_DESC_SM_COUNT_TARGET sweep. Separates "which algorithm config the heuristic picks for an SM
// count" from "how many SMs the context really has" (full GPU vs MPS active-thread-percentage share).
// usage: lt_test [m n k] -- targets...     (target 0 = leave the default)
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cublasLt.h>

typedef _Float16 h16;
#define CK(x) do { int _r = (int)(x); if (_r) { printf("ERR %s -> %d\n", #x, _r); exit(1); } } while (0)

static uint64_t cfg(const cublasLtMatmulAlgo_t *a, cublasLtMatmulAlgoConfigAttributes_t at) {
    uint64_t v = 0; size_t w = 0;
    if (cublasLtMatmulAlgoConfigGetAttribute(a, at, NULL, 0, &w) != CUBLAS_STATUS_SUCCESS || w == 0 || w > 8) return (uint64_t)-1;
    if (cublasLtMatmulAlgoConfigGetAttribute(a, at, &v, w, &w) != CUBLAS_STATUS_SUCCESS) return (uint64_t)-1;
    return v;
}

int main(int argc, char **argv) {
    int m = 1000, n = 16, k = 2048, ai = 1;
    if (argc > 4 && strcmp(argv[1], "--")) { m = atoi(argv[1]); n = atoi(argv[2]); k = atoi(argv[3]); ai = 4; }
    if (ai < argc && !strcmp(argv[ai], "--")) ai++;
    struct cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0));
    printf("device=%s physical/visible SMs=%d  MPS pct=%s  m=%d n=%d k=%d  cublasLt=%zu\n", p.name, p.multiProcessorCount,
           getenv("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE") ? getenv("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE") : "-", m, n, k, cublasLtGetVersion());

    h16 *A = malloc(sizeof(h16) * m * k), *B = malloc(sizeof(h16) * k * n), *C = malloc(sizeof(h16) * m * n);
    double *R = calloc((size_t)m * n, sizeof(double));
    srand(1);
    for (long i = 0; i < (long)m * k; i++) A[i] = (h16)((rand() / (double)RAND_MAX - 0.5) * 2.0);
    for (long i = 0; i < (long)k * n; i++) B[i] = (h16)((rand() / (double)RAND_MAX - 0.5) * 2.0 / sqrt((double)k));
    for (int j = 0; j < n; j++) for (int l = 0; l < k; l++) { double b = (double)B[l + (long)j * k];
        for (int i = 0; i < m; i++) R[i + (long)j * m] += (double)A[i + (long)l * m] * b; }

    h16 *dA, *dB, *dC; void *ws; size_t wsz = 32u << 20;
    CK(cudaMalloc((void **)&dA, sizeof(h16) * m * k)); CK(cudaMalloc((void **)&dB, sizeof(h16) * k * n));
    CK(cudaMalloc((void **)&dC, sizeof(h16) * m * n)); CK(cudaMalloc(&ws, wsz));
    CK(cudaMemcpy(dA, A, sizeof(h16) * m * k, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dB, B, sizeof(h16) * k * n, cudaMemcpyHostToDevice));
    cublasLtHandle_t lt; CK(cublasLtCreate(&lt));

    for (; ai < argc; ai++) {
        int32_t tgt = atoi(argv[ai]);
        cublasLtMatmulDesc_t op; cublasLtMatrixLayout_t la, lb, lc; cublasLtMatmulPreference_t pref;
        CK(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
        if (tgt > 0) CK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_SM_COUNT_TARGET, &tgt, sizeof(tgt)));
        CK(cublasLtMatrixLayoutCreate(&la, CUDA_R_16F, m, k, m));
        CK(cublasLtMatrixLayoutCreate(&lb, CUDA_R_16F, k, n, k));
        CK(cublasLtMatrixLayoutCreate(&lc, CUDA_R_16F, m, n, m));
        CK(cublasLtMatmulPreferenceCreate(&pref));
        CK(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsz, sizeof(wsz)));
        int nreq = getenv("LT_ENUM") ? atoi(getenv("LT_ENUM")) : 1; if (nreq > 64) nreq = 64;   // LT_ENUM=N: try the N best candidates
        cublasLtMatmulHeuristicResult_t hrs[64]; int nres = 0;
        cublasStatus_t hs = cublasLtMatmulAlgoGetHeuristic(lt, op, la, lb, lc, lc, pref, nreq, hrs, &nres);
        if (hs != CUBLAS_STATUS_SUCCESS || nres == 0) { printf("target=%-3d heuristic failed status=%d\n", tgt, (int)hs); continue; }
        for (int ri = 0; ri < nres; ri++) {
        cublasLtMatmulHeuristicResult_t hr = hrs[ri];
        if (hr.state != CUBLAS_STATUS_SUCCESS) continue;
        if (getenv("LT_CLUSTER") && *getenv("LT_CLUSTER")) {          // LT_CLUSTER=<cublasLtClusterShape_t>: override the cluster shape of the selected algo
            uint16_t cs = (uint16_t)atoi(getenv("LT_CLUSTER")); size_t w = 0;
            cublasLtMatmulAlgoConfigGetAttribute(&hr.algo, CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID, NULL, 0, &w);
            uint64_t v = cs;
            cublasStatus_t s1 = cublasLtMatmulAlgoConfigSetAttribute(&hr.algo, CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID, &v, w);
            cublasLtMatmulHeuristicResult_t chk; memset(&chk, 0, sizeof chk);
            cublasStatus_t s2 = cublasLtMatmulAlgoCheck(lt, op, la, lb, lc, lc, &hr.algo, &chk);
            printf("  override cluster=%u (attr size %zu) set=%d check=%d state=%d workspace=%zu (orig %zu)\n", cs, w, (int)s1, (int)s2, (int)chk.state, chk.workspaceSize, hr.workspaceSize);
        }
        CK(cudaMemset(dC, 0x7C, sizeof(h16) * m * n));           // sentinel 0x7C7C = NaN -> "never written"
        float alpha = 1.f, beta = 0.f;
        cublasStatus_t ms = cublasLtMatmul(lt, op, &alpha, dA, la, dB, lb, &beta, dC, lc, dC, lc, &hr.algo, ws, wsz, 0);
        cudaError_t se = cudaDeviceSynchronize();
        CK(cudaMemcpy(C, dC, sizeof(h16) * m * n, cudaMemcpyDeviceToHost));
        long bad = 0, unwritten = 0; int firstbad = -1, lastbad = -1;
        for (int i = 0; i < m; i++) for (int j = 0; j < n; j++) {
            uint16_t raw; memcpy(&raw, &C[i + (long)j * m], 2);
            double c = (double)C[i + (long)j * m], r = R[i + (long)j * m];
            int isbad = !(fabs(c - r) <= 5e-2 * (1.0 + fabs(r)));
            if (raw == 0x7C7C) unwritten++;
            if (isbad) { bad++; if (firstbad < 0) firstbad = i; lastbad = i; }
        }
        printf("target=%-3d #%-2d algo[id=%llu tile=%llu stages=%llu custom=%llu cluster=%llu] status=%d sync=%d  %s bad=%ld/%d unwritten=%ld bad_rows=[%d..%d]\n",
               tgt, ri, (unsigned long long)cfg(&hr.algo, CUBLASLT_ALGO_CONFIG_ID), (unsigned long long)cfg(&hr.algo, CUBLASLT_ALGO_CONFIG_TILE_ID),
               (unsigned long long)cfg(&hr.algo, CUBLASLT_ALGO_CONFIG_STAGES_ID), (unsigned long long)cfg(&hr.algo, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION),
               (unsigned long long)cfg(&hr.algo, CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID), (int)ms, (int)se,
               bad ? "CORRUPT" : "ok     ", bad, m * n, unwritten, firstbad, lastbad);
        fflush(stdout);
        }
        cublasLtMatmulPreferenceDestroy(pref); cublasLtMatrixLayoutDestroy(la); cublasLtMatrixLayoutDestroy(lb);
        cublasLtMatrixLayoutDestroy(lc); cublasLtMatmulDescDestroy(op);
    }
    return 0;
}
