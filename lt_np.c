// lt_np: cublasLtMatmul correctness for narrow-precision inputs (FP8 E4M3 / E5M2, FP4 E2M1 block-scaled) next to
// FP16 / BF16, each compared with a CPU reference built from the decoded operands.
//   D[m x n] = op(A) * B,  A stored k x m with TRANSA = T (the layout FP8 kernels require), B stored k x n.
// usage: lt_np <cfg> <m> <n> <k> [more "m n k" triples ...]
//   cfg: fp16 bf16 | fp8 (E4M3 -> FP16) fp8bf (E4M3 -> BF16) fp8f32 (E4M3 -> FP32) fp8e5 (E5M2 -> FP16)
//        | nvfp4 (E2M1, VEC16_UE4M3 scales -> BF16) nvfp4f32 (-> FP32) mxfp4 (E2M1, VEC32_UE8M0 scales -> BF16)
// All scale factors are 1, so only the data layout matters, not the scale-tensor layout.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cublasLt.h>

// IEEE half <-> double without _Float16 (so that the file also builds with gcc < 12)
static double dec_f16(uint16_t h) { int s = h >> 15, e = (h >> 10) & 31, m = h & 1023; double x = e == 0 ? ldexp((double)m, -24) : e == 31 ? (m ? NAN : INFINITY) : ldexp(1024.0 + m, e - 25); return s ? -x : x; }
static uint16_t enc_f16(float f) {
    if (f == 0) return 0;
    int s = f < 0, e; double a = fabs((double)f), fr = frexp(a, &e);
    int E = e - 1 + 15;
    if (E >= 31) return (uint16_t)((s << 15) | 0x7C00);
    if (E <= 0) return (uint16_t)((s << 15) | (int)lrint(ldexp(a, 24)));
    int m = (int)lrint((fr * 2 - 1) * 1024);
    if (m == 1024) { m = 0; E++; }
    return (uint16_t)((s << 15) | (E << 10) | m);
}
static double dec_e4m3(uint8_t v) { int s = v >> 7, e = (v >> 3) & 15, m = v & 7; double x = e ? (1 + m / 8.0) * ldexp(1.0, e - 7) : (m / 8.0) * ldexp(1.0, -6); return s ? -x : x; }
static double dec_e5m2(uint8_t v) { int s = v >> 7, e = (v >> 2) & 31, m = v & 3; double x = e ? (1 + m / 4.0) * ldexp(1.0, e - 15) : (m / 4.0) * ldexp(1.0, -14); return s ? -x : x; }
static double dec_e2m1(uint8_t v) { int s = (v >> 3) & 1, e = (v >> 1) & 3, m = v & 1; double x = e ? (1 + m / 2.0) * ldexp(1.0, e - 1) : m * 0.5; return s ? -x : x; }
static double dec_bf16(uint16_t v) { uint32_t u = (uint32_t)v << 16; float f; memcpy(&f, &u, 4); return f; }
static uint16_t enc_bf16(float f) { uint32_t u; memcpy(&u, &f, 4); return (uint16_t)((u + 0x8000u) >> 16); }

typedef struct { const char *name; cudaDataType_t in, out; int scale_mode; } cfg_t;
static const cfg_t CFGS[] = {
    {"fp16", CUDA_R_16F, CUDA_R_16F, -1}, {"bf16", CUDA_R_16BF, CUDA_R_16BF, -1},
    {"fp8", CUDA_R_8F_E4M3, CUDA_R_16F, -1}, {"fp8bf", CUDA_R_8F_E4M3, CUDA_R_16BF, -1}, {"fp8f32", CUDA_R_8F_E4M3, CUDA_R_32F, -1},
    {"fp8e5", CUDA_R_8F_E5M2, CUDA_R_16F, -1},
    {"nvfp4", CUDA_R_4F_E2M1, CUDA_R_16BF, CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3},
    {"nvfp4f32", CUDA_R_4F_E2M1, CUDA_R_32F, CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3},
    {"mxfp4", CUDA_R_4F_E2M1, CUDA_R_16BF, CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0},
};
#define CK(x) do { int _r = (int)(x); if (_r) { printf("ERR %s -> %d\n", #x, _r); return 1; } } while (0)
static uint64_t acfg(const cublasLtMatmulAlgo_t *a, cublasLtMatmulAlgoConfigAttributes_t at) {
    uint64_t v = 0; size_t w = 0;
    if (cublasLtMatmulAlgoConfigGetAttribute(a, at, NULL, 0, &w) != CUBLAS_STATUS_SUCCESS || w == 0 || w > 8) return 9999;
    if (cublasLtMatmulAlgoConfigGetAttribute(a, at, &v, w, &w) != CUBLAS_STATUS_SUCCESS) return 9999;
    return v;
}

int main(int argc, char **argv) {
    if (argc < 5) { printf("usage: %s cfg m n k [m n k ...]\n", argv[0]); return 1; }
    const cfg_t *c = NULL;
    for (unsigned i = 0; i < sizeof CFGS / sizeof CFGS[0]; i++) if (!strcmp(argv[1], CFGS[i].name)) c = &CFGS[i];
    if (!c) { printf("unknown cfg\n"); return 1; }
    struct cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0));
    const char *pct = getenv("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE");
    cublasLtHandle_t lt; CK(cublasLtCreate(&lt));
    int in_bits = c->in == CUDA_R_4F_E2M1 ? 4 : (c->in == CUDA_R_16F || c->in == CUDA_R_16BF) ? 16 : 8;
    int out_bytes = c->out == CUDA_R_32F ? 4 : 2;
    for (int ai = 2; ai + 2 < argc; ai += 3) {
        int m = atoi(argv[ai]), n = atoi(argv[ai + 1]), k = atoi(argv[ai + 2]);
        size_t na = (size_t)k * m, nb = (size_t)k * n, nd = (size_t)m * n;
        size_t ba = na * in_bits / 8, bb = nb * in_bits / 8;
        uint8_t *A = malloc(ba), *B = malloc(bb); double *a = malloc(na * 8), *b = malloc(nb * 8), *R = calloc(nd, 8);
        srand(1 + m + 7 * n + 13 * k);
        // random operands, decoded copies in a[] / b[] (element l of column i at index l + i*k)
        for (int w = 0; w < 2; w++) {
            uint8_t *X = w ? B : A; double *x = w ? b : a; size_t ne = w ? nb : na;
            for (size_t i = 0; i < ne; i++) {
                if (in_bits == 16) {
                    float f = (float)((rand() / (double)RAND_MAX - 0.5) * 2.0 / (w ? sqrt((double)k) : 1.0));
                    if (c->in == CUDA_R_16F) { uint16_t u = enc_f16(f); memcpy(X + 2 * i, &u, 2); x[i] = dec_f16(u); }
                    else { uint16_t u = enc_bf16(f); memcpy(X + 2 * i, &u, 2); x[i] = dec_bf16(u); }
                } else if (in_bits == 8) {
                    // moderate magnitudes: E4M3 exponent field 2..7 (2^-5 .. 2^0), E5M2 field 10..15
                    uint8_t v = c->in == CUDA_R_8F_E4M3 ? (uint8_t)(((rand() & 1) << 7) | ((2 + rand() % 6) << 3) | (rand() & 7))
                                                        : (uint8_t)(((rand() & 1) << 7) | ((10 + rand() % 6) << 2) | (rand() & 3));
                    X[i] = v; x[i] = c->in == CUDA_R_8F_E4M3 ? dec_e4m3(v) : dec_e5m2(v);
                } else {
                    uint8_t v = rand() & 15;                                       // element i in the low nibble for even i
                    if (i & 1) X[i / 2] |= (uint8_t)(v << 4); else X[i / 2] = v;
                    x[i] = dec_e2m1(v);
                }
            }
        }
        for (int j = 0; j < n; j++) for (int i = 0; i < m; i++) { double s = 0; const double *ac = a + (size_t)i * k, *bc = b + (size_t)j * k;
            for (int l = 0; l < k; l++) s += ac[l] * bc[l]; R[i + (size_t)j * m] = s; }

        void *dA, *dB, *dD, *ws, *dS; size_t wsz = 64u << 20, ssz = (na > nb ? na : nb) + 4096;
        CK(cudaMalloc(&dA, ba)); CK(cudaMalloc(&dB, bb)); CK(cudaMalloc(&dD, nd * out_bytes)); CK(cudaMalloc(&ws, wsz)); CK(cudaMalloc(&dS, ssz));
        CK(cudaMemcpy(dA, A, ba, cudaMemcpyHostToDevice)); CK(cudaMemcpy(dB, B, bb, cudaMemcpyHostToDevice));
        // scale tensors: every factor = 1 (UE4M3 0x38 = 1.0, UE8M0 0x7F = 2^0)
        CK(cudaMemset(dS, c->scale_mode == CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0 ? 0x7F : 0x38, ssz));
        CK(cudaMemset(dD, 0x7C, nd * out_bytes));
        cublasLtMatmulDesc_t op; cublasLtMatrixLayout_t la, lb, ld; cublasLtMatmulPreference_t pref;
        CK(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
        cublasOperation_t tr = CUBLAS_OP_T;
        CK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &tr, sizeof tr));
        if (c->scale_mode >= 0) {
            int32_t sm = c->scale_mode;
            CK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &sm, sizeof sm));
            CK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &sm, sizeof sm));
            CK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &dS, sizeof dS));
            CK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &dS, sizeof dS));
        }
        CK(cublasLtMatrixLayoutCreate(&la, c->in, k, m, k));
        CK(cublasLtMatrixLayoutCreate(&lb, c->in, k, n, k));
        CK(cublasLtMatrixLayoutCreate(&ld, c->out, m, n, m));
        CK(cublasLtMatmulPreferenceCreate(&pref));
        CK(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsz, sizeof wsz));
        cublasLtMatmulHeuristicResult_t hr; int nres = 0;
        cublasStatus_t hs = cublasLtMatmulAlgoGetHeuristic(lt, op, la, lb, ld, ld, pref, 1, &hr, &nres);
        if (hs != CUBLAS_STATUS_SUCCESS || nres == 0) {
            printf("NP %s sm=%d pct=%s %dx%dx%d UNSUPPORTED heuristic_status=%d\n", c->name, p.multiProcessorCount, pct ? pct : "-", m, n, k, (int)hs);
        } else {
            float alpha = 1.f, beta = 0.f;
            cublasStatus_t ms = cublasLtMatmul(lt, op, &alpha, dA, la, dB, lb, &beta, dD, ld, dD, ld, &hr.algo, ws, wsz, 0);
            cudaError_t se = cudaDeviceSynchronize();
            uint8_t *D = malloc(nd * out_bytes); CK(cudaMemcpy(D, dD, nd * out_bytes, cudaMemcpyDeviceToHost));
            long bad = 0, unwritten = 0; double tol = c->out == CUDA_R_16BF ? 1e-1 : 2e-2;
            for (size_t i = 0; i < nd; i++) {
                double v; uint16_t raw = 0;
                if (out_bytes == 4) { float f; memcpy(&f, D + 4 * i, 4); v = f; uint32_t u; memcpy(&u, D + 4 * i, 4); if (u == 0x7C7C7C7Cu) unwritten++; }
                else { memcpy(&raw, D + 2 * i, 2); if (raw == 0x7C7C) unwritten++; v = c->out == CUDA_R_16F ? dec_f16(raw) : dec_bf16(raw); }
                if (!(fabs(v - R[i]) <= tol * (1.0 + fabs(R[i])))) bad++;
            }
            printf("NP %s sm=%d pct=%s %dx%dx%d algo[id=%llu tile=%llu custom=%llu cluster=%llu] status=%d sync=%d %s bad=%ld/%zu unwritten=%ld\n",
                   c->name, p.multiProcessorCount, pct ? pct : "-", m, n, k, (unsigned long long)acfg(&hr.algo, CUBLASLT_ALGO_CONFIG_ID),
                   (unsigned long long)acfg(&hr.algo, CUBLASLT_ALGO_CONFIG_TILE_ID), (unsigned long long)acfg(&hr.algo, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION),
                   (unsigned long long)acfg(&hr.algo, CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID), (int)ms, (int)se,
                   (ms || se) ? "FAILED" : bad ? "CORRUPT" : "ok", bad, nd, unwritten);
            free(D);
        }
        fflush(stdout);
        cublasLtMatmulPreferenceDestroy(pref); cublasLtMatrixLayoutDestroy(la); cublasLtMatrixLayoutDestroy(lb); cublasLtMatrixLayoutDestroy(ld);
        cublasLtMatmulDescDestroy(op); cudaFree(dA); cudaFree(dB); cudaFree(dD); cudaFree(ws); cudaFree(dS);
        free(A); free(B); free(a); free(b); free(R);
    }
    return 0;
}
