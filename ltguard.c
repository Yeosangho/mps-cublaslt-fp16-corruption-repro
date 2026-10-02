// ltguard.so — LD_PRELOAD guard that keeps cuBLASLt from executing GEMM kernels with thread-block clusters of
// more than 2 CTAs.  No CUDA driver API is touched and nothing the driver reports is changed.
//
// It interposes the cuBLASLt *execution* entry points only (both for callers that link against libcublasLt and
// for callers that look the functions up with dlsym(), which is what cuDNN does):
//   cublasLtMatmul                       (public API; this is what cuDNN calls for 1x1 convolutions)
//   cublasLt{SSS,HSH,HHH,...}Matmul      (typed entry points libcublas uses for cublas*gemm / cublasGemmEx)
// For every call it reads the cluster shape of the algorithm that is about to run
// (cublasLtMatmulAlgoConfigGetAttribute, CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID).  If the shape has more than
// 2 CTAs (4x1x1, 1x4x1, 2x2x1, 2x4x1, 4x2x1, ...) it runs the SAME algorithm (same id, tile, stages, custom
// option) with the cluster shape replaced by 2x1x1 / 1x2x1 / 1x1x1, after cublasLtMatmulAlgoCheck accepted the
// replacement.  Everything else passes through unchanged.
//
// Scope: needed only on Hopper (compute capability 9.x) when the process runs under an MPS
// CUDA_MPS_ACTIVE_THREAD_PERCENTAGE share.  Blackwell (B200, RTX PRO 6000), MIG instances and 100 % / no MPS
// gave correct results without it.  The guard therefore checks the compute capability of the device of the
// current CUDA context (cuCtxGetDevice + cuDeviceGetAttribute, read-only; cuBLASLt is always called with a
// context in place) the first time it meets a large cluster shape, and rewrites only on compute capability
// LTGUARD_CC (default 9).  If the capability cannot be determined the guard stays active.
//
// Env:  LTGUARD_DISABLE=1   pass everything through
//       LTGUARD_CC=<major>  compute-capability major version to guard (default 9 = Hopper); 0 = every GPU
//       LTGUARD_LOG=1       print a summary at exit (2 = also every rewrite)
//       LTGUARD_MAXCTA=<n>  largest cluster (in CTAs) left alone, default 2
//       LTGUARD_NOCACHE=1   validate every call with cublasLtMatmulAlgoCheck instead of caching the decision
#define _GNU_SOURCE
#include <dlfcn.h>
#include <link.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <cuda.h>
#include <cublasLt.h>

typedef cublasStatus_t (*matmul_t)(cublasLtHandle_t, cublasLtMatmulDesc_t, const void *, const void *, cublasLtMatrixLayout_t,
                                   const void *, cublasLtMatrixLayout_t, const void *, const void *, cublasLtMatrixLayout_t,
                                   void *, cublasLtMatrixLayout_t, const cublasLtMatmulAlgo_t *, void *, size_t, cudaStream_t);
typedef cublasStatus_t (*cfgget_t)(const cublasLtMatmulAlgo_t *, cublasLtMatmulAlgoConfigAttributes_t, void *, size_t, size_t *);
typedef cublasStatus_t (*cfgset_t)(cublasLtMatmulAlgo_t *, cublasLtMatmulAlgoConfigAttributes_t, const void *, size_t);
typedef cublasStatus_t (*check_t)(cublasLtHandle_t, cublasLtMatmulDesc_t, cublasLtMatrixLayout_t, cublasLtMatrixLayout_t,
                                  cublasLtMatrixLayout_t, cublasLtMatrixLayout_t, const cublasLtMatmulAlgo_t *, cublasLtMatmulHeuristicResult_t *);

// own parser instead of atoi(): keeps the library loadable on glibc < 2.38 (no __isoc23_strtol dependency)
static int ltg_atoi(const char *s) {
    int v = 0;
    while (*s >= '0' && *s <= '9') v = v * 10 + (*s++ - '0');
    return v;
}

typedef void *(*dlsym_t)(void *, const char *);
static dlsym_t r_dlsym;
static void *rdlsym(void *h, const char *n) {
    if (!r_dlsym) r_dlsym = (dlsym_t)dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.34");
    if (!r_dlsym) r_dlsym = (dlsym_t)dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.2.5");
    return r_dlsym(h, n);
}
static void *g_lt;                 // handle of the already loaded libcublasLt
static cfgget_t r_get;
static cfgset_t r_set;
static check_t r_check;
static int g_disable, g_log, g_maxcta = 2, g_nocache, g_ccmode = 9;
static unsigned long g_calls, g_rewritten, g_kept_unsafe, g_arch_skipped, g_arch_unknown;
static int g_last_cc = -1;         // compute-capability major of the last device looked at (for the report)

static int find_lt(struct dl_phdr_info *info, size_t size, void *data) {
    (void)size;
    if (info->dlpi_name && strstr(info->dlpi_name, "libcublasLt.so")) { strncpy((char *)data, info->dlpi_name, 4095); return 1; }
    return 0;
}

static void init_once(void) {
    static int done = 0;
    if (done) return;
    done = 1;
    const char *e;
    if ((e = getenv("LTGUARD_DISABLE")) && ltg_atoi(e)) g_disable = 1;
    if ((e = getenv("LTGUARD_LOG"))) g_log = ltg_atoi(e);
    if ((e = getenv("LTGUARD_MAXCTA")) && ltg_atoi(e) > 0) g_maxcta = ltg_atoi(e);
    if ((e = getenv("LTGUARD_NOCACHE")) && ltg_atoi(e)) g_nocache = 1;
    if ((e = getenv("LTGUARD_CC")) && *e) g_ccmode = ltg_atoi(e);
    char path[4096] = "";
    dl_iterate_phdr(find_lt, path);                      // the copy the process already loaded (may be RTLD_LOCAL)
    if (path[0]) g_lt = dlopen(path, RTLD_NOW | RTLD_NOLOAD);
    if (g_lt) {
        r_get = (cfgget_t)rdlsym(g_lt, "cublasLtMatmulAlgoConfigGetAttribute");
        r_set = (cfgset_t)rdlsym(g_lt, "cublasLtMatmulAlgoConfigSetAttribute");
        r_check = (check_t)rdlsym(g_lt, "cublasLtMatmulAlgoCheck");
    }
    if (!g_lt || !r_get || !r_set || !r_check) {
        if (g_log) fprintf(stderr, "[LTGUARD] libcublasLt not resolved (%s) - passing through\n", path);
        g_disable = 1;
    }
}

// Is the device of the current context one the guard is meant for?  Read-only driver queries; the answer is
// cached per device ordinal.  Unknown (driver not resolved, no current context) counts as "yes".
static int arch_needs_guard(void) {
    typedef CUresult (*ctxdev_t)(CUdevice *);
    typedef CUresult (*devattr_t)(int *, CUdevice_attribute, CUdevice);
    static ctxdev_t ctxdev; static devattr_t devattr; static int resolved;
    static signed char cc[64];                                    // 0 = not queried yet
    if (g_ccmode == 0) return 1;
    if (!resolved) {
        void *h = dlopen("libcuda.so.1", RTLD_NOW | RTLD_NOLOAD);
        if (h) { ctxdev = (ctxdev_t)rdlsym(h, "cuCtxGetDevice"); devattr = (devattr_t)rdlsym(h, "cuDeviceGetAttribute"); }
        resolved = 1;
    }
    CUdevice dev = -1; int major = 0;
    if (!ctxdev || !devattr || ctxdev(&dev) != CUDA_SUCCESS || dev < 0) { __sync_add_and_fetch(&g_arch_unknown, 1); return 1; }
    if (dev < 64 && cc[dev]) major = cc[dev];
    else {
        if (devattr(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev) != CUDA_SUCCESS || major <= 0) {
            __sync_add_and_fetch(&g_arch_unknown, 1); return 1;
        }
        if (dev < 64) cc[dev] = (signed char)major;
    }
    g_last_cc = major;
    return major == g_ccmode;
}

// CTAs per cluster for a cublasLtClusterShape_t value; 0 = AUTO / unknown (left alone)
static int cta_of(unsigned cs, int *xdom) {
    static const unsigned char X[] = {0,0,1,2,4,1,2,4,1,2,4,8,1,8,2,16,1,3,5,6,7,9,10,11,12,13,14,15,3,5,6,7,1,2,3,4,5,3,1,2,3};
    static const unsigned char Y[] = {0,0,1,1,1,2,2,2,4,4,4,1,8,2,8,1,16,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,3,3,3,3,3,4,5,5,5};
    if (cs >= sizeof X) { *xdom = 1; return cs ? 99 : 0; }       // newer, unknown shapes: treat as large
    *xdom = X[cs] >= Y[cs];
    return X[cs] * Y[cs];
}

// Decision cache: algorithm (all 64 bytes) -> accepted replacement cluster shape.  A hit skips the two
// cublasLtMatmulAlgoCheck calls; if the library then rejects the cached replacement for this problem, the
// wrapper falls back to the checked path (a rejection is an error status, never a silent wrong result).
#define NCACHE 256
static struct { cublasLtMatmulAlgo_t key; uint16_t rep; int valid; } g_cache[NCACHE];
static volatile int g_cache_lock;
static unsigned long g_cache_hits;
static unsigned slot_of(const cublasLtMatmulAlgo_t *a) {
    uint64_t x = 1469598103934665603ULL;
    for (int i = 0; i < 8; i++) x = (x ^ a->data[i]) * 1099511628211ULL;
    return (unsigned)(x % NCACHE);
}
static int cache_get(const cublasLtMatmulAlgo_t *a, uint16_t *rep) {
    unsigned i = slot_of(a); int hit;
    while (__sync_lock_test_and_set(&g_cache_lock, 1)) ;
    hit = g_cache[i].valid && !memcmp(&g_cache[i].key, a, sizeof *a);
    if (hit) *rep = g_cache[i].rep;
    __sync_lock_release(&g_cache_lock);
    return hit;
}
static void cache_put(const cublasLtMatmulAlgo_t *a, uint16_t rep, int valid) {
    unsigned i = slot_of(a);
    while (__sync_lock_test_and_set(&g_cache_lock, 1)) ;
    g_cache[i].key = *a; g_cache[i].rep = rep; g_cache[i].valid = valid;
    __sync_lock_release(&g_cache_lock);
}

// from_cache == NULL: checked path only (used when a cached replacement was rejected by the library).
static const cublasLtMatmulAlgo_t *guard(cublasLtHandle_t h, cublasLtMatmulDesc_t d, cublasLtMatrixLayout_t A, cublasLtMatrixLayout_t B,
                                         cublasLtMatrixLayout_t C, cublasLtMatrixLayout_t D, const cublasLtMatmulAlgo_t *algo,
                                         size_t wsz, cublasLtMatmulAlgo_t *tmp, int *from_cache) {
    init_once();
    if (from_cache) __sync_add_and_fetch(&g_calls, 1);
    if (g_disable || !algo) return algo;
    uint16_t cs = 0; size_t w = 0;
    if (r_get(algo, CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID, &cs, sizeof cs, &w) != CUBLAS_STATUS_SUCCESS) return algo;
    int xdom, n = cta_of(cs, &xdom);
    if (n <= g_maxcta) return algo;                               // AUTO, 1x1x1, 2x1x1, 1x2x1
    if (!arch_needs_guard()) { if (from_cache) __sync_add_and_fetch(&g_arch_skipped, 1); return algo; }
    uint16_t crep;
    if (from_cache && !g_nocache && cache_get(algo, &crep)) {
        *tmp = *algo;
        if (r_set(tmp, CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID, &crep, sizeof crep) == CUBLAS_STATUS_SUCCESS) {
            *from_cache = 1;
            __sync_add_and_fetch(&g_rewritten, 1);
            __sync_add_and_fetch(&g_cache_hits, 1);
            return tmp;
        }
    }
    const uint16_t order[3] = { (uint16_t)(xdom ? CUBLASLT_CLUSTER_SHAPE_2x1x1 : CUBLASLT_CLUSTER_SHAPE_1x2x1),
                                (uint16_t)(xdom ? CUBLASLT_CLUSTER_SHAPE_1x2x1 : CUBLASLT_CLUSTER_SHAPE_2x1x1),
                                (uint16_t)CUBLASLT_CLUSTER_SHAPE_1x1x1 };
    // Workspace the library itself reports for the algorithm the caller chose; a replacement may need as much.
    size_t orig_ws = 0;
    {
        cublasLtMatmulHeuristicResult_t o;
        memset(&o, 0, sizeof o);
        if (r_check(h, d, A, B, C, D, algo, &o) == CUBLAS_STATUS_SUCCESS && o.state == CUBLAS_STATUS_SUCCESS) orig_ws = o.workspaceSize;
    }
    if (g_log > 2) fprintf(stderr, "[LTGUARD] original cluster %u reports ws=%zu (caller gave %zu)\n", cs, orig_ws, wsz);
    for (int i = (g_maxcta >= 2 ? 0 : 2); i < 3; i++) {
        cublasLtMatmulHeuristicResult_t res;
        memset(&res, 0, sizeof res);
        *tmp = *algo;
        cublasStatus_t s1 = r_set(tmp, CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID, &order[i], sizeof order[i]);
        cublasStatus_t s2 = s1 == CUBLAS_STATUS_SUCCESS ? r_check(h, d, A, B, C, D, tmp, &res) : s1;
        if (g_log > 2) {
            int id = -1, tile = -1, st = -1; uint32_t co = 0; size_t ww;
            r_get(algo, CUBLASLT_ALGO_CONFIG_ID, &id, sizeof id, &ww); r_get(algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &tile, sizeof tile, &ww);
            r_get(algo, CUBLASLT_ALGO_CONFIG_STAGES_ID, &st, sizeof st, &ww); r_get(algo, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, &co, sizeof co, &ww);
            fprintf(stderr, "[LTGUARD] try id=%d tile=%d stages=%d custom=%u cluster %u -> %u : set=%d check=%d state=%d ws=%zu (have %zu)\n",
                    id, tile, st, co, cs, order[i], (int)s1, (int)s2, (int)res.state, res.workspaceSize, wsz);
        }
        if (s2 != CUBLAS_STATUS_SUCCESS || res.state != CUBLAS_STATUS_SUCCESS) continue;
        if (res.workspaceSize > wsz && res.workspaceSize > orig_ws) continue;
        __sync_add_and_fetch(&g_rewritten, 1);
        if (!g_nocache) cache_put(algo, order[i], 1);
        if (g_log > 1) fprintf(stderr, "[LTGUARD] cluster shape %u -> %u\n", cs, order[i]);
        return tmp;
    }
    __sync_add_and_fetch(&g_kept_unsafe, 1);
    if (g_log > 1) fprintf(stderr, "[LTGUARD] cluster shape %u: no accepted replacement, left as is\n", cs);
    return algo;
}

#define WRAP(NAME)                                                                                                        \
    cublasStatus_t NAME(cublasLtHandle_t h, cublasLtMatmulDesc_t d, const void *alpha, const void *A, cublasLtMatrixLayout_t Ad, \
                        const void *B, cublasLtMatrixLayout_t Bd, const void *beta, const void *C, cublasLtMatrixLayout_t Cd,    \
                        void *D, cublasLtMatrixLayout_t Dd, const cublasLtMatmulAlgo_t *algo, void *ws, size_t wsz,              \
                        cudaStream_t stream) {                                                                            \
        static matmul_t real;                                                                                             \
        cublasLtMatmulAlgo_t tmp;                                                                                         \
        int fc = 0;                                                                                                       \
        const cublasLtMatmulAlgo_t *use = guard(h, d, Ad, Bd, Cd, Dd, algo, wsz, &tmp, &fc);                              \
        if (!real) real = (matmul_t)(g_lt ? rdlsym(g_lt, #NAME) : rdlsym(RTLD_NEXT, #NAME));                                \
        if (!real) return CUBLAS_STATUS_NOT_SUPPORTED;                                                                    \
        cublasStatus_t st = real(h, d, alpha, A, Ad, B, Bd, beta, C, Cd, D, Dd, use, ws, wsz, stream);                    \
        if (fc && st != CUBLAS_STATUS_SUCCESS) { /* cached replacement rejected for this problem: redo with checks */     \
            cache_put(algo, 0, 0);                                                                                        \
            use = guard(h, d, Ad, Bd, Cd, Dd, algo, wsz, &tmp, NULL);                                                     \
            st = real(h, d, alpha, A, Ad, B, Bd, beta, C, Cd, D, Dd, use, ws, wsz, stream);                               \
        }                                                                                                                 \
        return st;                                                                                                        \
    }

WRAP(cublasLtMatmul)
WRAP(cublasLtSSSMatmul) WRAP(cublasLtHSHMatmul) WRAP(cublasLtHHHMatmul) WRAP(cublasLtHSSMatmul)
WRAP(cublasLtBSSMatmul) WRAP(cublasLtBIIMatmul) WRAP(cublasLtTSSMatmul) WRAP(cublasLtTSTMatmul)
WRAP(cublasLtACCMatmul) WRAP(cublasLtCCCMatmul) WRAP(cublasLtDDDMatmul) WRAP(cublasLtZZZMatmul)

// Callers that resolve cuBLASLt through dlsym() (cuDNN) get the same wrappers.
#define ENTRY(NAME) { #NAME, (void *)NAME }
static const struct { const char *name; void *fn; } g_wrapped[] = {
    ENTRY(cublasLtMatmul),
    ENTRY(cublasLtSSSMatmul), ENTRY(cublasLtHSHMatmul), ENTRY(cublasLtHHHMatmul), ENTRY(cublasLtHSSMatmul),
    ENTRY(cublasLtBSSMatmul), ENTRY(cublasLtBIIMatmul), ENTRY(cublasLtTSSMatmul), ENTRY(cublasLtTSTMatmul),
    ENTRY(cublasLtACCMatmul), ENTRY(cublasLtCCCMatmul), ENTRY(cublasLtDDDMatmul), ENTRY(cublasLtZZZMatmul),
};
void *dlsym(void *handle, const char *name) {
    void *p = rdlsym(handle, name);
    if (p && name && !strncmp(name, "cublasLt", 8))
        for (unsigned i = 0; i < sizeof g_wrapped / sizeof g_wrapped[0]; i++)
            if (!strcmp(name, g_wrapped[i].name)) return g_wrapped[i].fn;
    return p;
}

// Counters for test harnesses (ctypes.CDLL(None).ltguard_rewritten()).
unsigned long ltguard_calls(void) { return g_calls; }
unsigned long ltguard_rewritten(void) { return g_rewritten; }
unsigned long ltguard_left_unsafe(void) { return g_kept_unsafe; }

__attribute__((destructor)) static void report(void) {
    if (g_log) fprintf(stderr, "[LTGUARD] pid=%d matmul calls=%lu cluster shape rewritten=%lu (cache hits %lu) left unsafe=%lu"
                       " | device CC major=%d guarded CC=%d skipped (other architecture)=%lu CC unknown=%lu%s\n", getpid(), g_calls,
                       g_rewritten, g_cache_hits, g_kept_unsafe, g_last_cc, g_ccmode, g_arch_skipped, g_arch_unknown,
                       g_disable ? " (disabled)" : "");
}
