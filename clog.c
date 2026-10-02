// clog.so — LD_PRELOAD logger for clustered launches and cluster-occupancy queries (no policy, pass-through).
// Logs to stderr (or CLOG_OUT):  every cuOccupancyMaxActiveClusters result and every cuLaunchKernelEx with
// cluster size > 1: grid, block, shared memory, cluster dims, kernel name.
// Interposition covers direct link, dlsym() and cuGetProcAddress[_v2]() (adapted from climit.c).
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <cuda.h>
#undef cuGetProcAddress

typedef CUresult (*pfn_launchex_t)(const CUlaunchConfig *, CUfunction, void **, void **);
typedef CUresult (*pfn_occ_t)(int *, CUfunction, const CUlaunchConfig *);
typedef CUresult (*pfn_gpa_v2_t)(const char *, void **, int, cuuint64_t, CUdriverProcAddressQueryResult *);
typedef CUresult (*pfn_gpa_v1_t)(const char *, void **, int, cuuint64_t);
typedef void *(*pfn_dlsym_t)(void *, const char *);
typedef CUresult (*pfn_getname_t)(const char **, CUfunction);

static pfn_launchex_t real_launchex;
static pfn_occ_t real_occ;
static pfn_gpa_v2_t real_gpa_v2;
static pfn_gpa_v1_t real_gpa_v1;
static pfn_dlsym_t real_dlsym;
static pfn_getname_t real_getname;
static FILE *g_out;
static int g_all = 0;

static void init_once(void) {
    static int done = 0;
    if (done) return;
    done = 1;
    const char *o = getenv("CLOG_OUT");
    g_out = o ? fopen(o, "a") : NULL;
    if (!g_out) g_out = stderr;
    const char *e = getenv("CLOG_ALL");
    if (e && atoi(e)) g_all = 1;
    real_dlsym = (pfn_dlsym_t)dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.34");
    if (!real_dlsym) real_dlsym = (pfn_dlsym_t)dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.2.5");
}

static void *real_cuda_sym(const char *name) {
    init_once();
    void *h = dlopen("libcuda.so.1", RTLD_NOW | RTLD_NOLOAD);
    if (!h) h = dlopen("libcuda.so.1", RTLD_NOW);
    void *p = h ? real_dlsym(h, name) : NULL;
    if (!p) p = real_dlsym(RTLD_NEXT, name);
    return p;
}

static const char *kname(CUfunction f) {
    const char *nm = NULL;
    if (!real_getname) real_getname = (pfn_getname_t)real_cuda_sym("cuFuncGetName");
    if (!real_getname || real_getname(&nm, f) != CUDA_SUCCESS || !nm) nm = "?";
    return nm;
}

static unsigned cluster_of(const CUlaunchConfig *cfg, unsigned *x, unsigned *y, unsigned *z) {
    unsigned cx = 1, cy = 1, cz = 1;
    if (cfg && cfg->attrs)
        for (unsigned i = 0; i < cfg->numAttrs; i++)
            if (cfg->attrs[i].id == CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION) {
                cx = cfg->attrs[i].value.clusterDim.x;
                cy = cfg->attrs[i].value.clusterDim.y;
                cz = cfg->attrs[i].value.clusterDim.z;
            }
    *x = cx; *y = cy; *z = cz;
    return cx * cy * cz;
}

static CUresult my_cuLaunchKernelEx(const CUlaunchConfig *cfg, CUfunction f, void **kp, void **extra) {
    init_once();
    unsigned x, y, z, sz = cluster_of(cfg, &x, &y, &z);
    if (!real_launchex) real_launchex = (pfn_launchex_t)real_cuda_sym("cuLaunchKernelEx");
    CUresult r = real_launchex(cfg, f, kp, extra);
    if (sz > 1 || g_all) {
        fprintf(g_out, "[CLOG] launch rc=%d grid=%ux%ux%u block=%ux%ux%u smem=%u cluster=%ux%ux%u %.120s\n", (int)r,
                cfg->gridDimX, cfg->gridDimY, cfg->gridDimZ, cfg->blockDimX, cfg->blockDimY, cfg->blockDimZ,
                cfg->sharedMemBytes, x, y, z, kname(f));
        // CLOG_PARAMS=1: dump the kernel arguments (cuFuncGetParamInfo gives offset/size of each one). Pointer-sized
        // arguments differ between runs; the scalar ones show what the library passed for this context.
        static int dump = -1;
        if (dump < 0) dump = getenv("CLOG_PARAMS") && atoi(getenv("CLOG_PARAMS"));
        if (dump && kp) {
            typedef CUresult (*pfn_pi_t)(CUfunction, size_t, size_t *, size_t *);
            static pfn_pi_t pi;
            if (!pi) pi = (pfn_pi_t)real_cuda_sym("cuFuncGetParamInfo");
            fprintf(g_out, "[CLOG] params:");
            for (size_t i = 0; pi && i < 64; i++) {
                size_t off = 0, sz = 0;
                if (pi(f, i, &off, &sz) != CUDA_SUCCESS) break;
                unsigned long long v = 0;
                memcpy(&v, kp[i], sz > 8 ? 8 : sz);
                fprintf(g_out, " %zu:%zu=%llx", i, sz, v);
            }
            fprintf(g_out, "\n");
        }
        if (dump && !kp && extra) {      // packed argument buffer (CU_LAUNCH_PARAM_BUFFER_POINTER / _SIZE), printed as 4-byte words
            const unsigned char *buf = NULL; size_t bsz = 0;
            for (int i = 0; i < 8 && extra[i] != CU_LAUNCH_PARAM_END; i += 2) {
                if (extra[i] == CU_LAUNCH_PARAM_BUFFER_POINTER) buf = (const unsigned char *)extra[i + 1];
                else if (extra[i] == CU_LAUNCH_PARAM_BUFFER_SIZE) bsz = *(size_t *)extra[i + 1];
            }
            fprintf(g_out, "[CLOG] argbuf %zu bytes:", bsz);
            for (size_t o = 0; buf && o + 4 <= bsz && o < 512; o += 4) { unsigned v; memcpy(&v, buf + o, 4); fprintf(g_out, " %x", v); }
            fprintf(g_out, "\n");
        }
        fflush(g_out);
    }
    return r;
}

static CUresult my_cuOccupancyMaxActiveClusters(int *num, CUfunction f, const CUlaunchConfig *cfg) {
    init_once();
    unsigned x, y, z;
    cluster_of(cfg, &x, &y, &z);
    if (!real_occ) real_occ = (pfn_occ_t)real_cuda_sym("cuOccupancyMaxActiveClusters");
    CUresult r = real_occ(num, f, cfg);
    fprintf(g_out, "[CLOG] occupancy rc=%d numClusters=%d block=%ux%ux%u smem=%u cluster=%ux%ux%u %.120s\n", (int)r,
            num ? *num : -1, cfg->blockDimX, cfg->blockDimY, cfg->blockDimZ, cfg->sharedMemBytes, x, y, z, kname(f));
    fflush(g_out);
    return r;
}

CUresult cuLaunchKernelEx(const CUlaunchConfig *cfg, CUfunction f, void **kp, void **extra) {
    return my_cuLaunchKernelEx(cfg, f, kp, extra);
}
CUresult cuOccupancyMaxActiveClusters(int *num, CUfunction f, const CUlaunchConfig *cfg) {
    return my_cuOccupancyMaxActiveClusters(num, f, cfg);
}

static int substitute(const char *sym, void **pfn) {
    if (!sym || !pfn || !*pfn) return 0;
    if (!strcmp(sym, "cuLaunchKernelEx")) { *pfn = (void *)my_cuLaunchKernelEx; return 1; }
    if (!strcmp(sym, "cuOccupancyMaxActiveClusters")) { *pfn = (void *)my_cuOccupancyMaxActiveClusters; return 1; }
    return 0;
}
CUresult cuGetProcAddress_v2(const char *sym, void **pfn, int ver, cuuint64_t flags, CUdriverProcAddressQueryResult *st) {
    init_once();
    if (!real_gpa_v2) real_gpa_v2 = (pfn_gpa_v2_t)real_cuda_sym("cuGetProcAddress_v2");
    if (sym && !strcmp(sym, "cuGetProcAddress")) { *pfn = (void *)cuGetProcAddress_v2; if (st) *st = CU_GET_PROC_ADDRESS_SUCCESS; return CUDA_SUCCESS; }
    CUresult r = real_gpa_v2(sym, pfn, ver, flags, st);
    if (r == CUDA_SUCCESS) substitute(sym, pfn);
    return r;
}
CUresult cuGetProcAddress(const char *sym, void **pfn, int ver, cuuint64_t flags) {
    init_once();
    if (!real_gpa_v1) real_gpa_v1 = (pfn_gpa_v1_t)real_cuda_sym("cuGetProcAddress");
    if (sym && !strcmp(sym, "cuGetProcAddress")) { *pfn = (void *)cuGetProcAddress; return CUDA_SUCCESS; }
    CUresult r = real_gpa_v1(sym, pfn, ver, flags);
    if (r == CUDA_SUCCESS) substitute(sym, pfn);
    return r;
}
void *dlsym(void *handle, const char *sym) {
    init_once();
    if (sym) {
        if (!strcmp(sym, "cuGetProcAddress_v2")) return (void *)cuGetProcAddress_v2;
        if (!strcmp(sym, "cuGetProcAddress")) return (void *)cuGetProcAddress;
        if (!strcmp(sym, "cuLaunchKernelEx")) return (void *)my_cuLaunchKernelEx;
        if (!strcmp(sym, "cuOccupancyMaxActiveClusters")) return (void *)my_cuOccupancyMaxActiveClusters;
    }
    return real_dlsym(handle, sym);
}
