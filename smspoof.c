// smspoof.so — LD_PRELOAD shim that overrides what the CUDA driver reports to libraries about the SM budget:
//   SPOOF_SM=<n>            cuDeviceGetAttribute(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT) returns n
//   SPOOF_OCC=<n1,n2,...>   cuOccupancyMaxActiveClusters returns n_c for cluster size c (1-based list);
//                           cluster sizes beyond the list return 0.  "SPOOF_OCC=noclust" = 0 for every size > 1.
//   SPOOF_MAXCLUSTER=<n>    cuOccupancyMaxActiveClusters returns 0 for every cluster size > n (real value otherwise),
//                           i.e. libraries stop selecting kernels that use larger thread block clusters.
//   SPOOF_SCOPE=<substr>    only override when the CALLER (return address) lives in a shared object whose path
//                           contains <substr>, e.g. SPOOF_SCOPE=cublas -> only cuBLAS/cuBLASLt see the override,
//                           every other library (cuDNN, the framework, ...) keeps the real MPS values.
// Unset variable = pass through.  SPOOF_LOG=1 prints every overridden answer.
// Interposition covers direct link, dlsym() and cuGetProcAddress[_v2]().
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>
#undef cuGetProcAddress

typedef CUresult (*pfn_attr_t)(int *, CUdevice_attribute, CUdevice);
typedef CUresult (*pfn_occ_t)(int *, CUfunction, const CUlaunchConfig *);
typedef CUresult (*pfn_gpa_v2_t)(const char *, void **, int, cuuint64_t, CUdriverProcAddressQueryResult *);
typedef CUresult (*pfn_gpa_v1_t)(const char *, void **, int, cuuint64_t);
typedef void *(*pfn_dlsym_t)(void *, const char *);

static pfn_attr_t real_attr;
static pfn_occ_t real_occ;
static pfn_gpa_v2_t real_gpa_v2;
static pfn_gpa_v1_t real_gpa_v1;
static pfn_dlsym_t real_dlsym;
static int g_sm = 0, g_log = 0, g_nocc = -1, g_noclust = 0, g_maxcl = 0, g_occ[256];
static const char *g_scope;

static int in_scope(void *ra) {
    if (!g_scope || !*g_scope) return 1;
    Dl_info di;
    return dladdr(ra, &di) && di.dli_fname && strstr(di.dli_fname, g_scope) != NULL;
}

static void init_once(void) {
    static int done = 0;
    if (done) return;
    done = 1;
    const char *e = getenv("SPOOF_SM");
    if (e) g_sm = atoi(e);
    g_scope = getenv("SPOOF_SCOPE");
    e = getenv("SPOOF_MAXCLUSTER");
    if (e) g_maxcl = atoi(e);
    e = getenv("SPOOF_LOG");
    if (e && atoi(e)) g_log = 1;
    e = getenv("SPOOF_OCC");
    if (e && !strcmp(e, "noclust")) g_noclust = 1;
    else if (e && *e) {
        char *s = strdup(e), *t = strtok(s, ",");
        g_nocc = 0;
        while (t && g_nocc < 256) { g_occ[g_nocc++] = atoi(t); t = strtok(NULL, ","); }
    }
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

static inline __attribute__((always_inline)) CUresult attr_impl(int *pi, CUdevice_attribute attrib, CUdevice dev, void *ra) {
    init_once();
    if (!real_attr) real_attr = (pfn_attr_t)real_cuda_sym("cuDeviceGetAttribute");
    CUresult r = real_attr(pi, attrib, dev);
    if (r == CUDA_SUCCESS && attrib == CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT && g_sm > 0 && pi && in_scope(ra)) {
        if (g_log) fprintf(stderr, "[SPOOF] SM count %d -> %d\n", *pi, g_sm);
        *pi = g_sm;
    }
    return r;
}

static inline __attribute__((always_inline)) CUresult occ_impl(int *num, CUfunction f, const CUlaunchConfig *cfg, void *ra) {
    init_once();
    if (!real_occ) real_occ = (pfn_occ_t)real_cuda_sym("cuOccupancyMaxActiveClusters");
    CUresult r = real_occ(num, f, cfg);
    unsigned c = 1;
    if (cfg && cfg->attrs)
        for (unsigned i = 0; i < cfg->numAttrs; i++)
            if (cfg->attrs[i].id == CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION)
                c = cfg->attrs[i].value.clusterDim.x * cfg->attrs[i].value.clusterDim.y * cfg->attrs[i].value.clusterDim.z;
    if (r == CUDA_SUCCESS && num && (g_nocc >= 0 || (g_noclust && c > 1) || (g_maxcl > 0 && (int)c > g_maxcl)) && in_scope(ra)) {
        int v = (g_noclust || (g_maxcl > 0 && (int)c > g_maxcl)) ? 0 : ((int)c <= g_nocc ? g_occ[c - 1] : 0);
        if (g_log) fprintf(stderr, "[SPOOF] occupancy cluster=%u %d -> %d\n", c, *num, v);
        *num = v;
    }
    return r;
}

// entry points handed out through cuGetProcAddress / dlsym
static CUresult my_cuDeviceGetAttribute(int *pi, CUdevice_attribute attrib, CUdevice dev) { return attr_impl(pi, attrib, dev, __builtin_return_address(0)); }
static CUresult my_cuOccupancyMaxActiveClusters(int *num, CUfunction f, const CUlaunchConfig *cfg) { return occ_impl(num, f, cfg, __builtin_return_address(0)); }
// exported entry points (direct-link path)
CUresult cuDeviceGetAttribute(int *pi, CUdevice_attribute attrib, CUdevice dev) { return attr_impl(pi, attrib, dev, __builtin_return_address(0)); }
CUresult cuOccupancyMaxActiveClusters(int *num, CUfunction f, const CUlaunchConfig *cfg) { return occ_impl(num, f, cfg, __builtin_return_address(0)); }

static int substitute(const char *sym, void **pfn) {
    if (!sym || !pfn || !*pfn) return 0;
    if (!strcmp(sym, "cuDeviceGetAttribute")) { *pfn = (void *)my_cuDeviceGetAttribute; return 1; }
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
        if (!strcmp(sym, "cuDeviceGetAttribute")) return (void *)my_cuDeviceGetAttribute;
        if (!strcmp(sym, "cuOccupancyMaxActiveClusters")) return (void *)my_cuOccupancyMaxActiveClusters;
    }
    return real_dlsym(handle, sym);
}
