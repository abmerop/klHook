// Copyright (c) 2026 Matt Poremba
// SPDX-License-Identifier: MIT
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <iostream>
#include <unordered_map>

#include "hip/hip_runtime.h"
#include "hip/hip_ext.h"
#include "hsa/hsa.h"
#include "hsa/hsa_ext_amd.h"


// Note: for hipMalloc there are two candidates so it seems decltype won't work
hipError_t (*hipMalloc_func)(void**, size_t);
decltype(::hipMemcpy)* hipMemcpy_func;
decltype(::hipMemcpyWithStream)* hipMemcpyWithStream_func;
decltype(::hipLaunchKernel)* hipLaunchKernel_func;
decltype(::hipModuleLaunchKernel)* hipModuleLaunchKernel_func;
decltype(::hipExtModuleLaunchKernel)* hipExtModuleLaunchKernel_func;

decltype(hsa_memory_allocate)* hsa_memory_allocate_func;
decltype(hsa_memory_free)* hsa_memory_free_func;
decltype(hsa_memory_assign_agent)* hsa_memory_assign_agent_func;
decltype(hsa_memory_copy)* hsa_memory_copy_func;
decltype(hsa_amd_memory_pool_allocate)* hsa_amd_memory_pool_allocate_func;
decltype(hsa_amd_memory_pool_free)* hsa_amd_memory_pool_free_func;
decltype(hsa_amd_memory_async_copy)* hsa_amd_memory_async_copy_func;
decltype(hsa_amd_memory_async_copy_rect)* hsa_amd_memory_async_copy_rect_func;

static bool initialized = false;
static bool verbose = false;
std::unordered_map<uint64_t, size_t> memory_map;

void init_hooks()
{
    const char *kl_hook_verbose = std::getenv("KL_HOOK_VERBOSE");
    if (kl_hook_verbose && !strncmp(kl_hook_verbose, "1", 2)) {
        verbose = true;
        std::cout << "kl-hook: verbose enabled\n";
    }

    const char *kl_hook_hip_path = std::getenv("KL_HOOK_HIP_LIB_PATH");
    std::string hip_lib_path;
    if (kl_hook_hip_path) {
        hip_lib_path = kl_hook_hip_path;
    } else {
        hip_lib_path = "/opt/rocm/lib/libamdhip64.so";
    }

    void *hip_handle = dlopen(hip_lib_path.c_str(), RTLD_NOW);
    if (!hip_handle) {
        std::cerr << "Error loading " << hip_lib_path << ": " << dlerror() << std::endl;
        exit(EXIT_FAILURE);
    }

    hipMalloc_func = (hipError_t (*)(void**, size_t))dlsym(hip_handle, "hipMalloc");
    if (!hipMalloc_func) { std::cerr << "Failed to load hipMalloc" << std::endl; exit(EXIT_FAILURE); }

    hipMemcpy_func = (decltype(::hipMemcpy)*)dlsym(hip_handle, "hipMemcpy");
    if (!hipMemcpy_func) { std::cerr << "Failed to load hipMemcpy" << std::endl; exit(EXIT_FAILURE); }

    hipMemcpyWithStream_func = (decltype(::hipMemcpyWithStream)*)dlsym(hip_handle, "hipMemcpyWithStream");
    if (!hipMemcpyWithStream_func) { std::cerr << "Failed to load hipMemcpyWithStream" << std::endl; exit(EXIT_FAILURE); }

    hipLaunchKernel_func = (decltype(::hipLaunchKernel)*)dlsym(hip_handle, "hipLaunchKernel");
    if (!hipLaunchKernel_func) { std::cerr << "Failed to load hipLaunchKernel" << std::endl; exit(EXIT_FAILURE); }

    hipModuleLaunchKernel_func = (decltype(::hipModuleLaunchKernel)*)dlsym(hip_handle, "hipModuleLaunchKernel");
    if (!hipModuleLaunchKernel_func) { std::cerr << "Failed to load hipModuleLaunchKernel" << std::endl; exit(EXIT_FAILURE); }

    // For C++ symbols use the mangled name from `strings /opt/rocm/lib/libamdhip64.so | grep <unmangled_name>`
    hipExtModuleLaunchKernel_func = (decltype(::hipExtModuleLaunchKernel)*)dlsym(hip_handle, "_Z24hipExtModuleLaunchKernelP18ihipModuleSymbol_tjjjjjjmP12ihipStream_tPPvS4_P11ihipEvent_tS6_j");
    if (!hipExtModuleLaunchKernel_func) { std::cerr << "Failed to load hipExtModuleLaunchKernel: " << dlerror() << std::endl; exit(EXIT_FAILURE); }


    // HSA memory functions
    const char *kl_hook_hsa_path = std::getenv("KL_HOOK_HSA_LIB_PATH");
    std::string hsa_lib_path;
    if (kl_hook_hsa_path) {
        hsa_lib_path = kl_hook_hsa_path;
    } else {
        hsa_lib_path = "/opt/rocm/lib/libhsa-runtime64.so";
    }

    void *hsa_handle = dlopen(hsa_lib_path.c_str(), RTLD_NOW);
    if (!hsa_handle) {
        std::cerr << "Error loading " << hsa_lib_path << ": " << dlerror() << std::endl;
        exit(EXIT_FAILURE);
    }

    hsa_memory_allocate_func = (decltype(hsa_memory_allocate)*)dlsym(hsa_handle, "hsa_memory_allocate");
    if (!hsa_memory_allocate_func) { std::cerr << "Failed to load hsa_memory_allocate" << std::endl; exit(EXIT_FAILURE); }

    hsa_memory_free_func = (decltype(hsa_memory_free)*)dlsym(hsa_handle, "hsa_memory_free");
    if (!hsa_memory_free_func) { std::cerr << "Failed to load hsa_memory_free" << std::endl; exit(EXIT_FAILURE); }

    hsa_memory_assign_agent_func = (decltype(hsa_memory_assign_agent)*)dlsym(hsa_handle, "hsa_memory_assign_agent");
    if (!hsa_memory_assign_agent_func) { std::cerr << "Failed to load hsa_memory_assign_agent" << std::endl; exit(EXIT_FAILURE); }

    hsa_memory_copy_func = (decltype(hsa_memory_copy)*)dlsym(hsa_handle, "hsa_memory_copy");
    if (!hsa_memory_copy_func) { std::cerr << "Failed to load hsa_memory_copy" << std::endl; exit(EXIT_FAILURE); }

    hsa_amd_memory_pool_allocate_func = (decltype(hsa_amd_memory_pool_allocate)*)dlsym(hsa_handle, "hsa_amd_memory_pool_allocate");
    if (!hsa_amd_memory_pool_allocate_func) { std::cerr << "Failed to load hsa_amd_memory_pool_allocate" << std::endl; exit(EXIT_FAILURE); }

    hsa_amd_memory_pool_free_func = (decltype(hsa_amd_memory_pool_free)*)dlsym(hsa_handle, "hsa_amd_memory_pool_free");
    if (!hsa_amd_memory_pool_free_func) { std::cerr << "Failed to load hsa_amd_memory_pool_free" << std::endl; exit(EXIT_FAILURE); }

    hsa_amd_memory_async_copy_func = (decltype(hsa_amd_memory_async_copy)*)dlsym(hsa_handle, "hsa_amd_memory_async_copy");
    if (!hsa_amd_memory_async_copy_func) { std::cerr << "Failed to load hsa_amd_memory_async_copy" << std::endl; exit(EXIT_FAILURE); }

    hsa_amd_memory_async_copy_rect_func = (decltype(hsa_amd_memory_async_copy_rect)*)dlsym(hsa_handle, "hsa_amd_memory_async_copy_rect");
    if (!hsa_amd_memory_async_copy_rect_func) { std::cerr << "Failed to load hsa_amd_memory_async_copy_rect" << std::endl; exit(EXIT_FAILURE); }

    initialized = true;
}


// Some functions have C linkage
extern "C" {

hipError_t hipMalloc(void** ptr, size_t size)
{
    if (!initialized) init_hooks();

    if (verbose) {
        std::cout << "kl-hook: hipMalloc called!\n\n";
    }

    return hipMalloc_func(ptr, size);
}

hipError_t hipMemcpy(void* dst, const void* src, size_t sizeBytes, hipMemcpyKind kind)
{
    if (!initialized) init_hooks();

    if (verbose) {
        std::cout << "kl-hook: hipMemcpy called!\n\n";
    }

    return hipMemcpy_func(dst, src, sizeBytes, kind);
}

hipError_t hipMemcpyWithStream(void* dst, const void* src, size_t sizeBytes,
                               hipMemcpyKind kind, hipStream_t stream)
{
    if (!initialized) init_hooks();

    if (verbose) {
        std::cout << "kl-hook: hipMemcpyWithStream called!\n\n";
    }

    return hipMemcpyWithStream_func(dst, src, sizeBytes, kind, stream);
}

hipError_t hipLaunchKernel(const void* function_address,
                           dim3 numBlocks,
                           dim3 dimBlocks,
                           void** args,
                           size_t sharedMemBytes,
                           hipStream_t stream)
{
    if (!initialized) init_hooks();

    if (verbose) {
        std::cout << "kl-hook: hipLaunchKernel called!\n\n"; 
        if (!args) {
            std::cout << " -- No kernargs\n";
        }
    }

    return hipLaunchKernel_func(function_address, numBlocks, dimBlocks, args, sharedMemBytes, stream);
}

hipError_t hipModuleLaunchKernel(hipFunction_t f, unsigned int gridDimX, unsigned int gridDimY,
                                 unsigned int gridDimZ, unsigned int blockDimX,
                                 unsigned int blockDimY, unsigned int blockDimZ,
                                 unsigned int sharedMemBytes, hipStream_t stream,
                                 void** kernelParams, void** extra)
{
    if (!initialized) init_hooks();

    if (verbose) {
        std::cout << "kl-hook: hipModuleLaunchKernel called!\n\n"; 
        if (!kernelParams) {
            std::cout << " -- No kernargs\n";
        }
    }

    return hipModuleLaunchKernel_func(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ,
                                      sharedMemBytes, stream, kernelParams, extra);
}

} // extern "C"



// Other functions have C++ linkage
HIP_PUBLIC_API
hipError_t hipExtModuleLaunchKernel(hipFunction_t f, uint32_t globalWorkSizeX,
                                    uint32_t globalWorkSizeY, uint32_t globalWorkSizeZ,
                                    uint32_t localWorkSizeX, uint32_t localWorkSizeY,
                                    uint32_t localWorkSizeZ, size_t sharedMemBytes,
                                    hipStream_t hStream, void** kernelParams, void** extra,
                                    hipEvent_t startEvent,
                                    hipEvent_t stopEvent,
                                    uint32_t flags)
{
    if (!initialized) init_hooks();

    if (verbose) {
        std::cout << "kl-hook: hipExtModuleLaunchKernel called!\n\n"; 
        if (!kernelParams) {
            std::cout << " -- No kernargs\n";
        }
    }

    return hipExtModuleLaunchKernel_func(f, globalWorkSizeX, globalWorkSizeY, globalWorkSizeZ, localWorkSizeX,
                                         localWorkSizeY, localWorkSizeZ, sharedMemBytes, hStream, kernelParams,
                                         extra, startEvent, stopEvent, flags);
}


// HSA functions are all C linkage
extern "C" {

hsa_status_t hsa_memory_allocate(hsa_region_t region, size_t size, void **ptr)
{
    if (!initialized) init_hooks();

    if (verbose) {
        std::cout << "kl-hook: hsa_memory_allocate called\n\n";
    }

    return hsa_memory_allocate_func(region, size, ptr);
}

hsa_status_t hsa_memory_free(void *ptr)
{
    if (!initialized) init_hooks();

    if (verbose) {
        std::cout << "kl-hook: hsa_memory_free called\n\n";
    }

    return hsa_memory_free_func(ptr);
}

hsa_status_t hsa_memory_assign_agent(void *ptr, hsa_agent_t agent, hsa_access_permission_t access)
{
    if (!initialized) init_hooks();

    if (verbose) {
        std::cout << "kl-hook: hsa_memory_assign_agent called\n\n";
    }

    return hsa_memory_assign_agent_func(ptr, agent, access);
}

hsa_status_t hsa_memory_copy(void *dst, const void *src, size_t size)
{
    if (!initialized) init_hooks();

    if (verbose) {
        std::cout << "kl-hook: hsa_memory_copy called\n\n";
    }

    return hsa_memory_copy_func(dst, src, size);
}

hsa_status_t hsa_amd_memory_pool_allocate(hsa_amd_memory_pool_t memory_pool, size_t size,
                                          uint32_t flags, void** ptr)
{
    if (!initialized) init_hooks();

    if (verbose) {
        std::cout << "kl-hook: hsa_amd_memory_pool_allocate called\n";
    }

    return hsa_amd_memory_pool_allocate_func(memory_pool, size, flags, ptr);
}

hsa_status_t hsa_amd_memory_pool_free(void *ptr)
{
    if (!initialized) init_hooks();

    if (verbose) {
        std::cout << "kl-hook: hsa_amd_memory_pool_free called\n\n";
    }

    return hsa_amd_memory_pool_free_func(ptr);
}

hsa_status_t hsa_amd_memory_async_copy(void* dst, hsa_agent_t dst_agent, const void* src,
                                       hsa_agent_t src_agent, size_t size,
                                       uint32_t num_dep_signals,
                                       const hsa_signal_t* dep_signals,
                                       hsa_signal_t completion_signal)
{
    if (!initialized) init_hooks();

    if (verbose) {
        std::cout << "kl-hook: hsa_amd_memory_async_copy called\n\n";
    }

    return hsa_amd_memory_async_copy_func(dst, dst_agent, src, src_agent, size, num_dep_signals,
                                          dep_signals, completion_signal);
}

hsa_status_t hsa_amd_memory_async_copy_rect(const hsa_pitched_ptr_t* dst, const hsa_dim3_t* dst_offset,
                                            const hsa_pitched_ptr_t* src, const hsa_dim3_t* src_offset,
                                            const hsa_dim3_t* range, hsa_agent_t copy_agent,
                                            hsa_amd_copy_direction_t dir, uint32_t num_dep_signals,
                                            const hsa_signal_t* dep_signals, hsa_signal_t completion_signal)
{
    if (!initialized) init_hooks();

    if (verbose) {
        std::cout << "kl-hook: hsa_amd_memory_async_copy_rect called\n\n";
    }

    return hsa_amd_memory_async_copy_rect_func(dst, dst_offset, src, src_offset, range, copy_agent,
                                               dir, num_dep_signals, dep_signals, completion_signal);
}

}
