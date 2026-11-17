#pragma once

#include <hip/hip_runtime.h>

#include <cstddef>

// Launchers of torch's own signal-pad kernels (CUDASymmetricMemory-inl.cuh),
// compiled into this extension so that the MORI backend follows the same CAS
// protocol as torch's built-in symm_mem ops without touching torch_hip.
namespace c10d::symmetric_memory::mori {

void launch_barrier(
    void** signal_pads_dev,
    size_t signal_pad_size,
    int channel,
    int rank,
    int world_size,
    size_t timeout_ms,
    hipStream_t stream);

void launch_put_signal(
    void** signal_pads_dev,
    size_t signal_pad_size,
    int dst_rank,
    int channel,
    int rank,
    int world_size,
    size_t timeout_ms,
    hipStream_t stream);

void launch_wait_signal(
    void** signal_pads_dev,
    size_t signal_pad_size,
    int src_rank,
    int channel,
    int rank,
    int world_size,
    size_t timeout_ms,
    hipStream_t stream);

} // namespace c10d::symmetric_memory::mori
