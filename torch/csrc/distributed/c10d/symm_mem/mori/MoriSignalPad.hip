#include <torch/csrc/distributed/c10d/symm_mem/mori/MoriSignalPad.hpp>

#include <c10/hip/HIPException.h>
#include <torch/csrc/distributed/c10d/symm_mem/SymmetricMemory.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/CUDASymmetricMemory-inl.cuh>

#include <algorithm>

namespace c10d::symmetric_memory::mori {

namespace {

uint32_t** pads(void** signal_pads_dev) {
  return reinterpret_cast<uint32_t**>(signal_pads_dev);
}

// One wavefront, as torch launches these, and at least one thread per rank.
int block_threads(int world_size) {
  int device = 0;
  int warp_size = 0;
  C10_HIP_CHECK(hipGetDevice(&device));
  C10_HIP_CHECK(
      hipDeviceGetAttribute(&warp_size, hipDeviceAttributeWarpSize, device));
  return std::max(warp_size, world_size);
}

} // namespace

void launch_barrier(
    void** signal_pads_dev,
    size_t signal_pad_size,
    int channel,
    int rank,
    int world_size,
    size_t timeout_ms,
    hipStream_t stream) {
  check_channel(channel, world_size, signal_pad_size);
  barrier_kernel<<<1, block_threads(world_size), 0, stream>>>(
      pads(signal_pads_dev), channel, rank, world_size, timeout_ms);
  C10_HIP_CHECK(hipGetLastError());
}

void launch_put_signal(
    void** signal_pads_dev,
    size_t signal_pad_size,
    int dst_rank,
    int channel,
    int rank,
    int world_size,
    size_t timeout_ms,
    hipStream_t stream) {
  check_channel(channel, world_size, signal_pad_size);
  check_rank(dst_rank, world_size);
  put_signal_kernel<<<1, block_threads(1), 0, stream>>>(
      pads(signal_pads_dev), dst_rank, channel, rank, world_size, timeout_ms);
  C10_HIP_CHECK(hipGetLastError());
}

void launch_wait_signal(
    void** signal_pads_dev,
    size_t signal_pad_size,
    int src_rank,
    int channel,
    int rank,
    int world_size,
    size_t timeout_ms,
    hipStream_t stream) {
  check_channel(channel, world_size, signal_pad_size);
  check_rank(src_rank, world_size);
  wait_signal_kernel<<<1, block_threads(1), 0, stream>>>(
      pads(signal_pads_dev), src_rank, channel, rank, world_size, timeout_ms);
  C10_HIP_CHECK(hipGetLastError());
}

} // namespace c10d::symmetric_memory::mori
