#pragma once

#include <torch/csrc/distributed/c10d/symm_mem/SymmetricMemory.hpp>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

// The "MORI" SymmetricMemory backend. Allocations are HIP VMM memory owned by
// this backend; MORI-CCO maps them into the peers of each group at rendezvous,
// over its C ABI, and gives every (allocation, group) its own signal pad.
// Every rank of a group must be in one load/store (LSA) team: one node, or one
// fabric domain.
namespace c10d::symmetric_memory::mori {

struct PeerAlloc;

class MoriSymmetricMemory : public SymmetricMemory {
 public:
  MoriSymmetricMemory(std::shared_ptr<PeerAlloc> peer, size_t offset);

  std::vector<void*> get_buffer_ptrs() override;
  std::vector<void*> get_signal_pad_ptrs() override;
  void** get_buffer_ptrs_dev() override;
  void** get_signal_pad_ptrs_dev() override;
  size_t get_buffer_size() override;
  size_t get_offset() override;

  bool has_multicast_support() override;
  void* get_multicast_ptr() override;

  void barrier(int channel, size_t timeout_ms) override;
  void put_signal(int dst_rank, int channel, size_t timeout_ms) override;
  void wait_signal(int src_rank, int channel, size_t timeout_ms) override;

  int get_rank() override;
  int get_world_size() override;
  c10::Device get_device() override;
  std::string get_group_name() override;
  const std::vector<int>& get_rank_to_global_rank() override;
  int* get_rank_to_global_rank_dev() override;
  bool world_within_direct_access() override;

  // MORI-CCO device windows (mori::cco::ccoWindow_t) of the buffer and of the
  // signal pad, for kernels written against MORI's device API.
  void* get_window();
  void* get_signal_pad_window();

  // A device communicator (mori::cco::ccoDevComm*) of the group for this
  // rendezvous, created on first use and cached by lsa_barrier_count; it lives
  // as long as the rendezvous. Collective on first use.
  void* get_dev_comm(int lsa_barrier_count);

 private:
  std::shared_ptr<PeerAlloc> peer_;
  size_t offset_;
};

// Registers the "MORI" backend. Idempotent. libmori_cco must be loaded.
void register_mori_backend();

// Releases every allocation and communicator; registered with atexit, while
// Python and HIP are still up.
void shutdown_mori_backend();

// Communicators currently alive.
int mori_live_comms();

} // namespace c10d::symmetric_memory::mori
