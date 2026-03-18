// Mori shmem backend for SymmetricMemory on AMD GPUs.
//
// Implements SymmetricMemoryAllocator and SymmetricMemory using Mori's
// host-side C++ API (shmem_api.hpp). HIP runtime is used for device-side
// pointer array management (get_buffer_ptrs_dev / get_signal_pad_ptrs_dev).
//
// Auto-loaded via torch_hip -> torch_mori dependency chain.
// Set TORCH_SYMMMEM=MORI or call set_backend("MORI") to activate.

#include <torch/csrc/distributed/c10d/GroupRegistry.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/CUDASymmetricMemoryUtils.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/SymmetricMemory.hpp>
#include <c10/hip/HIPCachingAllocator.h>
#include <c10/util/env.h>

#include <mori/shmem/shmem_api.hpp>

#include <hip/hip_runtime_api.h>
#include <mutex>

namespace c10d {
namespace symmetric_memory {

static std::string getMoriBackendEnv() {
  static auto val = c10::utils::get_env("TORCH_SYMMMEM");
  return val.has_value() ? val.value() : "";
}

static StoreExchange storeExchange("MORISymmetricMemory");

// group-local rank → global rank (== mori PE) mapping, cached per group_name
static std::mutex rank_map_mutex;
static std::unordered_map<std::string, std::vector<int>>
    rank_to_global_rank_map{};

struct MORIAllocation {
  void* ptr;
  size_t buffer_size;
  int device_idx;

  MORIAllocation(void* ptr, size_t buffer_size, int device_idx)
      : ptr(ptr), buffer_size(buffer_size), device_idx(device_idx) {}

  MORIAllocation(const MORIAllocation&) = delete;
  MORIAllocation& operator=(const MORIAllocation&) = delete;

  ~MORIAllocation() {
    if (is_finalizing()) {
      return;
    }
    mori::shmem::ShmemFree(ptr);
  }
};

class MORIPeerAllocInfo : public c10::intrusive_ptr_target {
 public:
  MORIPeerAllocInfo(
      MORIAllocation* allocation,
      const std::string& group_name)
      : base_ptr_(allocation->ptr), buffer_size_(allocation->buffer_size) {
    auto group = resolve_process_group(group_name);
    rank_ = group->getRank();
    world_size_ = group->getSize();
    auto store = group->getStore();

    int my_pe = mori::shmem::ShmemMyPe();

    // Exchange rank-to-global-rank mapping for this group (cached per group).
    // For the WORLD group, global_rank == local rank == mori PE.
    // For subgroups, this maps local ranks to the correct mori PEs.
    std::lock_guard<std::mutex> rank_map_lock(rank_map_mutex);
    auto it = rank_to_global_rank_map.find(group_name);
    if (it == rank_to_global_rank_map.end()) {
      auto global_group = resolve_process_group("0");
      auto global_rank = global_group->getRank();
      auto rank_to_global_rank =
          storeExchange.all_gather(store, rank_, world_size_, global_rank);
      it = rank_to_global_rank_map.emplace_hint(
          it, group_name, rank_to_global_rank);
    }
    auto& rank_to_global_rank = it->second;

    world_within_p2p_ = true;
    for (int r = 0; r < world_size_; ++r) {
      uint64_t peer_ptr = mori::shmem::ShmemPtrP2p(
          reinterpret_cast<uint64_t>(base_ptr_), my_pe,
          rank_to_global_rank[r]);
      if (peer_ptr != 0) {
        buffers_.push_back(reinterpret_cast<void*>(peer_ptr));
      } else {
        buffers_.push_back(nullptr);
        world_within_p2p_ = false;
      }
    }

    // Signal pads allocated via mori symmetric heap.
    const size_t signal_pad_size = get_signal_pad_size();
    signal_pad_ptr_ = mori::shmem::ShmemMalloc(signal_pad_size);
    TORCH_CHECK(signal_pad_ptr_ != nullptr, "mori ShmemMalloc failed for signal pad");
    C10_HIP_CHECK(hipMemset(signal_pad_ptr_, 0, signal_pad_size));

    for (int r = 0; r < world_size_; ++r) {
      uint64_t peer_sig = mori::shmem::ShmemPtrP2p(
          reinterpret_cast<uint64_t>(signal_pad_ptr_), my_pe,
          rank_to_global_rank[r]);
      signal_pads_.push_back(reinterpret_cast<void*>(peer_sig));
    }

    // Copy pointer arrays to device memory so GPU kernels can index them
    const size_t arr_size = sizeof(void*) * world_size_;
    buffers_dev_ = reinterpret_cast<void**>(
        c10::cuda::CUDACachingAllocator::raw_alloc(arr_size));
    signal_pads_dev_ = reinterpret_cast<void**>(
        c10::cuda::CUDACachingAllocator::raw_alloc(arr_size));

    C10_HIP_CHECK(hipMemcpy(
        buffers_dev_, buffers_.data(), arr_size, hipMemcpyHostToDevice));
    C10_HIP_CHECK(hipMemcpy(
        signal_pads_dev_,
        signal_pads_.data(),
        arr_size,
        hipMemcpyHostToDevice));
  }

 private:
  void* base_ptr_;
  void* signal_pad_ptr_{nullptr};
  size_t buffer_size_;
  int rank_;
  int world_size_;
  std::vector<void*> buffers_;
  std::vector<void*> signal_pads_;
  void** buffers_dev_{nullptr};
  void** signal_pads_dev_{nullptr};
  bool world_within_p2p_;

  friend class MORISymmetricMemory;
};

class MORISymmetricMemory : public SymmetricMemory {
 public:
  MORISymmetricMemory(
      MORIAllocation* allocation,
      const std::string& group_name)
      : device_idx_(allocation->device_idx), group_name_(group_name) {
    pai_ = c10::make_intrusive<MORIPeerAllocInfo>(allocation, group_name);
    offset_ = 0;
  }

  MORISymmetricMemory(const MORISymmetricMemory& other) = delete;

  MORISymmetricMemory(const MORISymmetricMemory& other, size_t offset)
      : device_idx_(other.device_idx_),
        group_name_(other.group_name_),
        pai_(other.pai_) {
    offset_ = offset;
  }

  ~MORISymmetricMemory() override = default;

  std::vector<void*> get_buffer_ptrs() override {
    return pai_->buffers_;
  }

  std::vector<void*> get_signal_pad_ptrs() override {
    return pai_->signal_pads_;
  }

  void** get_buffer_ptrs_dev() override {
    return pai_->buffers_dev_;
  }

  void** get_signal_pad_ptrs_dev() override {
    return pai_->signal_pads_dev_;
  }

  size_t get_buffer_size() override {
    return pai_->buffer_size_;
  }

  bool has_multicast_support() override {
    return false;
  }

  void* get_multicast_ptr() override {
    return nullptr;
  }

  size_t get_offset() override {
    return offset_;
  }

  void barrier(int channel, size_t timeout_ms) override {
    mori::shmem::ShmemBarrierAll();
  }

  void put_signal(int dst_rank, int channel, size_t timeout_ms) override {
    TORCH_CHECK(false, "MORISymmetricMemory::put_signal not yet implemented");
  }

  void wait_signal(int src_rank, int channel, size_t timeout_ms) override {
    TORCH_CHECK(false, "MORISymmetricMemory::wait_signal not yet implemented");
  }

  int get_rank() override {
    return pai_->rank_;
  }

  int get_world_size() override {
    return pai_->world_size_;
  }

  c10::Device get_device() override {
    return c10::Device(c10::DeviceType::CUDA, device_idx_);
  }

  const std::vector<int>& get_rank_to_global_rank() override {
    std::lock_guard<std::mutex> lock(rank_map_mutex);
    auto it = rank_to_global_rank_map.find(group_name_);
    TORCH_CHECK(
        it != rank_to_global_rank_map.end(),
        "Group name not found in rank_to_global_rank_map");
    return it->second;
  }

  bool world_within_direct_access() override {
    return pai_->world_within_p2p_;
  }

 private:
  int device_idx_;
  std::string group_name_;
  c10::intrusive_ptr<MORIPeerAllocInfo> pai_;
  size_t offset_{0};
};

static void initialize_mori(const std::string& group_name) {
  static bool is_initialized = false;
  if (is_initialized) {
    return;
  }

  // Try the requested group first, fall back to global group "0"
  c10::intrusive_ptr<c10d::ProcessGroup> group;
  try {
    group = resolve_process_group(group_name);
  } catch (...) {
    group = resolve_process_group("0");
  }
  int rank = group->getRank();
  int world_size = group->getSize();
  auto store = group->getStore();

  // Rank 0 generates a unique_id containing bootstrap socket info
  mori::shmem::mori_shmem_uniqueid_t unique_id{};
  if (rank == 0) {
    int ret = mori::shmem::ShmemGetUniqueId(&unique_id);
    TORCH_CHECK(ret == 0, "mori ShmemGetUniqueId failed: ", ret);
  }

  // Broadcast unique_id via the PyTorch distributed store
  const std::string store_key = "mori_shmem_uid";
  if (rank == 0) {
    std::vector<uint8_t> uid_vec(unique_id.begin(), unique_id.end());
    store->set(store_key, uid_vec);
  }
  store->wait({store_key});
  auto uid_vec = store->get(store_key);
  TORCH_CHECK(uid_vec.size() == unique_id.size(),
      "MORI uid size mismatch: got ", uid_vec.size(),
      " expected ", unique_id.size());
  std::copy(uid_vec.begin(), uid_vec.end(), unique_id.begin());

  // Initialize mori with the exchanged unique_id
  mori::shmem::mori_shmem_init_attr_t attr{};
  mori::shmem::ShmemSetAttrUniqueIdArgs(rank, world_size, &unique_id, &attr);

  int ret = mori::shmem::ShmemInitAttr(
      mori::shmem::MORI_SHMEM_INIT_WITH_UNIQUEID, &attr);
  TORCH_CHECK(ret == 0, "mori ShmemInitAttr failed: ", ret);

  is_initialized = true;
  LOG(INFO) << "[MORI] initialized: rank=" << rank
            << " world_size=" << world_size;
}

class MORISymmetricMemoryAllocator : public SymmetricMemoryAllocator {
 public:
  void* alloc(
      size_t size,
      int device_idx,
      const std::optional<std::string>& group_name) override {
    initialize_mori(group_name.value_or("default"));

    auto ptr = mori::shmem::ShmemMalloc(size);
    TORCH_CHECK(ptr != nullptr || size == 0, "mori ShmemMalloc failed");
    {
      std::lock_guard<std::mutex> lock(mutex_);
      allocations_.try_emplace(
          ptr, std::make_unique<MORIAllocation>(ptr, size, device_idx));
      // Remember the group_name used during alloc so rendezvous can use it
      // when group_name is not explicitly provided
      if (group_name.has_value()) {
        last_group_name_ = *group_name;
      }
    }
    return ptr;
  }

  void free(void* ptr) override {
    std::lock_guard<std::mutex> lock(mutex_);
    allocations_.erase(ptr);
  }

  size_t get_alloc_size(void* ptr) override {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = allocations_.find(ptr);
    TORCH_CHECK(
        it != allocations_.end(),
        ptr, " is not allocated with MORISymmetricMemoryAllocator");
    return it->second->buffer_size;
  }

  c10::intrusive_ptr<SymmetricMemory> rendezvous(
      void* ptr,
      const std::optional<std::string>& group_name) override {
    std::string gn = group_name.value_or(last_group_name_);
    std::lock_guard<std::mutex> lock(mutex_);
    {
      auto it = symm_mems_.find(std::make_tuple(ptr, gn));
      if (it != symm_mems_.end()) {
        return it->second;
      }
    }

    auto alloc_it = std::find_if(
        allocations_.begin(), allocations_.end(), [&](const auto& pair) {
          auto& allocation = pair.second;
          auto ptr_int = reinterpret_cast<uintptr_t>(ptr);
          auto base_ptr = reinterpret_cast<uintptr_t>(allocation->ptr);
          return ptr_int >= base_ptr &&
              ptr_int < base_ptr + allocation->buffer_size;
        });
    if (alloc_it == allocations_.end()) {
      TORCH_WARN(
          "Pointer not within any SymmetricMemory allocation, "
          "is the tensor allocated from SymmetricMemory?");
      return nullptr;
    }

    auto& allocation = alloc_it->second;

    auto it = symm_mems_.find(std::make_tuple(allocation->ptr, gn));
    c10::intrusive_ptr<MORISymmetricMemory> symm_mem;
    if (it != symm_mems_.end()) {
      symm_mem = it->second;
    } else {
      symm_mem = c10::make_intrusive<MORISymmetricMemory>(
          allocation.get(), gn);
    }

    symm_mems_[std::make_tuple(allocation->ptr, gn)] = symm_mem;

    if (ptr == allocation->ptr) {
      return symm_mem;
    } else {
      return c10::make_intrusive<MORISymmetricMemory>(
          *symm_mem, (uintptr_t)ptr - (uintptr_t)allocation->ptr);
    }
  }

  bool has_multicast_support(int device_idx) override {
    return false;
  }

  c10::DeviceType supported_device_type() override {
    return c10::DeviceType::CUDA;
  }

  std::string name() override {
    return "MORI";
  }

 private:
  std::mutex mutex_;
  std::string last_group_name_{"default"};
  std::unordered_map<void*, std::unique_ptr<MORIAllocation>> allocations_;
  std::map<
      std::tuple<void*, std::string>,
      c10::intrusive_ptr<MORISymmetricMemory>>
      symm_mems_;
};

struct RegisterMORISymmetricMemoryAllocator {
  RegisterMORISymmetricMemoryAllocator() {
    auto allocator = c10::make_intrusive<MORISymmetricMemoryAllocator>();
    register_availability("MORI", allocator);
    if (getMoriBackendEnv() == "MORI") {
      register_allocator(c10::DeviceType::CUDA, allocator);
    }
  }
};

__attribute__((used))
static RegisterMORISymmetricMemoryAllocator register_allocator_;

} // namespace symmetric_memory
} // namespace c10d
