#include <torch/csrc/distributed/c10d/symm_mem/mori/MoriSymmetricMemory.hpp>

#include <c10/hip/HIPException.h>
#include <c10/hip/HIPStream.h>
#include <c10/util/env.h>
#include <c10/util/irange.h>
#include <hip/hip_runtime.h>
#include <torch/csrc/distributed/c10d/GroupRegistry.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/GroupStreamGuard.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/mori/MoriCco.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/mori/MoriSignalPad.hpp>

#include <unistd.h>
#include <algorithm>
#include <atomic>
#include <climits>
#include <cstring>
#include <map>
#include <mutex>
#include <optional>
#include <random>
#include <set>
#include <unordered_map>
#include <utility>

namespace c10d::symmetric_memory::mori {

namespace {

std::atomic<int> live_comms{0};

// Teardown can run while HIP is already unwinding, when any VMM call faults.
// Leaking is deliberate there: the process is exiting.
bool runtime_usable() {
  if (is_finalizing()) {
    return false;
  }
  int device = -1;
  return hipGetDevice(&device) == hipSuccess;
}

// Never throws, so it can be used in destructors; a failed switch surfaces
// in the HIP calls that follow.
class DeviceGuard {
 public:
  explicit DeviceGuard(int device) {
    if (hipGetDevice(&prev_) != hipSuccess) {
      prev_ = -1;
    }
    if (prev_ != device) {
      restore_ = hipSetDevice(device) == hipSuccess && prev_ >= 0;
    }
  }
  ~DeviceGuard() {
    if (restore_) {
      (void)hipSetDevice(prev_);
    }
  }
  DeviceGuard(const DeviceGuard&) = delete;
  DeviceGuard& operator=(const DeviceGuard&) = delete;

 private:
  int prev_ = -1;
  bool restore_ = false;
};

#ifndef TORCH_MORI_ARCHS
#error "TORCH_MORI_ARCHS must list the archs torch._mori is built for"
#endif

// The signal-pad kernels exist only for the archs torch._mori is built for.
void check_device_arch(int device) {
  static std::mutex mutex;
  static std::set<int> checked;
  std::lock_guard<std::mutex> lock(mutex);
  if (checked.count(device)) {
    return;
  }
  hipDeviceProp_t prop{};
  C10_HIP_CHECK(hipGetDeviceProperties(&prop, device));
  std::string arch(prop.gcnArchName);
  arch = arch.substr(0, arch.find(':'));
  const std::string built = TORCH_MORI_ARCHS;
  TORCH_CHECK(
      ("," + built + ",").find("," + arch + ",") != std::string::npos,
      "MORI symmetric memory: device ",
      device,
      " is ",
      arch,
      ", but this torch._mori was built for ",
      built,
      ". MORI supports gfx942, gfx950 and gfx1250; rebuild with them in "
      "PYTORCH_ROCM_ARCH.");
  checked.insert(device);
}

hipStream_t current_stream(int device) {
  return c10::hip::getCurrentHIPStream(static_cast<c10::DeviceIndex>(device))
      .stream();
}

// alloc() and a first rendezvous synchronize and use the store.
void check_not_capturing(int device, const char* op, const char* hint) {
  hipStreamCaptureStatus status = hipStreamCaptureStatusNone;
  const hipError_t err = hipStreamIsCapturing(current_stream(device), &status);
  if (err != hipSuccess) {
    (void)hipGetLastError(); // e.g. the legacy stream under a global capture
  }
  TORCH_CHECK(
      err == hipSuccess && status == hipStreamCaptureStatusNone,
      "MORI symmetric memory: ",
      op,
      " is not supported while the current stream is capturing a graph. ",
      hint);
}

// A store all-gather counted per store, so ranks outside a group's
// rendezvous never desync the others.
class StoreExchange {
 public:
  explicit StoreExchange(std::string prefix) : prefix_(std::move(prefix)) {}

  template <typename T>
  std::vector<T> all_gather(
      const c10::intrusive_ptr<Store>& store,
      int rank,
      int world_size,
      const T& val) {
    static_assert(std::is_trivially_copyable_v<T>);
    const std::string base =
        prefix_ + "/" + std::to_string(next_seq(store)) + "/";
    std::vector<std::string> keys;
    keys.reserve(world_size);
    for (const auto r : c10::irange(world_size)) {
      keys.push_back(base + std::to_string(r));
    }
    const auto* bytes = reinterpret_cast<const uint8_t*>(&val);
    store->set(keys[rank], std::vector<uint8_t>(bytes, bytes + sizeof(T)));
    auto payloads = store->multiGet(keys);
    std::vector<T> out(world_size);
    for (const auto r : c10::irange(world_size)) {
      TORCH_CHECK(
          payloads[r].size() == sizeof(T),
          "MORI symmetric memory: rank ",
          r,
          " sent a rendezvous payload of the wrong size");
      std::memcpy(&out[r], payloads[r].data(), sizeof(T));
    }
    return out;
  }

 private:
  // Keyed by address; the weak reference keeps a live store's address from
  // being reused, and dead stores are pruned here.
  size_t next_seq(const c10::intrusive_ptr<Store>& store) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto it = seqs_.begin(); it != seqs_.end();) {
      it = it->second.ref.expired() ? seqs_.erase(it) : std::next(it);
    }
    const auto key = reinterpret_cast<uintptr_t>(store.get());
    auto it = seqs_.find(key);
    if (it == seqs_.end()) {
      it = seqs_.emplace(key, Entry{c10::weak_intrusive_ptr<Store>(store), 0})
               .first;
    }
    return it->second.seq++;
  }

  struct Entry {
    c10::weak_intrusive_ptr<Store> ref;
    size_t seq;
  };
  const std::string prefix_;
  std::mutex mutex_;
  std::unordered_map<uintptr_t, Entry> seqs_;
};

// Fabric handles, spelled out: older HIP headers lack the names.
constexpr auto kHandleTypeFabric = static_cast<hipMemAllocationHandleType>(0x8);
struct FabricHandle {
  unsigned char data[64];
};

hipMemAllocationProp make_prop(
    int device,
    hipMemAllocationHandleType handle_type) {
  hipMemAllocationProp prop = {};
  prop.type = hipMemAllocationTypePinned;
  prop.requestedHandleType = handle_type;
  prop.location.type = hipMemLocationTypeDevice;
  prop.location.id = device;
  return prop;
}

size_t granularity(const hipMemAllocationProp& prop) {
  size_t gran = 0;
  C10_HIP_CHECK(hipMemGetAllocationGranularity(
      &gran, &prop, hipMemAllocationGranularityRecommended));
  return gran;
}

// The handle type MORI-CCO exports windows with: fabric where the device can
// export it, else a POSIX fd. Probed the same way, once per device, since
// allocations precede the first communicator.
hipMemAllocationHandleType probe_handle_type(int device) {
  static std::mutex mutex;
  static std::unordered_map<int, hipMemAllocationHandleType> cache;
  std::lock_guard<std::mutex> lock(mutex);
  if (auto it = cache.find(device); it != cache.end()) {
    return it->second;
  }
  auto chosen = hipMemHandleTypePosixFileDescriptor;
  auto prop = make_prop(device, kHandleTypeFabric);
  size_t gran = 0;
  hipMemGenericAllocationHandle_t handle{};
  if (hipMemGetAllocationGranularity(
          &gran, &prop, hipMemAllocationGranularityRecommended) == hipSuccess &&
      gran > 0 && hipMemCreate(&handle, gran, &prop, 0) == hipSuccess) {
    FabricHandle blob;
    if (hipMemExportToShareableHandle(&blob, handle, kHandleTypeFabric, 0) ==
        hipSuccess) {
      chosen = kHandleTypeFabric;
    }
    (void)hipMemRelease(handle);
  }
  (void)hipGetLastError(); // the probe fails where fabric is unsupported
  cache[device] = chosen;
  return chosen;
}

// A mapped HIP VMM allocation, released explicitly.
struct VmmMemory {
  void* ptr = nullptr;
  size_t size = 0;
  int device = -1;
  hipMemGenericAllocationHandle_t handle{};
  bool live = false;

  void create(size_t bytes, int dev, hipMemAllocationHandleType handle_type) {
    device = dev;
    auto prop = make_prop(dev, handle_type);
    const size_t gran = granularity(prop);
    size = (std::max<size_t>(bytes, 1) + gran - 1) / gran * gran;
    C10_HIP_CHECK(hipMemCreate(&handle, size, &prop, 0));
    bool reserved = false;
    bool mapped = false;
    try {
      C10_HIP_CHECK(hipMemAddressReserve(&ptr, size, gran, nullptr, 0));
      reserved = true;
      C10_HIP_CHECK(hipMemMap(ptr, size, 0, handle, 0));
      mapped = true;
      hipMemAccessDesc desc = {};
      desc.location.type = hipMemLocationTypeDevice;
      desc.location.id = dev;
      desc.flags = hipMemAccessFlagsProtReadWrite;
      C10_HIP_CHECK(hipMemSetAccess(ptr, size, &desc, 1));
    } catch (...) {
      if (mapped) {
        (void)hipMemUnmap(ptr, size);
      }
      if (reserved) {
        (void)hipMemAddressFree(ptr, size);
      }
      (void)hipMemRelease(handle);
      throw;
    }
    live = true;
  }

  void release() noexcept {
    if (!live) {
      return;
    }
    live = false;
    (void)hipMemUnmap(ptr, size);
    (void)hipMemAddressFree(ptr, size);
    (void)hipMemRelease(handle);
  }
};

// Flat VA per rank that a communicator reserves for each LSA peer: the device
// memory size (0) for teams of up to 8, else a share of a fixed budget so a
// large team does not exhaust the user address space.
size_t per_rank_va(size_t device_mem, int lsa_bound) {
  constexpr int kUncappedTeam = 8;
  constexpr size_t k4GiB = size_t{1} << 32;
  constexpr size_t kBudget = size_t{4} << 40;
  if (lsa_bound <= kUncappedTeam) {
    return 0;
  }
  return std::max(
      k4GiB, std::min(device_mem, kBudget / lsa_bound) / k4GiB * k4GiB);
}

// What every rank publishes on the first rendezvous of an allocation with a
// group.
struct RendezvousInfo {
  uint64_t alloc_size;
  uint64_t buffer_size;
  uint64_t comm_epoch; // this rank's communicator for the group, 0 if none
  uint64_t device_mem;
  int32_t device_idx;
  int32_t global_rank; // rank in the default group, -1 without one
  int32_t handle_type;
  char host[HOST_NAME_MAX + 1];
};

bool group_registered(const std::string& name) {
  try {
    (void)resolve_process_group(name);
    return true;
  } catch (const std::exception&) {
    return false;
  }
}

// The default group is named "0".
int global_rank() {
  try {
    return resolve_process_group("0")->getRank();
  } catch (const std::exception&) {
    return -1;
  }
}

void sweep_dead_groups() noexcept;

} // namespace

// One communicator per (process group, device). It lives while the group is
// registered or an allocation is mapped through it.
struct CommGroup {
  moriCcoComm_t comm = nullptr;
  std::string name;
  int device = -1;
  int rank = -1;
  int world_size = 0;
  int lsa_size = 0; // ranks in this rank's load/store team
  int mem_handle_type = 0;
  // Drawn by rank 0; a communicator is reused only if all ranks hold it.
  uint64_t epoch = 0;
  std::vector<int> rank_to_global_rank;
  int* rank_to_global_rank_dev = nullptr;

  CommGroup() = default;
  CommGroup(const CommGroup&) = delete;
  CommGroup& operator=(const CommGroup&) = delete;

  ~CommGroup() {
    if (comm == nullptr) {
      return;
    }
    live_comms.fetch_sub(1);
    if (!runtime_usable()) {
      return;
    }
    DeviceGuard guard(device);
    (void)hipDeviceSynchronize();
    if (rank_to_global_rank_dev) {
      (void)hipFree(rank_to_global_rank_dev);
    }
    (void)mori_cco().commDestroy(comm);
  }
};

// An allocation's memory, shared by its block and the rendezvous handles, so
// a handle keeps it mapped after its tensor is freed. The windows importing
// it hold their own reference to it.
struct BlockMemory {
  VmmMemory memory;
  std::mutex mutex; // guards windows
  std::vector<std::pair<std::shared_ptr<CommGroup>, moriCcoWindow_t>> windows;

  BlockMemory() = default;
  BlockMemory(const BlockMemory&) = delete;
  BlockMemory& operator=(const BlockMemory&) = delete;

  void add_window(std::shared_ptr<CommGroup> group, moriCcoWindow_t window) {
    std::lock_guard<std::mutex> lock(mutex);
    windows.emplace_back(std::move(group), window);
  }

  ~BlockMemory() {
    if (runtime_usable()) {
      DeviceGuard guard(memory.device);
      // A symmetric tensor may die with kernels still using it, and
      // unmapping under them faults.
      (void)hipDeviceSynchronize();
      memory.release();
      for (auto& [group, window] : windows) {
        (void)mori_cco().windowRelease(window);
      }
    }
    windows.clear();
  }
};

// One rendezvous of an allocation with a group: its own signal pad, the peer
// tables and device communicators. Shared by the handles of that rendezvous.
struct PeerAlloc {
  std::shared_ptr<CommGroup> group;
  std::shared_ptr<BlockMemory> memory; // its window is released with it
  void* device_window = nullptr;
  VmmMemory pad;
  moriCcoWindow_t pad_window = nullptr;
  void* pad_device_window = nullptr;
  size_t pad_size = 0;
  size_t buffer_size = 0;
  int device = -1;
  std::vector<void*> buffers;
  std::vector<void*> pads;
  void** tables_dev = nullptr; // buffers, then pads
  // Created after this rendezvous's windows, so they know them, and
  // destroyed before those windows are released.
  std::mutex mutex; // guards dev_comms
  std::map<int, void*> dev_comms; // by lsa_barrier_count

  PeerAlloc() = default;
  PeerAlloc(const PeerAlloc&) = delete;
  PeerAlloc& operator=(const PeerAlloc&) = delete;

  ~PeerAlloc() {
    if (runtime_usable()) {
      DeviceGuard guard(device);
      // A handle may die with work in flight on its pad or buffer.
      (void)hipDeviceSynchronize();
      for (auto& [count, dev_comm] : dev_comms) {
        (void)mori_cco().devCommDestroy(group->comm, dev_comm);
      }
      if (tables_dev) {
        (void)hipFree(tables_dev);
      }
      // Our own handle first, as in ~BlockMemory.
      pad.release();
      if (pad_window) {
        (void)mori_cco().windowRelease(pad_window);
      }
    }
    memory.reset();
    group.reset();
    // A handle outliving its process group holds the last communicator ref.
    sweep_dead_groups();
  }
};

namespace {

struct Block {
  std::shared_ptr<BlockMemory> memory = std::make_shared<BlockMemory>();
  size_t buffer_size = 0;
  hipMemAllocationHandleType handle_type = hipMemHandleTypePosixFileDescriptor;
  std::optional<std::string> default_group_name;
  // Group name -> rendezvous. Guarded by the allocator's mutex.
  std::unordered_map<std::string, std::shared_ptr<PeerAlloc>> peers;

  // Unmaps now unless a rendezvous handle still holds the memory.
  void release() {
    peers.clear();
    memory.reset();
  }
};

class MoriSymmetricMemoryAllocator : public SymmetricMemoryAllocator {
 public:
  void* alloc(
      size_t size,
      int device_idx,
      const std::optional<std::string>& group_name) override {
    if (device_idx < 0) {
      C10_HIP_CHECK(hipGetDevice(&device_idx));
    }
    DeviceGuard guard(device_idx);
    check_device_arch(device_idx);
    check_not_capturing(
        device_idx,
        "alloc",
        "Allocate with torch.distributed._symmetric_memory.empty() before "
        "starting capture.");
    sweep();
    auto block = std::make_shared<Block>();
    block->handle_type = probe_handle_type(device_idx);
    block->memory->memory.create(size, device_idx, block->handle_type);
    block->buffer_size = size;
    block->default_group_name = group_name;
    void* ptr = block->memory->memory.ptr;
    std::lock_guard<std::mutex> lock(mutex_);
    blocks_[reinterpret_cast<uintptr_t>(ptr)] = std::move(block);
    return ptr;
  }

  void free(void* ptr) override {
    std::shared_ptr<Block> block;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      auto it = blocks_.find(reinterpret_cast<uintptr_t>(ptr));
      if (it == blocks_.end()) {
        return;
      }
      block = std::move(it->second);
      blocks_.erase(it);
    }
    block->release();
    sweep();
  }

  size_t get_alloc_size(void* ptr) override {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = blocks_.find(reinterpret_cast<uintptr_t>(ptr));
    TORCH_CHECK(
        it != blocks_.end(),
        "MORI symmetric memory: the pointer is not a MORI allocation");
    return it->second->buffer_size;
  }

  // Interior pointers count: a MemPool hands out tensors inside allocations.
  bool has_allocation(void* ptr) override {
    return find_covering(ptr, nullptr) != nullptr;
  }

  c10::intrusive_ptr<SymmetricMemory> rendezvous(
      void* ptr,
      const std::optional<std::string>& group_name) override {
    size_t offset = 0;
    auto block = find_covering(ptr, &offset);
    if (block == nullptr) {
      // Low-contention ops probe arbitrary tensors this way.
      return nullptr;
    }
    std::string name;
    if (group_name.has_value() && !group_name->empty()) {
      name = *group_name;
    } else {
      TORCH_CHECK(
          block->default_group_name.has_value() &&
              !block->default_group_name->empty(),
          "MORI symmetric memory: no group name was given at allocation or "
          "rendezvous");
      name = *block->default_group_name;
    }
    {
      std::lock_guard<std::mutex> lock(mutex_);
      auto it = block->peers.find(name);
      if (it != block->peers.end()) {
        return c10::make_intrusive<MoriSymmetricMemory>(it->second, offset);
      }
    }
    auto peer = first_rendezvous(*block, name);
    {
      std::lock_guard<std::mutex> lock(mutex_);
      block->peers.emplace(name, peer);
    }
    return c10::make_intrusive<MoriSymmetricMemory>(std::move(peer), offset);
  }

  bool has_multicast_support(int /*device_idx*/) override {
    return false;
  }

  c10::DeviceType supported_device_type() override {
    return c10::DeviceType::CUDA;
  }

  std::string name() override {
    return "MORI";
  }

  void shutdown() {
    std::map<uintptr_t, std::shared_ptr<Block>> blocks;
    std::map<GroupKey, std::shared_ptr<CommGroup>> groups;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      blocks.swap(blocks_);
      groups.swap(groups_);
    }
    for (auto& [addr, block] : blocks) {
      block->release();
    }
  }

  // Drops the communicators of groups no longer registered; each dies with
  // the last allocation mapped through it.
  void sweep() {
    std::vector<GroupKey> keys;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      for (const auto& [key, group] : groups_) {
        keys.push_back(key);
      }
    }
    std::vector<std::shared_ptr<CommGroup>> dead;
    for (const auto& key : keys) {
      if (group_registered(key.first)) {
        continue;
      }
      std::lock_guard<std::mutex> lock(mutex_);
      auto it = groups_.find(key);
      if (it != groups_.end()) {
        dead.push_back(std::move(it->second));
        groups_.erase(it);
      }
    }
  }

 private:
  using GroupKey = std::pair<std::string, int>; // (group name, device)

  std::shared_ptr<Block> find_covering(void* ptr, size_t* offset) {
    const auto p = reinterpret_cast<uintptr_t>(ptr);
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = blocks_.upper_bound(p);
    if (it == blocks_.begin()) {
      return nullptr;
    }
    --it;
    const size_t off = p - it->first;
    if (off != 0 && off >= it->second->buffer_size) {
      return nullptr;
    }
    if (offset) {
      *offset = off;
    }
    return it->second;
  }

  // The group's metadata exchange: its Store, or its own all-gather when the
  // group sets use_pg_for_symm_mem_rendezvous, which scales better.
  template <typename T>
  std::vector<T> all_gather(
      const c10::intrusive_ptr<ProcessGroup>& pg,
      int dev,
      const T& val) {
    const int world_size = pg->getSize();
    if (!pg->hasBackendForDeviceType(c10::DeviceType::CUDA) ||
        !pg->getBackend(c10::DeviceType::CUDA)
             ->getUsePgForSymmMemRendezvous()) {
      return exchange_.all_gather(
          pg->getStore(), pg->getRank(), world_size, val);
    }
    const c10::Device device(
        c10::DeviceType::CUDA, static_cast<c10::DeviceIndex>(dev));
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
    auto in = at::from_blob(
                  const_cast<T*>(&val),
                  {static_cast<int64_t>(sizeof(T))},
                  at::TensorOptions().dtype(at::kByte))
                  .to(device);
    auto out =
        at::empty({static_cast<int64_t>(world_size * sizeof(T))}, in.options());
    AllgatherOptions opts;
    opts.asyncOp = false;
    // Synchronous on the current stream; some backends return no Work for it.
    if (auto work = pg->all_gather_single(out, in, opts)) {
      work->wait();
    }
    const auto host = out.cpu();
    std::vector<T> vals(world_size);
    std::memcpy(vals.data(), host.data_ptr(), world_size * sizeof(T));
    return vals;
  }

  std::shared_ptr<CommGroup> lookup_group(const std::string& name, int dev) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = groups_.find({name, dev});
    return it == groups_.end() ? nullptr : it->second;
  }

  // Collective over the group: maps `block` into the group's communicator,
  // creating one unless every rank already shares one.
  std::shared_ptr<PeerAlloc> first_rendezvous(
      const Block& block,
      const std::string& name) {
    const int dev = block.memory->memory.device;
    DeviceGuard guard(dev);
    check_not_capturing(
        dev,
        "the first rendezvous of a tensor with a group",
        "Rendezvous once before starting capture; later calls on the same "
        "tensor and group return the cached handle.");
    sweep();
    auto pg = resolve_process_group(name);
    const int rank = pg->getRank();
    const int world_size = pg->getSize();

    auto cached = lookup_group(name, dev);
    if (cached && (cached->rank != rank || cached->world_size != world_size)) {
      cached = nullptr;
    }

    RendezvousInfo local{};
    local.alloc_size = block.memory->memory.size;
    local.buffer_size = block.buffer_size;
    local.comm_epoch = cached ? cached->epoch : 0;
    size_t device_mem = 0;
    C10_HIP_CHECK(hipDeviceTotalMem(&device_mem, dev));
    local.device_mem = device_mem;
    local.device_idx = dev;
    local.global_rank = global_rank();
    local.handle_type = static_cast<int32_t>(block.handle_type);
    (void)gethostname(local.host, HOST_NAME_MAX);
    auto infos = all_gather(pg, dev, local);

    // Window import is collective and would hang or mismap on a mismatch.
    for (const auto r : c10::irange(world_size)) {
      TORCH_CHECK(
          infos[r].alloc_size == local.alloc_size &&
              infos[r].buffer_size == local.buffer_size,
          "MORI symmetric memory: rank ",
          r,
          " allocated ",
          infos[r].buffer_size,
          " bytes but rank ",
          rank,
          " allocated ",
          local.buffer_size,
          "; allocations must be the same size on all ranks");
      TORCH_CHECK(
          infos[r].handle_type == local.handle_type,
          "MORI symmetric memory: ranks ",
          r,
          " and ",
          rank,
          " export memory with different handle types");
    }
    if (c10::utils::check_env("TORCH_SYMM_MEM_ALLOW_OVERLAPPING_DEVICES") !=
        true) {
      std::set<std::pair<std::string, int>> devices;
      for (const auto& info : infos) {
        devices.emplace(info.host, info.device_idx);
      }
      TORCH_CHECK(
          devices.size() == static_cast<size_t>(world_size),
          "MORI symmetric memory: rendezvous detected allocations from "
          "overlapping devices from different ranks");
    }

    bool reuse = cached != nullptr;
    for (const auto& info : infos) {
      reuse = reuse && info.comm_epoch == cached->epoch;
    }
    if (!reuse) {
      cached.reset();
    }
    auto group = reuse ? cached : create_group(name, dev, pg, infos);
    // torch's ops load from every peer. lsaSize is the same on every rank, so
    // all ranks fail here alike, before the collective window imports.
    TORCH_CHECK(
        group->lsa_size == world_size,
        "MORI symmetric memory: group ",
        name,
        " has ",
        world_size,
        " ranks, but only ",
        group->lsa_size,
        " of them share a load/store team; the MORI backend needs every rank "
        "of a group on one node or in one fabric domain");
    TORCH_CHECK(
        group->mem_handle_type == static_cast<int>(block.handle_type),
        "MORI symmetric memory: the communicator exports handle type ",
        group->mem_handle_type,
        " but the allocation was created with ",
        static_cast<int>(block.handle_type));
    return map_block(group, block);
  }

  // Collective: a new communicator for (name, dev), replacing this rank's.
  std::shared_ptr<CommGroup> create_group(
      const std::string& name,
      int dev,
      const c10::intrusive_ptr<ProcessGroup>& pg,
      const std::vector<RendezvousInfo>& infos) {
    const int rank = pg->getRank();
    const int world_size = pg->getSize();
    {
      // Released first, so its address space is back before the new one.
      std::shared_ptr<CommGroup> old;
      std::lock_guard<std::mutex> lock(mutex_);
      auto it = groups_.find({name, dev});
      if (it != groups_.end()) {
        old = std::move(it->second);
        groups_.erase(it);
      }
    }

    // The LSA team is bounded by the ranks per host, or by the group where a
    // fabric spans hosts.
    size_t device_mem = 0;
    std::map<std::string, int> per_host;
    bool fabric = false;
    for (const auto& info : infos) {
      device_mem = std::max<size_t>(device_mem, info.device_mem);
      per_host[info.host]++;
      fabric = fabric || info.handle_type == kHandleTypeFabric;
    }
    int lsa_bound = world_size;
    if (!fabric) {
      lsa_bound = 0;
      for (const auto& [host, n] : per_host) {
        lsa_bound = std::max(lsa_bound, n);
      }
    }

    struct CommId {
      moriCcoUniqueId_t uid;
      uint64_t epoch;
    };
    CommId mine{};
    const auto& api = mori_cco();
    if (rank == 0) {
      C10D_MORI_CHECK(api.getUniqueId(&mine.uid));
      std::random_device rd;
      mine.epoch = ((static_cast<uint64_t>(rd()) << 32) | rd()) | 1;
    }
    // The unique id travels the same way as the rendezvous metadata.
    auto ids = all_gather(pg, dev, mine);

    auto group = std::make_shared<CommGroup>();
    group->name = name;
    group->device = dev;
    group->rank = rank;
    group->world_size = world_size;
    group->epoch = ids[0].epoch;
    moriCcoCommConfig_t config = MORI_CCO_COMM_CONFIG_INIT;
    // No CPU pinning of the caller's thread; intra-node peers only.
    config.flags = MORI_CCO_COMM_NO_CPU_AFFINITY | MORI_CCO_COMM_LSA_ONLY;
    config.perRankVaBytes = per_rank_va(device_mem, lsa_bound);
    config.device = dev;
    C10D_MORI_CHECK(
        api.commCreate(&ids[0].uid, world_size, rank, &config, &group->comm));
    live_comms.fetch_add(1);
    moriCcoCommInfo_t info{};
    info.size = sizeof(info);
    C10D_MORI_CHECK(api.commGetInfo(group->comm, &info));
    group->mem_handle_type = info.memHandleType;
    group->lsa_size = info.lsaSize;

    bool have_global = true;
    for (const auto& i : infos) {
      have_global = have_global && i.global_rank >= 0;
    }
    group->rank_to_global_rank.resize(world_size);
    for (const auto r : c10::irange(world_size)) {
      group->rank_to_global_rank[r] = have_global ? infos[r].global_rank : r;
    }
    const size_t bytes = world_size * sizeof(int);
    C10_HIP_CHECK(hipMalloc(&group->rank_to_global_rank_dev, bytes));
    const hipStream_t stream = current_stream(dev);
    C10_HIP_CHECK(hipMemcpyAsync(
        group->rank_to_global_rank_dev,
        group->rank_to_global_rank.data(),
        bytes,
        hipMemcpyHostToDevice,
        stream));
    C10_HIP_CHECK(hipStreamSynchronize(stream));

    std::lock_guard<std::mutex> lock(mutex_);
    groups_[{name, dev}] = group;
    return group;
  }

  // Collective over the group: imports `block` and a new signal pad.
  std::shared_ptr<PeerAlloc> map_block(
      const std::shared_ptr<CommGroup>& group,
      const Block& block) {
    const auto& api = mori_cco();
    const int rank = group->rank;
    const int world_size = group->world_size;
    const int dev = block.memory->memory.device;
    const hipStream_t stream = current_stream(dev);
    auto peer = std::make_shared<PeerAlloc>();
    peer->group = group;
    peer->buffer_size = block.buffer_size;
    peer->device = dev;
    peer->pad_size = get_signal_pad_size();

    moriCcoWindowInfo_t info{};
    info.size = sizeof(info);
    moriCcoWindow_t window = nullptr;
    C10D_MORI_CHECK(api.windowImport(
        group->comm,
        block.memory->memory.ptr,
        block.memory->memory.size,
        0,
        &window,
        &info));
    block.memory->add_window(group, window);
    peer->memory = block.memory;
    peer->device_window = info.deviceWindow;

    // The pad is its own allocation, one per (allocation, group), so groups
    // never see each other's signals. It is zero before peers can reach it;
    // a copy rather than a memset, which fabric memory can reject.
    peer->pad.create(
        peer->pad_size,
        dev,
        static_cast<hipMemAllocationHandleType>(group->mem_handle_type));
    const std::vector<char> zeros(peer->pad_size, 0);
    C10_HIP_CHECK(hipMemcpyAsync(
        peer->pad.ptr,
        zeros.data(),
        peer->pad_size,
        hipMemcpyHostToDevice,
        stream));
    C10_HIP_CHECK(hipStreamSynchronize(stream));
    info = {};
    info.size = sizeof(info);
    C10D_MORI_CHECK(api.windowImport(
        group->comm,
        peer->pad.ptr,
        peer->pad.size,
        0,
        &peer->pad_window,
        &info));
    peer->pad_device_window = info.deviceWindow;

    // Every peer is in the LSA team (checked at rendezvous).
    peer->buffers.resize(world_size);
    peer->pads.resize(world_size);
    for (const auto r : c10::irange(world_size)) {
      C10D_MORI_CHECK(api.windowPeerPtr(window, r, &peer->buffers[r]));
      C10D_MORI_CHECK(api.windowPeerPtr(peer->pad_window, r, &peer->pads[r]));
    }
    // This rank's entries are its own mappings, not MORI's aliases of them.
    peer->buffers[rank] = block.memory->memory.ptr;
    peer->pads[rank] = peer->pad.ptr;

    std::vector<void*> tables(peer->buffers);
    tables.insert(tables.end(), peer->pads.begin(), peer->pads.end());
    const size_t bytes = tables.size() * sizeof(void*);
    C10_HIP_CHECK(
        hipMalloc(reinterpret_cast<void**>(&peer->tables_dev), bytes));
    C10_HIP_CHECK(hipMemcpyAsync(
        peer->tables_dev, tables.data(), bytes, hipMemcpyHostToDevice, stream));
    C10_HIP_CHECK(hipStreamSynchronize(stream));
    return peer;
  }

  std::mutex mutex_;
  std::map<uintptr_t, std::shared_ptr<Block>> blocks_; // ordered by address
  std::map<GroupKey, std::shared_ptr<CommGroup>> groups_;
  StoreExchange exchange_{"mori_symm_mem"};
};

// Immortal: torch's backend registry outlives this extension's statics.
c10::intrusive_ptr<MoriSymmetricMemoryAllocator>& allocator() {
  static auto* instance = new c10::intrusive_ptr<MoriSymmetricMemoryAllocator>(
      c10::make_intrusive<MoriSymmetricMemoryAllocator>());
  return *instance;
}

void sweep_dead_groups() noexcept {
  try {
    allocator()->sweep();
  } catch (...) {
  }
}

} // namespace

MoriSymmetricMemory::MoriSymmetricMemory(
    std::shared_ptr<PeerAlloc> peer,
    size_t offset)
    : peer_(std::move(peer)), offset_(offset) {}

std::vector<void*> MoriSymmetricMemory::get_buffer_ptrs() {
  return peer_->buffers;
}

std::vector<void*> MoriSymmetricMemory::get_signal_pad_ptrs() {
  return peer_->pads;
}

void** MoriSymmetricMemory::get_buffer_ptrs_dev() {
  return peer_->tables_dev;
}

void** MoriSymmetricMemory::get_signal_pad_ptrs_dev() {
  return peer_->tables_dev + peer_->buffers.size();
}

size_t MoriSymmetricMemory::get_buffer_size() {
  return peer_->buffer_size;
}

size_t MoriSymmetricMemory::get_offset() {
  return offset_;
}

bool MoriSymmetricMemory::has_multicast_support() {
  return false;
}

void* MoriSymmetricMemory::get_multicast_ptr() {
  return nullptr;
}

void MoriSymmetricMemory::barrier(int channel, size_t timeout_ms) {
  TORCH_CHECK(
      world_within_direct_access(),
      "MORI symmetric memory: barrier() needs every peer's signal pad, but "
      "some peers are outside this rank's load/store team");
  DeviceGuard guard(peer_->device);
  GroupStreamGuard stream_guard(peer_->group->name);
  launch_barrier(
      get_signal_pad_ptrs_dev(),
      peer_->pad_size,
      channel,
      get_rank(),
      get_world_size(),
      timeout_ms,
      current_stream(peer_->device));
}

void MoriSymmetricMemory::put_signal(
    int dst_rank,
    int channel,
    size_t timeout_ms) {
  check_rank(dst_rank, get_world_size());
  TORCH_CHECK(
      peer_->pads[dst_rank] != nullptr,
      "MORI symmetric memory: rank ",
      dst_rank,
      " is outside this rank's load/store team");
  DeviceGuard guard(peer_->device);
  GroupStreamGuard stream_guard(peer_->group->name);
  launch_put_signal(
      get_signal_pad_ptrs_dev(),
      peer_->pad_size,
      dst_rank,
      channel,
      get_rank(),
      get_world_size(),
      timeout_ms,
      current_stream(peer_->device));
}

void MoriSymmetricMemory::wait_signal(
    int src_rank,
    int channel,
    size_t timeout_ms) {
  check_rank(src_rank, get_world_size());
  DeviceGuard guard(peer_->device);
  GroupStreamGuard stream_guard(peer_->group->name);
  launch_wait_signal(
      get_signal_pad_ptrs_dev(),
      peer_->pad_size,
      src_rank,
      channel,
      get_rank(),
      get_world_size(),
      timeout_ms,
      current_stream(peer_->device));
}

int MoriSymmetricMemory::get_rank() {
  return peer_->group->rank;
}

int MoriSymmetricMemory::get_world_size() {
  return peer_->group->world_size;
}

c10::Device MoriSymmetricMemory::get_device() {
  return c10::Device(
      c10::DeviceType::CUDA, static_cast<c10::DeviceIndex>(peer_->device));
}

std::string MoriSymmetricMemory::get_group_name() {
  return peer_->group->name;
}

const std::vector<int>& MoriSymmetricMemory::get_rank_to_global_rank() {
  return peer_->group->rank_to_global_rank;
}

int* MoriSymmetricMemory::get_rank_to_global_rank_dev() {
  return peer_->group->rank_to_global_rank_dev;
}

bool MoriSymmetricMemory::world_within_direct_access() {
  return std::all_of(peer_->buffers.begin(), peer_->buffers.end(), [](void* p) {
    return p != nullptr;
  });
}

void* MoriSymmetricMemory::get_window() {
  return peer_->device_window;
}

void* MoriSymmetricMemory::get_signal_pad_window() {
  return peer_->pad_device_window;
}

void* MoriSymmetricMemory::get_dev_comm(int lsa_barrier_count) {
  TORCH_CHECK(
      lsa_barrier_count >= 0,
      "MORI symmetric memory: lsa_barrier_count must be non-negative");
  auto& peer = *peer_;
  std::lock_guard<std::mutex> lock(peer.mutex);
  auto it = peer.dev_comms.find(lsa_barrier_count);
  if (it != peer.dev_comms.end()) {
    return it->second;
  }
  moriCcoDevCommRequirements_t reqs = MORI_CCO_DEV_COMM_REQUIREMENTS_INIT;
  reqs.lsaBarrierCount = lsa_barrier_count;
  void* dev_comm = nullptr;
  DeviceGuard guard(peer.device);
  C10D_MORI_CHECK(mori_cco().devCommCreate(peer.group->comm, &reqs, &dev_comm));
  peer.dev_comms.emplace(lsa_barrier_count, dev_comm);
  return dev_comm;
}

void register_mori_backend() {
  static const bool registered = [] {
    register_availability("MORI", allocator());
    return true;
  }();
  (void)registered;
}

void shutdown_mori_backend() {
  allocator()->shutdown();
}

int mori_live_comms() {
  return live_comms.load();
}

} // namespace c10d::symmetric_memory::mori
