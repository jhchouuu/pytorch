#include <torch/csrc/distributed/c10d/symm_mem/mori/MoriCco.hpp>

#include <dlfcn.h>

#include <atomic>
#include <cstring>
#include <mutex>

namespace c10d::symmetric_memory::mori {

namespace {

std::mutex load_mutex;
moriCcoApi_t api_table;
std::atomic<bool> api_loaded{false};

// The name older amd-mori releases gave libmori_cco, before the C ABI.
constexpr const char* kUnversionedName = "libmori_cco.so";

std::string dl_error() {
  const char* err = dlerror();
  return err ? err : "unknown error";
}

// Fills api_table from `lib`; returns an error message, empty on success.
std::string bind_api(void* lib, const std::string& what) {
  auto get_api =
      reinterpret_cast<moriCcoGetApiFn_t>(dlsym(lib, MORI_CCO_GET_API_SYMBOL));
  if (get_api == nullptr) {
    return what + " has no " + MORI_CCO_GET_API_SYMBOL +
        "; it predates the MORI-CCO C ABI";
  }
  moriCcoApi_t table;
  std::memset(&table, 0, sizeof(table));
  const int rc = get_api(MORI_CCO_ABI_MAJOR, &table, sizeof(table));
  if (rc != MORI_CCO_SUCCESS) {
    return what + " does not implement MORI-CCO C ABI " +
        std::to_string(MORI_CCO_ABI_MAJOR) + " (error " + std::to_string(rc) +
        ")";
  }
  // Every ABI 1.x library fills all ABI 1.0 entries.
  if (!table.getVersion || !table.getLastError || !table.getUniqueId ||
      !table.commCreate || !table.commDestroy || !table.commGetInfo ||
      !table.windowImport || !table.windowRelease || !table.windowPeerPtr ||
      !table.devCommCreate || !table.devCommDestroy || !table.devCommGetInfo) {
    return what + " returned an incomplete MORI-CCO function table";
  }
  api_table = table;
  api_loaded.store(true, std::memory_order_release);
  return {};
}

} // namespace

bool load_mori_cco(const std::string& path) {
  std::lock_guard<std::mutex> lock(load_mutex);
  if (api_loaded.load(std::memory_order_acquire)) {
    return true;
  }
  // A copy already in the process, found by SONAME, so there is only one.
  if (void* lib = dlopen(
          MORI_CCO_LIBRARY_SONAME, RTLD_NOW | RTLD_LOCAL | RTLD_NOLOAD)) {
    const auto err = bind_api(lib, MORI_CCO_LIBRARY_SONAME);
    TORCH_CHECK(err.empty(), err);
    return true;
  }
  // A second copy next to an older, unversioned one would not share its state.
  if (void* lib =
          dlopen(kUnversionedName, RTLD_NOW | RTLD_LOCAL | RTLD_NOLOAD)) {
    const auto err = bind_api(lib, kUnversionedName);
    if (err.empty()) {
      return true;
    }
    dlclose(lib);
    TORCH_CHECK(
        false,
        "An amd-mori without the MORI-CCO C ABI is already loaded in this "
        "process (",
        err,
        "). The MORI symmetric memory backend needs an amd-mori that ships ",
        MORI_CCO_LIBRARY_SONAME,
        ".");
  }
  if (path.empty()) {
    return false;
  }
  void* lib = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
  TORCH_CHECK(lib != nullptr, "Failed to load ", path, ": ", dl_error());
  const auto err = bind_api(lib, path);
  if (!err.empty()) {
    dlclose(lib);
    TORCH_CHECK(false, err);
  }
  return true;
}

bool mori_cco_loaded() {
  return api_loaded.load(std::memory_order_acquire);
}

const moriCcoApi_t& mori_cco() {
  TORCH_CHECK(
      mori_cco_loaded(),
      "libmori_cco is not loaded; select the MORI backend with "
      "torch.distributed._symmetric_memory.set_backend('MORI') first");
  return api_table;
}

std::string mori_cco_path() {
  Dl_info info;
  if (!mori_cco_loaded() ||
      dladdr(reinterpret_cast<void*>(api_table.getVersion), &info) == 0 ||
      info.dli_fname == nullptr) {
    return {};
  }
  return info.dli_fname;
}

} // namespace c10d::symmetric_memory::mori
