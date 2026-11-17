// torch._mori: the MORI symmetric memory backend, as an optional extension.
//
// Imported only by torch/distributed/_symmetric_memory/_mori.py, the first
// time the "MORI" backend is named, so `import torch` never loads it. It does
// not link libmori_cco; load() reaches the runtime through its C ABI.
#include <torch/csrc/distributed/c10d/symm_mem/mori/MoriCco.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/mori/MoriSymmetricMemory.hpp>
#include <torch/csrc/utils/pybind.h>

#include <cstdint>
#include <string>
#include <tuple>

namespace py = pybind11;

namespace {

using c10d::symmetric_memory::SymmetricMemory;
using c10d::symmetric_memory::mori::MoriSymmetricMemory;

MoriSymmetricMemory* as_mori(const c10::intrusive_ptr<SymmetricMemory>& hdl) {
  auto* mori = dynamic_cast<MoriSymmetricMemory*>(hdl.get());
  TORCH_CHECK(mori != nullptr, "not a handle of the MORI backend");
  return mori;
}

} // namespace

PYBIND11_MODULE(_mori, m) {
  using namespace c10d::symmetric_memory::mori;

  m.attr("ABI_MAJOR") = MORI_CCO_ABI_MAJOR;
  m.attr("ABI_MINOR") = MORI_CCO_ABI_MINOR;
  m.attr("LIBRARY_SONAME") = MORI_CCO_LIBRARY_SONAME;

  m.def(
      "load",
      &load_mori_cco,
      py::arg("path"),
      "Load libmori_cco: the copy already in the process if path is empty "
      "(False if there is none), else the file at path.");
  m.def("is_loaded", &mori_cco_loaded);
  m.def(
      "version",
      []() {
        int abi_major = 0;
        int abi_minor = 0;
        const char* library = nullptr;
        C10D_MORI_CHECK(
            mori_cco().getVersion(&abi_major, &abi_minor, &library));
        return std::make_tuple(
            abi_major, abi_minor, std::string(library ? library : ""));
      },
      "(abi_major, abi_minor, library_version) of the loaded libmori_cco.");
  m.def("library_path", &mori_cco_path);

  m.def(
      "register_backend",
      &register_mori_backend,
      "Register the \"MORI\" backend (idempotent); libmori_cco must be loaded.");
  m.def(
      "shutdown",
      &shutdown_mori_backend,
      "Release every allocation and communicator (an atexit hook).");
  m.def("live_comms", &mori_live_comms, "Communicators alive (diagnostics).");

  // MORI-CCO device handles of a rendezvous, for kernels written against
  // MORI's device API. Opaque addresses; valid while the handle lives.
  m.def(
      "window",
      [](const c10::intrusive_ptr<SymmetricMemory>& hdl) {
        return reinterpret_cast<uintptr_t>(as_mori(hdl)->get_window());
      },
      py::arg("hdl"),
      "The buffer's device window (mori::cco::ccoWindow_t).");
  m.def(
      "signal_pad_window",
      [](const c10::intrusive_ptr<SymmetricMemory>& hdl) {
        return reinterpret_cast<uintptr_t>(
            as_mori(hdl)->get_signal_pad_window());
      },
      py::arg("hdl"),
      "The signal pad's device window (mori::cco::ccoWindow_t).");
  m.def(
      "dev_comm",
      [](const c10::intrusive_ptr<SymmetricMemory>& hdl,
         int lsa_barrier_count) {
        return reinterpret_cast<uintptr_t>(
            as_mori(hdl)->get_dev_comm(lsa_barrier_count));
      },
      py::arg("hdl"),
      py::arg("lsa_barrier_count") = 1,
      "The group's device communicator (mori::cco::ccoDevComm*), created on "
      "first use per lsa_barrier_count; collective then. It knows the windows "
      "imported before it was created.");
}
