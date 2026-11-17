// torch._mori: the MORI symmetric memory backend, as an optional extension.
//
// Imported only by torch/distributed/_symmetric_memory/_mori.py, the first
// time the "MORI" backend is named, so `import torch` never loads it. It does
// not link libmori_cco; load() reaches the runtime through its C ABI.
#include <torch/csrc/distributed/c10d/symm_mem/mori/MoriCco.hpp>
#include <torch/csrc/utils/pybind.h>

#include <string>
#include <tuple>

namespace py = pybind11;

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
}
