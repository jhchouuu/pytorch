#pragma once

#include <c10/util/Exception.h>
#include <mori/cco/cco_c.h>

#include <string>

// MORI's runtime, libmori_cco, is never linked into torch: this extension
// reaches it through the C ABI in third_party/mori, from a copy it loads at
// run time (see torch/distributed/_symmetric_memory/_mori.py).
namespace c10d::symmetric_memory::mori {

// Loads libmori_cco. With an empty `path`, reuses the copy already loaded in
// the process (e.g. by `import mori`) and returns false if there is none;
// otherwise loads the file at `path`. Idempotent once it has succeeded.
bool load_mori_cco(const std::string& path);

bool mori_cco_loaded();

// The function table. Throws if libmori_cco is not loaded.
const moriCcoApi_t& mori_cco();

// File the loaded libmori_cco was mapped from.
std::string mori_cco_path();

} // namespace c10d::symmetric_memory::mori

#define C10D_MORI_CHECK(call)                                       \
  do {                                                              \
    const int _rc = (call);                                         \
    TORCH_CHECK(                                                    \
        _rc == MORI_CCO_SUCCESS,                                    \
        #call " failed with MORI-CCO error ",                       \
        _rc,                                                        \
        ": ",                                                       \
        ::c10d::symmetric_memory::mori::mori_cco().getLastError()); \
  } while (0)
