"""Loader of the MORI symmetric memory backend.

The backend lives in the optional ``torch._mori`` extension, built with
``USE_MORI=1``, and its runtime, ``libmori_cco``, comes from the ``amd-mori``
package. ``import torch`` loads neither: :func:`_load_backend` runs the first
time ``set_backend("MORI")`` or ``TORCH_SYMMMEM=MORI`` names the backend.
"""

import atexit
import importlib.util
import os
import threading
from types import ModuleType


__all__: list[str] = []

_PACKAGE = "amd-mori"
_lock = threading.Lock()
_loaded = False


def _import_extension() -> ModuleType:
    try:
        # pyrefly: ignore [missing-import]  # built only with USE_MORI
        import torch._mori as ext
    except ImportError as e:
        raise ImportError(
            "The MORI symmetric memory backend is not available: this PyTorch "
            "was built without USE_MORI=1."
        ) from e
    return ext


def _find_library(soname: str) -> str:
    # find_spec locates the package without importing it.
    spec = importlib.util.find_spec("mori")
    dirs = list(spec.submodule_search_locations or []) if spec is not None else []
    for d in dirs:
        path = os.path.join(d, soname)
        if os.path.isfile(path):
            return path
    if not dirs:
        raise ImportError(
            f"The MORI symmetric memory backend needs the {_PACKAGE} package, "
            f"which provides {soname} (pip install {_PACKAGE})."
        )
    raise ImportError(
        f"The {_PACKAGE} package in {dirs[0]} does not provide {soname}; the "
        f"MORI symmetric memory backend needs a release of {_PACKAGE} with the "
        "MORI-CCO C ABI."
    )


def _unavailable_reason() -> str | None:
    """Why the MORI backend cannot load here, or None. Loads nothing."""
    if importlib.util.find_spec("torch._mori") is None:
        return "this PyTorch was built without USE_MORI=1"
    try:
        _find_library("libmori_cco.so.1")
    except ImportError as e:
        return str(e)
    return None


def _load_library(ext: ModuleType) -> None:
    # Reuse the copy `import mori` loaded, so the process holds only one. The
    # package's libmori_cco.so is for linkers and is never loaded.
    try:
        loaded = ext.load("")
    except RuntimeError as e:
        raise ImportError(
            f"The MORI symmetric memory backend cannot use the {_PACKAGE} "
            f"runtime already loaded in this process: {e}"
        ) from e
    if not loaded:
        path = _find_library(ext.LIBRARY_SONAME)
        try:
            ext.load(path)
        except RuntimeError as e:
            raise ImportError(
                f"The MORI symmetric memory backend failed to load {path} from "
                f"the {_PACKAGE} package: {e}"
            ) from e


def _load_backend() -> None:
    global _loaded
    with _lock:
        if _loaded:
            return
        ext = _import_extension()
        _load_library(ext)
        ext.register_backend()
        # Release MORI's allocations while HIP is still up.
        atexit.register(ext.shutdown)
        _loaded = True
