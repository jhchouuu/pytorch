# Owner(s): ["module: c10d"]

# Tests of the MORI symmetric memory backend (torch._mori, built with
# USE_MORI=1). Its runtime comes from the amd-mori package. Tests that need it
# skip when it is missing, unless TORCH_TEST_REQUIRE_MORI=1 makes that a
# failure.

import functools
import importlib.util
import os
import subprocess
import sys
import textwrap
from collections.abc import Callable
from typing import Any
from unittest import skipUnless

from torch.distributed._symmetric_memory._mori import _unavailable_reason
from torch.testing._internal.common_utils import (
    HardwareClassification,
    run_tests,
    TEST_WITH_ROCM,
    TestCase,
)


_REQUIRE_MORI = os.environ.get("TORCH_TEST_REQUIRE_MORI") == "1"


def requires_mori(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Skip without a MORI runtime; fail instead under TORCH_TEST_REQUIRE_MORI=1."""

    @functools.wraps(fn)
    def wrapper(self: TestCase, *args: Any, **kwargs: Any) -> Any:
        reason = _unavailable_reason()
        if reason is not None:
            if _REQUIRE_MORI:
                self.fail(f"TORCH_TEST_REQUIRE_MORI=1, but {reason}")
            self.skipTest(reason)
        return fn(self, *args, **kwargs)

    return wrapper


# Hides an installed amd-mori from the loader: find_spec("mori") is None.
_HIDE_MORI = 'import sys; sys.modules["mori"] = None\n'

# The libmori_cco files this process has mapped.
_MAPPED_MORI = """
def mapped_mori():
    with open("/proc/self/maps") as f:
        return sorted({l.split()[-1] for l in f if "libmori_cco" in l})
"""


@skipUnless(TEST_WITH_ROCM, "MORI is a ROCm backend")
class MoriLoaderTest(TestCase):
    hw_classification = HardwareClassification.CUDA

    # Each case runs in a fresh interpreter: loading is once per process.
    def _run(self, script: str, hide_mori: bool = False, **env: str) -> None:
        prologue = _MAPPED_MORI + (_HIDE_MORI if hide_mori else "")
        result = subprocess.run(
            [sys.executable, "-c", prologue + textwrap.dedent(script)],
            env={**os.environ, **env},
            capture_output=True,
            text=True,
            timeout=300,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_import_torch_does_not_load_mori(self) -> None:
        self._run(
            """
            import sys
            import torch
            import torch.distributed._symmetric_memory

            assert mapped_mori() == [], mapped_mori()
            assert "torch._mori" not in sys.modules
            assert "mori" not in sys.modules
            """,
            TORCH_SYMMMEM="MORI",
        )

    def test_without_extension(self) -> None:
        if importlib.util.find_spec("torch._mori") is not None:
            self.skipTest("this PyTorch was built with USE_MORI=1")
        self._run(
            """
            import torch.distributed._symmetric_memory as symm_mem

            try:
                symm_mem.set_backend("MORI")
            except RuntimeError as e:
                assert "does not find allocation backend MORI" in str(e), e
            else:
                raise AssertionError("selected MORI without torch._mori")
            """
        )

    @skipUnless(importlib.util.find_spec("torch._mori"), "needs USE_MORI=1")
    def test_without_runtime(self) -> None:
        # set_backend() and TORCH_SYMMMEM both name the package to install.
        self._run(
            """
            import torch
            import torch.distributed._symmetric_memory as symm_mem

            for select in (
                lambda: symm_mem.set_backend("MORI"),
                lambda: symm_mem.empty(16, device="cuda"),
            ):
                try:
                    select()
                except ImportError as e:
                    assert "amd-mori" in str(e), e
                else:
                    raise AssertionError("loaded MORI without amd-mori")
            assert mapped_mori() == [], mapped_mori()
            """,
            hide_mori=True,
            TORCH_SYMMMEM="MORI",
        )

    @requires_mori
    def test_loads_package_copy(self) -> None:
        # Without `import mori`, the loader maps libmori_cco.so.1 from the
        # amd-mori package (not its libmori_cco.so copy), and `import mori`
        # afterwards shares it.
        self._run(
            """
            import importlib.util, os
            import torch
            from torch.distributed._symmetric_memory import _mori

            _mori._load_backend()
            import torch._mori as ext

            pkg = importlib.util.find_spec("mori").submodule_search_locations[0]
            expected = os.path.realpath(os.path.join(pkg, "libmori_cco.so.1"))
            assert os.path.realpath(ext.library_path()) == expected, ext.library_path()
            assert ext.version()[0] == ext.ABI_MAJOR, ext.version()
            assert mapped_mori() == [expected], mapped_mori()

            import mori.cco  # noqa: F401

            assert mapped_mori() == [expected], mapped_mori()
            """
        )

    @requires_mori
    def test_reuses_loaded_copy(self) -> None:
        self._run(
            """
            import os
            import torch
            import mori.cco  # noqa: F401

            before = mapped_mori()
            assert len(before) == 1, before
            from torch.distributed._symmetric_memory import _mori

            _mori._load_backend()
            import torch._mori as ext

            assert mapped_mori() == before, mapped_mori()
            assert os.path.realpath(ext.library_path()) in before, ext.library_path()
            """
        )


if __name__ == "__main__":
    run_tests()
