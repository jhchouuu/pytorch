# Owner(s): ["module: c10d"]

# Tests of the MORI symmetric memory backend (torch._mori, built with
# USE_MORI=1). Its runtime comes from the amd-mori package. Tests that need it
# skip when it is missing, unless TORCH_TEST_REQUIRE_MORI=1 makes that a
# failure.

import functools
import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
from collections.abc import Callable
from typing import Any
from unittest import skipIf, skipUnless

import torch
import torch.distributed as c10d
import torch.distributed._symmetric_memory as symm_mem
from torch._C._autograd import DeviceType
from torch._C._distributed_c10d import _SymmetricMemory
from torch.distributed._symmetric_memory._mori import _unavailable_reason
from torch.testing._internal.common_distributed import (
    MultiProcContinuousTest,
    MultiProcessTestCase,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_utils import (
    HardwareClassification,
    instantiate_parametrized_tests,
    parametrize,
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


_MORI_UNAVAILABLE = (
    _unavailable_reason() if TEST_WITH_ROCM else "MORI is a ROCm backend"
)


def skip_without_mori(obj: Any) -> Any:
    """Skip without a MORI runtime, unless TORCH_TEST_REQUIRE_MORI=1."""
    return skipIf(
        _MORI_UNAVAILABLE is not None and not _REQUIRE_MORI, str(_MORI_UNAVAILABLE)
    )(obj)


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
        # set_backend() and TORCH_SYMMMEM both name the missing build flag.
        if importlib.util.find_spec("torch._mori") is not None:
            self.skipTest("this PyTorch was built with USE_MORI=1")
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
                    assert "USE_MORI=1" in str(e), e
                else:
                    raise AssertionError("selected MORI without torch._mori")
            """,
            TORCH_SYMMMEM="MORI",
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

    @skipUnless(importlib.util.find_spec("torch._mori"), "needs USE_MORI=1")
    def test_pre_c_abi_runtime_loaded(self) -> None:
        # A libmori_cco.so without the C ABI already in the process is an
        # ImportError, whatever was imported first.
        cc = shutil.which("cc") or shutil.which("gcc")
        if cc is None:
            self.skipTest("needs a C compiler")
        with tempfile.TemporaryDirectory() as tmp:
            stub = os.path.join(tmp, "libmori_cco.so")
            subprocess.run(
                [cc, "-shared", "-fPIC", "-Wl,-soname,libmori_cco.so"]
                + ["-o", stub, "-x", "c", "/dev/null"],
                check=True,
            )
            self._run(
                """
                import ctypes, os
                import torch
                import torch.distributed._symmetric_memory as symm_mem

                ctypes.CDLL(os.environ["MORI_STUB"])
                try:
                    symm_mem.set_backend("MORI")
                except ImportError as e:
                    assert "amd-mori" in str(e) and "C ABI" in str(e), e
                else:
                    raise AssertionError("loaded MORI next to a pre-C-ABI runtime")
                """,
                MORI_STUB=stub,
            )

    @requires_mori
    def test_loads_package_copy(self) -> None:
        # Without `import mori`, the loader maps libmori_cco.so.1 from the
        # amd-mori package (not its libmori_cco.so), and `import mori`
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


class _MoriTestMixin:
    rank: int
    world_size: int

    @property
    def device(self) -> torch.device:
        return torch.device("cuda", self.rank)

    def _world(self) -> str:
        symm_mem.set_backend("MORI")
        torch.cuda.set_device(self.device)
        return c10d.group.WORLD.group_name

    def _expected_sum(self, offset: float = 0.0) -> float:
        return self.world_size * offset + self.world_size * (self.world_size - 1) / 2


@skip_without_mori
@instantiate_parametrized_tests
class MoriSymmetricMemoryTest(_MoriTestMixin, MultiProcContinuousTest):
    hw_classification = HardwareClassification.CUDA

    @classmethod
    def _init_pg(cls, rank: int, world_size: int, rdvz_file: str | None) -> None:
        if rdvz_file is None:
            raise AssertionError("Expected rdvz_file to not be None")
        os.environ["LOCAL_RANK"] = str(rank)
        device = torch.device("cuda", rank)
        torch.cuda.set_device(device)
        c10d.init_process_group(
            backend="nccl",
            world_size=world_size,
            rank=rank,
            store=c10d.FileStore(rdvz_file, world_size),
            timeout=cls.timeout,
            device_id=device,
        )
        cls.pg = c10d.distributed_c10d._get_default_group()

    @skip_if_lt_x_gpu(2)
    def test_backend(self) -> None:
        self._world()
        self.assertEqual(symm_mem.get_backend(self.device), "MORI")
        t = symm_mem.empty(16, device=self.device)
        self.assertTrue(symm_mem.is_symm_mem_tensor(t))
        self.assertFalse(
            symm_mem.is_symm_mem_tensor(torch.empty(16, device=self.device))
        )
        self.assertFalse(
            _SymmetricMemory.has_multicast_support(DeviceType.CUDA, self.rank)
        )

    @skip_if_lt_x_gpu(2)
    def test_rendezvous_world(self) -> None:
        group_name = self._world()
        t = symm_mem.empty(64, device=self.device).fill_(self.rank)
        hdl = symm_mem.rendezvous(t, group=group_name)
        self.assertEqual(hdl.rank, self.rank)
        self.assertEqual(hdl.world_size, self.world_size)
        self.assertEqual(hdl.group_name, group_name)
        self.assertEqual(hdl.buffer_ptrs[self.rank], t.data_ptr())
        self.assertEqual(hdl.offset, 0)
        self.assertGreaterEqual(hdl.buffer_size, t.nbytes)
        self.assertEqual(hdl.signal_pad_size, symm_mem.get_signal_pad_size())
        self.assertTrue(all(p != 0 for p in hdl.buffer_ptrs))
        self.assertTrue(all(p != 0 for p in hdl.signal_pad_ptrs))
        hdl.barrier()
        for peer in range(self.world_size):
            buf = hdl.get_buffer(peer, (64,), torch.float32)
            self.assertTrue(buf.eq(peer).all())
        hdl.barrier()

        # A later rendezvous of the same tensor and group reuses the mapping.
        again = symm_mem.rendezvous(t, group=group_name)
        self.assertEqual(again.buffer_ptrs, hdl.buffer_ptrs)
        self.assertEqual(again.signal_pad_ptrs, hdl.signal_pad_ptrs)

    @skip_if_lt_x_gpu(2)
    def test_rendezvous_many_allocations(self) -> None:
        group_name = self._world()
        tensors = [symm_mem.empty(1, device=self.device) for _ in range(64)]
        for t in tensors[::8]:
            hdl = symm_mem.rendezvous(t, group=group_name)
            self.assertEqual(hdl.world_size, self.world_size)
        out = torch.ops.symm_mem.one_shot_all_reduce(
            tensors[-1].fill_(self.rank), "sum", group_name
        )
        self.assertEqual(out, torch.full_like(out, self._expected_sum()))

    @skip_if_lt_x_gpu(2)
    def test_barrier_channels(self) -> None:
        group_name = self._world()
        t = symm_mem.empty(64, device=self.device)
        hdl = symm_mem.rendezvous(t, group=group_name)
        num_channels = hdl.signal_pad_size // 4 // self.world_size
        for channel in (0, 1, num_channels - 1):
            hdl.barrier(channel=channel)
        with self.assertRaisesRegex(RuntimeError, "maximum supported channel"):
            hdl.barrier(channel=num_channels)
        with self.assertRaisesRegex(RuntimeError, "must be non-negative"):
            hdl.barrier(channel=-1)
        torch.cuda.synchronize()
        # Every signal was consumed, so the pads are zero again.
        pad = hdl.get_signal_pad(self.rank, (hdl.signal_pad_size // 4,), torch.int32)
        self.assertEqual(pad.count_nonzero().item(), 0)

    @skip_if_lt_x_gpu(2)
    def test_signal_rank_out_of_bounds(self) -> None:
        group_name = self._world()
        hdl = symm_mem.rendezvous(symm_mem.empty(64, device=self.device), group_name)
        for bad_rank in (-1, self.world_size):
            with self.assertRaisesRegex(RuntimeError, r"must be in \[0"):
                hdl.put_signal(dst_rank=bad_rank)
            with self.assertRaisesRegex(RuntimeError, r"must be in \[0"):
                hdl.wait_signal(src_rank=bad_rank)

    @skip_if_lt_x_gpu(2)
    @parametrize("channel", [0, 7])
    def test_handle_signal(self, channel: int) -> None:
        # A ring: write into the next rank's buffer, signal it, and wait for
        # the previous rank's write to land in this one.
        group_name = self._world()
        numel = 1024
        t = symm_mem.empty(numel, device=self.device).fill_(-1)
        hdl = symm_mem.rendezvous(t, group=group_name)
        hdl.barrier()
        dst = (self.rank + 1) % self.world_size
        src = (self.rank - 1) % self.world_size
        for step in range(3):
            hdl.get_buffer(dst, (numel,), torch.float32).fill_(self.rank + step)
            hdl.put_signal(dst_rank=dst, channel=channel)
            hdl.wait_signal(src_rank=src, channel=channel)
            self.assertTrue(t.eq(src + step).all())
            hdl.barrier(channel=channel)

    @skip_if_lt_x_gpu(2)
    def test_one_sided_apis(self) -> None:
        group_name = self._world()
        numel = 256
        src = symm_mem.empty(numel, device=self.device).fill_(self.rank)
        hdl = symm_mem.rendezvous(src, group=group_name)
        hdl.barrier()
        peer = (self.rank + 1) % self.world_size
        dst = torch.empty(numel, device=self.device)
        symm_mem.get(dst, hdl, peer=peer)
        self.assertEqual(dst, torch.full_like(dst, peer))

        buf = symm_mem.empty(numel, device=self.device).fill_(-1)
        hdl_buf = symm_mem.rendezvous(buf, group=group_name)
        hdl_buf.barrier()
        symm_mem.put_signal(src, hdl_buf, peer=peer)
        symm_mem.wait_signal(hdl_buf, peer=(self.rank - 1) % self.world_size)
        self.assertEqual(buf, torch.full_like(buf, (self.rank - 1) % self.world_size))

    @skip_if_lt_x_gpu(2)
    def test_collectives(self) -> None:
        group_name = self._world()
        numel = 1 << 16
        inp = symm_mem.empty(numel, device=self.device).fill_(self.rank)
        symm_mem.rendezvous(inp, group=group_name)
        expected = torch.full_like(inp, self._expected_sum())
        self.assertEqual(
            torch.ops.symm_mem.one_shot_all_reduce(inp, "sum", group_name), expected
        )
        torch.ops.symm_mem.two_shot_all_reduce_(inp, "sum", group_name)
        self.assertEqual(inp, expected)

        out = torch.empty(numel // self.world_size, device=self.device)
        inp.fill_(self.rank)
        torch.ops.symm_mem.reduce_scatter_out(inp, group_name, False, out)
        self.assertEqual(out, torch.full_like(out, self._expected_sum()))

    @skip_if_lt_x_gpu(2)
    def test_collective_cuda_graph(self) -> None:
        group_name = self._world()
        inp = symm_mem.empty(1024, device=self.device).fill_(self.rank)
        # The first rendezvous happens eagerly; captured calls reuse it.
        symm_mem.rendezvous(inp, group=group_name)
        torch.ops.symm_mem.one_shot_all_reduce(inp, "sum", group_name)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            res = torch.ops.symm_mem.one_shot_all_reduce(inp, "sum", group_name)
        for offset in range(3):
            inp.fill_(self.rank + offset)
            graph.replay()
            self.assertEqual(res, torch.full_like(res, self._expected_sum(offset)))

    @skip_if_lt_x_gpu(2)
    def test_capture_rejected(self) -> None:
        group_name = self._world()
        fresh = symm_mem.empty(64, device=self.device)
        stream = torch.cuda.Stream(device=self.device)
        with self.assertRaisesRegex(RuntimeError, "capturing"):
            with torch.cuda.graph(torch.cuda.CUDAGraph(), stream=stream):
                symm_mem.empty(64, device=self.device)
        with self.assertRaisesRegex(RuntimeError, "capturing"):
            with torch.cuda.graph(torch.cuda.CUDAGraph(), stream=stream):
                symm_mem.rendezvous(fresh, group=group_name)
        # Both still work outside capture, on every rank.
        hdl = symm_mem.rendezvous(fresh, group=group_name)
        hdl.barrier()

    @skip_if_lt_x_gpu(2)
    def test_subgroup(self) -> None:
        # One tensor, two groups: each rendezvous has its own pad and handle.
        group_name = self._world()
        sub = c10d.new_group(list(range(self.world_size)))
        t = symm_mem.empty(64, device=self.device).fill_(self.rank)
        world_hdl = symm_mem.rendezvous(t, group=group_name)
        sub_hdl = symm_mem.rendezvous(t, group=sub)
        self.assertEqual(sub_hdl.group_name, sub.group_name)
        self.assertEqual(world_hdl.group_name, group_name)
        self.assertEqual(sub_hdl.buffer_ptrs[self.rank], t.data_ptr())
        self.assertTrue(
            set(world_hdl.signal_pad_ptrs).isdisjoint(sub_hdl.signal_pad_ptrs)
        )
        world_hdl.barrier()
        sub_hdl.barrier()
        out = torch.ops.symm_mem.one_shot_all_reduce(t, "sum", sub.group_name)
        self.assertEqual(out, torch.full_like(out, self._expected_sum()))

    @skip_if_lt_x_gpu(2)
    def test_rendezvous_via_pg(self) -> None:
        # use_pg_for_symm_mem_rendezvous routes the exchange through the group.
        self._world()
        sub = c10d.new_group(list(range(self.world_size)))
        sub.use_pg_for_symm_mem_rendezvous = True
        t = symm_mem.empty(64, device=self.device).fill_(self.rank)
        hdl = symm_mem.rendezvous(t, group=sub)
        hdl.barrier()
        for peer in range(self.world_size):
            self.assertTrue(hdl.get_buffer(peer, (64,), torch.float32).eq(peer).all())
        hdl.barrier()

    @skip_if_lt_x_gpu(4)
    def test_split_subgroups(self) -> None:
        self._world()
        sub, _ = c10d.new_subgroups(group_size=2)
        t = symm_mem.empty(64, device=self.device).fill_(self.rank)
        hdl = symm_mem.rendezvous(t, group=sub)
        self.assertEqual(hdl.world_size, 2)
        self.assertEqual(hdl.rank, self.rank % 2)
        out = torch.ops.symm_mem.one_shot_all_reduce(t, "sum", sub.group_name)
        first = self.rank - self.rank % 2
        self.assertEqual(out, torch.full_like(out, 2 * first + 1))

    @skip_if_lt_x_gpu(2)
    def test_mempool(self) -> None:
        group_name = self._world()
        pool = symm_mem.get_mem_pool(self.device)
        with torch.cuda.use_mem_pool(pool):
            t = torch.full((4096,), float(self.rank), device=self.device)
        self.assertTrue(symm_mem.is_symm_mem_tensor(t))
        out = torch.ops.symm_mem.one_shot_all_reduce(t, "sum", group_name)
        self.assertEqual(out, torch.full_like(out, self._expected_sum()))

    @skip_if_lt_x_gpu(2)
    def test_rendezvous_size_mismatch(self) -> None:
        group_name = self._world()
        t = symm_mem.empty(64 + self.rank, device=self.device)
        with self.assertRaisesRegex(RuntimeError, "same size on all ranks"):
            symm_mem.rendezvous(t, group=group_name)

    @skip_if_lt_x_gpu(2)
    def test_device_handles(self) -> None:
        import torch._mori as ext

        group_name = self._world()
        hdl = symm_mem.rendezvous(symm_mem.empty(64, device=self.device), group_name)
        self.assertNotEqual(ext.window(hdl), 0)
        self.assertNotEqual(ext.signal_pad_window(hdl), 0)
        dev_comm = ext.dev_comm(hdl, lsa_barrier_count=1)
        self.assertNotEqual(dev_comm, 0)
        self.assertEqual(ext.dev_comm(hdl, lsa_barrier_count=1), dev_comm)
        # A later rendezvous gets its own, which knows its windows.
        later = symm_mem.rendezvous(symm_mem.empty(64, device=self.device), group_name)
        self.assertNotIn(ext.dev_comm(later), (0, dev_comm))
        del hdl
        torch.cuda.synchronize()
        self.assertNotEqual(ext.dev_comm(later, lsa_barrier_count=2), 0)


@skip_without_mori
class MoriLifecycleTest(_MoriTestMixin, MultiProcessTestCase):
    hw_classification = HardwareClassification.CUDA

    def setUp(self) -> None:
        super().setUp()
        self._spawn_processes()

    @property
    def world_size(self) -> int:
        return 2

    def _init(self, suffix: str = "") -> None:
        torch.cuda.set_device(self.device)
        c10d.init_process_group(
            backend="nccl",
            world_size=self.world_size,
            rank=self.rank,
            store=c10d.FileStore(self.file_name + suffix, self.world_size),
            device_id=self.device,
        )

    @skip_if_lt_x_gpu(2)
    def test_comm_released_with_group(self) -> None:
        import torch._mori as ext

        self._init()
        self._world()
        base = ext.live_comms()
        sub = c10d.new_group(list(range(self.world_size)))
        t = symm_mem.empty(64, device=self.device)
        hdl = symm_mem.rendezvous(t, group=sub)
        hdl.barrier()
        self.assertEqual(ext.live_comms(), base + 1)
        torch.cuda.synchronize()
        c10d.destroy_process_group(sub)
        del hdl, t
        self.assertEqual(ext.live_comms(), base)
        c10d.destroy_process_group()

    @skip_if_lt_x_gpu(2)
    def test_allocation_after_default_pg_restart(self) -> None:
        self._init()
        group_name = self._world()
        t = symm_mem.empty(4096, device=self.device)
        symm_mem.rendezvous(t, group=group_name).barrier()
        del t
        torch.cuda.synchronize()
        c10d.destroy_process_group()

        self._init("_restart")
        group_name = c10d.group.WORLD.group_name
        t = symm_mem.empty(4096, device=self.device).fill_(self.rank)
        hdl = symm_mem.rendezvous(t, group=group_name)
        hdl.barrier()
        peer = (self.rank + 1) % self.world_size
        self.assertTrue(hdl.get_buffer(peer, (4096,), torch.float32).eq(peer).all())
        hdl.barrier()
        c10d.destroy_process_group()

    @skip_if_lt_x_gpu(2)
    def test_cpu_affinity_unchanged(self) -> None:
        # The first communicator of a fresh process does not pin its thread.
        self._init()
        group_name = self._world()
        before = os.sched_getaffinity(0)
        symm_mem.rendezvous(symm_mem.empty(64, device=self.device), group_name)
        self.assertEqual(os.sched_getaffinity(0), before)
        c10d.destroy_process_group()

    @skip_if_lt_x_gpu(2)
    def test_group_spanning_teams_rejected(self) -> None:
        # Distinct vPOD ids put each rank in its own load/store team, as on
        # two nodes; torch's ops would load from unmapped peers.
        os.environ["MORI_VPOD_ID"] = str(self.rank)
        self._init()
        group_name = self._world()
        t = symm_mem.empty(64, device=self.device)
        with self.assertRaisesRegex(RuntimeError, "share a load/store team"):
            symm_mem.rendezvous(t, group=group_name)
        c10d.destroy_process_group()

    @skip_if_lt_x_gpu(2)
    def test_handle_outlives_tensor(self) -> None:
        # Freeing a rendezvoused tensor while a handle to it is alive keeps
        # the handle's signal pad usable.
        self._init()
        group_name = self._world()
        t = symm_mem.empty(64, device=self.device)
        hdl = symm_mem.rendezvous(t, group=group_name)
        del t
        torch.cuda.synchronize()
        hdl.barrier()
        torch.cuda.synchronize()
        del hdl
        c10d.destroy_process_group()


if __name__ == "__main__":
    run_tests()
