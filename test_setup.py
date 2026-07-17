#!/usr/bin/env python3
"""Network-free unit tests for setup lane selection and repair decisions."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("shap_e_setup", ROOT / "setup.py")
setup = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = setup
SPEC.loader.exec_module(setup)


def config(*, gpu_sm=None, cuda_version=None):
    return setup.SetupConfig(
        python_exe="/usr/bin/python3",
        ext_dir=Path("/tmp/shap-e-test"),
        gpu_sm=gpu_sm,
        cuda_version=cuda_version,
    )


class TorchLaneSelectionTests(unittest.TestCase):
    def select(self, *, gpu_sm=None, cuda_version=None, system_name="Linux", machine_name="x86_64"):
        return setup.select_torch_install_plan(
            config(gpu_sm=gpu_sm, cuda_version=cuda_version),
            system_name=system_name,
            machine_name=machine_name,
        )

    def test_no_cuda_signal_selects_cpu(self):
        plan = self.select()
        self.assertTrue(plan["supported"])
        self.assertEqual(plan["lane"], "cpu")
        self.assertEqual(plan["index_url"], setup.PYTORCH_CPU_INDEX_URL)
        self.assertFalse(plan["cuda_expected"])

    def test_gpu_sm_zero_forces_cpu_even_with_cuda_version(self):
        plan = self.select(gpu_sm=0, cuda_version=124)
        self.assertEqual(plan["lane"], "cpu")
        self.assertFalse(plan["cuda_expected"])
        self.assertIn("gpu_sm <= 0", plan["note"])

    def test_arm64_gpu_sm_zero_uses_pypi(self):
        plan = self.select(gpu_sm=0, system_name="Linux", machine_name="aarch64")
        self.assertEqual(plan["lane"], "pypi")
        self.assertIsNone(plan["index_url"])
        self.assertFalse(plan["cuda_expected"])

    def test_x86_cuda_118_to_123_selects_cu118(self):
        for cuda_version in (118, 121, 123):
            with self.subTest(cuda_version=cuda_version):
                plan = self.select(gpu_sm=86, cuda_version=cuda_version)
                self.assertEqual(plan["lane"], "cu118")
                self.assertEqual(plan["index_url"], setup.PYTORCH_CUDA_INDEX_URLS["cu118"])

    def test_x86_cuda_124_to_127_selects_cu124(self):
        for cuda_version in (124, 126, 127):
            with self.subTest(cuda_version=cuda_version):
                plan = self.select(gpu_sm=86, cuda_version=cuda_version)
                self.assertEqual(plan["lane"], "cu124")
                self.assertEqual(plan["index_url"], setup.PYTORCH_CUDA_INDEX_URLS["cu124"])

    def test_cuda_128_selects_cu128_at_cuda_boundary(self):
        plan = self.select(gpu_sm=119, cuda_version=128)
        self.assertEqual(plan["lane"], "cu128-blackwell")
        self.assertEqual(plan["index_url"], setup.PYTORCH_CUDA_INDEX_URLS["cu128-blackwell"])

    def test_sm_120_selects_cu128_at_sm_boundary(self):
        plan = self.select(gpu_sm=120, cuda_version=127)
        self.assertEqual(plan["lane"], "cu128-blackwell")
        self.assertEqual(plan["index_url"], setup.PYTORCH_CUDA_INDEX_URLS["cu128-blackwell"])

    def test_cuda_117_selects_cpu(self):
        plan = self.select(gpu_sm=86, cuda_version=117)
        self.assertEqual(plan["lane"], "cpu")
        self.assertEqual(plan["index_url"], setup.PYTORCH_CPU_INDEX_URL)

    def test_positive_non_blackwell_gpu_without_cuda_version_uses_cu118_on_x86(self):
        plan = self.select(gpu_sm=86, cuda_version=None)
        self.assertEqual(plan["lane"], "cu118")
        self.assertEqual(plan["index_url"], setup.PYTORCH_CUDA_INDEX_URLS["cu118"])

    def test_blackwell_uses_cu128_including_arm64(self):
        plan = self.select(gpu_sm=120, cuda_version=128, machine_name="aarch64")
        self.assertEqual(plan["lane"], "cu128-blackwell")
        self.assertEqual(plan["index_url"], setup.PYTORCH_CUDA_INDEX_URLS["cu128-blackwell"])
        self.assertEqual(plan["torch_version"], setup.BLACKWELL_TORCH_VERSION)
        self.assertEqual(plan["torchvision_version"], setup.BLACKWELL_TORCHVISION_VERSION)

    def test_non_blackwell_arm64_does_not_select_cuda_wheels(self):
        plan = self.select(gpu_sm=86, cuda_version=124, machine_name="arm64")
        self.assertEqual(plan["lane"], "pypi")
        self.assertIsNone(plan["index_url"])
        self.assertIn("ARM64 non-Blackwell CUDA wheels are not available", plan["note"])

    def test_macos_arm64_uses_pypi_without_index_url(self):
        plan = self.select(system_name="Darwin", machine_name="arm64")
        self.assertTrue(plan["supported"])
        self.assertEqual(plan["lane"], "pypi")
        self.assertIsNone(plan["index_url"])

    def test_macos_x86_64_fails_clearly(self):
        plan = self.select(system_name="Darwin", machine_name="x86_64")
        self.assertFalse(plan["supported"])
        self.assertEqual(plan["lane"], "unsupported")
        self.assertIn("macOS x86_64 is not supported", plan["note"])

    def test_unknown_architecture_is_unsupported_with_cuda_signal(self):
        plan = self.select(gpu_sm=86, cuda_version=124, system_name="Linux", machine_name="riscv64")
        self.assertFalse(plan["supported"])
        self.assertEqual(plan["lane"], "unsupported")
        self.assertIsNone(plan["index_url"])
        self.assertIn("linux/riscv64", plan["note"])

    def test_freebsd_is_unsupported_with_cuda_signal(self):
        plan = self.select(gpu_sm=86, cuda_version=124, system_name="FreeBSD", machine_name="x86_64")
        self.assertFalse(plan["supported"])
        self.assertEqual(plan["lane"], "unsupported")
        self.assertIsNone(plan["index_url"])
        self.assertIn("freebsd/x86_64", plan["note"])


class TorchRepairTests(unittest.TestCase):
    def test_reinstall_needed_for_stale_and_not_needed_for_current(self):
        plan = setup.select_torch_install_plan(
            config(gpu_sm=86, cuda_version=124),
            system_name="Linux",
            machine_name="x86_64",
        )
        current_probe = {
            "imports": {"torch": True, "torchvision": True},
            "torch": {"version": "2.6.0+cu124", "cuda_version": "12.4"},
            "torchvision": {"version": "0.21.0+cu124"},
        }
        stale_probe = {
            "imports": {"torch": True, "torchvision": True},
            "torch": {"version": "2.5.1+cu121", "cuda_version": "12.1"},
            "torchvision": {"version": "0.20.1+cu121"},
        }

        self.assertFalse(setup.torch_reinstall_needed(current_probe, plan))
        self.assertTrue(setup.torch_reinstall_needed(stale_probe, plan))

    def test_cuda_plan_reinstalls_mismatched_torchvision_local_tag(self):
        plan = setup.select_torch_install_plan(
            config(gpu_sm=86, cuda_version=124),
            system_name="Linux",
            machine_name="x86_64",
        )
        mismatched_probe = {
            "imports": {"torch": True, "torchvision": True},
            "torch": {"version": "2.6.0+cu124", "cuda_version": "12.4"},
            "torchvision": {"version": "0.21.0+cu118"},
        }

        self.assertEqual(plan["lane"], "cu124")
        self.assertTrue(setup.torch_reinstall_needed(mismatched_probe, plan))

    def test_cpu_plan_reinstalls_current_version_cuda_build(self):
        plan = setup.select_torch_install_plan(
            config(gpu_sm=0, cuda_version=124),
            system_name="Linux",
            machine_name="x86_64",
        )
        cuda_probe = {
            "imports": {"torch": True, "torchvision": True},
            "torch": {"version": "2.6.0+cu124", "cuda_version": "12.4"},
            "torchvision": {"version": "0.21.0+cu124"},
        }

        self.assertEqual(plan["lane"], "cpu")
        self.assertTrue(setup.torch_reinstall_needed(cuda_probe, plan))

    def test_arm64_pypi_plan_reinstalls_current_version_cuda_build(self):
        plan = setup.select_torch_install_plan(
            config(gpu_sm=86, cuda_version=124),
            system_name="Linux",
            machine_name="arm64",
        )
        cuda_probe = {
            "imports": {"torch": True, "torchvision": True},
            "torch": {"version": "2.6.0+cu124", "cuda_version": "12.4"},
            "torchvision": {"version": "0.21.0+cu124"},
        }

        self.assertEqual(plan["lane"], "pypi")
        self.assertTrue(setup.torch_reinstall_needed(cuda_probe, plan))

    def test_cpu_and_pypi_plans_reinstall_cuda_tagged_torchvision_without_runtime(self):
        plans = (
            (
                "cpu",
                setup.select_torch_install_plan(
                    config(gpu_sm=0, cuda_version=124),
                    system_name="Linux",
                    machine_name="x86_64",
                ),
            ),
            (
                "pypi",
                setup.select_torch_install_plan(
                    config(gpu_sm=86, cuda_version=124),
                    system_name="Linux",
                    machine_name="arm64",
                ),
            ),
        )
        mismatched_probe = {
            "imports": {"torch": True, "torchvision": True},
            "torch": {"version": "2.6.0+cpu", "cuda_version": None},
            "torchvision": {"version": "0.21.0+cu124"},
        }

        for expected_lane, plan in plans:
            with self.subTest(lane=expected_lane):
                self.assertEqual(plan["lane"], expected_lane)
                self.assertTrue(setup.torch_reinstall_needed(mismatched_probe, plan))

    def test_pypi_install_omits_index_url(self):
        plan = setup.select_torch_install_plan(
            config(),
            system_name="Darwin",
            machine_name="arm64",
        )
        completed = subprocess.CompletedProcess(args=[], returncode=0, stdout="")

        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.object(setup.subprocess, "run", return_value=completed) as run:
                setup.pip_install_torch(
                    "/venv/bin/python",
                    Path(tmpdir) / "setup.log",
                    plan,
                    force_reinstall=False,
                )

        cmd = run.call_args.args[0]
        self.assertNotIn("--index-url", cmd)
        self.assertIn("torch==2.6.0", cmd)
        self.assertIn("torchvision==0.21.0", cmd)


if __name__ == "__main__":
    unittest.main()
