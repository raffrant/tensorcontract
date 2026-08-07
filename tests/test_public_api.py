"""Compatibility tests for documented imports and the basic example."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import tensorcontract


EXPECTED_EXPORTS = {
    "HardwareProfile", "Index", "PlanConstraints", "RewriteEngine",
    "TensorKind", "TensorNetwork", "TensorNode", "plan_contraction",
}


def test_top_level_exports_remain_stable() -> None:
    assert set(tensorcontract.__all__) == EXPECTED_EXPORTS
    assert all(hasattr(tensorcontract, name) for name in EXPECTED_EXPORTS)


def test_base_and_symbolics_packages_import_when_torch_is_blocked() -> None:
    script = """
import builtins
real_import = builtins.__import__
def guarded(name, *args, **kwargs):
    if name == 'torch' or name.startswith('torch.'):
        raise ImportError('torch deliberately unavailable')
    return real_import(name, *args, **kwargs)
builtins.__import__ = guarded
import tensorcontract
import tensorcontract.symbolics
print('ok')
"""
    environment = dict(os.environ)
    source = str(Path(__file__).resolve().parents[1] / "src")
    environment["PYTHONPATH"] = source
    completed = subprocess.run(
        [sys.executable, "-c", script], check=True, capture_output=True, text=True, env=environment,
    )
    assert completed.stdout.strip() == "ok"


def test_repetition_example_remains_runnable() -> None:
    root = Path(__file__).resolve().parents[1]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(root / "src")
    completed = subprocess.run(
        [sys.executable, str(root / "examples" / "repetition_decode.py")],
        check=True, capture_output=True, text=True, env=environment,
    )
    assert "Exact verification: True" in completed.stdout
    assert "Selected plan:" in completed.stdout
    assert "Batch metrics:" in completed.stdout
