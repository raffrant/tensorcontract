"""Optional-dependency checks that do not require Matplotlib."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


def test_quantum_package_imports_when_plotting_libraries_are_blocked() -> None:
    script = """
import builtins
real_import = builtins.__import__
def guarded(name, *args, **kwargs):
    if name == 'matplotlib' or name.startswith('matplotlib.') or name == 'pandas' or name.startswith('pandas.'):
        raise ImportError(name + ' deliberately unavailable')
    return real_import(name, *args, **kwargs)
builtins.__import__ = guarded
import tensorcontract
import tensorcontract.quantum
from tensorcontract.quantum.visualization import EXPORT_COLUMNS, results_to_rows
assert len(EXPORT_COLUMNS) == 13
assert results_to_rows([]) == ()
print('optional plotting imports are lazy')
"""
    root = Path(__file__).resolve().parents[1]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(root / "src")
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert completed.stdout.strip() == "optional plotting imports are lazy"
