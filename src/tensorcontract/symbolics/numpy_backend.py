"""Compatibility functions for symbolic NumPy execution.

New code may instantiate :class:`tensorcontract.backends.NumPyBackend`
directly. These functions remain supported for the Stage 3 public API.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray

from tensorcontract.backends.numpy import NumPyBackend

from .ir import SymbolicGraph


def execute_symbolic_numpy_all(
    graph: SymbolicGraph,
    bindings: Mapping[str, ArrayLike],
) -> dict[str, NDArray[np.generic]]:
    """Execute a graph with the NumPy reference backend and return all values."""
    return NumPyBackend().execute_all(graph, bindings)


def execute_symbolic_numpy(
    graph: SymbolicGraph,
    bindings: Mapping[str, ArrayLike],
    output: str | None = None,
) -> NDArray[np.generic]:
    """Execute one graph output with the NumPy reference backend."""
    return NumPyBackend().execute(graph, bindings, output)
