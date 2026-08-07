"""Materialize SymPy expressions and execute their contraction with NumPy."""

from __future__ import annotations

import numpy as np
import sympy as sp

from tensorcontract.backends import NumPyBackend
from tensorcontract.symbolics import (
    ContractionNode,
    IndexRole,
    SymbolicGraph,
    SymbolicIndex,
    SymbolicTensor,
)


def main() -> None:
    row, inner, column = sp.symbols("row inner column", real=True)
    left_expression = 1 + row + 2 * inner
    right_expression = 1 + inner * column

    row_values = np.arange(2, dtype=np.float64)
    inner_values = np.arange(3, dtype=np.float64)
    column_values = np.arange(4, dtype=np.float64)
    left_function = sp.lambdify((row, inner), left_expression, modules="numpy")
    right_function = sp.lambdify((inner, column), right_expression, modules="numpy")
    left = np.asarray(left_function(row_values[:, None], inner_values[None, :]))
    right = np.asarray(right_function(inner_values[:, None], column_values[None, :]))

    graph = SymbolicGraph(metadata={"source": "SymPy expressions"})
    graph.add_index(SymbolicIndex("row", 2, IndexRole.FREE))
    graph.add_index(SymbolicIndex("inner", 3, IndexRole.CONTRACTED))
    graph.add_index(SymbolicIndex("column", 4, IndexRole.FREE))
    graph.add_tensor(SymbolicTensor("left", ("row", "inner"), (2, 3)))
    graph.add_tensor(SymbolicTensor("right", ("inner", "column"), (3, 4)))
    graph.add_operation(
        ContractionNode("result", ("left", "right"), ("row", "column"))
    )

    result = NumPyBackend().execute(graph, {"left": left, "right": right})
    reference = np.einsum("ri,ic->rc", left, right)
    print("left expression:", left_expression)
    print("right expression:", right_expression)
    print(graph.debug_string())
    print("result:\n", result)
    print("matches NumPy reference:", np.allclose(result, reference))


if __name__ == "__main__":
    main()
