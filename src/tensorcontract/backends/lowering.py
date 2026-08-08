"""Conservative high-level lowering recognition shared by eager backends."""

from __future__ import annotations

from dataclasses import dataclass

from tensorcontract.symbolics.ir import ContractionNode, IndexRole, SymbolicGraph


@dataclass(frozen=True, slots=True)
class BatchedMatmulLowering:
    """An exact ``(B,M,K) @ (B,K,N) -> (B,M,N)`` contraction match."""

    left: str
    right: str
    batch_index: str
    left_free_index: str
    contracted_index: str
    right_free_index: str


def recognize_batched_matmul(
    graph: SymbolicGraph,
    operation: ContractionNode,
) -> BatchedMatmulLowering | None:
    """Recognize only canonical rank-3 batched matrix multiplication.

    Returning ``None`` is always safe: the backend retains its general einsum
    implementation for every unrecognized structure or output ordering.
    """
    if len(operation.inputs) != 2 or len(operation.output_indices) != 3:
        return None
    left_name, right_name = operation.inputs
    left = graph.value_indices(left_name)
    right = graph.value_indices(right_name)
    if len(left) != 3 or len(right) != 3:
        return None
    batch_index, left_free, contracted = left
    expected_right = (batch_index, contracted, right[2])
    expected_output = (batch_index, left_free, right[2])
    if right != expected_right or operation.output_indices != expected_output:
        return None
    if len({batch_index, left_free, contracted, right[2]}) != 4:
        return None
    if graph.indices[batch_index].role != IndexRole.BATCH:
        return None
    if graph.indices[contracted].role != IndexRole.CONTRACTED:
        return None
    return BatchedMatmulLowering(
        left_name,
        right_name,
        batch_index,
        left_free,
        contracted,
        right[2],
    )
