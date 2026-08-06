# Symbolic tensor networks

This package constructs tensor factors as SymPy expressions, evaluates them on
a finite coordinate grid, and contracts the resulting tensors with explicit
NumPy or PyTorch plans.

The reference workload contains five rank-3 nodes over five variables. Node
`i` uses `(x_i, x_(i+1), x_(i+2))` modulo five. Every pair of nodes therefore
shares at least one variable: the induced node-interaction graph is the complete
graph `K5`, although each node still has exactly three symbolic arguments.

SymPy expression construction and materialization are CPU operations. PyTorch
executes each concrete pairwise contraction sequentially on the selected
device. CUDA timing synchronizes before and after each repetition, and CUDA
peak memory uses `torch.cuda.max_memory_allocated`. CPU fallback results never
claim measured GPU utilization or memory.

The contraction is an exact finite sum of the materialized floating-point
tensor entries. It is not symbolic integration and does not establish a
closed-form identity for the original continuous expressions.
