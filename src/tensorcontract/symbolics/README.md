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

## Optional PyTorch execution

Install with `python3 -m pip install -e '.[torch]'`, then request the backend
lazily with `get_backend("torch")`. Importing `tensorcontract` or
`tensorcontract.symbolics` does not import PyTorch. Tensor bindings are used
directly when possible, so backend autograd remains available; no `no_grad`,
detach, or conversion to Python scalars occurs in the general backend.

The current backend supports contractions, transpose, elementwise add/multiply,
and sum reductions. It does not yet implement broadcasting, sparse layouts,
fusion, compilation, custom kernels, or distributed devices. All tensors used
by one operation must be compatible under the corresponding PyTorch operation.

## Plan caching

`InMemoryPlanCache` specializes plans by the complete symbolic graph, index and
tensor declarations, runtime shape/dtype/device/layout/strides, backend
configuration, autograd state, and planner options. Each lookup returns a
`PlanCacheResult` with `hit`, `cache_enabled`, the stable key, and an isolated
copy of the plan. Cache statistics report hits, misses, stores, evictions, and
current size. Caching may be disabled or cleared explicitly.

There is no persistent cache. The current mutable graph representation lacks a
versioned stable serialization format, so cross-process reuse is deferred until
that compatibility boundary exists.
