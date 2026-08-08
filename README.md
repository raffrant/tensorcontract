# tensorcontract

`tensorcontract` is an early research implementation of a backend-independent
tensor-network IR, deterministic graph rewrites, hardware-aware contraction
planning, and exact small-code QEC decoding. It runs without CUDA; NumPy is the
correctness backend.

<p align="center">
  <img src="assets/qec-coset-probabilities.png" alt="Exact logical coset probabilities by syndrome" width="820">
</p>

<p align="center"><em>Tensor-network sector probabilities agree with exhaustive enumeration.</em></p>

## Working vertical slice

The current decoder constructs a tensor hypergraph for a classical repetition
code under independent Pauli noise, conditions parity factors on a syndrome,
and contracts each of the two logical sectors. It then verifies both sector
probabilities against exhaustive error enumeration. All reported repetition
results are exact up to floating-point rounding; no threshold or asymptotic
performance claim is made.

```bash
cd /home/raf/adap/tensorcontract
PYTHONPATH=src python3 examples/repetition_decode.py
PYTHONPATH=src python3 -m pytest
```

The example prints a deterministic rewrite trace, all candidate planning
reports, the selected multiobjective plan, and batched syndrome metrics.

## Visual reports

<p align="center">
  <img src="assets/rewrite-planner-report.png" alt="Rewrite reduction and contraction planner comparison" width="900">
</p>

<p align="center">
  <img src="assets/verified-scaling.png" alt="Verified decoder scaling measurements and planner estimates" width="820">
</p>

Regenerate these images from the implementation with:

```bash
PYTHONPATH=src python3 examples/generate_visualizations.py
```

The timing panel reports five local CPU runs per point and must not be treated
as a cross-machine benchmark. FLOPs and peak elements are planner estimates.

## Symbolic tensor-network ordering benchmark

The `tensorcontract.symbolics` package builds five reproducible random SymPy
functions, each with three variables. Cyclic variable triples make every pair
of nodes interact, producing a complete five-node interaction graph. The
expressions are evaluated on a finite grid and then contracted sequentially by
PyTorch using several explicit orders.

Display the Matplotlib report with automatic CUDA selection:

```bash
PYTHONPATH=src python3 examples/symbolic_gpu_benchmark.py \
  --device auto --dimension 8 --warmup 3 --repeats 20
```

Use `--device cuda` to require a GPU and fail rather than fall back. SymPy
construction and grid evaluation happen on CPU; the materialized tensor
contractions run on the selected PyTorch device. CUDA runs synchronize each
measurement and report peak allocated device memory. The interactive chart is
explicitly labeled when CUDA is unavailable and writes no PNG or JSON files.
Numerical results are checked against the NumPy backend.

The live figure compares the exact contraction sequence and every intermediate
size for the same graph. At dimension 8, the planner orders first contract
factors sharing two variables and peak at 4,096 elements. The included adverse
order first contracts factors sharing only one variable and peaks at 32,768
elements. On the development CPU, 10 warmups and 100 repetitions showed roughly
a 2× speedup for the planner orders; this is a local observation, not a portable
CPU or GPU performance claim.

## Optional PyTorch backend

The base installation does not require or import PyTorch. Install the optional
backend with:

```bash
python3 -m pip install -e '.[torch]'
```

Execute the same symbolic graph on PyTorch without changing its symbolic API:

```python
from tensorcontract.backends import get_backend

backend = get_backend("torch", device="cpu")
result = backend.execute(graph, {"A": torch_A, "B": torch_B})
```

Existing `torch.Tensor` bindings retain their dtype, device, and autograd graph
unless an explicit `dtype=` or `device=` conversion is requested. Supported
operations are contraction/einsum, transpose, elementwise addition and
multiplication, and sum reduction. Inputs to a contraction must have compatible
PyTorch dtypes and devices. Broadcasting, sparse tensors, explicit diagonal
indices, distributed execution, compilation, fusion, and general custom kernels
are not supported yet. The narrowly scoped experimental Triton exception is
documented below.

## Shape-specialized plan caching

The symbolic planner has an optional, thread-safe in-memory LRU cache:

```python
from tensorcontract.symbolics import InMemoryPlanCache

cache = InMemoryPlanCache(max_entries=128)
lookup = cache.get_or_plan(
    graph,
    {"A": array_a, "B": array_b},
    backend="numpy",
)
print("cache hit:", lookup.hit, "key:", lookup.key.digest)
result = backend.execute(lookup.plan.graph, {"A": array_a, "B": array_b})
```

Keys conservatively include graph and operation structure, declared tensor and
index shapes, index roles and symbolic-dimension metadata, runtime shape, dtype,
device, element strides, layout, storage offset, autograd state, backend and
backend configuration, and all planner options. Incompatible signatures always
miss. Disable reuse with `InMemoryPlanCache(enabled=False)` or temporarily with
`cache.set_enabled(False)`; inspect counters with `cache.info()`.

Persistent caching is intentionally not implemented yet. Plans contain mutable
graphs and do not have a versioned serialization format, so writing them to disk
would not currently provide a safe compatibility guarantee. In-memory entries
are deep-copied on storage and retrieval to prevent caller mutation from
poisoning future cache hits.

## NumPy batched-matrix lowering

The NumPy backend recognizes the exact symbolic pattern
`(batch, i, k) × (batch, k, j) → (batch, i, j)` and executes it with
`numpy.matmul`. Recognition requires explicit batch and contracted index roles;
different ranks, index layouts, or output orders retain the general `einsum`
implementation. This conservative rule can be disabled for comparison or
diagnosis:

```python
from tensorcontract.backends import NumPyBackend

baseline = NumPyBackend(enable_batched_matmul=False)
optimized = NumPyBackend(enable_batched_matmul=True)
```

Run the reproducible before/after benchmark with:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONPATH=src python3 benchmarks/benchmark_batched_matmul.py
```

It reports the selected implementation, median eager runtime, numerical error,
and any measured regression. NumPy performs no compilation, so compilation time
is reported as zero. Results depend on shapes, layouts, NumPy/BLAS versions, and
hardware; the benchmark is evidence for its listed workloads, not a general
speedup claim.

## Optional Triton backend

Stage 9 adds one deliberately narrow custom kernel for canonical rank-3
batched matrix multiplication. Triton and PyTorch remain optional:

```bash
python3 -m pip install -e '.[triton]'
```

Custom-kernel selection is never implicit. The default uses PyTorch, explicit
selection requests Triton, and automatic selection requires shapes approved by
an application benchmark:

```python
from tensorcontract.backends import get_backend

safe = get_backend("triton", device="cuda")  # policy="torch"
explicit = get_backend("triton", device="cuda", policy="triton")
measured = get_backend(
    "triton",
    device="cuda",
    policy="auto",
    approved_shapes={(16, 64, 64, 64)},  # (batch, M, K, N)
)
```

The custom path currently accepts contiguous CUDA `float16` and `float32`
inputs with no autograd requirement. Other contractions, CPU tensors,
unsupported dtypes, non-contiguous layouts, and differentiable inputs fall
back to PyTorch. `contraction_selection(...)` exposes the chosen implementation
and reason.

Run the three-way benchmark with:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONPATH=src python3 benchmarks/benchmark_triton_batched_matmul.py
```

On the Stage 9 development host, PyTorch 2.4.1 and Triton 3.0 were installed
but CUDA was unavailable. With one CPU thread, the measured medium workload
`(16,64,64,64)` took 0.3287 ms in NumPy and 0.0948 ms in PyTorch; the large
workload `(8,128,128,128)` took 1.0664 ms and 0.3242 ms respectively. Maximum
PyTorch errors against NumPy were `5.722e-06` and `1.717e-05`. Triton runtime
and compilation timing could not be measured on this host, so no Triton
speedup is claimed. CUDA tests validate the custom result against both NumPy
and PyTorch when suitable hardware is available and otherwise skip explicitly.
The benchmark reports the Triton cold first call as a compilation upper bound
because portable separation of compilation, allocation, and first launch is
not available through this eager interface.

## Implemented

- Typed named-index/hyperedge IR and tensor kinds for dense, diagonal, sparse,
  parity, copy/delta, permutation, stabilizer, Pauli-noise, identity, and scalar
  tensors.
- Open-index-preserving NumPy pairwise execution and optional symbolic PyTorch
  execution with autograd preservation.
- Conservative NumPy lowering of canonical rank-3 batched contractions to
  `matmul`, with an explicit opt-out and safe `einsum` fallback.
- Optional, policy-controlled Triton kernel for that same batched contraction,
  with an inspectable PyTorch fallback for unsupported inputs.
- Conservative scalar-folding, identity-elimination, and degree-one-absorption
  rewrites, each with preconditions and before/after storage estimates.
- Deterministic FLOP-, memory-, and balanced-greedy candidates; combined
  FLOP/peak-size selection, byte-traffic and arithmetic-intensity estimates,
  contraction shape classification, and hard intermediate/memory constraints.
- Exact repetition-code coset probabilities and batch statistics.

## Scientific status and limitations

This milestone is exact only for the implemented small repetition-code model
and exhaustive verifier. Planner costs are analytic estimates, not measured GPU
metrics. The candidate generator is deterministic greedy search (the `beam_width`
field is reserved for the next planner iteration). Slicing, PyTorch execution
for the legacy concrete IR, optional optimizer adapters, full correlated Pauli factors, and surface-code
construction are not implemented yet. The additional rewrite-rule names in the
product roadmap are likewise future work; this README does not claim them.

The code intentionally separates the tensor IR, rewrite engine, planner,
backend, and QEC construction. A future stabilizer design layer must validate
binary symplectic commutation independently; contraction output alone will not
be treated as proof that a construction is a valid quantum code.

## Next milestones

1. Add a validated CSS/stabilizer algebra layer and a small planar surface-code
   decoding network with exhaustive checks.
2. Qualify the experimental Triton path on CUDA hardware, then add slicing,
   repeated-syndrome reuse, and measured peak-memory/device metrics.
3. Expand the rewrite system and beam-search planner, then add `opt_einsum` and
   optional `cotengra`/cuTensorNet adapters plus representative benchmarks.
# tensorcontract
