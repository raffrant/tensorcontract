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

The three-qubit QEC workflow composes that cache with a backend-plan cache:

```python
from tensorcontract.quantum import QuantumExecutionPlanCache, run_monte_carlo

cache = QuantumExecutionPlanCache(max_entries=32)
cold = run_monte_carlo(
    100_000, 0.10, 0.05, batch_size=65_536, plan_cache=cache
)
warm = run_monte_carlo(
    100_000, 0.20, 0.15, batch_size=65_536, plan_cache=cache
)
print(cold.cache_hit, warm.cache_hit, cache.info())  # False, True, ...
```

Its stable backend key includes topology, tensor/batch shapes, index and noise
model structure, dtype, requested backend, device, batch size, fusion options,
and contraction options. The numerical values of `p` and `rho` remain runtime
bindings and therefore do not force replanning. Incompatible signatures always
miss. Pass `cache_enabled=False`, call `cache.set_enabled(False)`, inspect with
`cache.inspect()`, or clear with `cache.clear()`. A failed GPU plan construction
falls back to NumPy and is not cached, allowing a transient accelerator failure
to recover on a later call.

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

## Optional GPU Monte Carlo

The three-qubit correlated-noise example can process Monte Carlo batches with
high-level PyTorch CUDA operations:

```python
from tensorcontract.quantum import is_gpu_available, run_monte_carlo

print("CUDA available:", is_gpu_available())
result = run_monte_carlo(
    1_000_000,
    p=0.17,
    rho=0.23,
    seed=7,
    batch_size=262_144,
    backend="gpu",
)
print(result.backend, result.device, result.fallback_used)
```

This is an eager, vectorized PyTorch implementation. It samples the shared and
local variables on the GPU, then performs XOR, syndrome lookup, recovery,
residual classification, and reductions over whole batches. It does not launch
per shot or compile per shot. PyTorch remains optional;
install it with `python3 -m pip install -e '.[torch]'`.

An explicit optional fusion policy uses one Triton trajectory kernel per chunk:

```python
fused = run_monte_carlo(
    1_000_000,
    p=0.17,
    rho=0.23,
    seed=7,
    batch_size=262_144,
    backend="gpu",
    fusion_options={"enabled": True, "block_size": 256},
    plan_cache=cache,
)
print(fused.backend, fused.fusion_used, fused.kernel_count)
```

The kernel generates counter-based random values and computes errors,
syndromes, recovery, residuals, logical failures, and block reductions in
registers. It writes only two aggregate counters to global memory. Triton is
optional (`python3 -m pip install -e '.[triton]'`), selection is never implicit,
and unsupported schedules, missing Triton, sample-return requests, or kernel
construction failures fall back to eager PyTorch CUDA. CUDA absence still
falls back to NumPy. Fused RNG and reduction time are inseparable from kernel
time and are explicitly marked by `random_generation_fused=True` and
`reduction_fused=True` rather than being double-counted.

Compiled specialization selection is retained by the Stage 6 plan key. Triton
also owns its process-local binary cache. Numerical `p` and `rho` values remain
runtime arguments. The current int32 aggregate counters limit fused execution
to at most `(2**31-1)//3` shots per call; larger requests use eager CUDA.

When CUDA is unavailable, a GPU request falls back to NumPy and explicitly
reports `requested_backend="gpu"`, `backend="numpy"`, `device="cpu"`, and
`fallback_used=True`. Phase timings distinguish random generation,
host-to-device setup, vectorized kernel operations, reduction, and
device-to-host transfer. Full trajectory transfer occurs only when
`return_samples=True`.

Run the reproducible cold/warm benchmark with:

```bash
PYTHONPATH=src python3 benchmarks/benchmark_three_qubit_gpu.py
PYTHONPATH=src python3 benchmarks/benchmark_quantum_plan_cache.py
PYTHONPATH=src python3 benchmarks/benchmark_three_qubit_fused_gpu.py
```

Benchmark/export rows include `cache_hit`, `compilation_time`, `planning_time`,
`execution_time`, and `total_time`. The backend-performance plot labels cold
cache, warm cache, and uncached executions separately.

The CUDA path was validated on an NVIDIA GeForce RTX 4060 Laptop GPU with
PyTorch 2.4.1. Warm NumPy/GPU runtime ratios were 0.172x at 1,000 shots, 0.829x
at 10,000 shots, 7.448x at 100,000 shots, and 21.071x at 1,000,000 shots. Thus
GPU overhead dominated both small workloads, while the GPU was faster for the
two larger measured workloads. These are local results for one device,
software stack, seed, and batch size—not a general GPU speedup claim. The
1,000-shot GPU cold start was 0.272 seconds versus a 0.00105-second warm run.

On the same RTX 4060 Laptop GPU with Triton 3.0.0, the fused kernel used 40
registers/thread, reported 2 spills and 16 bytes of shared memory, with a 100%
register/thread-limit occupancy estimate. Its first 1,000-shot run took 0.376 s
including compilation, so fusion is inappropriate for one-off small jobs. Warm
total times for 1K/10K/100K/1M shots were respectively 0.512/0.406/0.386/0.697
ms, versus eager CUDA at 1.230/0.940/1.246/2.845 ms in this run. NumPy remained
faster at 1K shots (0.282 ms), while fused CUDA was fastest at 10K and above.
These are local measurements, include complete workflow costs, and do not imply
a general speedup on other GPUs, software versions, seeds, or batch sizes.
Generated PTX inspection for the measured specialization found two global
atomic operations and no ordinary global or local loads/stores; aggregate
counters are the only explicit trajectory outputs.

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
- Vectorized NumPy and optional high-level PyTorch CUDA Monte Carlo for the
  exact three-qubit correlated-XXX example, with explicit fallback and phase
  timings.
- Explicit, cached Triton fusion for aggregate three-qubit Monte Carlo, with
  eager CUDA fallback and register/shared-memory diagnostics when available.
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
