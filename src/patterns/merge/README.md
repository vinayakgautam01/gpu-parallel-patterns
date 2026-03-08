# Merge Pattern

Parallel merge of two sorted integer arrays using PMPP-style co-rank partitioning.

## Variants

| Variant | File | Strategy |
|---|---|---|
| Baseline | `baseline.cu` | Per-thread merge partition using two co-rank calls on global memory |
| Opt1 | `opt1_tiled.cu` | Block-level co-rank + shared-memory tiled merge |
| Opt2 | `opt2_circular_buf.cu` | Shared-memory tiled merge with circular buffering to avoid redundant loads |

## File layout

```
merge/
├── cpu_ref.hpp          # Serial CPU merge (correctness oracle)
├── cpu_ref_test.cpp     # CPU correctness tests
├── kernels.hpp          # Public API + co_rank helper
├── dispatch.cu          # Variant dispatcher
├── baseline.cu          # GPU baseline
├── opt1_tiled.cu        # Tiled shared-memory merge
├── opt2_circular_buf.cu # Circular-buffer tiled merge
├── gpu_test.cu          # GPU correctness tests vs CPU reference
├── bench.cu             # Benchmark binary
└── README.md
```

## Notes

- Inputs `A` and `B` must each be sorted in non-decreasing order.
- Merge is stable (`A` wins ties).
- `co_rank` is used at both global and tile scope to map output ranges to input ranges.

## Running

```bash
# Build
bash scripts/build.sh

# Tests
./build/bin/merge_cpu_test
./build/bin/merge_gpu_test

# Benchmark sweep
bash scripts/bench_merge.sh
```
