# OwnerMuon: matrix ownership for FSDP2

This fork adds an opt-in, one-dimensional FSDP2 adaptation of the matrix-owner idea used by Layer-Sharded Muon. Each matrix is reconstructed on one owner, orthogonalized once, and its update scattered back to the original shards. The default optimizer remains unchanged.

## Measured Stage-0 result

CSF3 job **24847964** completed successfully (exit `0:0`, elapsed `00:19:18`). Wan2.1 1.3B Stage-0, **4 A100 80GB**, SP=4, BF16, batch size 1, 20 latent frames (latent shape `20 × 16 × 60 × 104`), PRoPE and full-block checkpointing. Both variants use Muon, including the baseline: this is **not an AdamW-versus-Muon comparison**. HunyuanVideo 8B and Stage-1 were not measured.

Three warm-up steps precede five complete captured steps per variant. The owner run restores the baseline's initial local model shards and CPU/CUDA/tracker RNG states. There is no seed sweep or convergence run. Software: PyTorch `2.9.1+cu128`, FlashAttention `2.8.3`, Nsight Systems `2024.6.2.225`.

| Metric | Original Muon | OwnerMuon |
|---|---:|---:|
| Full-step median, ms | 4524.485 | 4276.321 |
| Forward, ms | 827.8 | 838.5 |
| Backward including FSDP, ms | 2487 | 2488 |
| Gradient clipping, ms | 3.215 | 3.226 |
| Optimizer step, ms | 1097 | 846.1 |
| Optimizer / full step | 24.25% | 19.79% |
| NS GPU compute union / step, ms | 390.8 | 101.1 |
| NCCL kernel union / step, ms | 568.9 | 644.7 |
| NCCL overlapping local compute / step, ms | 73.43 | 78.46 |
| NCCL without local compute / step, ms | 498.8 | 566.2 |
| GPU idle in capture envelope / step, ms | 747.8 | 732.3 |
| Peak allocated memory, GiB | 16.51 | 16.51 |
| Peak reserved memory, GiB | 18.86 | 18.86 |

Step latency falls **5.49%** (equivalent throughput increase **5.80%**); optimizer time falls approximately **22.9%**. The maximum-rank NS GPU compute union falls approximately **74.1%**, consistent with distributing formerly duplicated NS work over four owners. Backward remains the largest phase. **Peak memory does not improve in this measurement**, and NCCL kernel activity increases.

Full-step samples are the slowest rank's CUDA event duration per step. Phase values are maximum-rank medians and need not add to the step median. Kernel union values summarize interval unions per rank and then take the maximum-rank summary; they are not summed cluster compute or bytes transferred. NCCL-without-compute is computed per rank before aggregation, so the displayed maxima need not subtract exactly. Long NCCL kernels may include rank skew, dependency waits and device-side waiting; they do not by themselves establish network saturation. GPU idle also does not uniquely identify a network bottleneck.

These are short **Nsight-captured** timings, not unprofiled production throughput or convergence evidence. The checked-in [measurement record](experiments/owner_muon_24847964.json) is transcribed from the previously inspected job output, not a raw profiler export. Full eight rank traces, SQLite exports, environment and source-hash records were retained remotely under `~/scratch/minWM-20261102-new/outputs/muon-nsys-24847964/`; they are not bundled here. This publication checkout isolates the optimizer/profiler files from a workspace containing other edits; the upstream base is `2a54f4da5ffc9676c02bb61aa1a2cb22a52a84ff`. Exact measured workspace identity must be checked against the remote `source.sha256`, rather than treating this new publication commit as the measured commit.

## How it works

The original DTensor Muon path calls `full_tensor()` on each rank, runs full-matrix Newton–Schulz on each rank, then converts the result back to a DTensor for the parameter update. OwnerMuon changes this distributed execution, retaining the original optimizer mathematics:

1. Cache a deterministic ownership plan, greedily balancing estimated NS cost `rows * columns * min(rows, columns)`.
2. Update sharded momentum/Nesterov state and convert the direction to BF16, as in the original implementation.
3. Pad uneven shards and gather each matrix to its owner. Only that owner reconstructs the complete matrix.
4. Run the same normalization and five quintic Newton–Schulz iterations once per matrix. Windows allow at most one matrix per owner, so different owners process independent matrices.
5. Scatter the update, trim padding and construct a DTensor with the original shape, stride, mesh and placements. Apply the original scaling, weight decay and parameter update. Vectors/scalars retain the original AdamW fallback.

There is still full-matrix reconstruction, concentrated on the owner. Communication is per-matrix gather/scatter, not eliminated. Windows bound simultaneously reconstructed matrices, but the measured whole-step memory peak is unchanged. There is no explicit communication/NS pipeline or extra CUDA stream scheduling.

The supported distributed layout is one shared **1-D DeviceMesh with a single Shard placement**; row/column and uneven shards are handled. Multi-axis HSDP/TP layouts are unsupported. Parameter ordering and gradient presence must agree across ranks, as required by the fixed collective schedule. Use a fresh optimizer when switching implementations; optimizer-state checkpoint interchange has not been established by this experiment.

## Relationship to Layer-Sharded Muon

Primary reference: NVIDIA Megatron-LM [LayerShardedMuon](https://github.com/NVIDIA/Megatron-LM/blob/bfab98f4a3ca7bfab11f7fee7672e95be54a70b6/megatron/core/optimizer/layer_sharded_muon.py) and [routing implementation](https://github.com/NVIDIA/Megatron-LM/blob/bfab98f4a3ca7bfab11f7fee7672e95be54a70b6/megatron/core/optimizer/layer_sharded_a2a.py), pinned to commit `bfab98f4a3ca7bfab11f7fee7672e95be54a70b6`.

| Aspect | This implementation | Megatron reference |
|---|---|---|
| Matrix ownership | One owner per matrix | NS home per matrix |
| Full-matrix NS | Once on owner | Once on home |
| Return updates | Original FSDP shards | Original distributed shards |
| Parallel layout | 1-D FSDP2 DTensor | GTP-remat × TP, with additional parameter metadata |
| Routing | Per-matrix gather / scatter | Staged, packed all-to-all routing and reverse routing |
| NS scheduling | One matrix per owner per window | Same-shape batching; optional SYRK backend |
| Stream scheduling | No explicit overlap | Optional concurrent parameter-group streams with communicator constraints |

Thus the **core distributed algorithm is the same idea**, while this is a narrower FSDP2 implementation, not a port of the complete Megatron optimizer. Its NS coefficients/scaling remain minWM's original choices; equality to Megatron's configurable numerical implementation is not claimed. NVIDIA source was used as a conceptual reference, not copied into OwnerMuon. minWM's existing Keller Jordan Muon attribution is retained.

Megatron [issue #6719](https://github.com/NVIDIA/Megatron-LM/issues/6719) discusses pipelining layer-sharded communication to hide exchange latency and reduce temporary memory. It is an **issue, not PR #6719**, and this implementation does not implement that proposed pipeline.

## Use and reproduce

Select the optimizer through the existing recipe configuration:

```python
optimizer = dict(
    name="minwm.engine.optim.owner_muon:OwnerMuon",
    lr=2e-6, weight_decay=0.01, momentum=0.95,
    nesterov=True, ns_steps=5, adamw_betas=(0.0, 0.999),
)
```

The CSF script assumes model weights and preprocessed LMDB already exist in scratch, plus the named conda environment and site modules. Adapt those paths to your allocation; it does not download assets or build the environment. From an installed checkout with `dataset/benchmark160/data` populated:

```bash
mkdir -p slurm_logs
MINWM_ROOT="$PWD" sbatch scripts/csf3_muon_profile.sbatch
```

The script runs one small distributed update-equivalence check, then baseline and owner, each with three warm-up and five full captured steps. Captures include forward, backward/FSDP, clipping, optimizer and detailed Muon NVTX ranges. `MINWM_MUON_NVTX=1` enables markers without introducing synchronizations. The summarizer distinguishes GPU compute, NCCL overlap and idle intervals instead of treating all NCCL time as communication cost. Initial model/RNG snapshots are trusted local artifacts; do not load snapshots from untrusted sources.

The completed four-rank NCCL equivalence fixture checked three updates, shapes `(8,6)`, `(7,5)`, `(3,11)`, `(2,3,2)`, `(9,)`, and reported maximum absolute parameter error **0** versus the retained upstream reference. A two-rank CPU/Gloo fixture and local refactored-baseline comparison also reported zero error. These checks cover the exercised paths, not training convergence or every possible layout.
