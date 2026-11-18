"""Capture complete Stage-0 steps with asynchronous NVTX Muon phase labels."""

import argparse
import functools
import hashlib
import json
import os
import statistics
import time
from pathlib import Path

import torch
import torch.distributed as dist

from minwm.config import MWMConfig, apply_overrides, load
from minwm.engine import BaseTrainer, recipe_base
from minwm.engine.optim.muon_nvtx import muon_range


def main() -> None:
    """Warm up, then retain a complete n-step CUDA/NVTX capture on each rank."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["baseline", "owner"], required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("opts", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    os.environ.update(
        MINWM_OPT_SP_FUSED="0",
        MINWM_OPT_RECOMPUTE="full",
        MINWM_OPT_ATTN_BACKEND="original",
        MINWM_OPT_FSDP_PREFETCH="0",
        MINWM_MUON_NVTX="1",
    )
    cfg = apply_overrides(load("configs/wan21/action2v/train/stage0_bi_sft.py"), args.opts)
    cfg["training"].update(
        max_steps=0,
        output_dir="",
        seed=None,
        log_interval=0,
        ckpt_interval=0,
        eval_interval=0,
        sp_size=4,
    )
    cfg["data"].update(shuffle=False, num_workers=0, persistent_workers=False)
    cfg["checkpoint"] = dict(resume=False)
    cfg["recipe"]["optimizer"] = dict(
        name=(
            "minwm.engine.optim.muon:Muon"
            if args.mode == "baseline"
            else "minwm.engine.optim.owner_muon:OwnerMuon"
        ),
        lr=2e-6,
        weight_decay=0.01,
        momentum=0.95,
        nesterov=True,
        ns_steps=5,
        adamw_betas=(0.0, 0.999),
    )
    trainer = BaseTrainer(MWMConfig.from_dict(cfg))
    trainer.load_checkpoint()
    rank = dist.get_rank()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    initial_file = output / f"initial.rank{rank}.pt"
    from minwm.distributed.rng import get_rng_states_tracker

    if args.mode == "baseline":
        initial = dict(
            model={
                name: (value.to_local() if hasattr(value, "to_local") else value).detach().cpu()
                for name, value in trainer.model.state_dict().items()
            },
            cpu_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state(),
            tracker=get_rng_states_tracker().state_dict(),
        )
        torch.save(initial, initial_file)
        del initial
    else:
        initial = torch.load(initial_file, map_location="cpu", weights_only=False)
        with torch.no_grad():
            for name, value in trainer.model.state_dict().items():
                local = value.to_local() if hasattr(value, "to_local") else value
                local.copy_(initial["model"][name].to(local.device))
        torch.set_rng_state(initial["cpu_rng"])
        torch.cuda.set_rng_state(initial["cuda_rng"])
        get_rng_states_tracker().load_state_dict(initial["tracker"])
        del initial
    optimizer = trainer.optimizers["main"]
    records = []
    active = False

    def wrap(name, function):
        @functools.wraps(function)
        def run(*a, **kw):
            if not active:
                return function(*a, **kw)
            start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            with muon_range(name):
                start.record()
                result = function(*a, **kw)
                end.record()
            records.append((name, start, end))
            return result

        return run

    recipe = trainer.recipe
    recipe.adapter.denoise = wrap("Forward", recipe.adapter.denoise)
    recipe.adapter.conditioning = wrap("Text_conditioning", recipe.adapter.conditioning)
    recipe.preprocess = wrap("Preprocess", recipe.preprocess)
    backward = torch.autograd.backward
    clip = torch.nn.utils.clip_grad_norm_
    norm_value = recipe_base._grad_norm_value
    torch.autograd.backward = wrap("Backward_FSDP", backward)
    torch.nn.utils.clip_grad_norm_ = wrap("Gradient_clipping", clip)
    recipe_base._grad_norm_value = wrap("Gradient_clipping_global_norm", norm_value)
    optimizer.step = wrap("Optimizer_step", optimizer.step)
    step_events, losses, host_ms = [], [], []

    def train_step(index):
        with muon_range(f"Train_step/{index}"):
            with muon_range("Data"):
                batch = trainer._next_batch()
            return recipe.train_one_step(
                trainer.model,
                batch,
                trainer.optimizers,
                index,
                auxiliary_models=trainer.auxiliary_models,
            )

    try:
        # No EventStorage context: its production timers synchronize CUDA after
        # forward/backward, which would perturb the overlap being profiled.
        for index in range(args.warmup):
            metrics = train_step(index)
            print(f"warmup {args.mode} rank={rank} step={index} loss={metrics['loss']}", flush=True)
        dist.barrier()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        active = True
        torch.cuda.cudart().cudaProfilerStart()
        for index in range(args.steps):
            start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            start.record()
            clock = time.perf_counter()
            metrics = train_step(index)
            host_ms.append((time.perf_counter() - clock) * 1000)
            end.record()
            step_events.append((start, end))
            losses.append(metrics["loss"])
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()
        active = False
        phase_ms = {}
        for name, start, end in records:
            phase_ms.setdefault(name, []).append(start.elapsed_time(end))
        result = dict(
            mode=args.mode,
            rank=rank,
            steps=args.steps,
            warmup=args.warmup,
            step_ms=[start.elapsed_time(end) for start, end in step_events],
            host_step_ms=host_ms,
            phase_stream_elapsed_ms=phase_ms,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            losses=losses,
            optimizer=type(optimizer).__name__,
            torch=torch.__version__,
            initialization="same saved local weights and RNG state; no fixed seed argument",
            gpu=torch.cuda.get_device_name(),
            config=cfg,
            model_parameters=sum(p.numel() for p in trainer.model.parameters()),
            source_hashes={
                str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in Path("minwm/engine/optim").glob("*muon*.py")
            },
        )
        (output / f"{args.mode}.rank{rank}.json").write_text(
            json.dumps(result, indent=2, default=str)
        )
        ranks = [None] * dist.get_world_size()
        dist.all_gather_object(ranks, result)
        if rank == 0:
            summary = dict(
                mode=args.mode,
                slowest_rank_step_ms=[
                    max(r["step_ms"][i] for r in ranks) for i in range(args.steps)
                ],
                phase_stream_elapsed_ms={
                    name: max(statistics.median(r["phase_stream_elapsed_ms"][name]) for r in ranks)
                    for name in phase_ms
                },
                peak_allocated_bytes=max(r["peak_allocated_bytes"] for r in ranks),
                peak_reserved_bytes=max(r["peak_reserved_bytes"] for r in ranks),
                note="Phase event times include stream dependencies/waits; use Nsight for kernel attribution.",
            )
            (output / f"{args.mode}.summary.json").write_text(json.dumps(summary, indent=2))
            print(json.dumps(summary), flush=True)
    finally:
        torch.autograd.backward = backward
        torch.nn.utils.clip_grad_norm_ = clip
        recipe_base._grad_norm_value = norm_value
        trainer.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
