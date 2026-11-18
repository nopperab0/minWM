"""Check owner routing and multi-step updates against the original Muon path."""

import argparse
import json
import os

import torch
import torch.distributed as dist
from _muon_upstream_reference import Muon
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Shard, distribute_tensor

from minwm.engine.optim.owner_muon import OwnerMuon


def main() -> None:
    """Run a small distributed update-equivalence check, without training a model."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.device == "cuda":
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl" if args.device == "cuda" else "gloo")
    mesh = init_device_mesh(args.device, (dist.get_world_size(),))
    shapes = [(8, 6), (7, 5), (3, 11), (2, 3, 2), (9,)]
    left, right = [], []
    for index, shape in enumerate(shapes):
        full = torch.arange(torch.tensor(shape).prod().item(), device=args.device).float()
        full = (full.sin() * 0.1).reshape(shape)
        placement = [Shard(1 if index == 1 else 0)]
        left.append(nn.Parameter(distribute_tensor(full, mesh, placement)))
        right.append(nn.Parameter(distribute_tensor(full.clone(), mesh, placement)))
    baseline, owner = Muon(left), OwnerMuon(right)
    max_error = 0.0
    for step in range(3):
        for index, (a, b) in enumerate(zip(left, right)):
            count = a.numel()
            grad = torch.arange(count, device=args.device).float().reshape(a.shape)
            grad = (grad + step + index).cos() * 0.01
            a.grad = distribute_tensor(grad, mesh, a.placements)
            b.grad = distribute_tensor(grad.clone(), mesh, b.placements)
        baseline.step()
        owner.step()
        for a, b in zip(left, right):
            actual, expected = b.full_tensor(), a.full_tensor()
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
            max_error = max(max_error, (actual - expected).abs().max().item())
    if dist.get_rank() == 0:
        result = {"status": "PASS", "steps": 3, "shapes": shapes, "max_abs_error": max_error}
        with open(args.output, "w") as handle:
            json.dump(result, handle, indent=2)
        print(json.dumps(result), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
