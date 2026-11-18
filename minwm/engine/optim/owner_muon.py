"""Owner-routed matrix orthogonalization for one-dimensional FSDP2 meshes."""

import math

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor, Shard

from .muon import Muon, zeropower_via_newtonschulz5
from .muon_nvtx import muon_range


class OwnerMuon(Muon):
    """Keep optimizer state sharded and run each full-matrix NS on one owner.

    Windows contain at most one matrix per rank. All ranks gather in the same
    order, owners compute their independent matrices, then scatter in that order.
    The parent implements the unchanged momentum, scaling, decay and AdamW rules.

    Args:
        params (iterable): FSDP2 parameters with a one-dimensional shard mesh.
        **kwargs: Original Muon hyperparameters.
    """

    def __init__(self, params, **kwargs):
        super().__init__(params, **kwargs)
        self._owner_plans = {}

    def _make_plan(self, params):
        if not params:
            return []
        mesh = params[0].device_mesh
        if mesh.ndim != 1:
            raise ValueError("OwnerMuon currently requires a one-dimensional FSDP mesh")
        size = mesh.size()
        loads = [0] * size
        queues = [[] for _ in range(size)]
        for p in params:
            if not isinstance(p, DTensor) or p.device_mesh != mesh:
                raise ValueError("OwnerMuon matrices must share a DTensor device mesh")
            if len(p.placements) != 1 or not isinstance(p.placements[0], Shard):
                raise ValueError("OwnerMuon expects one Shard placement per matrix")
            owner = min(range(size), key=lambda rank: loads[rank])
            rows, columns = p.shape[0], math.prod(p.shape[1:])
            loads[owner] += rows * columns * min(rows, columns)
            queues[owner].append(p)
        windows = []
        for index in range(max(map(len, queues))):
            windows.append([(owner, q[index]) for owner, q in enumerate(queues) if index < len(q)])
        return windows

    def _matrix_updates(self, params, group):
        if not params or not isinstance(params[0], DTensor):
            yield from super()._matrix_updates(params, group)
            return
        # Parameter order and ownership depend only on metadata, never rank-local timing.
        if id(group) not in self._owner_plans:
            self._owner_plans[id(group)] = self._make_plan(params)
        for window in self._owner_plans[id(group)]:
            entries = []
            for owner, p in window:
                if p.grad is None:
                    continue
                g = self._momentum_update(p, group)
                mesh = g.device_mesh
                pg = mesh.get_group()
                rank, size = dist.get_rank(pg), dist.get_world_size(pg)
                dim = g.placements[0].dim
                chunk = (g.shape[dim] + size - 1) // size
                padded_shape = list(g.shape)
                padded_shape[dim] = chunk
                local = g.to_local()
                padded = local.new_zeros(padded_shape)
                padded.narrow(dim, 0, local.shape[dim]).copy_(local)
                received = (
                    [torch.empty_like(padded) for _ in range(size)] if rank == owner else None
                )
                root = dist.get_global_rank(pg, owner)
                with muon_range(f"Muon/gather_to_owner/{owner}"):
                    dist.gather(padded, gather_list=received, dst=root, group=pg)
                full = None
                if received is not None:
                    with muon_range("Muon/reconstruct_on_owner"):
                        full = (
                            torch.cat(received, dim=dim).narrow(dim, 0, g.shape[dim]).contiguous()
                        )
                    del received
                entries.append((owner, p, g, padded, full, pg, root, dim, chunk))

            updates = {}
            for index, (owner, p, g, padded, full, pg, root, dim, chunk) in enumerate(entries):
                if full is not None:
                    with muon_range(f"Muon/owner_NS/{owner}"):
                        updates[id(p)] = zeropower_via_newtonschulz5(full, group["ns_steps"])
                entries[index] = (owner, p, g, padded, None, pg, root, dim, chunk)
            full = None

            for owner, p, g, padded, full, pg, root, dim, chunk in entries:
                pieces = None
                if id(p) in updates:
                    update = updates.pop(id(p))
                    padded_shape = list(update.shape)
                    padded_shape[dim] = chunk * dist.get_world_size(pg)
                    send = update.new_zeros(padded_shape)
                    send.narrow(dim, 0, update.shape[dim]).copy_(update)
                    pieces = [piece.contiguous() for piece in send.split(chunk, dim=dim)]
                with muon_range(f"Muon/scatter_from_owner/{owner}"):
                    dist.scatter(padded, scatter_list=pieces, src=root, group=pg)
                with muon_range("Muon/from_local"):
                    local = padded.narrow(dim, 0, g.to_local().shape[dim]).contiguous()
                    u = DTensor.from_local(
                        local,
                        g.device_mesh,
                        g.placements,
                        run_check=False,
                        shape=g.shape,
                        stride=g.stride(),
                    )
                yield p, u
