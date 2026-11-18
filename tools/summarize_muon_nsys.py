"""Summarize complete-step timings and overlap-aware Nsight kernel intervals."""

import argparse
import bisect
import csv
import json
import sqlite3
import statistics
from collections import defaultdict
from pathlib import Path


def union_ns(intervals: list[tuple[int, int]]) -> int:
    """Return interval-union duration, avoiding overlap double counting."""
    total, end = 0, 0
    for start, stop in sorted(intervals):
        total += max(0, stop - max(start, end))
        end = max(end, stop)
    return total


def analyze(path: Path) -> dict:
    """Attribute kernels through launch correlations and preserve overlap caveats."""
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    tables = {r[0] for r in con.execute("select name from sqlite_master where type='table'")}
    if "CUPTI_ACTIVITY_KIND_KERNEL" not in tables or "NVTX_EVENTS" not in tables:
        raise RuntimeError(f"{path}: missing CUDA kernel or NVTX records")
    strings = dict(con.execute("select id,value from StringIds"))
    ranges = []
    by_thread = defaultdict(list)
    for row in con.execute("select * from NVTX_EVENTS where end > start"):
        r = dict(row)
        name = r.get("text") or strings.get(r.get("textId"), "")
        r["name"] = name
        if name.startswith(("Train_step/", "Muon/")) or name in {
            "Forward",
            "Backward_FSDP",
            "Gradient_clipping",
            "Gradient_clipping_global_norm",
            "Optimizer_step",
            "Text_conditioning",
            "Preprocess",
            "Data",
        }:
            ranges.append(r)
            by_thread[r["globalTid"]].append(r)
    phase_ranges = sorted(
        [r for r in ranges if r["name"] in {"Forward", "Backward_FSDP", "Optimizer_step"}],
        key=lambda r: r["start"],
    )
    phase_starts = [r["start"] for r in phase_ranges]
    launches = defaultdict(list)
    for table in ("CUPTI_ACTIVITY_KIND_RUNTIME", "CUPTI_ACTIVITY_KIND_DRIVER"):
        if table in tables:
            for r in con.execute(f"select * from {table}"):
                launches[r["globalTid"]].append(dict(r))
    correlations = {}
    inferred = 0
    for thread, calls in launches.items():
        rs = sorted(by_thread[thread], key=lambda r: (r["start"], -r["end"]))
        cursor, active = 0, []
        for call in sorted(calls, key=lambda r: r["start"]):
            time = call["start"]
            active = [r for r in active if r["end"] > time]
            while cursor < len(rs) and rs[cursor]["start"] <= time:
                if rs[cursor]["end"] > time:
                    active.append(rs[cursor])
                cursor += 1
            names = [r["name"] for r in active]
            phase = next(
                (
                    n
                    for n in names
                    if n
                    in {
                        "Forward",
                        "Backward_FSDP",
                        "Optimizer_step",
                        "Gradient_clipping",
                        "Text_conditioning",
                        "Preprocess",
                    }
                ),
                "unattributed",
            )
            if phase == "unattributed":
                index = bisect.bisect_right(phase_starts, time) - 1
                if index >= 0 and time < phase_ranges[index]["end"]:
                    phase = phase_ranges[index]["name"]
                    inferred += 1
            leaf = next((n for n in reversed(names) if n.startswith("Muon/")), "")
            step = next((n for n in names if n.startswith("Train_step/")), "")
            correlations[call["correlationId"]] = (phase, leaf, step)

    groups = defaultdict(list)
    kernels = []
    for row in con.execute("select * from CUPTI_ACTIVITY_KIND_KERNEL order by start"):
        r = dict(row)
        name = strings.get(r.get("demangledName"), strings.get(r.get("shortName"), ""))
        phase, leaf, step = correlations.get(r["correlationId"], ("unattributed", "", ""))
        interval = (r["start"], r["end"])
        is_nccl = "nccl" in name.lower()
        groups["all_gpu"].append(interval)
        groups["nccl" if is_nccl else "non_nccl"].append(interval)
        groups[f"phase/{phase}/gpu"].append(interval)
        if leaf:
            groups[f"leaf/{leaf}/gpu"].append(interval)
        if "/NS/" in leaf:
            groups["ns_compute_gpu"].append(interval)
        kernels.append(
            dict(
                start=r["start"],
                end=r["end"],
                name=name,
                phase=phase,
                leaf=leaf,
                step=step,
                stream=r.get("streamId"),
            )
        )
    nccl = union_ns(groups["nccl"])
    compute = union_ns(groups["non_nccl"])
    all_gpu = union_ns(groups["all_gpu"])
    overlap = nccl + compute - all_gpu
    envelope = max(r["end"] for r in kernels) - min(r["start"] for r in kernels)
    con.close()
    return dict(
        sqlite=str(path),
        gpu_union_ms=all_gpu / 1e6,
        nccl_union_ms=nccl / 1e6,
        non_nccl_union_ms=compute / 1e6,
        nccl_compute_overlap_ms=overlap / 1e6,
        nccl_without_local_compute_ms=(nccl - overlap) / 1e6,
        gpu_idle_in_capture_envelope_ms=(envelope - all_gpu) / 1e6,
        kernel_union_ms={k: union_ns(v) / 1e6 for k, v in groups.items()},
        nvtx_counts=dict(
            (n, sum(r["name"] == n for r in ranges))
            for n in {
                "Muon/full_tensor",
                "Muon/from_local",
                "Muon/NS/normalization",
            }
        ),
        nvtx_host_ms={
            name: sum(r["end"] - r["start"] for r in ranges if r["name"] == name) / 1e6
            for name in {r["name"] for r in ranges if r["name"].startswith("Muon/")}
        },
        cross_thread_api_attributions=inferred,
        kernels=kernels,
        note="NCCL without local compute is exposed activity, not measured network transfer time. "
        "It can contain rank skew and device-side waiting. GPU idle also includes CPU launch "
        "gaps and dependencies; cross-thread phase assignment uses enclosing host time.",
    )


def main() -> None:
    """Write A/B metrics alongside per-rank timeline attribution."""
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    root = args.directory
    rows, details = [], {}
    for mode in ("baseline", "owner"):
        summary = json.loads((root / f"{mode}.summary.json").read_text())
        step = statistics.median(summary["slowest_rank_step_ms"])
        metrics = {"Step (ms)": step}
        metrics.update(
            {f"{k} stream elapsed (ms)": v for k, v in summary["phase_stream_elapsed_ms"].items()}
        )
        metrics["Optimizer share (%)"] = (
            100 * summary["phase_stream_elapsed_ms"]["Optimizer_step"] / step
        )
        metrics["Peak allocated (GiB)"] = summary["peak_allocated_bytes"] / 2**30
        metrics["Peak reserved (GiB)"] = summary["peak_reserved_bytes"] / 2**30
        traces = []
        for path in sorted(root.glob(f"{mode}.rank*.sqlite")):
            detail = analyze(path)
            traces.append(detail)
            (root / (path.stem + ".attribution.json")).write_text(json.dumps(detail))
        if not traces:
            raise RuntimeError(f"No Nsight SQLite traces for {mode}")
        nsteps = len(summary["slowest_rank_step_ms"])
        for key in (
            "nccl_union_ms",
            "nccl_compute_overlap_ms",
            "nccl_without_local_compute_ms",
            "gpu_idle_in_capture_envelope_ms",
        ):
            metrics[f"{key} per step (ms)"] = max(t[key] / nsteps for t in traces)
        metrics["NS compute union per step (ms)"] = max(
            t["kernel_union_ms"].get("ns_compute_gpu", 0) / nsteps for t in traces
        )
        host_names = {name for trace in traces for name in trace["nvtx_host_ms"]}
        for name in sorted(host_names):
            metrics[f"{name} host range per step (ms)"] = max(
                trace["nvtx_host_ms"].get(name, 0) / nsteps for trace in traces
            )
        for metric, value in metrics.items():
            rows.append(dict(mode=mode, metric=metric, value=value))
        details[mode] = metrics
    with (root / "comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["mode", "metric", "value"])
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# Stage 0 Muon profiling", "", "| Metric | Original | Owner |", "|---|---:|---:|"]
    for metric in sorted(set(details["baseline"]) | set(details["owner"])):
        before, after = details["baseline"].get(metric), details["owner"].get(metric)
        left = "N/A" if before is None else f"{before:.4g}"
        right = "N/A" if after is None else f"{after:.4g}"
        lines.append(f"| {metric} | {left} | {right} |")
    lines += [
        "",
        "NCCL union is not a communication-bottleneck verdict. Non-overlapped NCCL "
        "can include waiting for other ranks. Inspect the retained complete traces and "
        "CUDA stream dependencies before interpreting it as exposed network cost.",
    ]
    (root / "comparison.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
