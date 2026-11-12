from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Mapping

import torch


def absorption_curve(values: torch.Tensor) -> torch.Tensor:
    """Normalize mean per-state KL by its step-zero value."""
    values = torch.as_tensor(values, dtype=torch.float64)
    if values.ndim != 2:
        raise ValueError("values must have shape [steps, states]")
    mean = values.mean(dim=1)
    if not torch.isfinite(mean).all() or mean[0] <= 0:
        raise ValueError("KL values must be finite and have positive step-zero mean")
    return mean / mean[0]


def absorption_times(curve: torch.Tensor, steps=None) -> dict:
    curve = torch.as_tensor(curve)
    if curve.ndim != 1 or not torch.isfinite(curve).all():
        raise ValueError("finite one-dimensional curve required")
    steps = list(range(len(curve))) if steps is None else list(steps)
    if len(steps) != len(curve): raise ValueError("step alignment mismatch")
    result = {"budget": steps[-1], "final_A": float(curve[-1])}
    for absorption, threshold in ((50,.5),(90,.1)):
        hits = torch.where(curve < threshold)[0]
        result[f"tau_{absorption}"] = steps[int(hits[0])] if len(hits) else None
        result[f"tau_{absorption}_censored"] = not bool(len(hits))
    return result


def aggregate_benchmark_accuracy(records: Iterable[Mapping]) -> dict:
    counts = defaultdict(lambda: [0, 0])
    for record in records:
        item = counts[str(record["benchmark"])]
        item[0] += int(bool(record["correct"]))
        item[1] += 1
    if not counts:
        raise ValueError("empty evaluation")
    scores = {name: correct / total for name, (correct, total) in counts.items()}
    return {"benchmarks": scores, "macro_accuracy": sum(scores.values()) / len(scores), "records": sum(v[1] for v in counts.values())}
