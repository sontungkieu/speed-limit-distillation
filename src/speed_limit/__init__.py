"""Conditioning and absorption experiments for on-policy distillation."""

from .metrics import absorption_curve, absorption_times, aggregate_benchmark_accuracy
from .lanczos import lanczos_top_eigenpairs

__all__ = ["absorption_curve", "absorption_times", "aggregate_benchmark_accuracy", "lanczos_top_eigenpairs"]
