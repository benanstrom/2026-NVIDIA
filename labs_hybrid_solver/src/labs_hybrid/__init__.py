"""NVIDIA LABS hybrid solver (MIT iQuHACK 2026).

Non-negotiable architecture:
  Quantum seeder -> (optional post-selection) -> fixed MTS backbone -> logs/plots

Seeders are the *only* experimental axis. The MTS loop and budgets are held fixed.
"""

from .config import BenchmarkPlan, RunConfig, default_benchmark_plan
from .pipeline import run_experiment

__all__ = ["RunConfig", "BenchmarkPlan", "default_benchmark_plan", "run_experiment"]
