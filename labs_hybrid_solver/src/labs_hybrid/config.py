from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Literal

DeviceStr = Literal["cpu", "gpu", "auto"]


@dataclass(frozen=True)
class TabuParams:
    """Parameters for the *fixed* Memetic Tabu Search loop.

    Do not tune these per seeder.
    """

    tenure: int = 7
    aspiration: bool = True
    # Evaluation batching knob. 0 means evaluate the full neighborhood (K*N) at once.
    neighbor_batch: int = 0
    # GPU path logs best-trace every `trace_stride` iterations to limit host transfers.
    trace_stride: int = 5


@dataclass(frozen=True)
class SeederConfig:
    name: str
    shots: int
    K_out: int
    K_select: int | None = None
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RunConfig:
    """Single experiment configuration (one seeder + fixed MTS)."""

    N: int
    seeder: SeederConfig

    backend: DeviceStr = "auto"  # requested backend for seeder execution
    device: DeviceStr = "auto"  # requested device for MTS + energy

    # Fixed budgets across seeders within a benchmark gate:
    budget_s: float | None = 2.0
    max_iters: int | None = None

    tabu: TabuParams = field(default_factory=TabuParams)

    rng_seed: int = 0

    # Output paths
    out_dir: str = "results"
    run_tag: str = "dev"

    # Logging/plot knobs
    save_traces: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GateConfig:
    """One stage of the scaling plan (Gate 1..3). Gate 0 = pytest."""

    name: str
    N_list: list[int]
    seeders: list[SeederConfig]
    budget_s: float | None
    max_iters: int | None
    device: DeviceStr


@dataclass(frozen=True)
class BenchmarkPlan:
    """Encodes the required gate workflow (0→3)."""

    gates: list[GateConfig]
    gpu_budget_cap_s: float = 1800.0  # stop Gate 3 early if exceeded


def default_benchmark_plan() -> BenchmarkPlan:
    """Default plan per handoff spec.

    Gate 0 is correctness via pytest (run by script).
    Gates 1..3 run end-to-end experiments.
    """

    def S(
        name: str,
        shots: int = 256,
        K_out: int = 256,
        K_select: int = 32,
        params: dict[str, Any] | None = None,
    ) -> SeederConfig:
        return SeederConfig(name=name, shots=shots, K_out=K_out, K_select=K_select, params=params or {})

    seeders_all = [
        S("random"),
        S("qaoa", params={"p": 1}),
        S("dcqo", params={"steps": 24}),
        S(
            "dcqo_plus",
            params={
                "steps": 24,
                "trotter_config": {
                    "time_grid": "right_endpoint",
                    "ordering": "fixed",
                    "K_perm": 1,
                },
            },
        ),
        S("pce", params={"layers": 15}),
    ]

    # Gate 1: CPU integration (small N) — Random/QAOA/DCQO/DCQO+
    gate1_seeders = [seeders_all[i] for i in [0, 1, 2, 3]]

    gates = [
        GateConfig(
            name="gate1_cpu_small",
            N_list=[12, 16],
            seeders=gate1_seeders,
            budget_s=1.0,
            max_iters=None,
            device="cpu",
        ),
        GateConfig(
            name="gate2_gpu_bringup",
            N_list=[16],
            seeders=seeders_all,
            budget_s=1.0,
            max_iters=None,
            device="gpu",
        ),
        GateConfig(
            name="gate3_gpu_matrix",
            N_list=[31, 35],
            seeders=seeders_all,
            budget_s=3.0,
            max_iters=None,
            device="gpu",
        ),
    ]

    return BenchmarkPlan(gates=gates, gpu_budget_cap_s=300.0)
