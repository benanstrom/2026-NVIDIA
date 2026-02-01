from __future__ import annotations

from typing import Any, Literal, Protocol

import numpy as np

Array = np.ndarray


class EnergyFn(Protocol):
    def __call__(self, seqs_pm1: Any, /) -> Any: ...


Device = Literal["cpu", "gpu"]
