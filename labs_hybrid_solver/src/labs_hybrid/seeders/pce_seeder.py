"""PCE seeder — Pauli Correlation Encoding (Sciorilli et al. 2025, arXiv:2506.17391).

Encodes N binary variables into n = ceil(log4(N+1)) qubits, optimizes a
relaxed LABS cost via brickwork ansatz on a classical state-vector simulator
(numpy + scipy).  No CUDA-Q required — n is tiny (e.g. n=4 for N<=45).
"""
from __future__ import annotations

import math
import time
from typing import Any

import numpy as np
from scipy.optimize import minimize

from ..labs_energy_cpu import labs_energy_batch
from ..rng import get_np_rng, sample_pm1

# ---------------------------------------------------------------------------
# Pauli utilities
# ---------------------------------------------------------------------------

def _num_qubits_for_N(N: int) -> int:
    """Smallest n such that 4^n - 1 >= N."""
    n = 1
    while 4**n - 1 < N:
        n += 1
    return n


def _enumerate_paulis(n: int) -> list[tuple[int, int]]:
    """All 4^n - 1 non-identity Pauli operators as (x, z) symplectic pairs.

    Each Pauli on n qubits is P = i^phase * X^{x} Z^{z}  where x,z are
    n-bit integers.  We enumerate all (x, z) != (0, 0).
    """
    paulis = []
    for code in range(1, 4**n):
        x, z = 0, 0
        c = code
        for q in range(n):
            r = c % 4
            c //= 4
            if r == 1:          # X
                x |= 1 << q
            elif r == 2:        # Z
                z |= 1 << q
            elif r == 3:        # Y = iXZ
                x |= 1 << q
                z |= 1 << q
        paulis.append((x, z))
    return paulis


def _symplectic_inner(x1: int, z1: int, x2: int, z2: int) -> int:
    """Symplectic inner product mod 2.  Returns 0 if operators commute, 1 otherwise."""
    return bin((x1 & z2) ^ (z1 & x2)).count("1") % 2


# ---------------------------------------------------------------------------
# Operator selection
# ---------------------------------------------------------------------------

def _select_commuting_set(
    n: int, N: int, rng: np.random.Generator,
) -> list[tuple[int, int]]:
    """Greedy selection of N commuting Pauli operators from the full set."""
    all_paulis = _enumerate_paulis(n)
    indices = list(range(len(all_paulis)))
    rng.shuffle(indices)

    selected: list[tuple[int, int]] = []
    for idx in indices:
        if len(selected) >= N:
            break
        x, z = all_paulis[idx]
        if all(_symplectic_inner(x, z, sx, sz) == 0 for sx, sz in selected):
            selected.append((x, z))
    # Should always succeed since 4^n - 1 >= N and a maximal commuting set
    # has size 2^n - 1 >= N for the n we choose.
    if len(selected) < N:
        # Fallback: pad with random non-identity Paulis (shouldn't happen).
        remaining = [all_paulis[i] for i in indices if all_paulis[i] not in selected]
        selected.extend(remaining[: N - len(selected)])
    return selected[:N]


def _select_noncommuting_set(
    n: int, N: int, rng: np.random.Generator,
) -> list[tuple[int, int]]:
    """Select N operators favouring anti-commutation.

    Build an anti-commuting core of up to 2n+1 operators, then extend by
    picking operators that maximise anti-commutation count with selected set.
    """
    all_paulis = _enumerate_paulis(n)
    indices = list(range(len(all_paulis)))
    rng.shuffle(indices)

    selected: list[tuple[int, int]] = []

    # Phase 1: greedy anti-commuting core (max 2n+1)
    for idx in indices:
        if len(selected) >= min(2 * n + 1, N):
            break
        x, z = all_paulis[idx]
        if all(_symplectic_inner(x, z, sx, sz) == 1 for sx, sz in selected):
            selected.append((x, z))

    if len(selected) >= N:
        return selected[:N]

    # Phase 2: extend by max anti-commutation count
    selected_set = set(selected)
    remaining = [(all_paulis[i], i) for i in indices if all_paulis[i] not in selected_set]

    while len(selected) < N and remaining:
        best_score = -1
        best_idx = 0
        for j, ((x, z), _) in enumerate(remaining):
            score = sum(
                _symplectic_inner(x, z, sx, sz) for sx, sz in selected
            )
            if score > best_score:
                best_score = score
                best_idx = j
        chosen, _ = remaining.pop(best_idx)
        selected.append(chosen)

    return selected[:N]


def _select_operators(
    n: int, N: int, op_set: str, rng: np.random.Generator,
) -> list[tuple[int, int]]:
    if op_set == "commuting":
        return _select_commuting_set(n, N, rng)
    return _select_noncommuting_set(n, N, rng)


# ---------------------------------------------------------------------------
# State-vector simulation (2^n complex arrays)
# ---------------------------------------------------------------------------

def _pauli_matrix(n: int, x: int, z: int) -> np.ndarray:
    """Build 2^n x 2^n Pauli matrix from symplectic (x, z) representation."""
    I2 = np.eye(2, dtype=np.complex128)
    X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)

    mat = np.array([[1.0]], dtype=np.complex128)
    for q in range(n):
        xq = (x >> q) & 1
        zq = (z >> q) & 1
        if xq == 0 and zq == 0:
            gate = I2
        elif xq == 1 and zq == 0:
            gate = X
        elif xq == 0 and zq == 1:
            gate = Z
        else:
            gate = Y
        mat = np.kron(mat, gate)
    return mat


def _apply_ry(state: np.ndarray, qubit: int, theta: float, n: int) -> np.ndarray:
    """Apply RY(theta) to qubit in-place on state vector of n qubits."""
    dim = 1 << n
    c = math.cos(theta / 2)
    s = math.sin(theta / 2)
    step = 1 << qubit
    new_state = state.copy()
    for i in range(dim):
        if i & step:
            continue
        j = i | step
        new_state[i] = c * state[i] - s * state[j]
        new_state[j] = s * state[i] + c * state[j]
    return new_state


def _apply_rz(state: np.ndarray, qubit: int, theta: float, n: int) -> np.ndarray:
    """Apply RZ(theta) to qubit on state vector."""
    dim = 1 << n
    step = 1 << qubit
    phase_0 = np.exp(-1j * theta / 2)
    phase_1 = np.exp(1j * theta / 2)
    new_state = state.copy()
    for i in range(dim):
        if i & step:
            new_state[i] = phase_1 * state[i]
        else:
            new_state[i] = phase_0 * state[i]
    return new_state


def _apply_rzz(state: np.ndarray, qa: int, qb: int, theta: float, n: int) -> np.ndarray:
    """Apply RZZ(theta) to qubits qa, qb.  Diagonal phase gate."""
    dim = 1 << n
    new_state = state.copy()
    for i in range(dim):
        ba = (i >> qa) & 1
        bb = (i >> qb) & 1
        parity = ba ^ bb
        if parity:
            new_state[i] = np.exp(1j * theta / 2) * state[i]
        else:
            new_state[i] = np.exp(-1j * theta / 2) * state[i]
    return new_state


def _entangling_pairs(n: int, layer_idx: int) -> list[tuple[int, int]]:
    """Brickwork entangling pairs for n qubits.

    Even layers: (0,1), (2,3), ...
    Odd layers:  (1,2), (3,0) for n=4; generally shifted pairs with wrap-around.
    """
    pairs = []
    if layer_idx % 2 == 0:
        for i in range(0, n - 1, 2):
            pairs.append((i, i + 1))
    else:
        for i in range(1, n - 1, 2):
            pairs.append((i, i + 1))
        if n >= 2:
            pairs.append((n - 1, 0))
    return pairs


def _brickwork_circuit(theta: np.ndarray, n: int, layers: int) -> np.ndarray:
    """Simulate brickwork ansatz from |0...0> and return state vector.

    Parameter layout per layer:
      n RY params + n RZ params + len(entangling_pairs) RZZ params
    """
    dim = 1 << n
    state = np.zeros(dim, dtype=np.complex128)
    state[0] = 1.0

    idx = 0
    for layer in range(layers):
        # Single-qubit RY layer
        for q in range(n):
            state = _apply_ry(state, q, theta[idx], n)
            idx += 1
        # Single-qubit RZ layer
        for q in range(n):
            state = _apply_rz(state, q, theta[idx], n)
            idx += 1
        # Two-qubit RZZ layer
        pairs = _entangling_pairs(n, layer)
        for qa, qb in pairs:
            state = _apply_rzz(state, qa, qb, theta[idx], n)
            idx += 1

    return state


def _params_per_layer(n: int, layer_idx: int) -> int:
    """Number of variational parameters in one brickwork layer."""
    return n + n + len(_entangling_pairs(n, layer_idx))


def _total_params(n: int, layers: int) -> int:
    """Total number of variational parameters for the ansatz."""
    return sum(_params_per_layer(n, l) for l in range(layers))


# ---------------------------------------------------------------------------
# Cost function + optimization
# ---------------------------------------------------------------------------

def _expectation(state: np.ndarray, pauli_mat: np.ndarray) -> float:
    """Re(<psi|Pi|psi>)."""
    return np.real(state.conj() @ pauli_mat @ state)


def _build_pauli_stack(n: int, operators: list[tuple[int, int]]) -> np.ndarray:
    """Pre-build (N, 2^n, 2^n) stack of Pauli matrices for vectorized expectations."""
    dim = 1 << n
    N = len(operators)
    stack = np.empty((N, dim, dim), dtype=np.complex128)
    for i, (x, z) in enumerate(operators):
        stack[i] = _pauli_matrix(n, x, z)
    return stack


def _all_expectations(state: np.ndarray, pauli_stack: np.ndarray) -> np.ndarray:
    """Vectorized expectations: returns (N,) real array of <psi|Pi_i|psi>."""
    return np.real(np.einsum("i,kij,j->k", state.conj(), pauli_stack, state))


def _relaxed_labs_cost(
    theta: np.ndarray,
    *,
    n: int,
    N: int,
    layers: int,
    pauli_stack: np.ndarray,
    alpha: float,
    beta: float,
) -> float:
    """Relaxed LABS cost function (Eq. 4 of Sciorilli et al. 2025).

    x_tilde_i = tanh(alpha * <Pi_i>)
    L = sum_{ell=1}^{N-1} [sum_i x_tilde_i * x_tilde_{i+ell}]^2
        - beta * sum_i x_tilde_i^2
    """
    state = _brickwork_circuit(theta, n, layers)
    exps = _all_expectations(state, pauli_stack)
    x_tilde = np.tanh(alpha * exps)

    # Autocorrelation sidelobes
    cost = 0.0
    for ell in range(1, N):
        c_ell = np.sum(x_tilde[:N - ell] * x_tilde[ell:])
        cost += c_ell * c_ell

    # Regularization: penalize small magnitudes to avoid trivial zero
    cost -= beta * np.sum(x_tilde ** 2)

    return float(cost)


def _optimize_single(
    n: int,
    N: int,
    layers: int,
    pauli_stack: np.ndarray,
    alpha: float,
    beta: float,
    rng: np.random.Generator,
    optimizer: str,
    maxiter: int,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Single optimization restart.

    Returns (theta_opt, cost, x_tilde).
    """
    n_params = _total_params(n, layers)
    theta0 = rng.uniform(-np.pi, np.pi, size=n_params)

    def cost_fn(theta):
        return _relaxed_labs_cost(
            theta, n=n, N=N, layers=layers,
            pauli_stack=pauli_stack, alpha=alpha, beta=beta,
        )

    result = minimize(
        cost_fn,
        theta0,
        method=optimizer,
        options={"maxiter": maxiter, "disp": False},
    )

    theta_opt = result.x
    cost = result.fun

    state = _brickwork_circuit(theta_opt, n, layers)
    exps = _all_expectations(state, pauli_stack)
    x_tilde = np.tanh(alpha * exps)

    return theta_opt, cost, x_tilde


def _decode_sequence(x_tilde: np.ndarray) -> np.ndarray:
    """Decode relaxed variables to +/-1 binary sequence."""
    return np.sign(x_tilde).astype(np.int8)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def _run_pce(
    N: int,
    K_out: int,
    rng: np.random.Generator,
    params: dict[str, Any],
) -> tuple[np.ndarray, dict]:
    """Multi-restart PCE optimization.

    Returns (seeds, pce_info) where seeds is (K_out, N) int8.
    """
    layers = int(params.get("layers", 15))
    n_qubits = params.get("n_qubits", None)
    op_set = str(params.get("operator_set", "non_commuting"))
    beta = float(params.get("beta", 15.0))
    max_restarts = int(params.get("max_restarts", max(K_out, 10)))
    optimizer = str(params.get("optimizer", "L-BFGS-B"))
    maxiter = int(params.get("maxiter", 500))

    n = int(n_qubits) if n_qubits is not None else _num_qubits_for_N(N)
    alpha = float(params.get("alpha", 1.5 * n))

    # Select Pauli operators
    operators = _select_operators(n, N, op_set, rng)

    # Pre-build Pauli matrix stack for vectorized expectations
    pauli_stack = _build_pauli_stack(n, operators)

    # Multi-restart optimization
    results: list[tuple[float, np.ndarray]] = []
    for _ in range(max_restarts):
        _, cost, x_tilde = _optimize_single(
            n, N, layers, pauli_stack, alpha, beta, rng, optimizer, maxiter,
        )
        seq = _decode_sequence(x_tilde)
        # Fix any zeros (from exact 0.0 in x_tilde) to +1
        seq[seq == 0] = 1
        results.append((cost, seq))

    # Sort by cost (lower is better) and take K_out unique seeds
    results.sort(key=lambda r: r[0])
    seen: set[bytes] = set()
    seeds_list: list[np.ndarray] = []
    for _, seq in results:
        key = seq.tobytes()
        if key not in seen:
            seen.add(key)
            seeds_list.append(seq)
        if len(seeds_list) >= K_out:
            break

    # Pad with random if not enough unique seeds
    while len(seeds_list) < K_out:
        seeds_list.append(sample_pm1((N,), rng))

    seeds = np.stack(seeds_list[:K_out], axis=0)

    pce_info = {
        "n_qubits": n,
        "n_params": _total_params(n, layers),
        "layers": layers,
        "operator_set": op_set,
        "alpha": alpha,
        "beta": beta,
        "max_restarts": max_restarts,
        "unique_seeds_found": len(seen),
        "optimizer": optimizer,
        "maxiter": maxiter,
    }

    return seeds, pce_info


def _fallback_classical(N: int, K_out: int, rng_seed: int, params: dict) -> np.ndarray:
    """Fallback: deterministic random +/-1 seeds."""
    rng = get_np_rng(rng_seed)
    return sample_pm1((K_out, N), rng)


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def generate_seeds(
    N: int,
    shots: int,
    K_out: int,
    *,
    backend: str,
    rng_seed: int,
    params: dict,
) -> tuple[np.ndarray, dict]:
    """PCE seeder (Pauli Correlation Encoding) — Sciorilli et al. 2025.

    Uses classical numpy state-vector simulation (n qubits is tiny) with
    scipy L-BFGS-B optimization of a relaxed LABS cost function.

    Parameters
    ----------
    N : int
        Sequence length.
    shots : int
        Ignored (kept for interface compatibility).
    K_out : int
        Number of seed sequences to return.
    backend : str
        Ignored (always uses numpy simulation).
    rng_seed : int
        Determinism seed.
    params : dict
        Algorithm parameters (layers, n_qubits, operator_set, alpha, beta,
        max_restarts, optimizer, maxiter).

    Returns
    -------
    seeds : ndarray of shape (K_out, N), dtype int8, values in {-1, +1}
    info : dict with run metadata
    """
    t0 = time.perf_counter()
    rng = get_np_rng(rng_seed)

    N = int(N)
    K_out = int(K_out)

    seeds, pce_info = _run_pce(N, K_out, rng, params)

    E = labs_energy_batch(seeds)
    elapsed = time.perf_counter() - t0

    info = {
        "seeder": "pce",
        "backend": "numpy",
        "cudaq_available": False,
        "compile_s": 0.0,
        "sampling_s": elapsed,
        "shots": int(shots),
        "K_out": K_out,
        "raw_energy_min": int(E.min()),
        "raw_energy_mean": float(E.mean()),
        "params": params,
        "pce": pce_info,
    }

    return seeds.astype(np.int8, copy=False), info
