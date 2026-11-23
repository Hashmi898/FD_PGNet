

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from Utils.h_sparse_creation import dirichlet_coefficient

if TYPE_CHECKING:  # type hints only (avoids circular imports)
    from Main import Config
    from Utils.h_sparse_creation import FractionalChannelBatch
    from Utils.sensing_matrix import PilotDesign


def _dd_grid_to_vec(grid: np.ndarray) -> np.ndarray:
    """Column-major vectorization consistent with MATLAB vec(X)."""
    grid = np.asarray(grid, dtype=np.complex128)
    if grid.ndim == 2:
        grid = grid[None, ...]
    return grid.transpose(0, 2, 1).reshape(grid.shape[0], -1)


def _dd_vec_to_grid(vector: np.ndarray, M: int, N: int) -> np.ndarray:
    """Inverse column-major vectorization for batched DD-domain vectors."""
    vector = np.atleast_2d(np.asarray(vector, dtype=np.complex128))
    return vector.reshape(vector.shape[0], N, M).transpose(0, 2, 1)


def build_Heff_from_hhat_fractional(
    M: int,
    N: int,
    l_max: int,
    k_max: int,
    Q: int,
    h_hat: np.ndarray,
) -> np.ndarray:
    """Build the DD-domain detection matrix from an estimated sparse channel.

    The sparse coefficient at ``(delay, k_eff)`` already contains the constant
    fractional phase ``exp(-j*2*pi*delay*kappa/(MN))`` and the leakage weight
    ``eta(q,kappa)``.  This matrix therefore implements the Q-truncated form of
    Eq. (2) in the DRSNN paper:

        y[k,l] = sum h_hat[l',k_eff]
                 exp(-j*2*pi*l'*k/(MN))
                 x[[k-k_eff]_N,[l-l']_M].

    MATLAB column-major vectorization is used: index = delay + doppler*M.
    """
    MN = M * N
    k_extended = k_max + Q
    expected_length = (l_max + 1) * (2 * k_extended + 1)
    h_hat = np.asarray(h_hat, dtype=np.complex128).reshape(-1)
    if h_hat.size != expected_length:
        raise ValueError(
            f"Invalid h_hat length: expected {expected_length}, got {h_hat.size}."
        )

    H_eff_hat = np.zeros((MN, MN), dtype=np.complex128)
    output_delay = np.arange(M, dtype=np.int64)
    column = 0

    for delay in range(l_max + 1):
        for k_eff in range(-k_extended, k_extended + 1):
            coefficient = h_hat[column]
            column += 1
            if abs(coefficient) <= 1e-12:
                continue

            input_delay = (output_delay - delay) % M
            for output_k in range(N):
                input_k = (output_k - k_eff) % N
                row_indices = output_delay + output_k * M
                column_indices = input_delay + input_k * M
                # Eq. (2) phase is a per-path constant already folded into
                # ``coefficient``; the operator itself is a pure circular shift.
                H_eff_hat[row_indices, column_indices] += coefficient

    return H_eff_hat


def build_Heff_from_physical_fractional(
    M: int,
    N: int,
    Q: int,
    chan_coef: np.ndarray,
    delay_taps: np.ndarray,
    integer_doppler_taps: np.ndarray,
    fractional_doppler_taps: np.ndarray,
) -> np.ndarray:
    """Build direct physical-path fractional-Doppler H_eff from Eq. (2).

    Each physical path is retained as ``(gain, delay, k, kappa)`` and its
    fractional-Doppler leakage is summed over ``q=-Q,...,Q``.  No fractional
    Doppler is rounded or cast to an integer.
    """
    MN = M * N
    chan_coef = np.asarray(chan_coef, dtype=np.complex128).reshape(-1)
    delay_taps = np.asarray(delay_taps, dtype=np.int64).reshape(-1)
    integer_doppler_taps = np.asarray(
        integer_doppler_taps, dtype=np.int64
    ).reshape(-1)
    fractional_doppler_taps = np.asarray(
        fractional_doppler_taps, dtype=np.float64
    ).reshape(-1)

    taps = chan_coef.size
    if not (
        delay_taps.size == taps
        and integer_doppler_taps.size == taps
        and fractional_doppler_taps.size == taps
    ):
        raise ValueError("All physical-path arrays must have equal length.")

    H_eff = np.zeros((MN, MN), dtype=np.complex128)
    output_delay = np.arange(M, dtype=np.int64)

    for path in range(taps):
        delay = int(delay_taps[path])
        integer_k = int(integer_doppler_taps[path])
        kappa = float(fractional_doppler_taps[path])
        gain = chan_coef[path]
        input_delay = (output_delay - delay) % M

        # Eq. (2): exp(-j2*pi*l'*(k'+kappa)/(MN)) uses TAP indices only, so it
        # is one constant per path and is independent of (k, l).
        path_phase = np.exp(-1j * 2 * np.pi * delay * (integer_k + kappa) / MN)

        for q in range(-Q, Q + 1):
            k_eff = integer_k - q
            eta_q = dirichlet_coefficient(q, kappa, N)
            for output_k in range(N):
                input_k = (output_k - k_eff) % N
                row_indices = output_delay + output_k * M
                column_indices = input_delay + input_k * M
                H_eff[row_indices, column_indices] += gain * eta_q * path_phase

    return H_eff


def apply_dd_channel(
    h: np.ndarray,
    x: np.ndarray,
    design: PilotDesign,
    cfg: Config,
    adjoint: bool = False,
) -> np.ndarray:
    """Apply fractional-Doppler H_eff(h) or its exact Hermitian adjoint.

    ``h`` is the Q-truncated sparse channel of length
    ``(l_max+1)(2*k_max+2*Q+1)``.  This matrix-free operator is exactly the
    action of :func:`build_Heff_from_hhat_fractional` and is used for the
    sensing matrix and the common LMMSE detector.
    """
    h = np.atleast_2d(np.asarray(h, dtype=np.complex128))
    x = np.asarray(x, dtype=np.complex128)
    if x.ndim == 2:
        x = x[None, ...]
    if x.shape[0] == 1 and h.shape[0] > 1:
        x = np.broadcast_to(x, (h.shape[0], cfg.M, cfg.N))
    if h.shape[0] != x.shape[0]:
        raise ValueError("h and x batch dimensions are incompatible.")
    if h.shape[1] != cfg.channel_length:
        raise ValueError("h has an invalid sparse-channel dimension.")

    y = np.zeros_like(x, dtype=np.complex128)

    for column, (delay, k_eff) in enumerate(
        zip(design.support_delay, design.support_doppler)
    ):
        delay = int(delay)
        k_eff = int(k_eff)
        if not adjoint:
            shifted = np.roll(x, shift=(delay, k_eff), axis=(-2, -1))
            y += h[:, column, None, None] * shifted
        else:
            shifted = np.roll(x, shift=(-delay, -k_eff), axis=(-2, -1))
            y += np.conj(h[:, column, None, None]) * shifted

    return y


def apply_physical_fractional_channel(
    channel: FractionalChannelBatch,
    x: np.ndarray,
    cfg: Config,
) -> np.ndarray:
    """Apply the direct physical-path fractional-Doppler channel of Eq. (2)."""
    x = np.asarray(x, dtype=np.complex128)
    if x.ndim == 2:
        x = x[None, ...]

    batch = channel.gains.shape[0]
    if x.shape[0] == 1 and batch > 1:
        x = np.broadcast_to(x, (batch, cfg.M, cfg.N))
    if x.shape[0] != batch:
        raise ValueError("Physical channel and x batch dimensions are incompatible.")

    y = np.zeros_like(x, dtype=np.complex128)
    batch_index = np.arange(batch)[:, None, None]
    output_delay = np.arange(cfg.M)[None, :, None]
    output_k = np.arange(cfg.N)[None, None, :]

    for path in range(cfg.paths):
        delay = channel.delay_taps[:, path]
        integer_k = channel.integer_doppler_taps[:, path]
        kappa = channel.fractional_doppler_taps[:, path]
        gain = channel.gains[:, path]

        # Eq. (2): one constant phase per path, independent of (k, l).
        path_phase = np.exp(
            -1j * 2 * np.pi * delay * (integer_k + kappa) / (cfg.M * cfg.N)
        )[:, None, None]

        source_delay = (output_delay - delay[:, None, None]) % cfg.M
        for q in range(-cfg.Q, cfg.Q + 1):
            k_eff = integer_k - q
            source_k = (output_k - k_eff[:, None, None]) % cfg.N
            shifted = x[batch_index, source_delay, source_k]
            eta_q = dirichlet_coefficient(q, kappa, cfg.N)
            y += (
                gain[:, None, None]
                * eta_q[:, None, None]
                * path_phase
                * shifted
            )

    return y
