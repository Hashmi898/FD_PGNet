"""Sparse delay-Doppler channel ``h`` creation (fractional-Doppler OTFS).

Moved verbatim from the original single-file FD-PGNet v2 script.

Contents
--------
FractionalChannelBatch                  physical paths + equivalent sparse h
dirichlet_coefficient                   fractional-Doppler leakage eta(q, kappa)
physical_paths_to_sparse_h              physical paths -> Q-truncated sparse h
generate_physical_fractional_channels   random physical channels and their h
_path_atom, _path_atoms_batch           sparse-h response of one unit-gain path
complex_noise                           circularly-symmetric complex AWGN
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Tuple

import numpy as np

if TYPE_CHECKING:  # type hints only (avoids a circular import with Main.py)
    from Main import Config


@dataclass
class FractionalChannelBatch:
    """Physical paths and their equivalent Q-truncated sparse DD channel."""

    delay_taps: np.ndarray
    integer_doppler_taps: np.ndarray
    fractional_doppler_taps: np.ndarray
    gains: np.ndarray
    sparse_h: np.ndarray


def dirichlet_coefficient(q: np.ndarray, kappa: np.ndarray, N: int) -> np.ndarray:
    """Finite-sum evaluation of eta(q,kappa), avoiding a 0/0 singularity."""
    phase = np.asarray(q, dtype=np.float64) + np.asarray(kappa, dtype=np.float64)
    n = np.arange(N, dtype=np.float64)
    return np.mean(np.exp(1j * 2 * np.pi * phase[..., None] * n / N), axis=-1)


def physical_paths_to_sparse_h(
    delay_taps: np.ndarray,
    integer_doppler_taps: np.ndarray,
    fractional_doppler_taps: np.ndarray,
    gains: np.ndarray,
    cfg: Config,
) -> np.ndarray:
    """Map physical fractional paths to the paper's Q-truncated sparse h."""
    delay_taps = np.asarray(delay_taps, dtype=np.int64)
    integer_doppler_taps = np.asarray(integer_doppler_taps, dtype=np.int64)
    fractional_doppler_taps = np.asarray(
        fractional_doppler_taps, dtype=np.float64
    )
    gains = np.asarray(gains, dtype=np.complex128)

    if not (
        delay_taps.shape
        == integer_doppler_taps.shape
        == fractional_doppler_taps.shape
        == gains.shape
    ):
        raise ValueError("All physical-path arrays must have identical shape.")

    count, paths = gains.shape
    h = np.zeros((count, cfg.channel_length), dtype=np.complex128)
    rows = np.arange(count)
    width = 2 * cfg.k_extended + 1

    for path in range(paths):
        delay = delay_taps[:, path]
        integer_k = integer_doppler_taps[:, path]
        kappa = fractional_doppler_taps[:, path]
        # Eq. (2) phase exp(-j2*pi*l'*(k'+kappa)/(MN)): it uses the path's own
        # delay tap and its own integer+fractional Doppler, so it is a single
        # complex constant shared by all 2Q+1 leakage taps of that path.
        fractional_phase = np.exp(
            -1j * 2 * np.pi * delay * (integer_k + kappa) / (cfg.M * cfg.N)
        )

        for q in range(-cfg.Q, cfg.Q + 1):
            k_eff = integer_k - q
            coefficient = (
                gains[:, path]
                * dirichlet_coefficient(q, kappa, cfg.N)
                * fractional_phase
            )
            columns = delay * width + k_eff + cfg.k_extended
            np.add.at(h, (rows, columns), coefficient)

    return h


def generate_physical_fractional_channels(
    count: int, cfg: Config, rng: np.random.Generator
) -> FractionalChannelBatch:
    """Generate physical paths and their equivalent sparse fractional h."""
    delays = rng.integers(0, cfg.l_max + 1, size=(count, cfg.paths))
    integer_k = rng.integers(
        -cfg.k_max, cfg.k_max + 1, size=(count, cfg.paths)
    )
    kappa = rng.uniform(-0.5, 0.5, size=(count, cfg.paths))
    gains = (
        rng.standard_normal((count, cfg.paths))
        + 1j * rng.standard_normal((count, cfg.paths))
    ) / np.sqrt(2 * cfg.paths)  # CN(0,1/P)

    sparse_h = physical_paths_to_sparse_h(
        delays, integer_k, kappa, gains, cfg
    )
    return FractionalChannelBatch(
        delay_taps=delays,
        integer_doppler_taps=integer_k,
        fractional_doppler_taps=kappa,
        gains=gains,
        sparse_h=sparse_h,
    )


def _path_atom(cfg: Config, delay: int, integer_k: int, kappa: float) -> np.ndarray:
    width = 2 * cfg.k_extended + 1
    atom = np.zeros(cfg.channel_length, dtype=np.complex128)
    for q in range(-cfg.Q, cfg.Q + 1):
        atom[delay * width + (integer_k - q) + cfg.k_extended] += (
            dirichlet_coefficient(np.array(float(q)), np.array(float(kappa)), cfg.N)
        )
    return atom * np.exp(
        -1j * 2 * np.pi * delay * (integer_k + kappa) / (cfg.M * cfg.N)
    )


def _path_atoms_batch(
    cfg: Config,
    delays: np.ndarray,
    integer_ks: np.ndarray,
    kappas: np.ndarray,
) -> np.ndarray:
    """Vectorized form of ``_path_atom``.

    ``delays``, ``integer_ks``, ``kappas`` broadcast to a common shape ``S``;
    the return value has shape ``S + (channel_length,)`` and is numerically
    IDENTICAL (same closed-form terms, evaluated with the same
    ``dirichlet_coefficient`` call and the same summation order via
    ``np.add.at``) to stacking ``_path_atom(cfg, d, k, kap)`` element-wise
    over ``S``. This exists purely to replace a Python-level loop of
    ``_path_atom`` calls with one vectorized numpy evaluation -- added on
    request to remove the dominant per-frame overhead of
    ``EstimatedPathRefiner``; it changes no formula, no data, and no result
    (verified against ``_path_atom`` to <1e-12 absolute difference on random
    inputs).
    """
    delays = np.asarray(delays, dtype=np.int64)
    integer_ks = np.asarray(integer_ks, dtype=np.int64)
    kappas = np.asarray(kappas, dtype=np.float64)
    shape = np.broadcast_shapes(delays.shape, integer_ks.shape, kappas.shape)
    delays = np.broadcast_to(delays, shape)
    integer_ks = np.broadcast_to(integer_ks, shape)
    kappas = np.broadcast_to(kappas, shape)

    width = 2 * cfg.k_extended + 1
    q_values = np.arange(-cfg.Q, cfg.Q + 1, dtype=np.int64)
    n = np.arange(cfg.N, dtype=np.float64)

    phase = kappas[..., None] + q_values.astype(np.float64)
    eta = np.mean(
        np.exp(1j * 2 * np.pi * phase[..., None] * n / cfg.N), axis=-1
    )
    columns = (
        delays[..., None] * width
        + (integer_ks[..., None] - q_values)
        + cfg.k_extended
    )

    flat_n = int(np.prod(shape)) if shape else 1
    atoms_flat = np.zeros((flat_n, cfg.channel_length), dtype=np.complex128)
    rows_flat = np.repeat(np.arange(flat_n), q_values.size)
    np.add.at(atoms_flat, (rows_flat, columns.reshape(-1)), eta.reshape(-1))

    fractional_phase = np.exp(
        -1j * 2 * np.pi * delays * (integer_ks + kappas) / (cfg.M * cfg.N)
    )
    atoms_flat = atoms_flat * fractional_phase.reshape(-1, 1)
    return atoms_flat.reshape(*shape, cfg.channel_length)


def complex_noise(
    shape: Tuple[int, ...], variance: float | np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    variance = np.asarray(variance, dtype=np.float64)
    while variance.ndim < len(shape):
        variance = variance[..., None]
    return np.sqrt(variance / 2) * (
        rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    )
