"""Common LMMSE data detector (applied identically to every channel estimate).

Moved verbatim from the original single-file FD-PGNet v2 script.

Contents
--------
lmmse_equalize    matrix-free CG solve of (H_D^H H_D + sigma^2 I) x_D = H_D^H y_res
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from Utils.H_eff_creation import apply_dd_channel
from Utils.sensing_matrix import PilotDesign

if TYPE_CHECKING:  # type hints only (avoids a circular import with Main.py)
    from Main import Config


def lmmse_equalize(
    y: np.ndarray,
    h_est: np.ndarray,
    design: PilotDesign,
    cfg: Config,
    noise_variance: float,
    return_iterations: bool = False,
):
    
    
    h_est = np.atleast_2d(np.asarray(h_est, dtype=np.complex128))
    y = np.asarray(y, dtype=np.complex128)
    if y.ndim == 2:
        y = y[None, ...]
    batch = h_est.shape[0]

    if h_est.shape[1] != cfg.channel_length:
        raise ValueError("h_est has an invalid sparse-channel dimension.")
    if y.shape != (batch, cfg.M, cfg.N):
        raise ValueError(
            "y must have shape [batch, M, N] matching h_est."
        )

    sigma2 = float(noise_variance)
    if not np.isfinite(sigma2) or sigma2 <= 0.0:
        raise ValueError("noise_variance must be finite and strictly positive.")

    detector_h = h_est

    def sparse_operator(values: np.ndarray, adjoint: bool = False) -> np.ndarray:
        return apply_dd_channel(
            detector_h, values, design, cfg, adjoint=adjoint
        )

    # Cancel only the known transmitted pilot symbols.  Guard REs are zero by
    # construction and DATA symbols remain unknown.
    pilots = np.broadcast_to(
        design.pilot_grid, (batch, cfg.M, cfg.N)
    )
    y_residual = y - sparse_operator(pilots)

    # P_D H^H y_res, where P_D keeps only DATA variables.
    rhs = sparse_operator(y_residual, adjoint=True)
    rhs[:, ~design.data_mask] = 0.0

    def normal_operator(values: np.ndarray) -> np.ndarray:
        # P_D (H^H H + sigma_w^2 I) P_D values.
        values = values.copy()
        values[:, ~design.data_mask] = 0.0
        result = sparse_operator(
            sparse_operator(values), adjoint=True
        )
        # MMSE regulariser is sigma^2 / E[|x_D|^2]. Here data_energy=1,
        # retained explicitly so the normalization is visible and testable.
        result += (sigma2 / float(design.data_energy)) * values
        result[:, ~design.data_mask] = 0.0
        return result

    # Batched conjugate gradients.  Every frame has its own convergence state;
    # already-converged frames are frozen while more difficult frames continue.
    estimate = np.zeros_like(rhs)
    residual_cg = rhs.copy()
    direction = residual_cg.copy()

    squared_norm = np.real(
        np.sum(np.conj(residual_cg) * residual_cg, axis=(1, 2))
    )
    initial_squared_norm = np.maximum(squared_norm.copy(), 1e-30)
    tolerance_squared = float(cfg.lmmse_cg_tolerance) ** 2
    active = squared_norm > tolerance_squared * initial_squared_norm
    tiny = np.finfo(np.float64).tiny

    performed = 0
    for _ in range(int(cfg.lmmse_cg_iterations)):
        if not np.any(active):
            break
        performed += 1

        normal_direction = normal_operator(direction)
        denominator = np.real(
            np.sum(
                np.conj(direction) * normal_direction,
                axis=(1, 2),
            )
        )

        step = np.zeros(batch, dtype=np.float64)
        valid = active & (denominator > tiny)
        step[valid] = (
            squared_norm[valid] / denominator[valid]
        )

        estimate += step[:, None, None] * direction
        residual_cg -= step[:, None, None] * normal_direction
        residual_cg[:, ~design.data_mask] = 0.0

        next_squared_norm = np.real(
            np.sum(
                np.conj(residual_cg) * residual_cg,
                axis=(1, 2),
            )
        )

        next_active = (
            next_squared_norm
            > tolerance_squared * initial_squared_norm
        )

        beta = np.zeros(batch, dtype=np.float64)
        continuing = next_active & active
        beta[continuing] = (
            next_squared_norm[continuing]
            / np.maximum(squared_norm[continuing], tiny)
        )

        direction = residual_cg + beta[:, None, None] * direction
        direction[~next_active] = 0.0
        squared_norm = next_squared_norm
        active = next_active

    estimate[:, ~design.data_mask] = 0.0
    if return_iterations:
        return estimate, performed
    return estimate
