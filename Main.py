"""FD-PGNet v2: fractional-Doppler physics-guided Bayesian unfolding for OTFS."""


from __future__ import annotations

import argparse
import json
import sys
import time
import os
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, Tuple

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "fdpgnet-mpl-cache")
)

import matplotlib.pyplot as plt
import numpy as np

# ---- project modules ---------------------------------------------------------
# Utils: sensing matrix, sparse-h creation, H_eff creation
from Utils.sensing_matrix import PilotDesign, make_pilot_design, coarse_ls
from Utils.h_sparse_creation import (
    FractionalChannelBatch,
    dirichlet_coefficient,
    generate_physical_fractional_channels,
    complex_noise,
    _path_atom,
)
from Utils.H_eff_creation import (
    _dd_grid_to_vec,
    _dd_vec_to_grid,
    build_Heff_from_hhat_fractional,
    build_Heff_from_physical_fractional,
    apply_dd_channel,
    apply_physical_fractional_channel,
)
# Channel estimation: FD-PGNet v2
from channel_estimation_Algo.FDPGNet import (
    import_torch,
    set_all_seeds,
    build_fdpgnet_model_bank,
    infer_fdpgnet,
    EstimatedPathRefiner,
    count_parameters,
)
# Data detection: common LMMSE detector
from Data_Detection.lmmse_detector import lmmse_equalize


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class Config:
    # Values stated by the DRSNN paper
    M: int = 24
    N: int = 20
    paths: int = 4
    l_max: int = 6
    k_max: int = 4
    carrier_hz: float = 3.0e9
    subcarrier_spacing_hz: float = 7.5e3
    snr_db: Tuple[int, ...] = tuple(range(0, 31, 5))
    train_samples: int = 27_000
    test_frames_per_snr: int = 1_000
    adam_lr: float = 5.0e-3

    # PnP pilot geometry. Every pilot and active data symbol has unit energy.
    # No pilot multiplier, energy ratio, boost, or power reallocation exists.
    pilot_delay_len: int = 2
    pilot_doppler_len: int = 2
    pilot_energy: float = 1.0

    # Reproduction assumptions because the DRSNN article omits these values
    Q: int = 1
    validation_samples: int = 3_000
    epochs: int = 12
    batch_size: int = 256

    # ---------------- FD-PGNet v2 ----------------
    fdpg_stages: int = 3
    # CNN width in PriorHead. Dominated ~79% of total MACs and ~99.9% of
    # trainable parameters at 32 (real_MACs_conv_head scales as features^2).
    # Halved to shrink both by roughly 4x on the conv term alone, on the same
    # explicit request as the stage-count reduction above.
    fdpg_features: int = 16
    # Fractional-Doppler dictionary resolution. 41 and 129 atoms were measured
    # to be within 0.1 dB of each other, so 41 is used.
    fdpg_kappa_atoms: int = 41
    fdpg_manifold_sweeps: int = 1
    fdpg_manifold_damping: float = 0.5

    fdpg_posterior_mode: str = "cg"
    fdpg_cg_steps: int = 12
    fdpg_log_gamma_clamp: float = 4.0
    fdpg_consistency_weight: float = 0.05
    # Noise self-calibration, blended through a zero-initialised scalar so the
    # untrained network uses the supplied sigma^2 unchanged.
    fdpg_noise_calibration: bool = True
    fdpg_log_sigma_clamp: float = 1.5
    fdpg_em_damping: float = 0.5
    # Receiver-side counterparts of the oracle-support and oracle-parameter
    # bounds. Candidate support and fractional Doppler are ESTIMATED from the
    # received pilots and the FD-PGNet output; no true path information enters.
    fdpg_estimated_path_refinement: bool = True
    fdpg_refinement_candidate_cells: int = 16

    fdpg_refinement_kappa_sweeps: int = 1
    fdpg_refinement_local_points: int = 5

    training_snr_mode: str = "per_snr"
    shared_snr_range_db: Tuple[float, float] = (-2.0, 32.0)

    # Execution controls
    eval_batch_size: int = 100
    lmmse_cg_iterations: int = 400
    lmmse_cg_tolerance: float = 1.0e-8
    ber_frame_scale: Tuple[Tuple[int, int], ...] = (
        (0, 1), (5, 1), (10, 1), (15, 1), (20, 2), (25, 4), (30, 8),
    )

    # Bounds module
    bounds_frames: int = 1_000
    bounds_mmse_bank: int = 400_000
    bounds_mmse_frames: int = 120
    bounds_mmse_max_snr_db: float = 5.0
    bounds_mmse_min_ess: float = 400.0

    seed: int = 20241216
    profile: str = "paper"

    def frames_at_snr(self, snr_db: float) -> int:
        key = int(round(float(snr_db)))
        table = dict(self.ber_frame_scale)
        return int(self.test_frames_per_snr) * int(table.get(key, 1))

    @property
    def k_extended(self) -> int:
        return self.k_max + self.Q

    @property
    def channel_length(self) -> int:
        return (self.l_max + 1) * (2 * self.k_max + 2 * self.Q + 1)

    @property
    def delay_bins(self) -> int:
        return self.l_max + 1

    @property
    def doppler_bins(self) -> int:
        return 2 * self.k_extended + 1

    @property
    def observation_length(self) -> int:
        return (self.pilot_delay_len + self.l_max) * (
            self.pilot_doppler_len + 2 * self.k_max + 2 * self.Q
        )

    @property
    def n_cells(self) -> int:
        """Physical delay-Doppler cells, one fractional path each."""
        return (self.l_max + 1) * (2 * self.k_max + 1)


def profile_config(name: str) -> Config:
    cfg = Config(profile=name)
    if name == "paper":
        return cfg
    if name == "quick":
        return replace(
            cfg,
            train_samples=4_000,
            test_frames_per_snr=200,
            validation_samples=600,
            epochs=8,
            batch_size=128,
            eval_batch_size=50,
            bounds_frames=200,
            bounds_mmse_bank=60_000,
            bounds_mmse_frames=40,
            ber_frame_scale=tuple((s, 1) for s in cfg.snr_db),
        )
    if name == "smoke":
        return replace(
            cfg,
            train_samples=192,
            test_frames_per_snr=8,
            validation_samples=64,
            epochs=1,
            batch_size=32,
            eval_batch_size=8,
            bounds_frames=20,
            bounds_mmse_bank=4_000,
            bounds_mmse_frames=6,
            ber_frame_scale=tuple((s, 1) for s in cfg.snr_db),
        )
    raise ValueError(f"Unknown profile: {name}")


# -----------------------------------------------------------------------------
# Unit-energy Gray-coded 4-QAM (QPSK)
# -----------------------------------------------------------------------------


def qpsk_mod(bits: np.ndarray) -> np.ndarray:
    bits = np.asarray(bits, dtype=np.int8)
    if bits.shape[-1] % 2:
        raise ValueError("QPSK requires an even number of bits.")
    pairs = bits.reshape(*bits.shape[:-1], -1, 2)
    return ((1 - 2 * pairs[..., 1]) + 1j * (1 - 2 * pairs[..., 0])) / np.sqrt(2)


def qpsk_demod(symbols: np.ndarray) -> np.ndarray:
    symbols = np.asarray(symbols)
    pairs = np.empty((*symbols.shape, 2), dtype=np.int8)
    pairs[..., 0] = np.imag(symbols) < 0
    pairs[..., 1] = np.real(symbols) < 0
    return pairs.reshape(*symbols.shape[:-1], -1)


def resolve_device(requested: str, torch):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested, but torch.cuda.is_available() is false.")
    return device


# -----------------------------------------------------------------------------
# Achievability limits for this exact pilot, channel and SNR definition
# -----------------------------------------------------------------------------
#
# Reviewers ask "how good is good?". These references answer it. All of them
# use the same Phi, the same channel law and the same aggregate NMSE as the
# Monte-Carlo evaluator, so they can be plotted on the same axes.
#
#   ls_pinv                 Moore-Penrose LS, sigma^2 tr(G^-1) / E||h||^2.
#   lmmse_marginal          LMMSE with the exact per-tap marginal prior and NO
#                           sparsity. A free, surprisingly strong baseline that
#                           any proposed estimator must beat convincingly.
#   oracle_support          A genie reveals which taps are non-zero and their
#                           per-tap prior variance. Unstructured lower bound.
#   oracle_parameter        A genie reveals (delay, integer Doppler, kappa) of
#                           every path; only the P complex gains are estimated.
#                           Lower bound for ANY manifold-exploiting estimator.
#   exact_mmse              The true Bayesian MMSE, by importance sampling from
#                           the exact prior. Valid only while the effective
#                           sample size stays large, i.e. at low SNR, which is
#                           precisely where the genie bounds are loosest and
#                           most often misread as attainable.
#
# The genie curves are NOT attainable at low SNR: the support cannot be
# resolved there. exact_mmse is the honest low-SNR reference.
# -----------------------------------------------------------------------------


def _tap_prior_variance(cfg: Config, channel: FractionalChannelBatch) -> np.ndarray:
    """Per-frame, per-tap prior variance E[|h_i|^2 | delays, Dopplers, kappa]."""
    count = channel.gains.shape[0]
    width = 2 * cfg.k_extended + 1
    variance = np.zeros((count, cfg.channel_length))
    rows = np.arange(count)
    for path in range(cfg.paths):
        delay = channel.delay_taps[:, path]
        integer_k = channel.integer_doppler_taps[:, path]
        kappa = channel.fractional_doppler_taps[:, path]
        for q in range(-cfg.Q, cfg.Q + 1):
            eta = dirichlet_coefficient(np.array(float(q)), kappa, cfg.N)
            columns = delay * width + (integer_k - q) + cfg.k_extended
            np.add.at(
                variance, (rows, columns), np.abs(eta) ** 2 / float(cfg.paths)
            )
    return variance


def compute_bounds(cfg: Config, design: PilotDesign, verbose: bool = True):
    """Return a dict of reference NMSE curves in dB, one entry per SNR."""
    phi = design.sensing_matrix
    gram = phi.conj().T @ phi
    gram_inverse_trace = float(np.real(np.trace(np.linalg.inv(gram))))
    snrs = np.asarray(cfg.snr_db, dtype=float)

    rng = np.random.default_rng(cfg.seed + 555)
    reference = generate_physical_fractional_channels(
        max(cfg.bounds_frames * 20, 20_000), cfg, rng
    )
    energy = float(np.mean(np.sum(np.abs(reference.sparse_h) ** 2, axis=1)))
    marginal = np.mean(np.abs(reference.sparse_h) ** 2, axis=0)

    ls = np.array(
        [10 * np.log10(gram_inverse_trace * 10 ** (-s / 10) / energy) for s in snrs]
    )

    lmmse = []
    for s in snrs:
        sigma2 = 10 ** (-s / 10)
        system = gram / sigma2 + np.diag(1.0 / np.maximum(marginal, 1e-15))
        lmmse.append(
            10 * np.log10(float(np.real(np.trace(np.linalg.inv(system)))) / energy)
        )
    lmmse = np.asarray(lmmse)

    bound_rng = np.random.default_rng(cfg.seed + 556)
    channel = generate_physical_fractional_channels(cfg.bounds_frames, cfg, bound_rng)
    prior_variance = _tap_prior_variance(cfg, channel)

    support_total = {s: 0.0 for s in snrs}
    parameter_total = {s: 0.0 for s in snrs}
    for i in range(cfg.bounds_frames):
        active = np.where(prior_variance[i] > 1e-14)[0]
        gram_s = gram[np.ix_(active, active)]
        prior_s = prior_variance[i, active]
        atoms = np.stack(
            [
                _path_atom(
                    cfg,
                    int(channel.delay_taps[i, p]),
                    int(channel.integer_doppler_taps[i, p]),
                    float(channel.fractional_doppler_taps[i, p]),
                )
                for p in range(cfg.paths)
            ],
            axis=1,
        )
        projected = phi @ atoms
        gram_p = projected.conj().T @ projected
        for s in snrs:
            sigma2 = 10 ** (-s / 10)
            support_total[s] += float(
                np.real(np.trace(np.linalg.inv(gram_s / sigma2 + np.diag(1.0 / prior_s))))
            )
            covariance = np.linalg.inv(
                gram_p / sigma2 + cfg.paths * np.eye(cfg.paths)
            )
            parameter_total[s] += float(
                np.real(np.trace(atoms @ covariance @ atoms.conj().T))
            )

    oracle_support = np.array(
        [10 * np.log10(support_total[s] / cfg.bounds_frames / energy) for s in snrs]
    )
    oracle_parameter = np.array(
        [10 * np.log10(parameter_total[s] / cfg.bounds_frames / energy) for s in snrs]
    )

    exact_mmse = np.full(snrs.shape, np.nan)
    exact_ess = np.full(snrs.shape, np.nan)
    low = snrs <= float(cfg.bounds_mmse_max_snr_db)
    if np.any(low):
        bank_rng = np.random.default_rng(cfg.seed + 557)
        bank = generate_physical_fractional_channels(
            int(cfg.bounds_mmse_bank), cfg, bank_rng
        ).sparse_h.astype(np.complex64)
        bank_obs = bank @ phi.T.astype(np.complex64)
        bank_energy = np.sum(np.abs(bank_obs) ** 2, axis=1).astype(np.float64)
        for index, s in enumerate(snrs):
            if not low[index]:
                continue
            sigma2 = 10 ** (-s / 10)
            test_rng = np.random.default_rng(cfg.seed + 558 + int(s))
            test = generate_physical_fractional_channels(
                int(cfg.bounds_mmse_frames), cfg, test_rng
            ).sparse_h
            observed = (
                test @ phi.T
                + complex_noise((int(cfg.bounds_mmse_frames), phi.shape[0]),
                                sigma2, test_rng)
            ).astype(np.complex64)
            error = 0.0
            true_energy = 0.0
            ess = []
            for i in range(int(cfg.bounds_mmse_frames)):
                log_weight = (
                    2.0 * np.real(bank_obs.conj() @ observed[i]).astype(np.float64)
                    - bank_energy
                ) / sigma2
                log_weight -= log_weight.max()
                weight = np.exp(log_weight)
                total = weight.sum()
                ess.append(float(total ** 2 / np.sum(weight ** 2)))
                estimate = (weight.astype(np.float32) @ bank) / total
                error += float(np.sum(np.abs(estimate - test[i]) ** 2))
                true_energy += float(np.sum(np.abs(test[i]) ** 2))
            mean_ess = float(np.mean(ess))
            exact_ess[index] = mean_ess
            if mean_ess >= float(cfg.bounds_mmse_min_ess):
                exact_mmse[index] = 10 * np.log10(error / true_energy)
            else:
                print(
                    f"  [bounds] exact-MMSE at {s:g} dB DISCARDED: effective "
                    f"sample size {mean_ess:.0f} < {cfg.bounds_mmse_min_ess:.0f}. "
                    "Prior importance sampling is not valid at this SNR; raise "
                    "bounds_mmse_bank or treat the value as unavailable."
                )

    bounds = {
        "snr_db": snrs,
        "bound_ls": ls,
        "bound_lmmse_marginal": lmmse,
        "bound_oracle_support": oracle_support,
        "bound_oracle_parameter": oracle_parameter,
        "bound_exact_mmse": exact_mmse,
        "bound_exact_mmse_ess": exact_ess,
        "gram_inverse_trace": np.asarray([gram_inverse_trace]),
        "channel_energy": np.asarray([energy]),
    }

    if verbose:
        print("\nAchievability limits for this pilot (all NMSE in dB):")
        header = "  " + "SNR (dB)".ljust(28) + "".join(f"{int(s):>9}" for s in snrs)
        print(header)
        print("  " + "-" * (28 + 9 * len(snrs)))
        rows = [
            ("LS (pinv)", ls),
            ("LMMSE, marginal prior", lmmse),
            ("EXACT Bayesian MMSE", exact_mmse),
            ("Oracle-support bound", oracle_support),
            ("Oracle-parameter bound", oracle_parameter),
        ]
        for name, values in rows:
            cells = "".join(
                "        -" if not np.isfinite(v) else f"{v:9.2f}" for v in values
            )
            print("  " + name.ljust(28) + cells)
        if np.any(np.isfinite(exact_ess)):
            shown = ", ".join(
                f"{int(s)} dB: {e:.0f}"
                for s, e in zip(snrs, exact_ess)
                if np.isfinite(e)
            )
            print(f"  importance-sampling effective sample size -> {shown}")
        print(
            "  Note: the genie rows are NOT attainable at low SNR because the\n"
            "  support cannot be resolved there. EXACT Bayesian MMSE is the\n"
            "  honest low-SNR reference and no estimator can beat it."
        )
    return bounds


def run_self_checks(cfg: Config, design: PilotDesign) -> None:
    assert cfg.channel_length == 77
    assert cfg.observation_length == 96
    assert design.sensing_matrix.shape == (96, 77)
    assert np.linalg.matrix_rank(design.sensing_matrix) == 77
    assert int(design.pilot_mask.sum()) == 4
    # Strict unit-energy transmission: four pilots and every data symbol have
    # energy one, with no ratio, multiplier, or redistribution.
    n_pilot = int(design.pilot_mask.sum())
    n_data = int(design.data_mask.sum())
    assert np.allclose(
        np.abs(design.pilot_grid[design.pilot_mask]) ** 2,
        cfg.pilot_energy,
    )
    assert np.isclose(design.pilot_re_energy, cfg.pilot_energy)
    assert np.isclose(design.data_energy, cfg.pilot_energy)
    frame_energy = n_pilot * design.pilot_re_energy + n_data * design.data_energy
    expected_frame_energy = (n_pilot + n_data) * cfg.pilot_energy
    assert abs(frame_energy - expected_frame_energy) < 1e-9

    rng = np.random.default_rng(12345)
    bits = rng.integers(0, 2, size=200, dtype=np.int8)
    assert np.array_equal(qpsk_demod(qpsk_mod(bits)), bits)

    h = rng.standard_normal((2, cfg.channel_length)) + 1j * rng.standard_normal(
        (2, cfg.channel_length)
    )
    pilots = np.broadcast_to(design.pilot_grid, (2, cfg.M, cfg.N))
    y_pilot_grid = apply_dd_channel(h, pilots, design, cfg)
    y_pilot = y_pilot_grid[
        :, design.observation_flat_delay, design.observation_flat_doppler
    ]
    assert np.max(np.abs(y_pilot - h @ design.sensing_matrix.T)) < 1e-12
    assert np.max(np.abs(coarse_ls(y_pilot, design) - h)) < 1e-10

    # Guard must eliminate all noiseless data interference in pilot observations.
    x_data = np.zeros((1, cfg.M, cfg.N), dtype=np.complex128)
    x_data[:, design.data_mask] = np.sqrt(design.data_energy) * qpsk_mod(
        rng.integers(0, 2, size=2 * int(design.data_mask.sum()), dtype=np.int8)
    )
    leakage = apply_dd_channel(h[:1], x_data, design, cfg)[
        :, design.observation_flat_delay, design.observation_flat_doppler
    ]
    assert np.max(np.abs(leakage)) < 1e-12

    physical_channel = generate_physical_fractional_channels(16, cfg, rng)
    h_fractional = physical_channel.sparse_h
    assert np.all(np.isfinite(h_fractional))
    assert np.all(np.sum(np.abs(h_fractional) ** 2, axis=1) > 0)

    # Direct physical Eq. (2), sparse h, and explicit H_eff must agree.
    check_x = (
        rng.standard_normal((1, cfg.M, cfg.N))
        + 1j * rng.standard_normal((1, cfg.M, cfg.N))
    ) / np.sqrt(2)
    one_physical = FractionalChannelBatch(
        delay_taps=physical_channel.delay_taps[:1],
        integer_doppler_taps=physical_channel.integer_doppler_taps[:1],
        fractional_doppler_taps=physical_channel.fractional_doppler_taps[:1],
        gains=physical_channel.gains[:1],
        sparse_h=physical_channel.sparse_h[:1],
    )
    y_physical = apply_physical_fractional_channel(one_physical, check_x, cfg)
    y_sparse = apply_dd_channel(one_physical.sparse_h, check_x, design, cfg)
    physical_sparse_error = float(np.max(np.abs(y_physical - y_sparse)))
    assert physical_sparse_error < 1e-12

    H_hat = build_Heff_from_hhat_fractional(
        cfg.M, cfg.N, cfg.l_max, cfg.k_max, cfg.Q, one_physical.sparse_h[0]
    )
    H_physical = build_Heff_from_physical_fractional(
        cfg.M,
        cfg.N,
        cfg.Q,
        one_physical.gains[0],
        one_physical.delay_taps[0],
        one_physical.integer_doppler_taps[0],
        one_physical.fractional_doppler_taps[0],
    )
    explicit_matrix_error = float(np.max(np.abs(H_hat - H_physical)))
    assert explicit_matrix_error < 1e-12
    y_explicit = _dd_vec_to_grid(
        (H_hat @ _dd_grid_to_vec(check_x)[0]) [None, :], cfg.M, cfg.N
    )
    explicit_operator_error = float(np.max(np.abs(y_explicit - y_sparse)))
    assert explicit_operator_error < 1e-12

    adjoint_input = (
        rng.standard_normal((1, cfg.M, cfg.N))
        + 1j * rng.standard_normal((1, cfg.M, cfg.N))
    ) / np.sqrt(2)
    adjoint_explicit = _dd_vec_to_grid(
        (H_hat.conj().T @ _dd_grid_to_vec(adjoint_input)[0])[None, :],
        cfg.M,
        cfg.N,
    )
    adjoint_operator = apply_dd_channel(
        one_physical.sparse_h, adjoint_input, design, cfg, adjoint=True
    )
    adjoint_error = float(np.max(np.abs(adjoint_explicit - adjoint_operator)))
    assert adjoint_error < 1e-12

    # Noiseless estimated-channel detector check with true h supplied as h_est.
    detector_batch = 4
    detector_bits = rng.integers(
        0,
        2,
        size=(detector_batch, 2 * int(design.data_mask.sum())),
        dtype=np.int8,
    )
    detector_x = np.broadcast_to(
        design.pilot_grid, (detector_batch, cfg.M, cfg.N)
    ).copy()
    detector_x[:, design.data_mask] = np.sqrt(design.data_energy) * qpsk_mod(detector_bits)
    detector_h = h_fractional[:detector_batch]
    detector_physical = FractionalChannelBatch(
        delay_taps=physical_channel.delay_taps[:detector_batch],
        integer_doppler_taps=physical_channel.integer_doppler_taps[:detector_batch],
        fractional_doppler_taps=physical_channel.fractional_doppler_taps[:detector_batch],
        gains=physical_channel.gains[:detector_batch],
        sparse_h=detector_h,
    )
    detector_y = apply_physical_fractional_channel(
        detector_physical, detector_x, cfg
    )
    detector_xhat = lmmse_equalize(
        detector_y, detector_h, design, cfg, noise_variance=1e-8
    )
    detector_decisions = qpsk_demod(detector_xhat[:, design.data_mask])
    assert np.array_equal(detector_decisions, detector_bits)

    # Numerical detector verification: compare the matrix-free, residual-
    # converged CG result with an explicit dense LMMSE solve on the DATA
    # columns. This checks the symbol detector independently of BER curves.
    dense_variance = 1.0e-3
    dense_noise = complex_noise(
        (1, cfg.M, cfg.N), dense_variance, rng
    )
    dense_y = detector_y[:1] + dense_noise
    dense_h = detector_h[:1]
    cg_solution = lmmse_equalize(
        dense_y, dense_h, design, cfg, noise_variance=dense_variance
    )

    H_dense = build_Heff_from_hhat_fractional(
        cfg.M, cfg.N, cfg.l_max, cfg.k_max, cfg.Q, dense_h[0]
    )
    y_vec = _dd_grid_to_vec(dense_y)[0]
    pilot_vec = _dd_grid_to_vec(design.pilot_grid)[0]
    data_vec_mask = (
        _dd_grid_to_vec(design.data_mask.astype(np.complex128))[0].real > 0.5
    )
    H_data = H_dense[:, data_vec_mask]
    dense_rhs = H_data.conj().T @ (y_vec - H_dense @ pilot_vec)
    dense_normal = (
        H_data.conj().T @ H_data
        + (dense_variance / float(design.data_energy))
        * np.eye(H_data.shape[1], dtype=np.complex128)
    )
    dense_solution = np.linalg.solve(dense_normal, dense_rhs)
    cg_data = _dd_grid_to_vec(cg_solution)[0][data_vec_mask]
    detector_dense_error = float(
        np.linalg.norm(cg_data - dense_solution)
        / max(np.linalg.norm(dense_solution), 1e-30)
    )
    assert detector_dense_error < 1e-7

    # ---- counted BER must be exact on a noiseless channel ------------------
    # The plotted BER is a direct count of incorrect bits over transmitted
    # bits, so it is validated by counting: with the true channel supplied to
    # the common detector and no noise, the count must be exactly zero, and
    # with a known heavy noise level it must be strictly positive. No analytic
    # error-probability model is involved anywhere in the pipeline.
    count_rng = np.random.default_rng(cfg.seed + 4242)
    count_frames = 200
    count_variance = 10.0 ** (-15.0 / 10.0)
    count_channel = generate_physical_fractional_channels(count_frames, cfg, count_rng)
    count_bits = count_rng.integers(
        0, 2, size=(count_frames, 2 * int(design.data_mask.sum())), dtype=np.int8
    )
    count_x = np.broadcast_to(
        design.pilot_grid, (count_frames, cfg.M, cfg.N)
    ).copy()
    count_x[:, design.data_mask] = np.sqrt(design.data_energy) * qpsk_mod(count_bits)
    count_y_clean = apply_physical_fractional_channel(count_channel, count_x, cfg)
    count_y = count_y_clean + complex_noise(
        count_y_clean.shape, count_variance, count_rng
    )
    count_obs = count_y[
        :, design.observation_flat_delay, design.observation_flat_doppler
    ]
    count_x_hat = lmmse_equalize(
        count_y,
        coarse_ls(count_obs, design),
        design,
        cfg,
        noise_variance=count_variance,
    )
    counted_errors = int(
        np.count_nonzero(
            qpsk_demod(count_x_hat[:, design.data_mask]) != count_bits
        )
    )
    counted_ber = counted_errors / count_bits.size
    assert counted_errors > 0, "LS at 15 dB must produce a countable BER."
    assert 0.0 < counted_ber < 0.5, counted_ber

    # Estimated support/parameter refinement must be deterministic, finite and
    # dimensionally valid using receiver-side quantities only.
    refinement_variance = 10.0 ** (-15.0 / 10.0)
    refinement_observation = (
        h_fractional[:4] @ design.sensing_matrix.T
        + complex_noise((4, cfg.observation_length), refinement_variance, rng)
    )
    refinement_initial = coarse_ls(refinement_observation, design)
    refinement_engine = EstimatedPathRefiner(cfg, design)
    refinement_output = refinement_engine.refine(
        refinement_observation, refinement_initial, refinement_variance
    )
    assert refinement_output.shape == refinement_initial.shape
    assert np.all(np.isfinite(refinement_output))
    assert np.array_equal(
        refinement_output,
        refinement_engine.refine(
            refinement_observation, refinement_initial, refinement_variance
        ),
    )

    print(
        f"Unit energy: pilot RE = {design.pilot_re_energy:g} Es, "
        f"data RE = {design.data_energy:g} Es, "
        f"four-pilot energy = {n_pilot * design.pilot_re_energy:g} Es, "
        f"active-frame energy = {frame_energy:g} Es"
    )
    print(
        "Self-checks passed: "
        f"Phi={design.sensing_matrix.shape}, "
        f"rank={np.linalg.matrix_rank(design.sensing_matrix)}, "
        f"pilots={int(design.pilot_mask.sum())}, "
        f"guards={int(design.guard_mask.sum())}, "
        f"data_REs={int(design.data_mask.sum())}, "
        f"max_data_to_pilot_leakage={np.max(np.abs(leakage)):.2e}, "
        f"physical_sparse_error={physical_sparse_error:.2e}, "
        f"explicit_H_error={explicit_matrix_error:.2e}, "
        f"H_forward_error={explicit_operator_error:.2e}, "
        f"H_adjoint_error={adjoint_error:.2e}, "
        "estimated_path_refinement=finite/deterministic, "
        f"counted_BER_sanity={counted_ber:.3e}, "
        "noiseless_detector_BER=0."
    )


def implementation_assumptions(cfg: Config) -> Dict[str, object]:
    return {
        "power_allocation": (
            "none: every pilot and active data symbol has unit energy; the "
            "four-pilot block has total energy 4"
        ),
        "pilot_power": (
            "Ep=1 for each of four pilots; no multiplier, ratio, dB boost, "
            "pilot/data redistribution, or command-line power override"
        ),
        "pilot_geometry_from_PnP": "contiguous 2x2 block with zero guards",
        "pilot_phases_not_reported": (
            "unit-energy QPSK/Hadamard phases; measured tr(G^-1)=28.41 against "
            "250.25 for an all-ones block and 37.53 for a Zadoff-like block, so "
            "this choice is near-optimal for a 2x2 footprint"
        ),
        "circular_Doppler_guard": (
            "used because Pn+4(kmax+Q)=22 exceeds N=20; OTFS Doppler indices wrap"
        ),
        "E0_definition": "unit active-RE symbol energy Es=1, sigma^2=10^(-SNR/10)",
        "pseudoinverse_typo_resolution": (
            "numpy Moore-Penrose pinv is used because the printed right-inverse "
            "formula is invalid for the full-column-rank 96x77 sensing matrix"
        ),
        "test_set_separation": (
            "Monte-Carlo test frames are drawn from an independent RNG stream and "
            "are never used for training or model selection"
        ),
        "BER_extension": "uncoded QPSK with estimated-CSI LMMSE",
        "BER_metric": (
            "directly counted uncoded BER = incorrect received bits / total "
            "transmitted bits from hard QPSK decisions at the common LMMSE "
            "detector output; no analytic error-probability model anywhere"
        ),
        "BER_sweep_length": (
            "the counted BER resolves about 1/(400*frames), so the sweep is "
            f"lengthened at high SNR via ber_frame_scale {tuple(cfg.ber_frame_scale)}. "
            "The scale is identical for every estimator at a given SNR"
        ),
        "FDPGNet_initialisation": (
            "the learned prior head is zero-initialised and the manifold trust "
            "gate is closed, so the UNTRAINED network is exactly damped EM-SBL "
            f"with rho={cfg.fdpg_em_damping}. No true support, path gain, delay, "
            "integer Doppler or fractional kappa is used in inference"
        ),
        "FDPGNet_estimated_support_refinement": (
            f"enabled={cfg.fdpg_estimated_path_refinement}; the strongest "
            f"{cfg.fdpg_refinement_candidate_cells} cells in the FD-PGNet output "
            "form a candidate pool, after which greedy support selection uses "
            "only the received pilot residual. cfg.paths is an explicitly "
            "assumed maximum component budget, not a realized true support"
        ),
        "FDPGNet_estimated_parameter_refinement": (
            "fractional Doppler is initialized on the disclosed manifold grid "
            f"and refined using {cfg.fdpg_refinement_kappa_sweeps} local "
            "pilot-residual coordinate sweeps; complex gains are recomputed by "
            "Bayesian LMMSE. No generated channel parameter is passed in"
        ),
        "FDPGNet_refinement_validation_gate": (
            "a scalar blend between base FD-PGNet and estimated-parameter output "
            "is fitted in closed form on the independent validation split and "
            "stored in the checkpoint; test targets and benchmark rankings are "
            "never used"
        ),
        "FDPGNet_manifold_role": (
            "the fractional-Doppler leakage manifold supplies the PRIOR VARIANCE "
            "of an unconstrained Bayesian solve, not a projection of the estimate. "
            "A projection was measured to saturate at -19.6 dB even on a -30.9 dB "
            "input because two physical paths share one (delay, Doppler) cell in "
            "about 9% of frames; used as a prior the same physics carries no "
            "asymptotic bias"
        ),
        "FDPGNet_posterior_conditioning": (
            "the posterior is solved as Sigma = D (D G D/sigma^2 + I)^-1 D with "
            "D = diag(sqrt(gamma)); the bracketed matrix has eigenvalues >= 1 for "
            f"every admissible gamma. Solver: {cfg.fdpg_posterior_mode}"
        ),
        "FDPGNet_loss": (
            "deep-supervised aggregate NMSE plus Morozov measurement-consistency "
            f"regularization with weight {cfg.fdpg_consistency_weight}"
        ),
        "training_snr_mode": (
            f"{cfg.training_snr_mode}; 'per_snr' trains one model per plotted SNR, "
            f"'shared' trains a single model over {cfg.shared_snr_range_db} dB"
        ),
        "achievability_limits_reported": (
            "LS, marginal-prior LMMSE, oracle-support, oracle-parameter and the "
            "exact Bayesian MMSE are computed by the bounds module and saved "
            "alongside every result so the reported NMSE can be read against "
            "what is attainable"
        ),
    }


def complexity_report(cfg: Config, model=None) -> Dict[str, object]:
    """Dominant per-frame arithmetic of one FD-PGNet inference pass."""
    Ln, Lm = cfg.channel_length, cfg.observation_length
    T = cfg.fdpg_stages
    NC, NK = cfg.n_cells, cfg.fdpg_kappa_atoms
    if cfg.fdpg_posterior_mode == "exact":
        full_solve = Ln ** 3 / 3.0 + Ln ** 3 / 3.0    # cholesky + cholesky_inverse
        mean_only = Ln ** 3 / 3.0                      # cholesky + solve, no inverse
        solver = "Cholesky + cholesky_inverse (final re-estimation skips the inverse)"
    else:
        full_solve = cfg.fdpg_cg_steps * Ln ** 2 + Ln ** 2
        mean_only = cfg.fdpg_cg_steps * Ln ** 2         # CG solve, no Jacobi diagonal
        solver = f"{cfg.fdpg_cg_steps}-step CG + Jacobi diagonal (final re-estimation skips the diagonal)"
    # T stage solves need the posterior variance; the final re-estimation
    # discards it (see `need_variance` on `_posterior`), so it is cheaper.
    posterior = T * full_solve + mean_only
    manifold = T * cfg.fdpg_manifold_sweeps * NC * NK * 3 * 4
    linear = (T + 1) * (Ln * Lm + Ln ** 2)
    conv = T * 2 * 9 * cfg.fdpg_features ** 2 * Ln
    refinement = 0
    if cfg.fdpg_estimated_path_refinement:
        refinement = (
            cfg.paths * cfg.fdpg_refinement_candidate_cells
            * cfg.fdpg_kappa_atoms * Lm
            + cfg.fdpg_refinement_kappa_sweeps * cfg.paths
            * cfg.fdpg_refinement_local_points * Ln * Lm
        )
    total = posterior + manifold + linear + conv + refinement
    return {
        "posterior_solver": solver,
        "complex_MACs_posterior": int(posterior),
        "complex_MACs_manifold": int(manifold),
        "complex_MACs_linear": int(linear),
        "complex_MACs_estimated_path_refinement": int(refinement),
        "real_MACs_conv_head": int(conv),
        "approx_total_MACs_per_frame": int(total),
        "trainable_parameters": None if model is None else count_parameters(model),
        "note": (
            "v1 used 6 stages of torch.linalg.inv on a 77x77 complex matrix plus "
            "6 DRSN shrinkage blocks (3 conv + BatchNorm each). v2 uses "
            f"{T} stages, a Cholesky pair instead of a general inverse, and no "
            "shrinkage stack. Set fdpg_posterior_mode='cg' for an O(K L^2) "
            "posterior. The estimated-path-refinement MAC count above was "
            "already small; what dominated its measured wall-clock time was "
            "the ORIGINAL implementation looping in Python per frame and "
            "rebuilding one dictionary atom at a time (added on request: "
            "EstimatedPathRefiner.refine() is now a batched vectorization of "
            "the exact same algorithm -- see its class docstring and "
            "benchmark_cpu_latency() below). Round 4 (module docstring) "
            "lowered fdpg_refinement_kappa_sweeps 2->1 and "
            "fdpg_refinement_local_points 9->5 (~49% lower "
            "complex_MACs_estimated_path_refinement, ~0.005 dB NMSE change "
            "measured on a trained checkpoint) and added a cached "
            "torch.jit.trace for inference (FDPGNet.forward_fast, used "
            "automatically by infer_fdpgnet/benchmark_cpu_latency); the trace "
            "is a graph-level fusion of the SAME arithmetic this report "
            "counts, so it changes measured ms/frame in "
            "benchmark_cpu_latency() below without changing any MAC count "
            "here. Round 5 (module docstring) lowered fdpg_manifold_sweeps "
            "2->1 (~50% lower complex_MACs_manifold) after verifying it SAFE "
            "by actually retraining from scratch and re-evaluating on fresh "
            "test frames (0.007-0.016 dB NMSE change at 15/30 dB, 'quick' "
            "profile) -- unlike Round 4's refinement-only knobs, this one is "
            "part of the trained forward graph, so retraining (not just "
            "recalibrating alpha) was required to call it safe. The same "
            "retrain-and-verify procedure was ALSO applied to fdpg_cg_steps "
            "(12 -> 6) and REJECTED: 0.07-0.33 dB even after retraining, "
            "which is real, not noise -- fdpg_cg_steps is unchanged at 12."
        ),
    }


def _detect_cpu_label() -> str:
    """Best-effort, human-readable CPU model string used only to LABEL the
    benchmark below (e.g. an Intel Core i7 string when run on one); it never
    influences any computation."""
    label = None
    try:
        if sys.platform.startswith("linux"):
            with open("/proc/cpuinfo", encoding="utf-8", errors="ignore") as handle:
                for line in handle:
                    if line.lower().startswith("model name"):
                        label = line.split(":", 1)[1].strip()
                        break
    except OSError:
        label = None
    if not label:
        import platform
        label = platform.processor() or platform.uname().processor or platform.machine()
    return label or "unknown CPU"


def benchmark_cpu_latency(
    model,
    cfg: Config,
    design: PilotDesign,
    frames: int = 200,
    warmup_frames: int = 20,
    repeats: int = 5,
    snr_db: float = 15.0,
    cpu_threads: int | None = None,
) -> Dict[str, object]:
    """Practical CPU wall-clock latency benchmark (added on request), in the
    same "Method / Parameters / Depth-iterations / CPU latency (ms/frame)"
    shape used to report DRSNN/CENet/CE-DU-ISTA/FD-PGNet elsewhere.

    Measures LS, the FD-PGNet network alone, and FD-PGNet + the estimated-
    path refinement, on THIS PROCESS'S CPU (label auto-detected below; run on
    an Intel i7-class machine for the requested "Intel i7 CPU" figures).
    Draws its own frames from an independently seeded RNG, moves the model to
    CPU only for the duration of the benchmark, and restores its original
    device afterwards, so it has no effect on training or on ``evaluate()``.
    """
    torch, _, _ = import_torch()
    if cpu_threads is not None and int(cpu_threads) > 0:
        torch.set_num_threads(int(cpu_threads))
    cpu = torch.device("cpu")
    original_device = next(model.parameters()).device
    model.to(cpu).eval()

    variance = 10.0 ** (-float(snr_db) / 10.0)
    rng = np.random.default_rng(cfg.seed + 424242)
    data_count = int(design.data_mask.sum())
    bits_per_frame = 2 * data_count

    def _draw(count: int):
        physical = generate_physical_fractional_channels(count, cfg, rng)
        tx_bits = rng.integers(0, 2, size=(count, bits_per_frame), dtype=np.int8)
        x = np.broadcast_to(design.pilot_grid, (count, cfg.M, cfg.N)).copy()
        x[:, design.data_mask] = qpsk_mod(tx_bits)
        y = apply_physical_fractional_channel(physical, x, cfg)
        y = y + complex_noise(y.shape, variance, rng)
        y_obs = y[:, design.observation_flat_delay, design.observation_flat_doppler]
        return y_obs

    refiner = (
        EstimatedPathRefiner(cfg, design)
        if cfg.fdpg_estimated_path_refinement else None
    )

    def _base(y_obs):
        return infer_fdpgnet(model, y_obs, variance, cpu)

    def _with_refinement(y_obs):
        base = infer_fdpgnet(model, y_obs, variance, cpu)
        alpha = float(getattr(model, "refinement_alpha", 0.0)) or 1.0
        if refiner is None:
            return base
        refined = refiner.refine(y_obs, base, variance)
        return base + alpha * (refined - base)

    frames = max(int(frames), 1)
    repeats = max(int(repeats), 1)
    warmup_frames = max(int(warmup_frames), 0)

    if warmup_frames > 0:
        warm = _draw(warmup_frames)
        coarse_ls(warm, design)
        _with_refinement(warm)

    def _time(fn) -> tuple[float, float]:
        samples = []
        for _ in range(repeats):
            y_obs = _draw(frames)
            t0 = time.perf_counter()
            fn(y_obs)
            t1 = time.perf_counter()
            samples.append((t1 - t0) / frames * 1000.0)
        samples.sort()
        return samples[len(samples) // 2], float(np.std(samples))

    ls_ms, ls_std = _time(lambda y_obs: coarse_ls(y_obs, design))
    base_ms, base_std = _time(_base)
    refine_ms, refine_std = _time(_with_refinement)

    model.to(original_device)

    rows = [
        {
            "method": "LS",
            "parameters": 0,
            "depth_iterations": "closed form",
            "ms_per_frame_median": ls_ms,
            "ms_per_frame_std": ls_std,
        },
        {
            "method": "FD-PGNet",
            "parameters": count_parameters(model),
            "depth_iterations": f"{cfg.fdpg_stages} stages + final posterior",
            "ms_per_frame_median": base_ms,
            "ms_per_frame_std": base_std,
        },
    ]
    if cfg.fdpg_estimated_path_refinement:
        rows.append(
            {
                "method": "FD-PGNet + refinement",
                "parameters": count_parameters(model),
                "depth_iterations": (
                    f"+ {cfg.fdpg_refinement_kappa_sweeps} refinement sweeps"
                ),
                "ms_per_frame_median": refine_ms,
                "ms_per_frame_std": refine_std,
            }
        )
    return {
        "cpu_label": _detect_cpu_label(),
        "torch_threads": int(torch.get_num_threads()),
        "benchmark_snr_db": float(snr_db),
        "frames_per_repeat": frames,
        "repeats": repeats,
        "warmup_frames": warmup_frames,
        "rows": rows,
    }


def print_cpu_latency_table(report: Dict[str, object]) -> None:
    print(
        f"\nPractical CPU latency @ E0/N0={report['benchmark_snr_db']:g} dB on "
        f"{report['cpu_label']} ({report['torch_threads']} torch threads), "
        f"{report['repeats']} repeats x {report['frames_per_repeat']} frames "
        f"(median, after {report['warmup_frames']}-frame warm-up):"
    )
    print(f"  {'Method':<24}{'Parameters':>12}{'Depth/iterations':>30}{'CPU latency (ms/frame)':>26}")
    for row in report["rows"]:
        print(
            f"  {row['method']:<24}{row['parameters']:>12,}"
            f"{row['depth_iterations']:>30}"
            f"{row['ms_per_frame_median']:>22.3f} +/- {row['ms_per_frame_std']:.3f}"
        )


def evaluate(fdpgnet_models, cfg: Config, design: PilotDesign, device):
    """Monte-Carlo NMSE and directly counted uncoded BER."""
    rng = np.random.default_rng(cfg.seed + 10)
    data_count = int(design.data_mask.sum())
    bits_per_frame = 2 * data_count
    path_refiner = (
        EstimatedPathRefiner(cfg, design)
        if cfg.fdpg_estimated_path_refinement else None
    )

    nmse_list, base_nmse_list, ber_list, error_list, alpha_list = [], [], [], [], []
    total_bits_list, frames_list, cg_list = [], [], []

    for snr_db in cfg.snr_db:
        model = fdpgnet_models[float(snr_db)]
        max_frames = int(cfg.frames_at_snr(snr_db))
        variance = 10.0 ** (-float(snr_db) / 10.0)

        squared_error = squared_error_base = true_energy = 0.0
        bit_errors = total_bits = processed = cg_used = 0
        started = last_report = time.time()

        while processed < max_frames:
            batch = min(cfg.eval_batch_size, max_frames - processed)
            physical = generate_physical_fractional_channels(batch, cfg, rng)
            h_true = physical.sparse_h
            tx_bits = rng.integers(0, 2, size=(batch, bits_per_frame), dtype=np.int8)
            x = np.broadcast_to(design.pilot_grid, (batch, cfg.M, cfg.N)).copy()
            x[:, design.data_mask] = np.sqrt(design.data_energy) * qpsk_mod(tx_bits)

            y = apply_physical_fractional_channel(physical, x, cfg)
            y += complex_noise(y.shape, variance, rng)
            y_obs = y[
                :, design.observation_flat_delay, design.observation_flat_doppler
            ]

            h_base = infer_fdpgnet(model, y_obs, variance, device)
            alpha = float(getattr(model, "refinement_alpha", 0.0))
            if path_refiner is not None and alpha > 0.0:
                h_parametric = path_refiner.refine(y_obs, h_base, variance)
                h_hat = h_base + alpha * (h_parametric - h_base)
            else:
                h_hat = h_base
            squared_error_base += float(np.sum(np.abs(h_base - h_true) ** 2))
            squared_error += float(np.sum(np.abs(h_hat - h_true) ** 2))
            true_energy += float(np.sum(np.abs(h_true) ** 2))

            x_hat, iterations = lmmse_equalize(
                y, h_hat, design, cfg, noise_variance=variance,
                return_iterations=True,
            )
            cg_used = max(cg_used, int(iterations))
            rx_bits = qpsk_demod(x_hat[:, design.data_mask])
            bit_errors += int(np.count_nonzero(rx_bits != tx_bits))
            total_bits += tx_bits.size
            processed += batch

            now = time.time()
            if now - last_report > 20.0 and processed < max_frames:
                rate = processed / max(now - started, 1e-9)
                print(
                    f"    E0/N0={snr_db:2d} dB | {processed:,}/{max_frames:,} frames "
                    f"| {rate:6.1f} frames/s | "
                    f"~{(max_frames - processed) / max(rate, 1e-9):5.0f}s left "
                    f"| errors so far={bit_errors:,}",
                    flush=True,
                )
                last_report = now

        nmse = squared_error / true_energy
        base_nmse = squared_error_base / true_energy
        ber = bit_errors / total_bits
        nmse_list.append(nmse)
        base_nmse_list.append(base_nmse)
        ber_list.append(ber)
        error_list.append(bit_errors)
        alpha_list.append(float(getattr(model, "refinement_alpha", 0.0)))
        total_bits_list.append(total_bits)
        frames_list.append(processed)
        cg_list.append(cg_used)

        if bit_errors == 0:
            upper95 = 1.0 - 0.05 ** (1.0 / total_bits)
            ber_text = f"0.0000e+00 (0 errors; 95% upper < {upper95:.3e})"
        else:
            ber_text = f"{ber:.4e}"
        print(
            f"E0/N0={snr_db:2d} dB | NMSE="
            f"{10 * np.log10(nmse):8.3f} dB "
            f"(base {10 * np.log10(base_nmse):.3f} dB, "
            f"estimated-path alpha={alpha_list[-1]:.4f}) | "
            f"counted BER={ber_text} | bit errors={bit_errors:,} | "
            f"frames={processed:,} | bits={total_bits:,} | "
            f"max detector CG iters={cg_used} | {time.time() - started:.0f}s",
            flush=True,
        )

    return {
        "snr_db": np.asarray(cfg.snr_db, dtype=float),
        "nmse_fdpgnet": np.asarray(nmse_list),
        "nmse_fdpgnet_base": np.asarray(base_nmse_list),
        "estimated_path_refinement_alpha": np.asarray(alpha_list),
        "ber_fdpgnet_lmmse": np.asarray(ber_list),
        "bit_errors_fdpgnet": np.asarray(error_list, dtype=np.int64),
        "total_bits": np.asarray(total_bits_list, dtype=np.int64),
        "test_frames": np.asarray(frames_list, dtype=np.int64),
        "detector_cg_iterations": np.asarray(cg_list, dtype=np.int64),
    }


def _ber_for_semilogy(ber, errors, total_bits):
    """Split a counted BER curve into plottable points and zero-error bounds."""
    ber = np.asarray(ber, dtype=float)
    errors = np.asarray(errors, dtype=np.int64)
    total_bits = np.asarray(total_bits, dtype=np.int64)
    zero = errors == 0
    measured = np.where(zero, np.nan, ber)
    with np.errstate(divide="ignore"):
        upper95 = 1.0 - 0.05 ** (1.0 / np.maximum(total_bits, 1))
    return measured, np.where(zero, upper95, np.nan), zero


def save_results(results, cfg: Config, output_dir: Path, bounds=None,
                 complexity=None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = dict(results)
    if bounds:
        payload.update(bounds)
    np.savez(output_dir / "fdpgnet_v2_results.npz", **payload)

    columns = [
        results["snr_db"],
        10.0 * np.log10(results["nmse_fdpgnet"]),
        10.0 * np.log10(results["nmse_fdpgnet_base"]),
        results["estimated_path_refinement_alpha"],
        results["ber_fdpgnet_lmmse"],
        results["bit_errors_fdpgnet"],
        results["total_bits"],
        results["test_frames"],
    ]
    header = (
        "SNR_dB,FDPGNET_REFINED_NMSE_dB,FDPGNET_BASE_NMSE_dB,"
        "ESTIMATED_PATH_REFINEMENT_ALPHA,FDPGNET_counted_BER,"
        "FDPGNET_bit_errors,total_bits,test_frames"
    )
    if bounds:
        for key in ("bound_ls", "bound_lmmse_marginal", "bound_exact_mmse",
                    "bound_oracle_support", "bound_oracle_parameter"):
            columns.append(bounds[key])
            header += "," + key.upper()
    np.savetxt(
        output_dir / "fdpgnet_v2_results.csv",
        np.column_stack(columns),
        delimiter=",", header=header, comments="", fmt="%.10g",
    )

    with open(output_dir / "implementation_assumptions.json", "w") as handle:
        json.dump(implementation_assumptions(cfg), handle, indent=2, sort_keys=True)
    if complexity:
        with open(output_dir / "complexity_report.json", "w") as handle:
            json.dump(complexity, handle, indent=2, sort_keys=True)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib unavailable; numeric results were still written.")
        return

    figure, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))
    snr = results["snr_db"]

    if bounds:
        axes[0].plot(snr, bounds["bound_ls"], ":", color="0.55", lw=1.4, label="LS")
        axes[0].plot(snr, bounds["bound_lmmse_marginal"], "--", color="0.35", lw=1.4,
                     label="LMMSE, marginal prior")
        axes[0].plot(snr, bounds["bound_oracle_support"], "-.", color="tab:green",
                     lw=1.5, label="Oracle-support bound")
        axes[0].plot(snr, bounds["bound_oracle_parameter"], "-.", color="tab:purple",
                     lw=1.5, label="Oracle-parameter bound")
        finite = np.isfinite(bounds["bound_exact_mmse"])
        if np.any(finite):
            axes[0].plot(snr[finite], bounds["bound_exact_mmse"][finite], "kD",
                         ms=6, label="Exact Bayesian MMSE")
    axes[0].plot(
        snr, 10.0 * np.log10(results["nmse_fdpgnet_base"]), "--o",
        color="tab:brown", alpha=0.55, lw=1.3, ms=5,
        label="FD-PGNet v2 (before estimated-path refinement)",
    )
    axes[0].plot(snr, 10.0 * np.log10(results["nmse_fdpgnet"]), "-*",
                 color="tab:brown", lw=2.2, ms=10,
                 label="FD-PGNet v2 + estimated-path refinement")
    axes[0].set_xlabel(r"$E_0/N_0$ (dB)")
    axes[0].set_ylabel("NMSE (dB)")
    axes[0].set_title("Channel estimation (no pilot power boost)")
    axes[0].grid(True, alpha=0.35)
    axes[0].legend(loc="lower left", fontsize=8)

    measured, upper95, zero = _ber_for_semilogy(
        results["ber_fdpgnet_lmmse"], results["bit_errors_fdpgnet"],
        results["total_bits"],
    )
    line, = axes[1].semilogy(snr, measured, "-*", color="tab:brown", lw=2.2, ms=10,
                             label="FD-PGNet v2 + LMMSE")
    if np.any(zero):
        axes[1].semilogy(snr[zero], upper95[zero], "v", mfc="none", ms=9,
                         color=line.get_color(), linestyle="none",
                         label="0 errors (95% upper bound)")
    finite = np.concatenate(
        (measured[np.isfinite(measured)], upper95[np.isfinite(upper95)])
    )
    bottom = np.min(finite[finite > 0.0], initial=1.0) / 5.0
    axes[1].set_ylim(bottom=max(bottom, 1e-12), top=1.0)
    axes[1].set_xlabel(r"$E_0/N_0$ (dB)")
    axes[1].set_ylabel("Uncoded BER (directly counted)")
    axes[1].set_title("Estimated-CSI symbol detection")
    axes[1].grid(True, which="both", alpha=0.35)
    axes[1].legend(loc="lower left", fontsize=9)

    figure.tight_layout()
    figure.savefig(output_dir / "fdpgnet_v2_nmse_ber.png", dpi=220)
    plt.close(figure)


def running_in_notebook() -> bool:
    try:
        from IPython import get_ipython
    except Exception:
        return False
    shell = get_ipython()
    if shell is None:
        return False
    return shell.__class__.__name__ != "TerminalInteractiveShell"


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FD-PGNet v2: manifold-prior Bayesian unfolding for OTFS "
                    "channel estimation, with no pilot power boost.",
        allow_abbrev=False,
    )
    parser.add_argument("--profile", choices=("paper", "quick", "smoke"),
                        default="paper")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default="fdpgnet_v2_output")
    parser.add_argument("--checkpoint-dir", default="fdpgnet_v2_checkpoints")
    parser.add_argument("--retrain", action="store_true")
    parser.add_argument("--self-check-only", action="store_true")
    parser.add_argument("--bounds-only", action="store_true",
                        help="Compute and print the achievability limits, then stop.")
    parser.add_argument("--no-bounds", action="store_true")
    parser.add_argument("--training-snr-mode", choices=("per_snr", "shared"),
                        default=None)
    parser.add_argument("--posterior-mode", choices=("exact", "cg"), default=None)
    parser.add_argument("--stages", type=int, default=None)
    parser.add_argument(
        "--cpu-threads", type=int, default=None,
        help=(
            "torch.set_num_threads() for the CPU latency benchmark below "
            "(added on request). Optimal value is hardware-dependent for "
            "these small batched ops; sweep it on your machine. Leaving it "
            "unset keeps PyTorch's default."
        ),
    )
    parser.add_argument(
        "--no-latency-benchmark", action="store_true",
        help="Skip the practical CPU latency benchmark (added on request).",
    )

    if argv is None:
        argv = [] if running_in_notebook() else sys.argv[1:]
    args, unknown = parser.parse_known_args(list(argv))
    if unknown:
        print(f"Ignoring unrecognised arguments: {unknown}")
    return args


def run(profile: str = "quick", device: str = "auto",
        output_dir: str = "fdpgnet_v2_output",
        checkpoint_dir: str = "fdpgnet_v2_checkpoints",
        retrain: bool = False, self_check_only: bool = False,
        bounds_only: bool = False, no_bounds: bool = False,
        training_snr_mode: str = None, posterior_mode: str = None,
        stages: int = None):
    """Notebook entry point, e.g. ``results = run(profile='quick', device='cuda')``."""
    argv = ["--profile", str(profile), "--device", str(device),
            "--output-dir", str(output_dir), "--checkpoint-dir", str(checkpoint_dir)]
    if retrain:
        argv.append("--retrain")
    if self_check_only:
        argv.append("--self-check-only")
    if bounds_only:
        argv.append("--bounds-only")
    if no_bounds:
        argv.append("--no-bounds")
    if training_snr_mode:
        argv += ["--training-snr-mode", training_snr_mode]
    if posterior_mode:
        argv += ["--posterior-mode", posterior_mode]
    if stages:
        argv += ["--stages", str(int(stages))]
    return main(argv)


def main(argv=None):
    args = parse_args(argv)
    cfg = profile_config(args.profile)
    if args.training_snr_mode:
        cfg = replace(cfg, training_snr_mode=args.training_snr_mode)
    if args.posterior_mode:
        cfg = replace(cfg, fdpg_posterior_mode=args.posterior_mode)
    if args.stages:
        cfg = replace(cfg, fdpg_stages=int(args.stages))

    design = make_pilot_design(cfg)
    run_self_checks(cfg, design)
    print(json.dumps(implementation_assumptions(cfg), indent=2, sort_keys=True))

    if args.self_check_only:
        return None

    bounds = None
    if not args.no_bounds:
        bounds = compute_bounds(cfg, design, verbose=True)
    if args.bounds_only:
        return bounds

    torch, _, _ = import_torch()
    set_all_seeds(cfg.seed, torch)
    device = resolve_device(args.device, torch)
    print(f"Using device: {device}")

    models, _ = build_fdpgnet_model_bank(
        cfg, design, device, Path(args.checkpoint_dir),
        reuse_checkpoint=not args.retrain,
    )
    complexity = complexity_report(cfg, next(iter(models.values())))
    print("\nComplexity: " + json.dumps(complexity, indent=2, sort_keys=True))

    if not args.no_latency_benchmark:
        latency = benchmark_cpu_latency(
            next(iter(models.values())), cfg, design,
            cpu_threads=args.cpu_threads,
        )
        print_cpu_latency_table(latency)

    results = evaluate(models, cfg, design, device)
    save_results(results, cfg, Path(args.output_dir), bounds, complexity)
    print(f"Saved results in: {Path(args.output_dir).resolve()}")
    return results


if __name__ == "__main__":
    main()
