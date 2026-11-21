"""PnP embedded pilot design and the DRSNN sensing matrix ``Phi``.

Moved verbatim from the original single-file FD-PGNet v2 script.

Contents
--------
PilotDesign          pilot / guard / data masks, Phi and its pseudoinverse
make_pilot_design    builds the 2 x 2 unit-energy pilot, zero guards and Phi
coarse_ls            DRSNN Eq. (8): LS estimate h = pinv(Phi) y
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from Utils.H_eff_creation import apply_dd_channel

if TYPE_CHECKING:  # type hints only (avoids a circular import with Main.py)
    from Main import Config


# -----------------------------------------------------------------------------
# DRSNN DD channel and PnP embedded pilot
# -----------------------------------------------------------------------------


@dataclass
class PilotDesign:
    pilot_grid: np.ndarray
    pilot_mask: np.ndarray
    guard_mask: np.ndarray
    data_mask: np.ndarray
    observation_delay: np.ndarray
    observation_doppler: np.ndarray
    observation_flat_delay: np.ndarray
    observation_flat_doppler: np.ndarray
    support_delay: np.ndarray
    support_doppler: np.ndarray
    sensing_matrix: np.ndarray
    sensing_pinv: np.ndarray
    data_energy: float = 1.0
    pilot_re_energy: float = 1.0


def make_pilot_design(cfg: Config) -> PilotDesign:
    """Create the 2 x 2 embedded pilot, zero guards, and DRSNN Phi matrix."""
    M, N = cfg.M, cfg.N
    pm, pn = cfg.pilot_delay_len, cfg.pilot_doppler_len
    k_ext = cfg.k_extended

    pilot_delay = np.arange(cfg.l_max, cfg.l_max + pm)
    pilot_k0 = N // 2 - pn // 2
    pilot_doppler = np.arange(pilot_k0, pilot_k0 + pn)

    pilot_grid = np.zeros((M, N), dtype=np.complex128)
    # A unit-energy 4-QAM Hadamard block; neither article discloses phases.
    pilot_block = np.exp(1j * np.pi / 4) * np.array(
        [[1.0, 1.0], [1.0, -1.0]], dtype=np.complex128
    )
    # Strict unit-energy pilots: no multiplier or pilot/data reallocation.
    pilot_re_energy = float(cfg.pilot_energy)
    pilot_grid[np.ix_(pilot_delay, pilot_doppler)] = (
        np.sqrt(pilot_re_energy) * pilot_block
    )
    pilot_mask = np.abs(pilot_grid) > 0

    support_delay = np.repeat(np.arange(cfg.l_max + 1), 2 * k_ext + 1)
    support_doppler = np.tile(np.arange(-k_ext, k_ext + 1), cfg.l_max + 1)

    # Exactly L_m=(P_m+l_max)(P_n+2k_max+2Q), not PnP Eq. (19).
    observation_delay = np.arange(
        pilot_delay[0], pilot_delay[-1] + cfg.l_max + 1
    ) % M
    observation_doppler = np.arange(
        pilot_doppler[0] - k_ext, pilot_doppler[-1] + k_ext + 1
    ) % N
    od, ok = np.meshgrid(observation_delay, observation_doppler, indexing="ij")
    obs_l = od.reshape(-1)
    obs_k = ok.reshape(-1)

    # Zero every transmitted RE that can enter a pilot observation.
    reserved = np.zeros((M, N), dtype=bool)
    for delay, doppler in zip(support_delay, support_doppler):
        reserved[(obs_l - delay) % M, (obs_k - doppler) % N] = True
    guard_mask = reserved & ~pilot_mask
    data_mask = ~reserved

    provisional = PilotDesign(
        pilot_grid=pilot_grid,
        pilot_mask=pilot_mask,
        guard_mask=guard_mask,
        data_mask=data_mask,
        observation_delay=observation_delay,
        observation_doppler=observation_doppler,
        observation_flat_delay=obs_l,
        observation_flat_doppler=obs_k,
        support_delay=support_delay,
        support_doppler=support_doppler,
        sensing_matrix=np.empty((0, 0), dtype=np.complex128),
        sensing_pinv=np.empty((0, 0), dtype=np.complex128),
    )

    columns = []
    for column in range(cfg.channel_length):
        basis = np.zeros((1, cfg.channel_length), dtype=np.complex128)
        basis[0, column] = 1.0
        response = apply_dd_channel(basis, pilot_grid, provisional, cfg)[0]
        columns.append(response[obs_l, obs_k])
    sensing = np.stack(columns, axis=1)
    pinv = np.linalg.pinv(sensing, rcond=1e-12)
    provisional.sensing_matrix = sensing
    provisional.sensing_pinv = pinv

    # Active data symbols use the same unit energy as every pilot.
    provisional.pilot_re_energy = pilot_re_energy
    provisional.data_energy = float(cfg.pilot_energy)
    return provisional


def coarse_ls(y_observed: np.ndarray, design: PilotDesign) -> np.ndarray:
    """DRSNN Eq. (8), evaluated with the Moore-Penrose pseudoinverse."""
    return np.asarray(y_observed) @ design.sensing_pinv.T
