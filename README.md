# FD-PGNet v2 — Fractional-Doppler Physics-Guided Bayesian Unfolding for OTFS

Channel estimation and data detection for OTFS with fractional Doppler, using
a 2×2 unit-energy embedded pilot (no pilot power boost). FD-PGNet v2 is an
unrolled EM-SBL network whose prior comes from the fractional-Doppler leakage
manifold. A pilot-only estimated-path refinement step follows the network.

## Project structure

```
FDPGNet_Project/
├── Main.py                         # Config, bounds, self-checks, evaluation, plots, CLI entry point
├── Utils/
│   ├── sensing_matrix.py           # PilotDesign, make_pilot_design (Phi), coarse_ls
│   ├── h_sparse_creation.py        # dirichlet_coefficient, physical paths -> sparse h, channel generation
│   └── H_eff_creation.py           # build_Heff_* (explicit H_eff), apply_dd_channel, apply_physical_fractional_channel
├── channel_estimation_Algo/
│   └── FDPGNet.py                  # FD-PGNet v2 model, training, checkpoints, inference, path refinement
├── Data_Detection/
│   └── lmmse_detector.py           # Common LMMSE detector (lmmse_equalize)
├── requirements.txt
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

To use a GPU, install the PyTorch build that matches your CUDA version from
https://pytorch.org/.

## Usage

Run all commands from the project root, the folder that contains `Main.py`.

```bash
python Main.py --self-check-only                      # model / operator consistency checks
python Main.py --bounds-only                          # LS, LMMSE, oracle and exact-MMSE references
python Main.py --profile smoke --device cpu           # fast end-to-end test run
python Main.py --profile quick --device cpu
python Main.py --profile paper --device cuda          # full-scale run
python Main.py --profile paper --device cpu --cpu-threads 4
```

Useful options: `--retrain`, `--no-bounds`, `--no-latency-benchmark`,
`--posterior-mode {exact,cg}`, `--training-snr-mode {per_snr,shared}`,
`--output-dir`, `--checkpoint-dir`.

From a Jupyter notebook opened in the project root:

```python
from Main import run
results = run(profile="quick", device="cuda")
```

## Outputs

Written to `fdpgnet_v2_output/` by default:

- `fdpgnet_v2_results.csv` / `.npz`: NMSE, counted BER, bit errors and the bounds for each SNR
- `fdpgnet_v2_nmse_ber.png`: NMSE and BER figure
- `complexity_report.json`, `implementation_assumptions.json`

Trained models are saved in `fdpgnet_v2_checkpoints/` and reused on later
runs unless you pass `--retrain`.
