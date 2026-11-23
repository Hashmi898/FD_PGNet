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


## Usage1 if you are using kaggle
1) Step one just download ZIP File
2) Upload the Zip file on kaggle.
3) Turn on the GPU
4) Run this Command (also check you file directory)
      from pathlib import Path
      import shutil
      import zipfile
      import os
      
      input_dir = Path("/kaggle/input")
      project_files = list(input_dir.rglob("Main.py"))
      
      # Extract the upload if Kaggle has kept it as a ZIP.
      if not project_files:
          extract_dir = Path("/kaggle/working/project_upload")
          for archive in input_dir.rglob("*.zip"):
              with zipfile.ZipFile(archive) as z:
                  z.extractall(extract_dir)
          project_files = list(extract_dir.rglob("Main.py"))
      
      if len(project_files) != 1:
          raise RuntimeError(
              f"Found {len(project_files)} Main.py files. "
              "Attach only the intended project dataset and check its files."
          )
      
      project_dir = Path("/kaggle/working/FD_PGNet")
      shutil.copytree(project_files[0].parent, project_dir, dirs_exist_ok=True)
      os.chdir(project_dir)
      
      print("Project folder:", project_dir)
      print("Files:", os.listdir())
5)Run the last command:
      %cd /kaggle/working/FD_PGNet
      !python -u Main.py


## Usage2 if you have a GPU Directly.
1) Local GPU execution: Install the required dependencies and a compatible CUDA-enabled PyTorch build, then run python Main.py from the project directory.



## Outputs
Written to `fdpgnet_v2_output/` by default:
- `fdpgnet_v2_results.csv` / `.npz`: NMSE, counted BER, bit errors and the bounds for each SNR
- `fdpgnet_v2_nmse_ber.png`: NMSE and BER figure
- `complexity_report.json`, `implementation_assumptions.json`

Trained models are saved in `fdpgnet_v2_checkpoints/` and reused on later
runs unless you pass `--retrain`.


<img width="1920" height="1440" alt="fdpgnet_nmse_vs_snr (1)" src="https://github.com/user-attachments/assets/a04f196d-7ade-45ba-adf5-42ca0184c6de" />
<img width="1920" height="1440" alt="fdpgnet_ber_vs_snr" src="https://github.com/user-attachments/assets/66492e8d-0273-47d4-a7c3-3a3f02c62a39" />




