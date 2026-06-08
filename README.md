# MF-Net Core Reproduction

Minimal standalone code for reproducing the main synthetic MF-Net runs.

## Install

```bash
pip install -r requirements.txt
```

## Data

The repository includes generated data in `data/`. To regenerate it:

```bash
python generate_data.py
```

Main files:

- `data/lv_main_log.csv`: Lotka--Volterra log-state trajectory used by MF-Net.
- `data/lv_A_true_target_source.npy`: true LV interaction matrix, stored as `A[target, source]`.
- `data/lorenz96_N40.csv`: Lorenz--96 trajectory with `N=40`.
- `data/lorenz96_N40_local_mask_source_target.npy`: true Lorenz--96 local directed support, stored as `[source, target]`.

## Run

Lotka--Volterra:

```bash
python mfnet.py --config config_lv_main.json
```

Lorenz--96 `N=40`:

```bash
python mfnet.py --config config_lorenz96_N40_main.json
```

Each run writes a timestamped folder under `runs/` with `metrics.csv`, `training_log.csv`, `D_source_target.npy`, `checkpoint.pt`, and the resolved `config.json`.

## Reproducing Multi-Seed Results

Use the same config and rerun with `seed = 0, 1, 2, 3, 4`. The main hyperparameters are fully specified in:

- `config_lv_main.json`
- `config_lorenz96_N40_main.json`

Expected reference values are listed in `expected_results.json`.

## Conventions

`D_source_target.npy` uses `D[source, target]`. For LV structural comparison, compare `D.T` with `data/lv_A_true_target_source.npy`.

With pinned dependencies, fixed generated data, fixed seeds, and `deterministic=true`, forecast metrics and structural support/correlation diagnostics are expected to reproduce within about 5% relative tolerance. Exact bitwise equality is not claimed. The absolute scale of `D` can vary slightly because structural readout is scale-calibrated. For stricter cross-machine reproducibility, set `device="cpu"` and `num_threads=1`.
