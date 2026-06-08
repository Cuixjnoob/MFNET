import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def generate_lv(T, n_trajectories, seed):
    rng = np.random.default_rng(seed)
    skew = np.array(
        [
            [0.00, -0.90, 0.55, -0.25, 0.15, 0.75],
            [0.90, 0.00, -0.85, 0.42, -0.18, 0.12],
            [-0.55, 0.85, 0.00, -0.95, 0.36, -0.22],
            [0.25, -0.42, 0.95, 0.00, -0.78, 0.48],
            [-0.15, 0.18, -0.36, 0.78, 0.00, -0.88],
            [-0.75, -0.12, 0.22, -0.48, 0.88, 0.00],
        ],
        dtype=np.float64,
    )
    dt = 0.035
    A = dt * (skew - 0.01 * np.eye(6, dtype=np.float64))
    x_star = np.array([1.05, 0.90, 1.15, 0.82, 1.02, 0.96], dtype=np.float64)
    r = -(A @ x_star)
    X = np.empty((n_trajectories, T, 6), dtype=np.float64)
    for q in range(n_trajectories):
        accepted = None
        for amp in (0.32, 0.24, 0.18, 0.12, 0.08):
            y = np.empty((T, 6), dtype=np.float64)
            y[0] = np.log(x_star) + rng.normal(0.0, amp, size=6)
            stable = True
            for t in range(T - 1):
                x = np.exp(y[t])
                y[t + 1] = y[t] + r + A @ x
                if not np.all(np.isfinite(y[t + 1])) or np.max(y[t + 1]) > 5 or np.min(y[t + 1]) < -7:
                    stable = False
                    break
            if stable:
                accepted = y
                break
        if accepted is None:
            raise RuntimeError("LV generator became unstable")
        X[q] = np.exp(accepted)
    return X, A, r


def l96_rhs(x, forcing):
    return (np.roll(x, -1) - np.roll(x, 2)) * np.roll(x, 1) - x + forcing


def rk4_l96(x, dt, forcing):
    k1 = l96_rhs(x, forcing)
    k2 = l96_rhs(x + 0.5 * dt * k1, forcing)
    k3 = l96_rhs(x + 0.5 * dt * k2, forcing)
    k4 = l96_rhs(x + dt * k3, forcing)
    return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def generate_lorenz96(T, N, forcing, dt, sample_stride, burn, seed):
    rng = np.random.default_rng(seed)
    x = forcing * np.ones(N, dtype=np.float64)
    x += rng.normal(0.0, 0.01, size=N)
    x[0] += 0.1
    samples = []
    total_steps = burn + T * sample_stride
    for step in range(total_steps):
        x = rk4_l96(x, dt, forcing)
        if step >= burn and (step - burn) % sample_stride == 0:
            samples.append(x.copy())
    return np.asarray(samples[:T], dtype=np.float32)


def l96_local_mask(N):
    mask = np.zeros((N, N), dtype=bool)
    for i in range(N):
        for j in ((i - 2) % N, (i - 1) % N, (i + 1) % N):
            mask[j, i] = True
    np.fill_diagonal(mask, False)
    return mask


def save_panel(path, arr, prefix):
    df = pd.DataFrame(arr, columns=[f"{prefix}{i}" for i in range(arr.shape[1])])
    df.insert(0, "t", np.arange(len(df), dtype=int))
    df.to_csv(path, index=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="data")
    p.add_argument("--lv-T", type=int, default=900)
    p.add_argument("--lv-train-T", type=int, default=260)
    p.add_argument("--lv-trajectories", type=int, default=24)
    p.add_argument("--lv-seed", type=int, default=123)
    p.add_argument("--lv-trajectory-index", type=int, default=0)
    p.add_argument("--l96-T", type=int, default=2200)
    p.add_argument("--l96-N", type=int, default=40)
    p.add_argument("--l96-forcing", type=float, default=8.0)
    p.add_argument("--l96-dt", type=float, default=0.01)
    p.add_argument("--l96-sample-stride", type=int, default=5)
    p.add_argument("--l96-burn", type=int, default=1000)
    p.add_argument("--l96-seed", type=int, default=0)
    args = p.parse_args()
    out = Path(args.out_dir)
    if not out.is_absolute():
        out = Path(__file__).resolve().parent / out
    out.mkdir(parents=True, exist_ok=True)
    X, A, r = generate_lv(args.lv_T, args.lv_trajectories, args.lv_seed)
    x = X[int(args.lv_trajectory_index), : args.lv_train_T]
    z_lv = np.log(np.clip(x, 1e-8, np.inf) + 1e-8)
    save_panel(out / "lv_main_log.csv", z_lv.astype(np.float32), "z")
    save_panel(out / "lv_main_abundance.csv", x.astype(np.float32), "x")
    np.save(out / "lv_A_true_target_source.npy", A.astype(np.float32))
    np.save(out / "lv_r_true.npy", r.astype(np.float32))
    l96 = generate_lorenz96(args.l96_T, args.l96_N, args.l96_forcing, args.l96_dt, args.l96_sample_stride, args.l96_burn, args.l96_seed)
    save_panel(out / "lorenz96_N40.csv", l96, "x")
    np.save(out / "lorenz96_N40_local_mask_source_target.npy", l96_local_mask(args.l96_N).astype(bool))
    meta = {
        "lv": {
            "generator": "discrete generalized Lotka-Volterra in log state",
            "T_generated": args.lv_T,
            "T_used": args.lv_train_T,
            "n_trajectories": args.lv_trajectories,
            "seed": args.lv_seed,
            "trajectory_index": args.lv_trajectory_index,
            "dt_absorbed_in_A": 0.035,
            "csv_for_training": "lv_main_log.csv",
            "state_used_by_model": "z=log(x+1e-8)",
        },
        "lorenz96_N40": {
            "equation": "dx_i/dt=(x_{i+1}-x_{i-2})x_{i-1}-x_i+F",
            "N": args.l96_N,
            "T": args.l96_T,
            "forcing": args.l96_forcing,
            "dt": args.l96_dt,
            "sample_stride": args.l96_sample_stride,
            "burn": args.l96_burn,
            "seed": args.l96_seed,
            "csv_for_training": "lorenz96_N40.csv",
        },
    }
    (out / "generation_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(str(out.resolve()))


if __name__ == "__main__":
    main()
