import argparse
import json
import math
import random
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def set_deterministic(enabled, num_threads=None):
    if num_threads is not None:
        torch.set_num_threads(int(num_threads))
    if enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)


def mlp(d_in, hidden, d_out, depth=1, zero_last=False):
    layers = []
    d = d_in
    for _ in range(depth):
        layers += [nn.Linear(d, hidden), nn.SiLU()]
        d = hidden
    layers.append(nn.Linear(d, d_out))
    net = nn.Sequential(*layers)
    if zero_last:
        nn.init.zeros_(net[-1].weight)
        nn.init.zeros_(net[-1].bias)
    return net


class MFNet(nn.Module):
    def __init__(self, n, L, u_dim, chi_dim, field_modes, hidden, token_hidden, source_mode, integrator, dt, y_min, y_max, seed):
        super().__init__()
        if integrator not in {"euler", "heun"}:
            raise ValueError("integrator must be euler or heun")
        if source_mode not in {"identity", "positive_level", "lv_level"}:
            raise ValueError("source_mode must be identity, positive_level, or lv_level")
        gen = torch.Generator().manual_seed(seed)
        self.n = n
        self.L = L
        self.u_dim = u_dim
        self.eta_dim = 1 + u_dim
        self.chi_dim = chi_dim
        self.field_modes = field_modes
        self.source_mode = source_mode
        self.integrator = integrator
        self.dt = float(dt)
        self.y_min = float(y_min)
        self.y_max = float(y_max)
        self.D_param = nn.Parameter(torch.randn(n, n, generator=gen) * 1e-3)
        self.r = nn.Parameter(torch.zeros(n))
        self.a_z = nn.Parameter(torch.zeros(n))
        self.raw_lambda_eta = nn.Parameter(torch.full((n,), -0.1))
        self.raw_lambda_chi = nn.Parameter(torch.full((chi_dim,), -0.1))
        self.u0 = nn.Parameter(torch.randn(n, u_dim, generator=gen) * 1e-2)
        self.eta0 = nn.Parameter(torch.zeros(n, self.eta_dim))
        self.register_buffer("chi0", torch.zeros(n, chi_dim))
        mech_dim = 1 + u_dim + self.eta_dim + chi_dim
        self.source_emitter = mlp(mech_dim, hidden, field_modes)
        self.source_bias = nn.Parameter(torch.zeros(field_modes))
        self.reception = mlp(mech_dim, hidden, self.eta_dim * field_modes)
        self.R_const = nn.Parameter(torch.randn(self.eta_dim, field_modes, generator=gen) * 1e-2)
        self.u_local = mlp(1 + u_dim + chi_dim, hidden, u_dim, zero_last=True)
        self.token_dim = 1 + u_dim + self.eta_dim + chi_dim + 3 + field_modes + self.eta_dim + field_modes + 3
        self.token_encoder = mlp(self.token_dim, token_hidden, token_hidden)
        self.tape_gru = nn.GRU(token_hidden, token_hidden, batch_first=True)
        self.u_head = mlp(token_hidden, token_hidden, u_dim, zero_last=True)
        self.eta_head = mlp(token_hidden, token_hidden, self.eta_dim, zero_last=True)
        self.chi_head = mlp(token_hidden, token_hidden, chi_dim, zero_last=True)

    def D(self):
        mask = 1.0 - torch.eye(self.n, device=self.D_param.device, dtype=self.D_param.dtype)
        return self.D_param * mask

    def source_strength(self, z):
        if self.source_mode in {"positive_level", "lv_level"}:
            return torch.clamp(torch.exp(torch.clamp(z, -12.0, 8.0)) - 1e-3, 0.0, 200.0)
        return torch.clamp(z, -20.0, 20.0)

    def latent0(self, z):
        b = z.shape[0]
        return self.u0[None].expand(b, -1, -1), self.eta0[None].expand(b, -1, -1), self.chi0[None].expand(b, -1, -1)

    def source_emission(self, z, u, eta, chi):
        b, n = z.shape
        x = torch.cat([z[..., None], u, eta, chi], dim=-1)
        raw = self.source_emitter(x.reshape(b * n, -1)).reshape(b, n, self.field_modes)
        return F.layer_norm(raw + self.source_bias.reshape(1, 1, -1), (self.field_modes,), eps=1e-5)

    def reception_field(self, z, u, eta, chi):
        b, n = z.shape
        x = torch.cat([z[..., None], u, eta, chi], dim=-1)
        raw = self.reception(x.reshape(b * n, -1)).reshape(b, n, self.eta_dim, self.field_modes)
        return 0.1 * raw + self.R_const.reshape(1, 1, self.eta_dim, self.field_modes)

    def vector_field(self, z, u, eta, chi, ablation=None, d_perm=None):
        b, n = z.shape
        z = torch.clamp(z, self.y_min, self.y_max)
        D = self.D()
        if ablation == "shuffle_D":
            off = ~torch.eye(n, dtype=torch.bool, device=D.device)
            vals = D[off]
            D = torch.zeros_like(D)
            D[off] = vals[d_perm]
        if ablation == "zero_D":
            D = torch.zeros_like(D)
        J = D.reshape(1, n, n) * self.source_strength(z)[:, :, None]
        incoming_J = J.sum(dim=1)
        outgoing_J = J.sum(dim=2)
        emission = self.source_emission(z, u, eta, chi)
        h_field = torch.einsum("bsn,bsk->bnk", J, emission)
        R = self.reception_field(z, u, eta, chi)
        G = torch.einsum("bnck,bnk->bnc", R, h_field)
        z_dot = self.r.reshape(1, n) + self.a_z.reshape(1, n) * z + incoming_J + eta[..., 0]
        u_dot = self.u_local(torch.cat([z[..., None], u, chi], dim=-1).reshape(b * n, -1)).reshape(b, n, self.u_dim) + eta[..., 1:]
        lambda_eta = F.softplus(self.raw_lambda_eta).reshape(1, n, 1) + 1e-3
        lambda_chi = F.softplus(self.raw_lambda_chi).reshape(1, 1, self.chi_dim) + 1e-3
        eta_dot = lambda_eta * (G - eta)
        chi_dot = -lambda_chi * chi
        return {
            "J": J,
            "incoming_J": incoming_J,
            "outgoing_J": outgoing_J,
            "emission": emission,
            "h_field": h_field,
            "R": R,
            "G": G,
            "z_dot": z_dot,
            "u_dot": u_dot,
            "eta_dot": eta_dot,
            "chi_dot": chi_dot,
        }

    def step_with(self, z, u, eta, chi, vf, scale):
        z2 = torch.clamp(z + scale * self.dt * vf["z_dot"], self.y_min, self.y_max)
        return z2, u + scale * self.dt * vf["u_dot"], eta + scale * self.dt * vf["eta_dot"], chi + scale * self.dt * vf["chi_dot"]

    def step(self, z, u, eta, chi, ablation=None, d_perm=None):
        k1 = self.vector_field(z, u, eta, chi, ablation, d_perm)
        if self.integrator == "euler":
            z2, u2, eta2, chi2 = self.step_with(z, u, eta, chi, k1, 1.0)
            used = k1
        else:
            zt, ut, etat, chit = self.step_with(z, u, eta, chi, k1, 1.0)
            k2 = self.vector_field(zt, ut, etat, chit, ablation, d_perm)
            z2 = torch.clamp(z + 0.5 * self.dt * (k1["z_dot"] + k2["z_dot"]), self.y_min, self.y_max)
            u2 = u + 0.5 * self.dt * (k1["u_dot"] + k2["u_dot"])
            eta2 = eta + 0.5 * self.dt * (k1["eta_dot"] + k2["eta_dot"])
            chi2 = chi + 0.5 * self.dt * (k1["chi_dot"] + k2["chi_dot"])
            used = k2
        return {**used, "z_next": z2, "u_next": u2, "eta_next": eta2, "chi_next": chi2}

    def build_tape(self, window):
        b, L, n = window.shape
        u, eta, chi = self.latent0(window[:, 0])
        prev_obs = window[:, 0]
        prev_prior = window[:, 0]
        tokens = []
        for k in range(L):
            z = window[:, k]
            out = self.step(z, u, eta, chi)
            prior_delta = out["z_next"] - z
            past_delta = torch.zeros_like(z) if k == 0 else z - prev_obs
            prior_error = torch.zeros_like(z) if k == 0 else z - prev_prior
            token = torch.cat(
                [
                    z[..., None],
                    u,
                    eta,
                    chi,
                    out["incoming_J"][..., None],
                    out["J"].abs().sum(dim=1)[..., None],
                    out["outgoing_J"][..., None],
                    out["h_field"],
                    out["G"],
                    out["emission"],
                    prior_delta[..., None],
                    past_delta[..., None],
                    prior_error[..., None],
                ],
                dim=-1,
            )
            tokens.append(token)
            prev_obs = z
            prev_prior = out["z_next"]
            u, eta, chi = out["u_next"], out["eta_next"], out["chi_next"]
        return torch.stack(tokens, dim=1)

    def encode(self, window):
        b, L, n = window.shape
        tape = self.build_tape(window)
        seq = tape.permute(0, 2, 1, 3).reshape(b * n, L, self.token_dim)
        emb = self.token_encoder(seq.reshape(b * n * L, self.token_dim)).reshape(b * n, L, -1)
        _, h_last = self.tape_gru(emb)
        h = h_last[-1].reshape(b, n, -1)
        u0, eta0, chi0 = self.latent0(window[:, -1])
        u = u0 + self.u_head(h.reshape(b * n, -1)).reshape(b, n, self.u_dim)
        eta = eta0 + self.eta_head(h.reshape(b * n, -1)).reshape(b, n, self.eta_dim)
        chi = chi0 + self.chi_head(h.reshape(b * n, -1)).reshape(b, n, self.chi_dim)
        return u, eta, chi

    def rollout(self, z0, horizons, window, ablation=None):
        u, eta, chi = self.encode(window)
        u0, eta0, chi0 = u, eta, chi
        z = z0
        preds = {}
        first = None
        d_perm = None
        if ablation == "shuffle_D":
            d_perm = torch.randperm(self.n * (self.n - 1), device=z0.device)
        for h in range(1, max(horizons) + 1):
            out = self.step(z, u, eta, chi, ablation, d_perm)
            z, u, eta, chi = out["z_next"], out["u_next"], out["eta_next"], out["chi_next"]
            if first is None:
                first = {**out, "u0": u0, "eta0": eta0, "chi0": chi0}
            if h in horizons:
                preds[h] = z
        return {"preds": preds, **first}


def read_csv(cfg):
    df = pd.read_csv(cfg["csv_path"])
    if cfg.get("max_rows") is not None:
        df = df.iloc[: int(cfg["max_rows"])]
    cols = cfg.get("value_columns") or []
    if not cols:
        drop = {cfg.get("time_column")} if cfg.get("time_column") else set()
        cols = [c for c in df.columns if c not in drop and pd.api.types.is_numeric_dtype(df[c])]
    x = df[cols].apply(pd.to_numeric, errors="coerce")
    if cfg.get("fill_missing", True):
        x = x.interpolate(limit_direction="both").ffill().bfill()
    x = x.to_numpy(dtype=np.float32)
    if len(cols) < 2:
        raise ValueError("at least two value columns are required")
    if not np.isfinite(x).all():
        raise ValueError("non-finite values remain after preprocessing")
    return x, list(cols)


def make_data(cfg):
    raw, cols = read_csv(cfg)
    T = raw.shape[0]
    train_end = int(cfg["train_frac"] * T)
    val_end = int((cfg["train_frac"] + cfg["val_frac"]) * T)
    if cfg.get("standardize", True):
        mu = raw[:train_end].mean(axis=0, keepdims=True)
        sd = raw[:train_end].std(axis=0, keepdims=True) + 1e-6
        z = ((raw - mu) / sd).astype(np.float32)
    else:
        mu = np.zeros((1, raw.shape[1]), dtype=np.float32)
        sd = np.ones((1, raw.shape[1]), dtype=np.float32)
        z = raw.astype(np.float32)
    L = int(cfg["L"])
    train_h = sorted({int(h) for h in cfg["horizons"] if int(h) <= int(cfg["H"])})
    report_h = sorted({int(h) for h in cfg.get("report_horizons", train_h) if int(h) <= int(cfg["H"])})
    rollout_h = sorted(set(train_h) | set(report_h))
    if not train_h or not report_h:
        raise ValueError("horizons and report_horizons cannot be empty")
    starts = np.arange(L - 1, T - max(rollout_h), dtype=np.int64)
    train_idx = starts[starts < train_end]
    val_idx = starts[(starts >= train_end) & (starts < val_end)]
    test_idx = starts[starts >= val_end]
    if len(train_idx) == 0 or len(val_idx) == 0 or len(test_idx) == 0:
        raise ValueError("empty train, validation, or test set; reduce L/H or use more rows")
    pad = np.pad(z, ((L - 1, 0), (0, 0)), mode="edge")
    windows = np.empty((T, L, z.shape[1]), dtype=np.float32)
    for t in range(T):
        windows[t] = pad[t : t + L]
    return {
        "z": z,
        "mu": mu.astype(np.float32),
        "sd": sd.astype(np.float32),
        "columns": cols,
        "windows": windows,
        "train_h": train_h,
        "report_h": report_h,
        "rollout_h": rollout_h,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
        "train_end": train_end,
        "val_end": val_end,
    }


def horizon_weights(cfg, horizons):
    raw = {int(k): float(v) for k, v in dict(cfg.get("horizon_weights", {})).items()}
    return {h: raw.get(h, 1.0 / math.sqrt(h)) for h in horizons}


def pred_loss(pred, target, cfg):
    if str(cfg.get("loss", "huber")).lower() in {"huber", "smooth_l1", "smoothl1"}:
        return F.smooth_l1_loss(pred, target)
    return F.mse_loss(pred, target)


def loss_fn(model, idx, z, windows, data, cfg):
    out = model.rollout(z[idx], data["rollout_h"], windows[idx])
    loss = torch.zeros((), device=z.device)
    for h, w in horizon_weights(cfg, data["train_h"]).items():
        loss = loss + w * pred_loss(out["preds"][h], z[idx + h], cfg)
    D_l2 = model.D().pow(2).mean()
    chi_l2 = out["chi0"].pow(2).mean()
    eta_l2 = 0.5 * (out["eta0"].pow(2).mean() + out["eta_next"].pow(2).mean())
    S_reg = out["emission"].mean().pow(2) + (torch.sqrt(out["emission"].square().mean() + 1e-8) - 1.0).pow(2)
    R_l2 = out["R"].pow(2).mean()
    total = loss + cfg["lambda_D"] * D_l2 + cfg["lambda_chi"] * chi_l2 + cfg["lambda_eta"] * eta_l2 + cfg["lambda_S"] * S_reg + cfg["lambda_R"] * R_l2
    scalars = {
        "loss": float(total.detach().cpu()),
        "pred_loss": float(loss.detach().cpu()),
        "mean_abs_D": float(model.D().abs().mean().detach().cpu()),
        "mean_abs_J": float(out["incoming_J"].abs().mean().detach().cpu()),
        "mean_abs_G": float(out["G"].abs().mean().detach().cpu()),
    }
    return total, scalars


def metric(pred, target):
    p = pred.reshape(-1).astype(np.float64)
    t = target.reshape(-1).astype(np.float64)
    err = p - t
    denom = float(np.sum((t - t.mean()) ** 2)) + 1e-12
    corr = float(np.corrcoef(p, t)[0, 1]) if p.size > 2 and np.std(p) > 0 and np.std(t) > 0 else float("nan")
    return {
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mae": float(np.mean(np.abs(err))),
        "r2": float(1.0 - np.sum(err ** 2) / denom),
        "corr": corr,
    }


@torch.no_grad()
def predict(model, data, device, cfg, ablation=None):
    z = torch.tensor(data["z"], dtype=torch.float32, device=device)
    windows = torch.tensor(data["windows"], dtype=torch.float32, device=device)
    preds = {h: [] for h in data["report_h"]}
    for s in range(0, len(data["test_idx"]), int(cfg["eval_batch_size"])):
        idx = torch.tensor(data["test_idx"][s : s + int(cfg["eval_batch_size"])], dtype=torch.long, device=device)
        out = model.rollout(z[idx], data["rollout_h"], windows[idx], ablation)
        for h in data["report_h"]:
            preds[h].append(out["preds"][h].detach().cpu().numpy())
    return {h: np.concatenate(v, axis=0) for h, v in preds.items()}


def evaluate(model, data, device, cfg):
    rows = []
    z = data["z"]
    starts = data["test_idx"]
    for name, ablation in [("MFNet", None), ("zero-D", "zero_D"), ("D-shuffle", "shuffle_D")]:
        preds = predict(model, data, device, cfg, ablation)
        for h in data["report_h"]:
            rows.append({"model": name, "horizon": h, **metric(preds[h], z[starts + h])})
    for h in data["report_h"]:
        rows.append({"model": "Persistence", "horizon": h, **metric(z[starts], z[starts + h])})
    return pd.DataFrame(rows)


def train(cfg):
    set_seed(int(cfg["seed"]))
    set_deterministic(bool(cfg.get("deterministic", True)), cfg.get("num_threads"))
    device_name = cfg["device"] if cfg["device"] != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    data = make_data(cfg)
    model = MFNet(
        data["z"].shape[1],
        int(cfg["L"]),
        int(cfg["u_dim"]),
        int(cfg["chi_dim"]),
        int(cfg["field_modes"]),
        int(cfg["hidden"]),
        int(cfg["token_hidden"]),
        str(cfg["source_mode"]),
        str(cfg["integrator"]),
        float(cfg.get("dt", 1.0)),
        float(cfg.get("y_min", -8.0)),
        float(cfg.get("y_max", 8.0)),
        int(cfg["seed"]),
    ).to(device)
    z = torch.tensor(data["z"], dtype=torch.float32, device=device)
    windows = torch.tensor(data["windows"], dtype=torch.float32, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"]))
    rng = np.random.default_rng(int(cfg["seed"]) + 17)
    best_loss = float("inf")
    best_state = None
    logs = []
    for epoch in range(int(cfg["epochs"])):
        model.train()
        step_rows = []
        for _ in range(int(cfg["steps_per_epoch"])):
            batch = rng.choice(data["train_idx"], size=min(int(cfg["batch_size"]), len(data["train_idx"])), replace=False)
            idx = torch.tensor(batch, dtype=torch.long, device=device)
            loss, row = loss_fn(model, idx, z, windows, data, cfg)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["grad_clip"]))
            opt.step()
            step_rows.append(row)
        if epoch % int(cfg["log_every"]) == 0 or epoch == int(cfg["epochs"]) - 1:
            model.eval()
            vals = []
            with torch.no_grad():
                for s in range(0, len(data["val_idx"]), int(cfg["eval_batch_size"])):
                    idx = torch.tensor(data["val_idx"][s : s + int(cfg["eval_batch_size"])], dtype=torch.long, device=device)
                    v, _ = loss_fn(model, idx, z, windows, data, cfg)
                    vals.append(float(v.detach().cpu()))
            log = {"epoch": epoch, "val_loss": float(np.mean(vals))}
            for k in step_rows[0]:
                log[k] = float(np.mean([r[k] for r in step_rows]))
            if log["val_loss"] < best_loss:
                best_loss = log["val_loss"]
                best_state = deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
            logs.append(log)
            print(json.dumps(log), flush=True)
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    out = Path(cfg["output_dir"]) / datetime.now().strftime("%Y%m%d_%H%M%S_mfnet")
    out.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(logs).to_csv(out / "training_log.csv", index=False)
    evaluate(model, data, device, cfg).to_csv(out / "metrics.csv", index=False)
    np.save(out / "D_source_target.npy", model.D().detach().cpu().numpy())
    np.save(out / "train_mu.npy", data["mu"])
    np.save(out / "train_sd.npy", data["sd"])
    torch.save({"model_state": model.state_dict(), "config": cfg}, out / "checkpoint.pt")
    saved = dict(cfg)
    saved["resolved_value_columns"] = data["columns"]
    saved["train_horizons"] = data["train_h"]
    saved["report_horizons"] = data["report_h"]
    saved["train_end"] = int(data["train_end"])
    saved["val_end"] = int(data["val_end"])
    saved["device_used"] = str(device)
    (out / "config.json").write_text(json.dumps(saved, indent=2), encoding="utf-8")
    (out / "value_columns.json").write_text(json.dumps(data["columns"], indent=2), encoding="utf-8")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    args = parser.parse_args()
    cfg_path = Path(args.config)
    cfg = json.loads(cfg_path.read_text(encoding="utf-8-sig"))
    base = cfg_path.parent
    for key in ("csv_path", "output_dir"):
        if key in cfg and cfg[key] is not None:
            p = Path(str(cfg[key]))
            if not p.is_absolute():
                cfg[key] = str((base / p).resolve())
    print(str(train(cfg).resolve()))


if __name__ == "__main__":
    main()
