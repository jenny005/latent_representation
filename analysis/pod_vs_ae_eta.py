#!/usr/bin/env python3
"""
pod_vs_ae_eta.py — Chapter 3 representation-faithfulness comparison.

Computes eta (Definition 3.1, relative reconstruction error) for POD and for
a matched-capacity convolutional autoencoder, at several latent dimensions,
on the variable-well-location darcyFoam dataset.

Data layout expected (as produced by extract_fields.py):
    sample_XXXX_p.npy     shape (n_snapshots, n_cells)   float32, physical Pa
    sample_XXXX_meta.npy  dict with well_cx_m, well_cy_m, ...

Split is by WELL (sample id), not by snapshot: all snapshots from a given
well go entirely into train or entirely into test, so the held-out set
tests generalization to unseen well locations, not just unseen timesteps
of an already-seen well.

Usage:
    python3 pod_vs_ae_eta.py /work/jing/LDO_project/ldo_data_v2 \
        --n-dims 8 16 32 64 \
        --held-out-frac 0.15 \
        --max-samples 5001 \
        --snapshots-per-sample all \
        --epochs 60 \
        --out results.json

For a fast first pass on a login/dev node before committing a full job:
    python3 pod_vs_ae_eta.py <data_dir> --max-samples 300 --epochs 20
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.utils.extmath import randomized_svd

# ---- mesh layout constants (must match blockMeshDict / generate_samples_v2.py) ----
NX, NY = 40, 40
NZ_BOT, NZ_TOP = 12, 8
NZ_TOTAL = NZ_BOT + NZ_TOP
N_CELLS = NX * NY * NZ_TOTAL


def load_snapshot_matrix(data_dir, max_samples=None, snapshots_per_sample="all"):
    """Returns X (n_snapshots_total, N_CELLS) float32, and well_id per row
    (int array) so we can split by well, not by snapshot."""
    data_dir = Path(data_dir)
    p_files = sorted(data_dir.glob("sample_*_p.npy"))
    if max_samples is not None:
        p_files = p_files[:max_samples]

    rows = []
    well_id = []
    meta_records = []
    print(f"Loading {len(p_files)} samples ...", flush=True)
    t0 = time.time()
    for k, f in enumerate(p_files):
        p = np.load(f)  # (n_snap, N_CELLS)
        if snapshots_per_sample == "all":
            idx = np.arange(p.shape[0])
        elif snapshots_per_sample == "last":
            idx = np.array([p.shape[0] - 1])
        else:
            idx = np.array(snapshots_per_sample)
            idx = idx[idx < p.shape[0]]

        sample_id = f.name.split("_")[1]
        meta_path = f.parent / f"sample_{sample_id}_meta.npy"
        meta = np.load(meta_path, allow_pickle=True).item()
        meta_records.append(meta)

        for i in idx:
            rows.append(p[i].astype(np.float32))
            well_id.append(k)

        if (k + 1) % 500 == 0:
            print(f"  ... {k+1}/{len(p_files)} loaded, {time.time()-t0:.0f}s", flush=True)

    X = np.stack(rows, axis=0)
    well_id = np.array(well_id)
    print(f"Snapshot matrix: {X.shape}, {time.time()-t0:.0f}s", flush=True)
    return X, well_id, meta_records


def well_split(well_id, held_out_frac, seed=0):
    rng = np.random.default_rng(seed)
    n_wells = well_id.max() + 1
    perm = rng.permutation(n_wells)
    n_held = int(n_wells * held_out_frac)
    held_wells = set(perm[:n_held].tolist())
    is_test = np.array([w in held_wells for w in well_id])
    return ~is_test, is_test  # train_mask, test_mask


def eta_relative(X_true, X_recon):
    """Definition 3.1, aggregated over the full set (single number)."""
    num = np.linalg.norm(X_true - X_recon)
    den = np.linalg.norm(X_true)
    return float(num / den)


# ---------------- POD ----------------

def pod_eta(X_train, X_test, n_components_list, seed=0):
    mean = X_train.mean(axis=0, keepdims=True)
    Xc_train = X_train - mean
    Xc_test = X_test - mean

    n_max = max(n_components_list)
    print(f"Running randomized SVD (n_components={n_max}) on {X_train.shape} ...", flush=True)
    t0 = time.time()
    U, S, Vt = randomized_svd(Xc_train, n_components=n_max, random_state=seed)
    print(f"  SVD done in {time.time()-t0:.0f}s", flush=True)

    results = {}
    for n in n_components_list:
        basis = Vt[:n]  # (n, N_CELLS)
        z_test = Xc_test @ basis.T          # encode
        recon_test = z_test @ basis + mean  # decode
        results[n] = eta_relative(X_test, recon_test)
    return results


# ---------------- Autoencoder ----------------

class ConvAE(nn.Module):
    """Treats the z-direction (20 layers) as channels, convolves over (x, y).
    Lightweight design choice for a first pass; upgrade to Conv3d if the
    z-direction structure turns out to matter for the saturation-front case.
    """
    def __init__(self, latent_dim, nz=NZ_TOTAL, nxy=NX):
        super().__init__()
        self.nz, self.nxy = nz, nxy
        self.enc = nn.Sequential(
            nn.Conv2d(nz, 32, 3, stride=2, padding=1), nn.ReLU(),   # 40->20
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),   # 20->10
            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.ReLU(),   # 10->5
            nn.Flatten(),
            nn.Linear(64 * 5 * 5, latent_dim),
        )
        self.dec = nn.Sequential(
            nn.Linear(latent_dim, 64 * 5 * 5), nn.ReLU(),
            nn.Unflatten(1, (64, 5, 5)),
            nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, nz, 3, stride=2, padding=1, output_padding=1),
        )

    def forward(self, x):
        z = self.enc(x)
        return self.dec(z)


def to_grid(X):
    """(n, N_CELLS) -> (n, NZ_TOTAL, NY, NX), matching the bottom+top block
    ordering confirmed against real well-location data (i fastest, then j,
    then k; bottom block first, then top block)."""
    n = X.shape[0]
    bottom = X[:, :NX * NY * NZ_BOT].reshape(n, NZ_BOT, NY, NX)
    top = X[:, NX * NY * NZ_BOT:].reshape(n, NZ_TOP, NY, NX)
    return np.concatenate([bottom, top], axis=1)  # (n, NZ_TOTAL, NY, NX)


def from_grid(G):
    n = G.shape[0]
    bottom = G[:, :NZ_BOT].reshape(n, -1)
    top = G[:, NZ_BOT:].reshape(n, -1)
    return np.concatenate([bottom, top], axis=1)


def ae_eta(X_train, X_test, n_components_list, epochs=150, batch_size=64,
           lr=1e-3, seed=0, device=None, val_frac=0.1, patience=20,
           min_epochs=40):
    """Trains with a held-out validation slice carved out of X_train (test
    stays untouched for final eta only), a ReduceLROnPlateau scheduler, and
    best-checkpoint selection by validation loss -- so an unstable late
    epoch can't silently become the reported result. Also runs each n for a
    minimum number of epochs before patience-based early stopping kicks in,
    so a small n with a fast-plateauing loss doesn't stop before a fair
    comparison against larger n."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training AE on device={device}", flush=True)

    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True) + 1e-6
    Xn_train_full = (X_train - mean) / std
    Xn_test = (X_test - mean) / std

    rng = np.random.default_rng(seed)
    n_all = Xn_train_full.shape[0]
    perm = rng.permutation(n_all)
    n_val = max(1, int(n_all * val_frac))
    val_idx, fit_idx = perm[:n_val], perm[n_val:]

    G_fit = to_grid(Xn_train_full[fit_idx])
    G_val = to_grid(Xn_train_full[val_idx])
    G_test = to_grid(Xn_test)

    Gt_fit = torch.tensor(G_fit, dtype=torch.float32)
    Gt_val = torch.tensor(G_val, dtype=torch.float32).to(device)
    Gt_test = torch.tensor(G_test, dtype=torch.float32).to(device)

    results = {}
    diagnostics = {}
    for n in n_components_list:
        torch.manual_seed(seed)
        model = ConvAE(n).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=0.5, patience=8)
        lossf = nn.MSELoss()

        n_fit = Gt_fit.shape[0]
        best_val = float("inf")
        best_state = None
        best_epoch = -1
        epochs_since_best = 0
        val_history = []

        for ep in range(epochs):
            model.train()
            perm_ep = torch.randperm(n_fit)
            total_loss = 0.0
            for i in range(0, n_fit, batch_size):
                idx = perm_ep[i:i + batch_size]
                batch = Gt_fit[idx].to(device)
                opt.zero_grad()
                out = model(batch)
                loss = lossf(out, batch)
                loss.backward()
                opt.step()
                total_loss += loss.item() * len(idx)
            train_mse = total_loss / n_fit

            model.eval()
            with torch.no_grad():
                val_out = model(Gt_val)
                val_mse = lossf(val_out, Gt_val).item()
            val_history.append(val_mse)
            sched.step(val_mse)

            if val_mse < best_val - 1e-6:
                best_val = val_mse
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                best_epoch = ep
                epochs_since_best = 0
            else:
                epochs_since_best += 1

            if (ep + 1) % max(1, epochs // 10) == 0:
                cur_lr = opt.param_groups[0]["lr"]
                print(f"  n={n:3d}  epoch {ep+1}/{epochs}  "
                      f"train_mse={train_mse:.5f}  val_mse={val_mse:.5f}  "
                      f"lr={cur_lr:.2e}  best@{best_epoch+1}", flush=True)

            if ep + 1 >= min_epochs and epochs_since_best >= patience:
                print(f"  n={n:3d}  early stop at epoch {ep+1} "
                      f"(no val improvement in {patience} epochs)", flush=True)
                break

        model.load_state_dict(best_state)
        final_val = val_history[-1]
        flag = "OK" if best_epoch == len(val_history) - 1 or \
            abs(final_val - best_val) / best_val < 0.05 else \
            f"WARNING: best val_mse was at epoch {best_epoch+1}, " \
            f"{len(val_history) - best_epoch - 1} epochs before stopping " \
            f"(final val_mse={final_val:.5f} vs best={best_val:.5f}) -- " \
            f"using best checkpoint, not final weights"
        print(f"  n={n:3d}  {flag}", flush=True)
        diagnostics[n] = {"best_epoch": best_epoch + 1, "best_val_mse": best_val,
                           "n_epochs_run": len(val_history), "flag": flag}

        model.eval()
        with torch.no_grad():
            recon_test_n = model(Gt_test).cpu().numpy()
        recon_test = from_grid(recon_test_n) * std + mean
        results[n] = eta_relative(X_test, recon_test)
    return results, diagnostics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir")
    ap.add_argument("--n-dims", type=int, nargs="+", default=[8, 16, 32, 64])
    ap.add_argument("--held-out-frac", type=float, default=0.15)
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--snapshots-per-sample", default="all",
                     help="'all', 'last', or space-separated indices")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--min-epochs", type=int, default=40)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--num-threads", type=int, default=None,
                     help="torch CPU thread count; default lets torch pick, "
                          "but on a fully-allocated node set this explicitly "
                          "(e.g. --num-threads 64) to actually use the cores")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="pod_vs_ae_results.json")
    args = ap.parse_args()

    if args.num_threads is not None:
        torch.set_num_threads(args.num_threads)
    print(f"torch using {torch.get_num_threads()} CPU threads", flush=True)

    X, well_id, _ = load_snapshot_matrix(
        args.data_dir, max_samples=args.max_samples,
        snapshots_per_sample=args.snapshots_per_sample,
    )
    train_mask, test_mask = well_split(well_id, args.held_out_frac, seed=args.seed)
    X_train, X_test = X[train_mask], X[test_mask]
    print(f"Train snapshots: {X_train.shape[0]}  Test snapshots: {X_test.shape[0]}", flush=True)

    print("\n=== POD ===", flush=True)
    pod_results = pod_eta(X_train, X_test, args.n_dims, seed=args.seed)
    for n, e in pod_results.items():
        print(f"  n={n:3d}  eta_POD={e:.5f}", flush=True)

    print("\n=== Autoencoder ===", flush=True)
    ae_results, ae_diag = ae_eta(
        X_train, X_test, args.n_dims, epochs=args.epochs, seed=args.seed,
        val_frac=args.val_frac, patience=args.patience, min_epochs=args.min_epochs,
        batch_size=args.batch_size,
    )
    for n, e in ae_results.items():
        print(f"  n={n:3d}  eta_AE={e:.5f}", flush=True)

    print("\n=== Training diagnostics ===", flush=True)
    any_warning = False
    for n in args.n_dims:
        d = ae_diag[n]
        print(f"  n={n:3d}  best_epoch={d['best_epoch']}/{d['n_epochs_run']}  "
              f"best_val_mse={d['best_val_mse']:.5f}  {d['flag']}", flush=True)
        if d["flag"] != "OK":
            any_warning = True
    if any_warning:
        print("\n  NOTE: one or more n used a checkpoint well before the final epoch.")
        print("  This is handled automatically (best-val weights are what's evaluated),")
        print("  but if best_epoch is very early relative to n_epochs_run, consider")
        print("  more epochs, a lower --lr, or more training data for that n.")

    print("\n=== Summary (Table 3.1 candidate) ===", flush=True)
    print(f"{'n':>4}  {'eta_POD':>10}  {'eta_AE':>10}  {'gap_%':>8}")
    summary = []
    for n in args.n_dims:
        pod_e, ae_e = pod_results[n], ae_results[n]
        gap_pct = 100 * (pod_e - ae_e) / pod_e
        print(f"{n:4d}  {pod_e:10.5f}  {ae_e:10.5f}  {gap_pct:7.1f}%")
        summary.append({"n": n, "eta_pod": pod_e, "eta_ae": ae_e, "gap_pct": gap_pct})

    with open(args.out, "w") as f:
        json.dump({
            "n_train_snapshots": int(X_train.shape[0]),
            "n_test_snapshots": int(X_test.shape[0]),
            "held_out_frac": args.held_out_frac,
            "results": summary,
            "ae_diagnostics": ae_diag,
        }, f, indent=2)
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
