#!/usr/bin/env python3
import argparse
import numpy as np
from pathlib import Path
from sklearn.utils.extmath import randomized_svd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NX, NY = 40, 40
NZ_TOTAL = 20
N_CELLS = NX * NY * NZ_TOTAL


def load_train_matrix(data_dir, max_samples, held_out_frac, seed):
    data_dir = Path(data_dir)
    p_files = sorted(data_dir.glob("sample_*_p.npy"))
    if max_samples is not None:
        p_files = p_files[:max_samples]
    rows, well_id = [], []
    for k, f in enumerate(p_files):
        p = np.load(f)
        for i in range(p.shape[0]):
            rows.append(p[i].astype(np.float32))
            well_id.append(k)
    X = np.stack(rows, axis=0)
    well_id = np.array(well_id)
    rng = np.random.default_rng(seed)
    n_wells = well_id.max() + 1
    perm = rng.permutation(n_wells)
    n_held = int(n_wells * held_out_frac)
    held = set(perm[:n_held].tolist())
    is_test = np.array([w in held for w in well_id])
    X_train = X[~is_test]
    return X_train


def spectrum(X_train, n_sv, seed):
    mean = X_train.mean(axis=0, keepdims=True)
    _, S, _ = randomized_svd(X_train - mean, n_components=n_sv, random_state=seed)
    return S / S[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir-A", required=True)
    ap.add_argument("--dir-B", required=True)
    ap.add_argument("--n-sv", type=int, default=64)
    ap.add_argument("--held-out-frac", type=float, default=0.15)
    ap.add_argument("--max-samples", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="singular_values.png")
    args = ap.parse_args()

    print("Loading Experiment A (fixed well) ...", flush=True)
    XA = load_train_matrix(args.dir_A, args.max_samples, args.held_out_frac, args.seed)
    print("Loading Experiment B (variable well) ...", flush=True)
    XB = load_train_matrix(args.dir_B, args.max_samples, args.held_out_frac, args.seed)

    print("Computing SVD (A) ...", flush=True)
    sA = spectrum(XA, args.n_sv, args.seed)
    print("Computing SVD (B) ...", flush=True)
    sB = spectrum(XB, args.n_sv, args.seed)

    np.savez("singular_values.npz", sigma_A=sA, sigma_B=sB)

    idx = np.arange(1, len(sA) + 1)
    plt.figure(figsize=(5, 4))
    plt.semilogy(idx, sA, "o-", label="Experiment A (fixed well)", markersize=4)
    plt.semilogy(idx, sB, "s-", label="Experiment B (variable well)", markersize=4)
    plt.xlabel("index i")
    plt.ylabel("normalized singular value")
    plt.title("POD singular-value decay")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    print("Saved -> " + args.out, flush=True)

    for label, s in [("A", sA), ("B", sB)]:
        print("Exp " + label +
              ": sigma_8/sigma_1=%.2e, sigma_16/sigma_1=%.2e, sigma_64/sigma_1=%.2e"
              % (s[7], s[15], s[-1]), flush=True)


if __name__ == "__main__":
    main()
