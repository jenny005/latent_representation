#!/usr/bin/env python3
"""
extract_fields_v3.py — darcyFoam v3 ensemble extractor

Changes from the v2 extractor:
  - N_SNAPS 10 -> 100, spacing 50 days -> 5 days (matches
    generate_samples_v3.py's controlDict: 5000 steps, writeInterval=50
    steps -> snapshots every 432000 s / 5 days)
  - Added ct field extraction (constant/ct) -- new in v3, physically
    coupled to phi. Chapter 4/5 need it available alongside K/phi.
  - p unit conversion is now an explicit --units flag instead of a
    silent hardcoded RHO_OIL multiply. Default is 'kinematic' (raw
    solver units, m2/s2) -- that's what the PDE actually solves and
    what ct's dimensions are defined relative to in createFields.H.
    Pass --units pa if you want the old p*RHO_OIL Pa-converted output
    for display/interpretation purposes.
"""
import argparse, json
import numpy as np
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--case",   required=True)
parser.add_argument("--sample", type=int, required=True)
parser.add_argument("--out",    default="ldo_data_v3")
parser.add_argument("--units",  choices=["kinematic", "pa"], default="kinematic",
                     help="kinematic = raw solver p [m2/s2] (default); "
                          "pa = p*RHO_OIL for display/interpretation")
args = parser.parse_args()

CASE_DIR    = Path(args.case)
SID         = args.sample
OUT_DIR     = Path(args.out)
OUT_DIR.mkdir(parents=True, exist_ok=True)

NX, NY, NZ  = 40, 40, 20
N_CELLS     = NX * NY * NZ      # 32000
RHO_OIL     = 850.0             # [kg/m3], only used if --units pa
DARCY_TO_M2 = 9.869e-13         # 1 Darcy in m2

# v3 timing: 5000 steps, writeInterval=50 steps -> 100 snapshots,
# 5 days (432000 s) apart, out to 500 days (43200000 s)
N_SNAPS      = 100
SNAP_SPACING_S = 432000
SNAP_TIMES_S = [SNAP_SPACING_S * k for k in range(1, N_SNAPS + 1)]

K_MIN_D = 1e-4    # 0.1 mD minimum
K_MAX_D = 100.0   # 100 D maximum (well zone is 10 D)
CT_MIN  = 0.7e-9  # matches generate_samples_v3.py CT_MIN/CT_MAX
CT_MAX  = 1.3e-9


# ── OpenFOAM field readers (unchanged from v2) ────────────────────────────

def _detect_format(path):
    with open(path, 'rb') as f:
        h = f.read(2048).decode('utf-8', errors='ignore')
    return 'binary' if ('format      binary' in h or
                        'format binary' in h) else 'ascii'


def _parse_binary_block(raw, marker):
    idx = raw.find(marker)
    if idx == -1:
        return None, None
    rest  = raw[idx + len(marker):]
    nl1   = rest.find(b'\n')
    nl2   = rest.find(b'\n', nl1 + 1)
    try:
        n = int(rest[nl1:nl2].strip())
    except ValueError:
        return None, None
    paren = rest.find(b'(', nl2)
    if paren == -1:
        return None, None
    return n, rest[paren + 1:]


def read_foam_scalar(path):
    if _detect_format(path) == 'binary':
        with open(path, 'rb') as f:
            raw = f.read()
        n, data = _parse_binary_block(raw, b'nonuniform List<scalar>')
        if data is not None and n == N_CELLS:
            return np.frombuffer(data[:n * 8], dtype='<f8').astype(np.float32)

    values   = []
    in_list  = False
    in_paren = False
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if 'nonuniform List<scalar>' in line:
                in_list = True
                continue
            if in_list:
                if line == '(' or line == '':
                    in_paren = True
                    continue
                if line == ');' or line == ')':
                    break
                if in_paren:
                    try:
                        values.append(float(line))
                    except ValueError:
                        continue
    return np.array(values, dtype=np.float32)


def read_foam_vector(path):
    if _detect_format(path) == 'binary':
        with open(path, 'rb') as f:
            raw = f.read()
        n, data = _parse_binary_block(raw, b'nonuniform List<vector>')
        if data is not None and n == N_CELLS:
            return np.frombuffer(data[:n * 3 * 8],
                                 dtype='<f8').reshape(n, 3).astype(np.float32)

    values   = []
    in_list  = False
    in_paren = False
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if 'nonuniform List<vector>' in line:
                in_list = True
                continue
            if in_list:
                if line == '(' or line == '':
                    in_paren = True
                    continue
                if line == ');' or line == ')':
                    break
                if in_paren and line.startswith('(') and line.endswith(')'):
                    try:
                        values.append([float(x) for x in line[1:-1].split()])
                    except ValueError:
                        continue
    return np.array(values, dtype=np.float32)


def find_time_dir(case_dir, t_s):
    candidates = [
        str(t_s),
        f"{float(t_s):.1f}",
        f"{float(t_s):g}",
        f"{float(t_s):.3e}",
        f"{float(t_s):.2e}",
        f"{float(t_s):.1e}",
    ]
    for name in candidates:
        d = case_dir / name
        if d.is_dir():
            return d
    try:
        dirs = [d for d in case_dir.iterdir()
                if d.is_dir() and d.name[0].isdigit()]
        for d in dirs:
            try:
                if abs(float(d.name) - t_s) < 1.0:
                    return d
            except ValueError:
                pass
    except Exception:
        pass
    return None


def load_static(path, field_name, valid_min=None, valid_max=None):
    if not path.exists():
        print(f"  WARNING: {field_name} not found at {path}")
        return np.full(N_CELLS, np.nan, dtype=np.float32)

    arr = read_foam_scalar(path)

    if arr.shape[0] == N_CELLS + 1:
        arr = arr[:N_CELLS]

    if arr.shape[0] != N_CELLS:
        print(f"  ERROR: {field_name} shape {arr.shape} != {N_CELLS} — returning NaN")
        return np.full(N_CELLS, np.nan, dtype=np.float32)

    if valid_min is not None and valid_max is not None:
        n_bad = int(((arr < valid_min) | (arr > valid_max)).sum())
        if n_bad > 0:
            bad_vals = arr[(arr < valid_min) | (arr > valid_max)]
            print(f"  WARNING: {field_name} has {n_bad} out-of-range values "
                  f"(min={bad_vals.min():.4g}, max={bad_vals.max():.4g}) — clamping")
            arr = np.clip(arr, valid_min, valid_max)

    return arr


# ── Load pressure snapshots ───────────────────────────────────────────────
p_snaps, U_snaps, found_times = [], [], []

for t_s in SNAP_TIMES_S:
    td = find_time_dir(CASE_DIR, t_s)
    if td is None:
        print(f"  WARNING: missing time dir for t={t_s//86400}d")
        continue
    pf = td / "p"
    uf = td / "U"
    if not pf.exists():
        print(f"  WARNING: no p field at t={t_s//86400}d")
        continue
    p_raw = read_foam_scalar(pf)   # kinematic [m2/s2] as solved
    p_out = p_raw * RHO_OIL if args.units == "pa" else p_raw
    if p_out.shape[0] != N_CELLS:
        print(f"  WARNING: p shape {p_out.shape} at t={t_s//86400}d — skipping")
        continue
    if uf.exists():
        U_arr = read_foam_vector(uf)
        if U_arr.shape[0] != N_CELLS:
            U_arr = np.zeros((N_CELLS, 3), dtype=np.float32)
    else:
        U_arr = np.zeros((N_CELLS, 3), dtype=np.float32)
    p_snaps.append(p_out)
    U_snaps.append(U_arr)
    found_times.append(t_s)

if not p_snaps:
    print(f"ERROR: no valid snapshots found in {CASE_DIR}")
    exit(1)

p_array = np.stack(p_snaps)   # (N_snaps, N_CELLS)
U_array = np.stack(U_snaps)   # (N_snaps, N_CELLS, 3) [m/s]
unit_label = "Pa" if args.units == "pa" else "kinematic (m2/s2)"
print(f"  p ({unit_label}): {p_array.shape}  "
      f"{p_array.min():.4g}–{p_array.max():.4g}  "
      f"({len(found_times)}/{N_SNAPS} snaps)")

# ── Load static fields ────────────────────────────────────────────────────
K_darcy = load_static(CASE_DIR / "constant" / "K",   "K",
                      valid_min=K_MIN_D, valid_max=K_MAX_D)
K_m2    = K_darcy * DARCY_TO_M2

phi_arr = load_static(CASE_DIR / "constant" / "phi", "phi",
                      valid_min=0.01, valid_max=1.0)

ct_arr  = load_static(CASE_DIR / "constant" / "ct",  "ct",
                      valid_min=CT_MIN * 0.9, valid_max=CT_MAX * 1.1)

print(f"  K:   {K_darcy.shape}  "
      f"[{K_darcy[K_darcy < 5].min():.4f}, {K_darcy[K_darcy < 5].max():.4f}] D  "
      f"(excl. well)  well cells: {(K_darcy >= 5).sum()}")
print(f"  phi: {phi_arr.shape}  "
      f"[{phi_arr.min():.3f}, {phi_arr.max():.3f}]")
print(f"  ct:  {ct_arr.shape}  "
      f"[{ct_arr.min():.3e}, {ct_arr.max():.3e}]")

# ── Load sample metadata ──────────────────────────────────────────────────
meta = {}
for p in [CASE_DIR / "sample_meta.json",
          CASE_DIR.parent / "metadata.json"]:
    if p.exists():
        with open(p) as f:
            d = json.load(f)
        meta = d[SID] if isinstance(d, list) else d
        break

# ── Save outputs ──────────────────────────────────────────────────────────
sid = f"sample_{SID:04d}"
np.save(OUT_DIR / f"{sid}_p.npy",    p_array)   # (100, N_CELLS), units per --units
np.save(OUT_DIR / f"{sid}_U.npy",    U_array)   # (100, N_CELLS, 3) [m/s]
np.save(OUT_DIR / f"{sid}_K.npy",    K_darcy)   # (N_CELLS,) [Darcy]
np.save(OUT_DIR / f"{sid}_K_m2.npy", K_m2)      # (N_CELLS,) [m2]
np.save(OUT_DIR / f"{sid}_phi.npy",  phi_arr)   # (N_CELLS,) [-]
np.save(OUT_DIR / f"{sid}_ct.npy",   ct_arr)    # (N_CELLS,) [1/Pa]
np.save(OUT_DIR / f"{sid}_meta.npy", {
    "sample_id":      SID,
    "snap_times_s":   found_times,
    "snap_times_d":   [t // 86400 for t in found_times],
    "n_snapshots":    len(p_snaps),
    "p_units":        args.units,
    **meta
})
print(f"  Saved -> {OUT_DIR}/  [{sid}_p/U/K/K_m2/phi/ct/meta.npy]")
