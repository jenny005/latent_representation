#!/usr/bin/env python3
"""
generate_samples_v3_reconfirm.py — darcyFoam VARIABLE-WELL ensemble,
reconfirmation run on the CURRENT solver binary (Ch.3 main experiment)

Derived from generate_samples_v3_fixedwell.py -- restores random per-sample
well location (as in the original ldo_data_v3 generator), while keeping
100% of the rest of that script's code identical: the event mechanism stays
neutralized to a no-op (rate_multiplier=1.0), and reservoirProperties is
still written via the Function1-table code path. This means ldo_data_v3
(this script's output) and ldo_data_v3_fixedwell share the exact same
compiled darcyFoam binary and code path -- the only difference between the
two Table 3.1 columns is genuinely just well location, not an incidental
solver-version difference.

Derived from generate_samples_v3.py (the variable-well/event generator).
Two changes, both deliberate:
  1. Well location (cx, cy) is FIXED at the domain center for every sample,
     not randomized. K/phi/ct still vary per sample -- amplitude-type
     variation only, no geometric/translation variation.
  2. The event mechanism is neutralized (rate_multiplier forced to 1.0,
     wellRate_before == wellRate_after) -- this run has nothing to do with
     Chapter 4's event experiment; it's pure Chapter 3 representation
     faithfulness under smooth, constant-rate depletion.

Purpose: direct test of Chapter 3's mechanism claim (Kolmogorov n-width for
a translated-function family explains the POD/AE gap). If that mechanism
is right, removing well-location translation here should collapse most of
the variable-well gap (88.6% at n=8 down to 50.8% at n=64) toward parity,
since there is no longer a shifted-manifold problem for a linear method to
struggle with.

Single ensemble, used for both:
  - Chapter 4 (Dynamical Faithfulness): needs time-resolved trajectories
    with heterogeneous K/phi/ct so latent dynamics models are genuinely
    stressed by multiple coupled diffusion timescales.
  - Chapter 5 (Decoder Transport Faithfulness): needs a decoder trained
    across the full geometric + physical diversity of this ensemble so
    it develops real curvature worth doing Jacobian analysis on.

Per sample, independently:
  - well location (cx, cy)      -- random, margin-constrained
  - K field                     -- lognormal correlated noise, by layer
  - phi field                   -- correlated noise, by layer
  - ct field                    -- NOT independent noise. Physically
                                    coupled to phi: ct = ct(phi), linear
                                    map from [phi_min, phi_max] to
                                    [ct_min, ct_max].
  - EVENT (Ch.4 Sec.4.6):       -- well-rate step, independently randomized
                                    per sample: event time in
                                    [EVENT_TIME_MIN_DAYS, EVENT_TIME_MAX_DAYS],
                                    post-event rate = WELL_RATE * multiplier,
                                    multiplier in [RATE_MULT_MIN, RATE_MULT_MAX].
                                    Requires darcyFoam compiled against the
                                    Function1<scalar>-based createFields.H/
                                    darcyFoam.C (wellRate is no longer a plain
                                    scalar) -- recompile before regenerating.

WELL_R (wellbore radius) and WELL_K_D (well-zone permeability) are FIXED
across the ensemble -- only location, background rock properties, and the
event schedule vary.

Each case runs 5000 timesteps (deltaT = 0.1 day, endTime = 500 days,
unchanged physical duration from the original tuning), writing 100
snapshots per trajectory (writeInterval = 50 steps, i.e. 5 days apart).
"""

import shutil, argparse, json
import numpy as np
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--n",    type=int, default=1000)
parser.add_argument("--base", type=str, default="base_case")
parser.add_argument("--out",  type=str, default="ensemble_ch4ch5")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

N_SAMPLES    = args.n
BASE_CASE    = Path(args.base)
ENSEMBLE_DIR = Path(args.out)
np.random.seed(args.seed)

# ── Domain / mesh ────────────────────────────────────────────────────────
LX, LY, LZ = 10000.0, 10000.0, 100.0
NX, NY, NZ  = 40, 40, 20
NZ_BOT      = 12
N_CELLS     = NX * NY * NZ

MU_OIL      = 5e-3
RHO_OIL     = 850.0
WELL_RATE   = 2e-6

# ── K, phi statistics (unchanged from v2) ───────────────────────────────
K_BOT_MEAN_LOG = np.log(0.30);  K_BOT_STD_LOG = 0.8
K_BOT_MIN, K_BOT_MAX = 0.05, 0.5
PHI_BOT_MEAN, PHI_BOT_STD = 0.2, 0.04
PHI_BOT_MIN,  PHI_BOT_MAX = 0.1, 0.25

K_TOP_MEAN_LOG = np.log(0.02);  K_TOP_STD_LOG = 0.7
K_TOP_MIN, K_TOP_MAX = 0.005, 0.03
PHI_TOP_MEAN, PHI_TOP_STD = 0.10, 0.03
PHI_TOP_MIN,  PHI_TOP_MAX = 0.05, 0.20

# ── ct(phi) physical coupling ────────────────────────────────────────────
# Softer / more porous rock -> higher pore compressibility. Linear map
# from the full phi range seen across both layers to a ct range centred
# on the original 1.0e-9 baseline.
PHI_GLOBAL_MIN = PHI_TOP_MIN   # 0.05
PHI_GLOBAL_MAX = PHI_BOT_MAX   # 0.25
CT_MIN, CT_MAX = 0.7e-9, 1.3e-9

def ct_from_phi(phi):
    frac = (phi - PHI_GLOBAL_MIN) / (PHI_GLOBAL_MAX - PHI_GLOBAL_MIN)
    frac = np.clip(frac, 0.0, 1.0)
    return CT_MIN + (CT_MAX - CT_MIN) * frac

# ── Well geometry — FIXED across the ensemble ───────────────────────────
WELL_R      = 200.0   # m, wellbore/near-well zone radius
WELL_K_D    = 10.0    # Darcy, well-zone permeability
WELL_MARGIN = 2000.0  # m, min distance of well centre from boundary

# ── Event schedule (Ch.4 Sec.4.6) — randomized PER SAMPLE ───────────────
# Each well independently gets a rate-step event: baseline WELL_RATE steps
# to WELL_RATE * multiplier at a randomized time. Randomizing both time and
# magnitude per sample (rather than a single fixed event for the whole
# ensemble) is what lets a held-out well test generalization to unseen
# event timing/magnitude, not just unseen well location.
EVENT_TIME_MIN_DAYS = 150.0
EVENT_TIME_MAX_DAYS = 350.0
RATE_MULT_MIN = 1.5
RATE_MULT_MAX = 4.0

P_RES_KIN = 30e6 / RHO_OIL
CORR_LEN  = 0.25

# ── Time stepping (Ch.4/5 decision: same 500-day physical window,
#    finer resolution -> 5000 steps, 100 saved snapshots) ───────────────
DELTA_T_DAYS   = 0.1
END_TIME_DAYS  = 500.0
N_STEPS        = int(round(END_TIME_DAYS / DELTA_T_DAYS))   # 5000
N_SNAPSHOTS    = 100
WRITE_INTERVAL_STEPS = N_STEPS // N_SNAPSHOTS                # 50
DELTA_T_S  = DELTA_T_DAYS * 86400.0
END_TIME_S = END_TIME_DAYS * 86400.0

ENSEMBLE_DIR.mkdir(parents=True, exist_ok=True)

FOAM_HEADER = """\
/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  2212                                  |
|   \\\\  /    A nd           | Web:      www.OpenFOAM.com                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       {cls};
    location    "{loc}";
    object      {obj};
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
"""

def correlated_field(nx, ny, nz, corr_len):
    fx = np.fft.fftfreq(nx); fy = np.fft.fftfreq(ny); fz = np.fft.fftfreq(nz)
    Fx, Fy, Fz = np.meshgrid(fx, fy, fz, indexing="ij")
    power = np.exp(-2 * np.pi**2 * corr_len**2 * (Fx**2+Fy**2+Fz**2))
    power[0,0,0] = 0.0
    noise = np.random.randn(nx,ny,nz) + 1j*np.random.randn(nx,ny,nz)
    field = np.real(np.fft.ifftn(np.fft.fftn(noise) * np.sqrt(power)))
    return (field - field.mean()) / (field.std() + 1e-12)

def make_well_mask(nx, ny, nz, cx, cy, r, lx, ly):
    x = (np.arange(nx)+0.5)*lx/nx
    y = (np.arange(ny)+0.5)*ly/ny
    X,Y = np.meshgrid(x, y, indexing="ij")
    mask2d = np.sqrt((X-cx)**2+(Y-cy)**2) <= r
    return np.broadcast_to(mask2d[:,:,None],(nx,ny,nz)).copy()

def write_p_field(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(values)
    with open(path,"w") as f:
        f.write(FOAM_HEADER.format(cls="volScalarField",loc="0",obj="p"))
        f.write("dimensions      [0 2 -2 0 0 0 0];\n\n")
        f.write(f"internalField   nonuniform List<scalar>\n{n}\n(\n")
        for v in values: f.write(f"    {v:.8e}\n")
        f.write(");\n\nboundaryField\n{\n")
        f.write("    reservoir\n    {\n        type    zeroGradient;\n    }\n}\n")

def write_U_field(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path,"w") as f:
        f.write(FOAM_HEADER.format(cls="volVectorField",loc="0",obj="U"))
        f.write("dimensions      [0 1 -1 0 0 0 0];\n\n")
        f.write("internalField   uniform (0 0 0);\n\n")
        f.write("boundaryField\n{\n")
        f.write("    reservoir\n    {\n        type    zeroGradient;\n    }\n}\n")

def write_scalar_field(path, name, values, dims):
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(values)
    with open(path,"w") as f:
        f.write(FOAM_HEADER.format(cls="volScalarField",loc="constant",obj=name))
        f.write(f"dimensions      {dims};\n\n")
        f.write(f"internalField   nonuniform List<scalar>\n{n}\n(\n")
        for v in values: f.write(f"    {v:.8e}\n")
        f.write(");\n\nboundaryField\n{\n")
        f.write("    reservoir\n    {\n        type zeroGradient;\n    }\n}\n")

def snap_to_timestep(t_days, delta_t_days=DELTA_T_DAYS):
    """Snap an event time to the nearest valid timestep boundary, so the
    step lands exactly on a solved step rather than being silently
    interpolated/smeared by the table's linear interpolation between
    non-aligned points."""
    return round(t_days / delta_t_days) * delta_t_days

def write_reservoirProperties(path, mu, wellRate_before, wellRate_after,
                              event_time_s):
    """ct is no longer a scalar/zone constant -- it's derived from the phi
    field at generation time (ct_from_phi) and written as its own field.
    wellRate is now a time-varying Function1 table implementing a rate-step
    EVENT (Ch.4 Sec.4.6): wellRate_before until just before event_time_s,
    then wellRate_after for the remainder of the run. The two adjacent
    table points (event_time_s - DELTA_T_S, event_time_s) approximate an
    instantaneous step landing on a single timestep boundary, matching the
    approach validated in the manually-edited single-well reservoirProperties.
    Requires darcyFoam compiled against the Function1<scalar>-based
    createFields.H/darcyFoam.C -- a plain-scalar-reading binary will fail
    to parse this dictionary."""
    path.parent.mkdir(parents=True, exist_ok=True)
    step_pre_s = event_time_s - DELTA_T_S
    with open(path,"w") as f:
        f.write(FOAM_HEADER.format(cls="dictionary",loc="constant",
                                   obj="reservoirProperties"))
        f.write(f"mu          {mu:.6e};\n\n")
        f.write("// EVENT (Ch.4 Sec.4.6): well-rate step, randomized per\n")
        f.write("// sample -- see generate_samples_v3.py EVENT_TIME_*/RATE_MULT_*.\n")
        f.write("// Sink coefficient only (still linear in p): the point is a\n")
        f.write("// genuine change in the effective discrete-time operator A(t)\n")
        f.write("// mid-trajectory, not an attempt to make the physics itself\n")
        f.write("// nonlinear.\n")
        f.write(f"// event_time = {event_time_s/86400.0:.2f} days, "
                f"rate {wellRate_before:.4e} -> {wellRate_after:.4e} [1/s]\n")
        f.write("wellRate\n{\n")
        f.write("    type    table;\n")
        f.write("    values\n    (\n")
        f.write(f"        (0             {wellRate_before:.6e})\n")
        f.write(f"        ({step_pre_s:.1f}  {wellRate_before:.6e})\n")
        f.write(f"        ({event_time_s:.1f}  {wellRate_after:.6e})\n")
        f.write(f"        ({END_TIME_S:.1f}  {wellRate_after:.6e})\n")
        f.write("    );\n")
        f.write("    outOfBounds clamp;\n")
        f.write("}\n\n")
        f.write("// K, phi, ct are all read as spatially-resolved fields\n")
        f.write("// (constant/K, constant/phi, constant/ct). ct is physically\n")
        f.write("// coupled to phi at generation time (ct_from_phi), not an\n")
        f.write("// independent random field. Zone entries below are for\n")
        f.write("// reference only and are not read by the solver.\n")
        f.write("zones\n{\n")
        f.write("    bottomLayer { K  2.961e-13;  phi  0.20; }\n")
        f.write("    topLayer    { K  1.974e-14;  phi  0.10; }\n")
        f.write(f"    wellCells   {{ K  9.869e-12;  phi  1.0; }}\n")
        f.write("}\n")

def write_topoSetDict(path, cx_m, cy_m, r_m):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path,"w") as f:
        f.write(FOAM_HEADER.format(cls="dictionary",loc="system",obj="topoSetDict"))
        f.write("actions\n(\n")
        f.write("""
    {
        name    bottomLayer;
        type    cellZoneSet;
        action  new;
        source  boxToCell;
        sourceInfo { box (0 0 0) (10000 10000 60); }
    }
""")
        f.write("""
    {
        name    topLayer;
        type    cellZoneSet;
        action  new;
        source  boxToCell;
        sourceInfo { box (0 0 60) (10000 10000 100); }
    }
""")
        f.write(f"""
    {{
        name    wellCells;
        type    cellZoneSet;
        action  new;
        source  cylinderToCell;
        sourceInfo
        {{
            p1 ({cx_m:.1f} {cy_m:.1f} 0);
            p2 ({cx_m:.1f} {cy_m:.1f} 100);
            radius {r_m:.1f};
        }}
    }}
""")
        f.write(");\n")

def write_controlDict(path):
    """Written explicitly per sample so the ensemble's time-stepping is
    guaranteed consistent regardless of what's sitting in base_case/system.
    5000 steps @ deltaT=0.1 day over the original 500-day window;
    writeInterval=50 steps -> 100 saved snapshots per trajectory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path,"w") as f:
        f.write(FOAM_HEADER.format(cls="dictionary",loc="system",obj="controlDict"))
        f.write("application     darcyFoam;\n\n")
        f.write("startFrom       startTime;\n")
        f.write("startTime       0;\n")
        f.write("stopAt          endTime;\n")
        f.write(f"endTime         {END_TIME_S:.1f};       // {END_TIME_DAYS:.0f} days\n\n")
        f.write(f"deltaT          {DELTA_T_S:.2f};          // {DELTA_T_DAYS} day timestep\n\n")
        f.write("writeControl    timeStep;\n")
        f.write(f"writeInterval   {WRITE_INTERVAL_STEPS};             // -> {N_SNAPSHOTS} snapshots\n")
        f.write("purgeWrite      0;\n\n")
        f.write("writeFormat     binary;\n")
        f.write("writePrecision  8;\n")
        f.write("writeCompression off;\n\n")
        f.write("runTimeModifiable true;\n\n")
        f.write("functions\n{\n")
        f.write("    wellPressure\n    {\n")
        f.write("        type            volFieldValue;\n")
        f.write('        libs            ("libfieldFunctionObjects.so");\n')
        f.write("        fields          (p);\n")
        f.write("        operation       volAverage;\n")
        f.write("        regionType      cellZone;\n")
        f.write("        name            wellCells;\n")
        f.write("        writeFields     false;\n")
        f.write("        writeControl    timeStep;\n")
        f.write(f"        writeInterval   {WRITE_INTERVAL_STEPS};\n")
        f.write("    }\n")
        f.write("    reservoirPressure\n    {\n")
        f.write("        type            volFieldValue;\n")
        f.write('        libs            ("libfieldFunctionObjects.so");\n')
        f.write("        fields          (p);\n")
        f.write("        operation       volAverage;\n")
        f.write("        regionType      all;\n")
        f.write("        writeFields     false;\n")
        f.write("        writeControl    timeStep;\n")
        f.write(f"        writeInterval   {WRITE_INTERVAL_STEPS};\n")
        f.write("    }\n")
        f.write("}\n")

# ── Main loop ───────────────────────────────────────────────────────────
metadata = []
BOT_MASK = np.zeros((NX,NY,NZ),dtype=bool); BOT_MASK[:,:,:NZ_BOT]=True
TOP_MASK = ~BOT_MASK

print(f"Generating {N_SAMPLES} full-realistic samples (Ch.4/Ch.5) -> {ENSEMBLE_DIR}/")
print(f"  {N_STEPS} steps @ dt={DELTA_T_DAYS} day, endTime={END_TIME_DAYS} day, "
      f"{N_SNAPSHOTS} snapshots/trajectory")

for i in range(N_SAMPLES):
    if i % 100 == 0: print(f"  Sample {i:04d} / {N_SAMPLES}")
    case_dir = ENSEMBLE_DIR / f"sample_{i:04d}"
    if BASE_CASE.exists():
        if case_dir.exists(): shutil.rmtree(case_dir)
        shutil.copytree(str(BASE_CASE), str(case_dir))
    for d in ["0","constant","system"]:
        (case_dir/d).mkdir(parents=True, exist_ok=True)

    # Random well location -- independent per sample
    # VARIABLE-WELL RECONFIRMATION RUN (Ch.3 main experiment, rerun on the
    # CURRENT solver binary): well location is randomized per sample, same
    # as the original ldo_data_v3 generator -- but this script otherwise
    # shares 100% of its code with generate_samples_v3_fixedwell.py (event
    # neutralized to a no-op, same Function1-table reservoirProperties code
    # path), so both Table 3.1 columns are now provably produced by the
    # identical compiled darcyFoam binary, not two different solver versions.
    cx_m = np.random.uniform(WELL_MARGIN, LX - WELL_MARGIN)
    cy_m = np.random.uniform(WELL_MARGIN, LY - WELL_MARGIN)
    WELL_MASK = make_well_mask(NX,NY,NZ,cx_m,cy_m,WELL_R,LX,LY)
    bot_nonwell = BOT_MASK & ~WELL_MASK
    top_nonwell = TOP_MASK & ~WELL_MASK

    # K field -- independent correlated noise, by layer
    raw = correlated_field(NX,NY,NZ,CORR_LEN)
    K_darcy = np.zeros((NX,NY,NZ))
    K_darcy[:,:,:NZ_BOT] = np.clip(
        np.exp(raw[:,:,:NZ_BOT]*K_BOT_STD_LOG+K_BOT_MEAN_LOG),K_BOT_MIN,K_BOT_MAX)
    K_darcy[:,:,NZ_BOT:] = np.clip(
        np.exp(raw[:,:,NZ_BOT:]*K_TOP_STD_LOG+K_TOP_MEAN_LOG),K_TOP_MIN,K_TOP_MAX)
    K_darcy[WELL_MASK] = WELL_K_D
    K_flat = K_darcy.flatten(order="F")

    # phi field -- independent correlated noise, by layer (partially
    # decorrelated from K via its own noise draw, as in v2)
    raw2 = 0.6*raw + 0.4*correlated_field(NX,NY,NZ,CORR_LEN)
    phi_field = np.zeros((NX,NY,NZ))
    phi_field[:,:,:NZ_BOT] = np.clip(
        raw2[:,:,:NZ_BOT]*PHI_BOT_STD+PHI_BOT_MEAN,PHI_BOT_MIN,PHI_BOT_MAX)
    phi_field[:,:,NZ_BOT:] = np.clip(
        raw2[:,:,NZ_BOT:]*PHI_TOP_STD+PHI_TOP_MEAN,PHI_TOP_MIN,PHI_TOP_MAX)
    phi_flat = phi_field.flatten(order="F")

    # ct field -- NOT independent noise. Deterministic function of phi.
    ct_field = ct_from_phi(phi_field)
    ct_flat = ct_field.flatten(order="F")

    # Event: rate step, randomized per sample (Ch.4 Sec.4.6)
    # NO-OP EVENT (Ch.3 fixed-well control has nothing to do with Ch.4's
    # event experiment): rate_multiplier forced to 1.0 so wellRate_before
    # == wellRate_after. This keeps the same Function1-table reservoirProperties
    # code path (still compatible with the already-compiled darcyFoam binary)
    # without introducing a rate change -- Chapter 3 only needs smooth,
    # constant-rate depletion with fixed well location.
    event_time_days_raw = np.random.uniform(EVENT_TIME_MIN_DAYS, EVENT_TIME_MAX_DAYS)
    event_time_days = snap_to_timestep(event_time_days_raw)
    event_time_s = event_time_days * 86400.0
    rate_multiplier = 1.0
    wellRate_before = WELL_RATE
    wellRate_after = WELL_RATE * rate_multiplier

    # Write fields
    write_p_field(case_dir/"0"/"p", np.full(N_CELLS, P_RES_KIN))
    write_U_field(case_dir/"0"/"U")
    write_scalar_field(case_dir/"constant"/"K",   "K",   K_flat,   "[0 2 0 0 0 0 0]")
    write_scalar_field(case_dir/"constant"/"phi", "phi", phi_flat, "[0 0 0 0 0 0 0]")
    write_scalar_field(case_dir/"constant"/"ct",  "ct",  ct_flat,  "[-1 1 2 0 0 0 0]")
    write_reservoirProperties(case_dir/"constant"/"reservoirProperties",
                              mu=MU_OIL, wellRate_before=wellRate_before,
                              wellRate_after=wellRate_after,
                              event_time_s=event_time_s)
    write_topoSetDict(case_dir/"system"/"topoSetDict", cx_m, cy_m, WELL_R)
    write_controlDict(case_dir/"system"/"controlDict")

    well_cx_norm = cx_m / LX
    well_cy_norm = cy_m / LY

    sample_meta = {
        "sample_id":   i,
        "well_cx_m":   float(cx_m),
        "well_cy_m":   float(cy_m),
        "well_cx_norm": float(well_cx_norm),
        "well_cy_norm": float(well_cy_norm),
        "well_R_m":    WELL_R,          # fixed across ensemble
        "K_bot_mean_D": float(K_darcy[bot_nonwell].mean()),
        "K_top_mean_D": float(K_darcy[top_nonwell].mean()),
        "phi_bot_mean": float(phi_field[:,:,:NZ_BOT].mean()),
        "phi_top_mean": float(phi_field[:,:,NZ_BOT:].mean()),
        "ct_bot_mean": float(ct_field[:,:,:NZ_BOT].mean()),
        "ct_top_mean": float(ct_field[:,:,NZ_BOT:].mean()),
        "ct_coupling": "linear in phi, ct in [%.2e, %.2e]" % (CT_MIN, CT_MAX),
        "mu_oil": MU_OIL,
        "wellRate_before": float(wellRate_before),
        "wellRate_after": float(wellRate_after),
        "rate_multiplier": float(rate_multiplier),
        "event_time_days": float(event_time_days),
        "event_time_s": float(event_time_s),
        "event_snapshot_index": int(round(event_time_days /
                                    (END_TIME_DAYS / N_SNAPSHOTS))),
        "n_steps": N_STEPS, "n_snapshots": N_SNAPSHOTS,
    }
    with open(case_dir/"sample_meta.json","w") as f:
        json.dump(sample_meta, f, indent=2)
    metadata.append(sample_meta)

with open(ENSEMBLE_DIR/"metadata.json","w") as f:
    json.dump(metadata, f, indent=2)
print(f"Done! {N_SAMPLES} cases -> {ENSEMBLE_DIR}/")
print(f"Well location: variable, x,y in [{WELL_MARGIN:.0f}, {LX-WELL_MARGIN:.0f}] m, WELL_R={WELL_R} m")
print(f"ct physically coupled to phi: ct in [{CT_MIN:.2e}, {CT_MAX:.2e}]")
