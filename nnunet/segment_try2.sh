#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./segment_try3.sh
#   ./segment_try3.sh /Users/.../CT_Normal_Volumes_Anonymized "CT_Normal_1"
#
# Defaults = run all CT_Normal_* under BASE_DIR

ENV_PREFIX="$HOME/miniconda3/envs/sam2"
TS="$ENV_PREFIX/bin/TotalSegmentator"
PY="$ENV_PREFIX/bin/python"

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BASE_DIR="${1:-$SCRIPT_DIR}"
CASE_GLOB="${2:-CT_Normal_*}"

# Inputs relative to each case
REL="vessel_seg/reg_ants_art_to_ven"
VEN_NAME="venous_pad.nii.gz"
ART_NAME="arterial_reg_to_venous.nii.gz"
LIV_NAME="liver.nii.gz"

# Device + threads
DEVICE="${DEVICE:-gpu}"    # use GPU (TotalSegmentator uses 'gpu' not 'cuda')
NR="${NR:-4}"
NS="${NS:-4}"

# Postprocess knobs (mm / mm^3)
EDGE_ERODE_MM="${EDGE_ERODE_MM:-2.0}"   # increase if you still see edge artifacts
OPEN_MM="${OPEN_MM:-0.8}"               # reduce speckle
CLOSE_MM="${CLOSE_MM:-1.2}"             # connect tiny gaps
MINVOL_A_MM3="${MINVOL_A_MM3:-30}"      # keep more A by lowering
MINVOL_V_MM3="${MINVOL_V_MM3:-60}"      # keep more V by lowering
KEEP_N_A="${KEEP_N_A:-6}"               # keep more components to reduce “missing branches”
KEEP_N_V="${KEEP_N_V:-6}"

OPEN_PREVIEW="${OPEN_PREVIEW:-1}"       # auto-open PNGs on macOS

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"

if [[ ! -x "$TS" ]]; then
  echo "[ERROR] TotalSegmentator not found: $TS"
  exit 1
fi
if [[ ! -x "$PY" ]]; then
  echo "[ERROR] python not found: $PY"
  exit 1
fi

shopt -s nullglob
CASES=( "$BASE_DIR"/$CASE_GLOB )
if (( ${#CASES[@]} == 0 )); then
  echo "[ERROR] No cases matched: $BASE_DIR/$CASE_GLOB"
  exit 1
fi

for CASE in "${CASES[@]}"; do
  [[ -d "$CASE" ]] || continue

  REGDIR="$CASE/$REL"
  VEN="$REGDIR/$VEN_NAME"
  ART="$REGDIR/$ART_NAME"
  LIV="$REGDIR/$LIV_NAME"

  if [[ ! -f "$VEN" || ! -f "$ART" || ! -f "$LIV" ]]; then
    echo "[SKIP] Missing inputs in $REGDIR"
    continue
  fi

  OUTROOT="$REGDIR/ts_liver_vessels"
  OUTV="$OUTROOT/venous"
  OUTA="$OUTROOT/arterial"
  OUTC="$OUTROOT/combined"
  rm -rf "$OUTROOT"
  mkdir -p "$OUTV" "$OUTA" "$OUTC"

  echo "============================================================"
  echo "[CASE] $(basename "$CASE")"
  echo "DEVICE=$DEVICE  NR=$NR  NS=$NS"
  echo "============================================================"

  echo "[1/3] TotalSegmentator venous (veins)"
  # IMPORTANT: no -p (preview) to avoid fury dependency
  "$TS" -i "$VEN" -o "$OUTV" -ta liver_vessels -cp "$LIV" -rc -d "$DEVICE" -nr "$NR" -ns "$NS"

  echo "[2/3] TotalSegmentator arterial (arteries)"
  "$TS" -i "$ART" -o "$OUTA" -ta liver_vessels -cp "$LIV" -rc -d "$DEVICE" -nr "$NR" -ns "$NS"

  echo "[3/3] Postprocess + QC (no fury needed)"
  "$PY" - <<PY
import os, glob
import numpy as np
import nibabel as nib
import scipy.ndimage as ndi

# connected-components-3d is installed in your env (you showed it in pip freeze)
import cc3d

from skimage.measure import marching_cubes
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

VEN = r"$VEN"
ART = r"$ART"
LIV = r"$LIV"
OUTV = r"$OUTV"
OUTA = r"$OUTA"
OUTC = r"$OUTC"

EDGE_ERODE_MM = float(r"$EDGE_ERODE_MM")
OPEN_MM = float(r"$OPEN_MM")
CLOSE_MM = float(r"$CLOSE_MM")
MINVOL_A_MM3 = float(r"$MINVOL_A_MM3")
MINVOL_V_MM3 = float(r"$MINVOL_V_MM3")
KEEP_N_A = int(r"$KEEP_N_A")
KEEP_N_V = int(r"$KEEP_N_V")

def load_nii(p):
    img = nib.load(p)
    data = img.get_fdata().astype(np.float32)
    return img, data

ref_img, ven = load_nii(VEN)
_, art = load_nii(ART)
_, liv = load_nii(LIV)

spacing = np.array(ref_img.header.get_zooms()[:3], dtype=float)  # x,y,z
liv = liv > 0.5

# Erode liver in *mm* using distance transform (kills edge artifacts)
dt = ndi.distance_transform_edt(liv.astype(np.uint8), sampling=spacing)
liv_interior = dt >= EDGE_ERODE_MM

def ellipsoid_struct(r_mm, spacing):
    sx, sy, sz = spacing
    rx = max(1, int(np.ceil(r_mm / sx)))
    ry = max(1, int(np.ceil(r_mm / sy)))
    rz = max(1, int(np.ceil(r_mm / sz)))
    x, y, z = np.ogrid[-rx:rx+1, -ry:ry+1, -rz:rz+1]
    d2 = (x*sx)**2 + (y*sy)**2 + (z*sz)**2
    return d2 <= (r_mm**2)

st_open  = ellipsoid_struct(OPEN_MM, spacing)
st_close = ellipsoid_struct(CLOSE_MM, spacing)

def union_masks_from_dir(d, prefer_keywords=(), forbid_keywords=()):
    files = sorted(glob.glob(os.path.join(d, "*.nii*")))
    if not files:
        return np.zeros(ref_img.shape, bool), []

    chosen = []
    if prefer_keywords:
        for f in files:
            bn = os.path.basename(f).lower()
            if any(k in bn for k in prefer_keywords) and not any(k in bn for k in forbid_keywords):
                chosen.append(f)

    if not chosen:
        chosen = files

    u = np.zeros(ref_img.shape, bool)
    used = []
    for f in chosen:
        try:
            m = nib.load(f).get_fdata() > 0.5
            if m.shape == u.shape:
                u |= m
                used.append(os.path.basename(f))
        except Exception:
            pass
    return u, used

# Heuristic: pick veins from venous output; arteries from arterial output
maskV_raw, usedV = union_masks_from_dir(OUTV, prefer_keywords=("vein","vena","portal"), forbid_keywords=("arter",))
maskA_raw, usedA = union_masks_from_dir(OUTA, prefer_keywords=("arter",), forbid_keywords=("vein","vena","portal"))

# Constrain to eroded liver interior
maskV = maskV_raw & liv_interior
maskA = maskA_raw & liv_interior

# Open then close to reduce speckle + connect tiny gaps
if maskV.any():
    maskV = ndi.binary_opening(maskV, structure=st_open, iterations=1)
    maskV = ndi.binary_closing(maskV, structure=st_close, iterations=1)
if maskA.any():
    maskA = ndi.binary_opening(maskA, structure=st_open, iterations=1)
    maskA = ndi.binary_closing(maskA, structure=st_close, iterations=1)

def keep_components(mask, min_mm3, keep_n):
    if not mask.any():
        return mask
    lab = cc3d.connected_components(mask.astype(np.uint8), connectivity=26)
    n = int(lab.max())
    if n == 0:
        return mask
    voxvol = float(np.prod(spacing))
    counts = np.bincount(lab.ravel())
    vols = [(i, counts[i]*voxvol) for i in range(1, n+1)]
    vols = [v for v in vols if v[1] >= float(min_mm3)]
    vols.sort(key=lambda x: x[1], reverse=True)
    keep_ids = [i for i,_ in vols[:max(1,int(keep_n))]]
    return np.isin(lab, keep_ids)

maskA = keep_components(maskA, MINVOL_A_MM3, KEEP_N_A)
maskV = keep_components(maskV, MINVOL_V_MM3, KEEP_N_V)

def save_mask(mask, out_path):
    out = nib.Nifti1Image(mask.astype(np.uint8), ref_img.affine, ref_img.header)
    out.set_data_dtype(np.uint8)
    nib.save(out, out_path)

save_mask(maskA_raw, os.path.join(OUTC, "arteries_raw.nii.gz"))
save_mask(maskV_raw, os.path.join(OUTC, "veins_raw.nii.gz"))
save_mask(maskA,     os.path.join(OUTC, "arteries_clean.nii.gz"))
save_mask(maskV,     os.path.join(OUTC, "veins_clean.nii.gz"))

# QC 1: MIP overlay on venous background (red/blue)
v = np.max(ven, axis=2)
v = np.clip(v, np.percentile(v, 5), np.percentile(v, 99))
v = (v - v.min()) / (v.max() - v.min() + 1e-6)
A_mip = np.max(maskA, axis=2)
V_mip = np.max(maskV, axis=2)

rgb = np.stack([v,v,v], axis=-1)
rgb[A_mip, :] = [1,0,0]
rgb[V_mip, :] = [0,0,1]

plt.figure(figsize=(7,7))
plt.imshow(rgb)
plt.axis("off")
plt.title("MIP overlay: arteries red / veins blue")
mip_png = os.path.join(OUTC, "qc_mip_red_blue.png")
plt.tight_layout()
plt.savefig(mip_png, dpi=200)
plt.close()

# QC 2: 3D isosurface PNG
def add_mesh(ax, mask, color_rgba, alpha):
    if mask.sum() < 50:
        return
    verts, faces, _, _ = marching_cubes(mask.astype(np.uint8), level=0.5, spacing=spacing)
    mesh = Poly3DCollection(verts[faces], alpha=alpha)
    mesh.set_facecolor(color_rgba)
    mesh.set_edgecolor((0,0,0,0))
    ax.add_collection3d(mesh)

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection="3d")
ax.set_axis_off()
add_mesh(ax, maskV, (0.1,0.3,0.95,0.35), 0.35)
add_mesh(ax, maskA, (0.95,0.2,0.2,0.65), 0.65)

# autoscale
pts = []
for m in (maskA, maskV):
    if m.any():
        p = np.argwhere(m)
        pts.append(p * spacing[None,:])
if pts:
    pts = np.vstack(pts)
    mn, mx = pts.min(0), pts.max(0)
    c = (mn+mx)/2
    span = (mx-mn).max()/2
    ax.set_xlim(c[0]-span, c[0]+span)
    ax.set_ylim(c[1]-span, c[1]+span)
    ax.set_zlim(c[2]-span, c[2]+span)

ax.view_init(elev=18, azim=35)
png3d = os.path.join(OUTC, "qc_3d_red_blue.png")
plt.tight_layout()
plt.savefig(png3d, dpi=200)
plt.close(fig)

# Summary
def mm3(m): return float(m.sum()) * float(np.prod(spacing))
with open(os.path.join(OUTC, "summary.txt"), "w") as f:
    f.write(f"Spacing (mm): {spacing.tolist()}\\n")
    f.write(f"EDGE_ERODE_MM={EDGE_ERODE_MM}, OPEN_MM={OPEN_MM}, CLOSE_MM={CLOSE_MM}\\n")
    f.write(f"MINVOL_A_MM3={MINVOL_A_MM3}, MINVOL_V_MM3={MINVOL_V_MM3}\\n")
    f.write(f"KEEP_N_A={KEEP_N_A}, KEEP_N_V={KEEP_N_V}\\n")
    f.write(f"Used artery files: {usedA}\\n")
    f.write(f"Used vein files: {usedV}\\n")
    f.write(f"Arteries raw mm3: {mm3(maskA_raw):.1f}, clean mm3: {mm3(maskA):.1f}\\n")
    f.write(f"Veins raw mm3: {mm3(maskV_raw):.1f}, clean mm3: {mm3(maskV):.1f}\\n")
    f.write(f"QC: {mip_png}\\nQC: {png3d}\\n")

print("Wrote:", os.path.join(OUTC, "arteries_clean.nii.gz"))
print("Wrote:", os.path.join(OUTC, "veins_clean.nii.gz"))
print("QC:", mip_png)
print("QC:", png3d)
PY

  if [[ "$OPEN_PREVIEW" == "1" ]]; then
    xdg-open "$OUTC/qc_mip_red_blue.png" 2>/dev/null || open "$OUTC/qc_mip_red_blue.png" 2>/dev/null || true
    xdg-open "$OUTC/qc_3d_red_blue.png" 2>/dev/null || open "$OUTC/qc_3d_red_blue.png" 2>/dev/null || true
  fi

  echo "[DONE] $CASE"
done

echo
echo "All done."
echo "If you publish with TotalSegmentator, cite: https://pubs.rsna.org/doi/10.1148/ryai.230024"

