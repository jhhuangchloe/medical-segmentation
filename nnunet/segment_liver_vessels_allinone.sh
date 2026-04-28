#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# segment_liver_vessels_allinone.sh
#
# Runs TotalSegmentator liver_vessels on venous+arterial CT (cropped by liver mask),
# postprocesses masks (edge-avoid + denoise + bridge gaps + CC filtering),
# then generates:
#   - qc_mip_red_blue.png
#   - qc_3d_red_blue.png
#   - qc_3d_red_blue.html  (interactive Plotly)
#
# TotalSegmentator citation: https://pubs.rsna.org/doi/10.1148/ryai.230024
#
# USAGE:
#   chmod +x segment_liver_vessels_allinone.sh
#   ./segment_liver_vessels_allinone.sh \
#     "/Users/davidberry/Desktop/ARPA_H/CT/CT_Normal_Volumes_Anonymized" "CT_Normal_1"
#
# Defaults:
#   BASE_DIR = /Users/davidberry/Desktop/ARPA_H/CT/CT_Normal_Volumes_Anonymized
#   CASE_GLOB = CT_Normal_*
# ------------------------------------------------------------

# ---- EDIT IF NEEDED ----
ENV_PREFIX="${ENV_PREFIX:-$HOME/miniconda3/envs/sam2}"
TS="${TS:-$ENV_PREFIX/bin/TotalSegmentator}"
PY="${PY:-$ENV_PREFIX/bin/python}"

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BASE_DIR="${1:-$SCRIPT_DIR}"
CASE_GLOB="${2:-CT_Normal_*}"

REL="vessel_seg/reg_ants_art_to_ven"
VEN_NAME="venous_pad.nii.gz"
ART_NAME="arterial_reg_to_venous.nii.gz"
LIV_NAME="liver.nii.gz"

# Reliability on macOS:
DEVICE="${DEVICE:-gpu}"   # use GPU (TotalSegmentator uses 'gpu' not 'cuda')
NR="${NR:-4}"
NS="${NS:-4}"

# Postprocess knobs (mm-aware where relevant)
EDGE_ERODE_MM="${EDGE_ERODE_MM:-2.0}"     # shrink liver mask inward to avoid edge artifacts
OPEN_MM="${OPEN_MM:-0.7}"                  # remove speckle
CLOSE_MM="${CLOSE_MM:-1.2}"                # bridge small gaps
MINVOL_A_MM3="${MINVOL_A_MM3:-20}"         # drop tiny arterial islands
MINVOL_V_MM3="${MINVOL_V_MM3:-40}"         # drop tiny venous islands
KEEP_N_A="${KEEP_N_A:-0}"                  # 0 = keep ALL comps above MINVOL (recommended)
KEEP_N_V="${KEEP_N_V:-0}"                  # 0 = keep ALL comps above MINVOL (recommended)

# HTML mesh knobs (visualization only)
HTML_MAX_FACES="${HTML_MAX_FACES:-200000}"
HTML_SMOOTH="${HTML_SMOOTH:-1.0 1.0 0.6}"  # Gaussian sigma in VOXELS for viz-only smoothing

OPEN_PREVIEW="${OPEN_PREVIEW:-1}"          # open outputs automatically (macOS "open")

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"

# ---- Sanity checks ----
if [[ ! -x "$TS" ]]; then
  echo "[ERROR] TotalSegmentator not found/executable: $TS"
  exit 1
fi
if [[ ! -x "$PY" ]]; then
  echo "[ERROR] Python not found/executable: $PY"
  exit 1
fi

# Require plotly (you already installed it)
"$PY" - <<'PY'
import plotly
print("plotly OK")
PY

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
    echo "[SKIP] Missing inputs in: $REGDIR"
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
  echo "REGDIR: $REGDIR"
  echo "DEVICE=$DEVICE NR=$NR NS=$NS"
  echo "EDGE_ERODE_MM=$EDGE_ERODE_MM OPEN_MM=$OPEN_MM CLOSE_MM=$CLOSE_MM"
  echo "MINVOL_A_MM3=$MINVOL_A_MM3 MINVOL_V_MM3=$MINVOL_V_MM3 KEEP_N_A=$KEEP_N_A KEEP_N_V=$KEEP_N_V"
  echo "HTML_MAX_FACES=$HTML_MAX_FACES HTML_SMOOTH=$HTML_SMOOTH"
  echo "============================================================"

  echo "[1/4] TotalSegmentator on VENOUS (for veins/portal tree)"
  "$TS" -i "$VEN" -o "$OUTV" -ta liver_vessels -cp "$LIV" -rc -d "$DEVICE" -nr "$NR" -ns "$NS"

  echo "[2/4] TotalSegmentator on ARTERIAL (for arteries)"
  "$TS" -i "$ART" -o "$OUTA" -ta liver_vessels -cp "$LIV" -rc -d "$DEVICE" -nr "$NR" -ns "$NS"

  echo "[3/4] Postprocess + QC PNGs"
  export VEN ART LIV OUTV OUTA OUTC
  export EDGE_ERODE_MM OPEN_MM CLOSE_MM MINVOL_A_MM3 MINVOL_V_MM3 KEEP_N_A KEEP_N_V

  "$PY" - <<'PY'
import os, glob
import numpy as np
import nibabel as nib
import scipy.ndimage as ndi
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# connected-components-3d provides cc3d
import cc3d
from skimage.measure import marching_cubes
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

VEN = os.environ["VEN"]
ART = os.environ["ART"]
LIV = os.environ["LIV"]
OUTV = os.environ["OUTV"]
OUTA = os.environ["OUTA"]
OUTC = os.environ["OUTC"]

EDGE_ERODE_MM = float(os.environ["EDGE_ERODE_MM"])
OPEN_MM = float(os.environ["OPEN_MM"])
CLOSE_MM = float(os.environ["CLOSE_MM"])
MINVOL_A_MM3 = float(os.environ["MINVOL_A_MM3"])
MINVOL_V_MM3 = float(os.environ["MINVOL_V_MM3"])
KEEP_N_A = int(os.environ["KEEP_N_A"])
KEEP_N_V = int(os.environ["KEEP_N_V"])

def load_nii(p):
    img = nib.load(p)
    return img, img.get_fdata().astype(np.float32)

ref_img, ven = load_nii(VEN)
_, art = load_nii(ART)
_, liv = load_nii(LIV)

spacing = np.array(ref_img.header.get_zooms()[:3], dtype=float)
liv = liv > 0.5

# Inward erosion in mm using distance transform (handles anisotropic spacing)
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

def union_all_masks(folder):
    files = sorted(glob.glob(os.path.join(folder, "*.nii")) + glob.glob(os.path.join(folder, "*.nii.gz")))
    u = np.zeros(ref_img.shape, dtype=bool)
    used = []
    for f in files:
        # avoid accidentally unioning non-mask files if present
        bn = os.path.basename(f).lower()
        if bn.startswith("preview") or bn.endswith(".json"):
            continue
        try:
            m = nib.load(f).get_fdata() > 0.5
            if m.shape == u.shape:
                u |= m
                used.append(os.path.basename(f))
        except Exception:
            pass
    return u, used

# Key simplifying assumption for batch:
# - VENOUS run mask union approximates venous/portal tree
# - ARTERIAL run mask union approximates arterial tree
maskV_raw, usedV = union_all_masks(OUTV)
maskA_raw, usedA = union_all_masks(OUTA)

# Restrict to interior liver (reduces edge artifacts a lot)
maskV = maskV_raw & liv_interior
maskA = maskA_raw & liv_interior

# Denoise + bridge small gaps
if maskV.any():
    maskV = ndi.binary_opening(maskV, structure=st_open, iterations=1)
    maskV = ndi.binary_closing(maskV, structure=st_close, iterations=1)
if maskA.any():
    maskA = ndi.binary_opening(maskA, structure=st_open, iterations=1)
    maskA = ndi.binary_closing(maskA, structure=st_close, iterations=1)

voxvol = float(np.prod(spacing))

def filter_components(mask, min_mm3, keep_n):
    if not mask.any():
        return mask
    lab = cc3d.connected_components(mask.astype(np.uint8), connectivity=26)
    n = int(lab.max())
    if n == 0:
        return mask
    counts = np.bincount(lab.ravel())
    vols = [(i, counts[i]*voxvol) for i in range(1, n+1)]
    vols = [v for v in vols if v[1] >= float(min_mm3)]
    vols.sort(key=lambda x: x[1], reverse=True)

    if keep_n <= 0:
        keep_ids = [i for i,_ in vols]
    else:
        keep_ids = [i for i,_ in vols[:max(1, keep_n)]]

    out = np.isin(lab, keep_ids)
    return out

maskA = filter_components(maskA, MINVOL_A_MM3, KEEP_N_A)
maskV = filter_components(maskV, MINVOL_V_MM3, KEEP_N_V)

def save_mask(mask, out_path):
    out = nib.Nifti1Image(mask.astype(np.uint8), ref_img.affine, ref_img.header)
    out.set_data_dtype(np.uint8)
    nib.save(out, out_path)

save_mask(maskA_raw, os.path.join(OUTC, "arteries_raw.nii.gz"))
save_mask(maskV_raw, os.path.join(OUTC, "veins_raw.nii.gz"))
save_mask(maskA,     os.path.join(OUTC, "arteries_clean.nii.gz"))
save_mask(maskV,     os.path.join(OUTC, "veins_clean.nii.gz"))

# QC MIP overlay (arteries red / veins blue) on venous background MIP
bg = np.max(ven, axis=2)
p5, p99 = np.percentile(bg, 5), np.percentile(bg, 99)
bg = np.clip(bg, p5, p99)
bg = (bg - bg.min()) / (bg.max() - bg.min() + 1e-6)

A_mip = np.max(maskA, axis=2)
V_mip = np.max(maskV, axis=2)

rgb = np.stack([bg, bg, bg], axis=-1)
rgb[A_mip] = [1, 0, 0]
rgb[V_mip] = [0, 0, 1]

mip_png = os.path.join(OUTC, "qc_mip_red_blue.png")
plt.figure(figsize=(7,7))
plt.imshow(rgb)
plt.axis("off")
plt.title("MIP overlay: arteries red / veins blue")
plt.tight_layout()
plt.savefig(mip_png, dpi=200)
plt.close()

# Static 3D PNG (voxel-y in z due to 2.5mm slice thickness)
def add_mesh(ax, mask, face_rgba, alpha):
    if mask.sum() < 50:
        return
    verts, faces, _, _ = marching_cubes(mask.astype(np.uint8), level=0.5, spacing=spacing)
    poly = Poly3DCollection(verts[faces], alpha=alpha)
    poly.set_facecolor(face_rgba)
    poly.set_edgecolor((0,0,0,0))
    ax.add_collection3d(poly)

png3d = os.path.join(OUTC, "qc_3d_red_blue.png")
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection="3d")
ax.set_axis_off()
add_mesh(ax, maskV, (0.1,0.3,0.95,0.30), 0.30)
add_mesh(ax, maskA, (0.95,0.2,0.2,0.65), 0.65)

# auto-fit limits
pts = []
for m in (maskA, maskV):
    if m.any():
        p = np.argwhere(m)
        pts.append(p * spacing[None,:])
if pts:
    pts = np.vstack(pts)
    mn, mx = pts.min(0), pts.max(0)
    c = (mn + mx)/2
    span = (mx - mn).max()/2
    ax.set_xlim(c[0]-span, c[0]+span)
    ax.set_ylim(c[1]-span, c[1]+span)
    ax.set_zlim(c[2]-span, c[2]+span)

ax.view_init(elev=18, azim=35)
plt.tight_layout()
plt.savefig(png3d, dpi=200)
plt.close(fig)

def mm3(mask): return float(mask.sum()) * voxvol
with open(os.path.join(OUTC, "summary.txt"), "w") as f:
    f.write(f"Spacing (mm): {spacing.tolist()}\n")
    f.write(f"EDGE_ERODE_MM={EDGE_ERODE_MM} OPEN_MM={OPEN_MM} CLOSE_MM={CLOSE_MM}\n")
    f.write(f"MINVOL_A_MM3={MINVOL_A_MM3} MINVOL_V_MM3={MINVOL_V_MM3}\n")
    f.write(f"KEEP_N_A={KEEP_N_A} KEEP_N_V={KEEP_N_V}\n")
    f.write(f"Used venous masks: {usedV}\n")
    f.write(f"Used arterial masks: {usedA}\n")
    f.write(f"Arteries raw mm3: {mm3(maskA_raw):.1f}, clean mm3: {mm3(maskA):.1f}\n")
    f.write(f"Veins raw mm3: {mm3(maskV_raw):.1f}, clean mm3: {mm3(maskV):.1f}\n")
    f.write(f"QC: {mip_png}\nQC: {png3d}\n")

print("Wrote:", os.path.join(OUTC, "arteries_clean.nii.gz"))
print("Wrote:", os.path.join(OUTC, "veins_clean.nii.gz"))
print("QC:", mip_png)
print("QC:", png3d)
PY

  echo "[4/4] Interactive HTML (Plotly)"
  export HTML_MAX_FACES HTML_SMOOTH

  "$PY" - <<'PY'
import os
import numpy as np
import nibabel as nib
import scipy.ndimage as ndi
from skimage.measure import marching_cubes
import plotly.graph_objects as go

OUTC = os.environ["OUTC"]
ART = os.path.join(OUTC, "arteries_clean.nii.gz")
VEN = os.path.join(OUTC, "veins_clean.nii.gz")

out_html = os.path.join(OUTC, "qc_3d_red_blue.html")

max_faces = int(os.environ.get("HTML_MAX_FACES", "200000"))
smooth = tuple(map(float, os.environ.get("HTML_SMOOTH", "1.0 1.0 0.6").split()))

def mesh_from_mask(mask, affine, smooth_sigma, max_faces=200000, seed=0):
    if mask is None or mask.sum() < 50:
        return None

    vol = mask.astype(np.float32)

    # Visualization-only smoothing to reduce voxel blockiness
    if smooth_sigma is not None:
        sx, sy, sz = smooth_sigma
        if sx > 0 or sy > 0 or sz > 0:
            vol = ndi.gaussian_filter(vol, sigma=(sx, sy, sz))

    verts, faces, _, _ = marching_cubes(vol, level=0.5)

    # Cap faces for speed + smaller HTML
    if faces.shape[0] > max_faces:
        rng = np.random.default_rng(seed)
        idx = rng.choice(faces.shape[0], size=max_faces, replace=False)
        faces = faces[idx]
        used = np.unique(faces.ravel())
        remap = -np.ones((verts.shape[0],), dtype=np.int64)
        remap[used] = np.arange(used.size, dtype=np.int64)
        verts = verts[used]
        faces = remap[faces]

    verts_world = nib.affines.apply_affine(affine, verts)
    return verts_world, faces

a_img = nib.load(ART)
v_img = nib.load(VEN)
A = (a_img.get_fdata() > 0.5)
V = (v_img.get_fdata() > 0.5)

a_mesh = mesh_from_mask(A, a_img.affine, smooth, max_faces=max_faces, seed=1)
v_mesh = mesh_from_mask(V, v_img.affine, smooth, max_faces=max_faces, seed=2)

fig = go.Figure()

# Veins (blue)
if v_mesh is not None:
    verts, faces = v_mesh
    fig.add_trace(go.Mesh3d(
        x=verts[:,0], y=verts[:,1], z=verts[:,2],
        i=faces[:,0], j=faces[:,1], k=faces[:,2],
        color="blue", opacity=0.25, name="Veins",
        lighting=dict(ambient=0.35, diffuse=0.6, specular=0.15, roughness=0.8),
        flatshading=False
    ))

# Arteries (red)
if a_mesh is not None:
    verts, faces = a_mesh
    fig.add_trace(go.Mesh3d(
        x=verts[:,0], y=verts[:,1], z=verts[:,2],
        i=faces[:,0], j=faces[:,1], k=faces[:,2],
        color="red", opacity=0.65, name="Arteries",
        lighting=dict(ambient=0.35, diffuse=0.7, specular=0.2, roughness=0.7),
        flatshading=False
    ))

fig.update_layout(
    title="Liver vessels (interactive): arteries (red) / veins (blue)",
    scene=dict(
        aspectmode="data",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        zaxis=dict(visible=False),
        bgcolor="white",
    ),
    legend=dict(itemsizing="constant"),
    margin=dict(l=0, r=0, t=40, b=0),
)

fig.write_html(out_html, include_plotlyjs="cdn", full_html=True, auto_open=False)
print("Wrote:", out_html)
PY

  if [[ "$OPEN_PREVIEW" == "1" ]]; then
    xdg-open "$OUTC/qc_mip_red_blue.png" 2>/dev/null || open "$OUTC/qc_mip_red_blue.png" 2>/dev/null || true
    xdg-open "$OUTC/qc_3d_red_blue.png" 2>/dev/null || open "$OUTC/qc_3d_red_blue.png" 2>/dev/null || true
    xdg-open "$OUTC/qc_3d_red_blue.html" 2>/dev/null || open "$OUTC/qc_3d_red_blue.html" 2>/dev/null || true
  fi

  echo "[DONE] $(basename "$CASE") -> $OUTC"
done

echo
echo "All done."
echo "If you publish with TotalSegmentator, cite: https://pubs.rsna.org/doi/10.1148/ryai.230024"

