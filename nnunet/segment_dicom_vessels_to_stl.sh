#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./segment_dicom_vessels_to_stl.sh /path/to/dicom_folder /path/to/output_dir [liver_mask.nii.gz]
#
# If you have a liver mask, pass it as the third argument to use it for cropping.
# Output:
#   output_dir/totalsegmentator/   -> TotalSegmentator raw nifti output
#   output_dir/combined_mask.nii.gz -> unioned vessel mask
#   output_dir/vessels.stl         -> exported STL mesh
#   output_dir/summary.txt         -> summary of results

ENV_PREFIX="${ENV_PREFIX:-$HOME/miniconda3/envs/sam2}"
TS="${TS:-$ENV_PREFIX/bin/TotalSegmentator}"
PY="${PY:-$ENV_PREFIX/bin/python}"

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <dicom_folder> <output_dir> [liver_mask.nii.gz]"
  exit 1
fi

DICOM_INPUT="$1"
OUTROOT="$2"
MASK_PATH="${3:-}"  # optional liver mask

DEVICE="${DEVICE:-auto}"
NR="${NR:-4}"
NS="${NS:-4}"

if [[ ! -x "$TS" ]]; then
  echo "[ERROR] TotalSegmentator not found or not executable: $TS"
  exit 1
fi

if [[ ! -x "$PY" ]]; then
  echo "[ERROR] Python not found or not executable: $PY"
  exit 1
fi

# Auto-detect GPU support unless DEVICE is explicitly set
if [[ "$DEVICE" == "auto" ]]; then
  echo "[INFO] Auto-detecting device..."
  if "$PY" -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null | grep -q True; then
    DEVICE=gpu
  else
    DEVICE=cpu
  fi
fi

if [[ "$DEVICE" == "gpu" ]]; then
  if ! "$PY" -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null | grep -q True; then
    echo "[WARN] GPU requested but torch.cuda is not available; falling back to cpu."
    DEVICE=cpu
  fi
fi

echo "[INFO] Using device: $DEVICE"

echo "[INFO] Checking TotalSegmentator binary..."

mkdir -p "$OUTROOT"
TS_OUT="$OUTROOT/totalsegmentator"
mkdir -p "$TS_OUT"

echo "[INFO] Input DICOM folder: $DICOM_INPUT"
echo "[INFO] Output root: $OUTROOT"
echo "[INFO] Totalsegmentator output: $TS_OUT"
if [[ -n "$MASK_PATH" ]]; then
  echo "[INFO] Liver crop mask: $MASK_PATH"
fi

echo "[INFO] Running TotalSegmentator..."
"$TS" -i "$DICOM_INPUT" -o "$TS_OUT" -ot nifti -ta liver_vessels -d "$DEVICE" -nr "$NR" -ns "$NS" ${MASK_PATH:+-cp "$MASK_PATH"}

echo "[INFO] Converting masks to union and STL..."
export OUTROOT DICOM_INPUT
"$PY" - <<'PY'
import glob
import os
import numpy as np
import nibabel as nib
from skimage.measure import marching_cubes

out_root = os.environ['OUTROOT']
output_dir = out_root
mask_files = sorted(glob.glob(os.path.join(output_dir, 'totalsegmentator', '*.nii')) +
                    glob.glob(os.path.join(output_dir, 'totalsegmentator', '*.nii.gz')))
if not mask_files:
    raise SystemExit(f'No NIfTI output found in {os.path.join(output_dir, "totalsegmentator")}')

print('Found output masks:')
for f in mask_files:
    print('  ', os.path.basename(f))

# Union all masks into one vessel mask
ref_img = nib.load(mask_files[0])
shape = ref_img.shape
spacing = np.array(ref_img.header.get_zooms()[:3], dtype=float)
affine = ref_img.affine
union_mask = np.zeros(shape, dtype=bool)

for p in mask_files:
    try:
        img = nib.load(p)
        data = img.get_fdata()
        if tuple(img.shape) != tuple(shape):
            print(f'WARNING: skipping {os.path.basename(p)} with mismatched shape {img.shape}')
            continue
        union_mask |= (data > 0.5)
    except Exception as e:
        print(f'WARNING: failed to load {p}: {e}')

combined_path = os.path.join(output_dir, 'combined_mask.nii.gz')
combined_img = nib.Nifti1Image(union_mask.astype(np.uint8), affine)
combined_img.set_data_dtype(np.uint8)
nib.save(combined_img, combined_path)
print('Saved combined vessel mask to', combined_path)

if not union_mask.any():
    raise SystemExit('Combined vessel mask is empty; no STL will be created.')

print('Running marching cubes...')
verts, faces, normals, values = marching_cubes(union_mask.astype(np.uint8), level=0.5, spacing=spacing)

# Convert to world coordinates using affine transform if needed
# marching_cubes with spacing already scales voxel coordinates, but affine may include origin/orientation.
verts_world = nib.affines.apply_affine(affine, verts)

stl_path = os.path.join(output_dir, 'vessels.stl')

# Compute normals if not provided
if normals is None or normals.shape[0] != faces.shape[0]:
    v0 = verts_world[faces[:, 1]] - verts_world[faces[:, 0]]
    v1 = verts_world[faces[:, 2]] - verts_world[faces[:, 0]]
    normals = np.cross(v0, v1)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normals = normals / norms

with open(stl_path, 'w') as f:
    f.write('solid vessels\n')
    for i, face in enumerate(faces):
        n = normals[i]
        f.write(f'  facet normal {n[0]:.6e} {n[1]:.6e} {n[2]:.6e}\n')
        f.write('    outer loop\n')
        for vid in face:
            v = verts_world[vid]
            f.write(f'      vertex {v[0]:.6e} {v[1]:.6e} {v[2]:.6e}\n')
        f.write('    endloop\n')
        f.write('  endfacet\n')
    f.write('endsolid vessels\n')

print('Saved STL to', stl_path)

summary_path = os.path.join(output_dir, 'summary.txt')
with open(summary_path, 'w') as summary_file:
    summary_file.write(f'Input DICOM: {os.path.abspath(os.environ["DICOM_INPUT"])}\n')
    summary_file.write(f'TotalSegmentator output dir: {os.path.abspath(os.path.join(output_dir, "totalsegmentator"))}\n')
    summary_file.write(f'Combined mask: {combined_path}\n')
    summary_file.write(f'STL: {stl_path}\n')
    summary_file.write(f'Found {len(mask_files)} NIfTI files in TotalSegmentator output\n')
    summary_file.write(f'Total vessel voxels: {int(union_mask.sum())}\n')
    summary_file.write(f'Voxel spacing: {spacing.tolist()}\n')
print('Saved summary to', summary_path)
PY

echo "[DONE] 3D mesh exported to $OUTROOT/vessels.stl"

echo "[NOTE] If the output is not good, you may need a liver mask or better phase selection."
