#!/usr/bin/env python3
"""
Micro-CT liver vessel segmentation using multiscale Frangi vesselness filtering.
Designed for large volumes (e.g. 1304x1304x1495) with ~24 µm isotropic voxel spacing.
No training data required.

Input  : DICOM folder  hybrid_ultra/          (hardcoded, relative to script location)
Output : vessels.stl + vesselness_map.nii.gz  (written to the current working directory)

Usage:
    python segment_vessels_frangi.py \
        [--sigma_min 0.05] \
        [--sigma_max 0.5] \
        [--num_scales 6] \
        [--threshold 0.05] \
        [--tile_size 256] \
        [--overlap 32] \
        [--device cuda]

Dependencies:
    pip install nibabel numpy scipy scikit-image pydicom
    pip install torch  # optional, for GPU-accelerated Hessian via torch
"""

import argparse
import os
import sys
import time
import warnings
import numpy as np
import nibabel as nib
from pathlib import Path
from skimage.measure import marching_cubes

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# Hardcoded I/O paths
# ─────────────────────────────────────────────────────────────

DICOM_INPUT_FOLDER = Path("../data/hybrid_ultra")   # relative to cwd
OUTPUT_DIR         = Path("./output")              # current working directory


# ─────────────────────────────────────────────────────────────
# DICOM loader
# ─────────────────────────────────────────────────────────────

def load_dicom_series(folder: Path):
    """
    Load a DICOM series from a folder into a (Z, Y, X) float32 volume.
    Returns (volume, affine, mean_spacing_mm).
    Slices are sorted by ImagePositionPatient[2], falling back to InstanceNumber.
    """
    try:
        import pydicom
    except ImportError:
        sys.exit("[ERROR] pydicom not found.  Install with:  pip install pydicom")

    candidates = list(folder.glob("*.dcm")) + list(folder.glob("*.DCM"))
    if not candidates:
        candidates = [p for p in folder.iterdir() if p.is_file()]
    if not candidates:
        sys.exit(f"[ERROR] No files found in {folder}")

    total = len(candidates)
    print(f"[INFO] Reading {total} files from {folder} ...", flush=True)
    slices = []
    t0 = time.time()
    for i, p in enumerate(sorted(candidates), 1):
        try:
            slices.append(pydicom.dcmread(str(p)))
        except Exception as e:
            print(f"  [WARN] Skipping {p.name}: {e}", flush=True)
        if i % 100 == 0 or i == total:
            elapsed = time.time() - t0
            eta     = elapsed / i * (total - i)
            print(f"  [DICOM] {i}/{total} files read  "
                  f"elapsed {elapsed:.0f}s  ETA {eta:.0f}s",
                  flush=True)

    if not slices:
        sys.exit("[ERROR] Could not read any DICOM slices.")

    def _zpos(ds):
        try:
            return float(ds.ImagePositionPatient[2])
        except Exception:
            try:
                return float(ds.InstanceNumber)
            except Exception:
                return 0.0

    slices.sort(key=_zpos)

    ds0 = slices[0]
    try:
        row_sp, col_sp = float(ds0.PixelSpacing[0]), float(ds0.PixelSpacing[1])
    except Exception:
        print("[WARN] PixelSpacing missing — assuming 0.0244 mm", flush=True)
        row_sp = col_sp = 0.0244

    if len(slices) > 1:
        try:
            slice_th = abs(float(slices[1].ImagePositionPatient[2]) -
                           float(slices[0].ImagePositionPatient[2]))
        except Exception:
            slice_th = float(getattr(ds0, "SliceThickness", row_sp))
    else:
        slice_th = float(getattr(ds0, "SliceThickness", row_sp))

    spacing_zyx = (slice_th, row_sp, col_sp)
    print(f"[INFO] Spacing Z={slice_th:.4f}  Y={row_sp:.4f}  X={col_sp:.4f} mm",
          flush=True)

    print("[INFO] Stacking pixel arrays into volume ...", flush=True)
    t0 = time.time()
    frames = []
    for i, ds in enumerate(slices, 1):
        arr = ds.pixel_array.astype(np.float32)
        arr = arr * float(getattr(ds, "RescaleSlope", 1.0)) \
                  + float(getattr(ds, "RescaleIntercept", 0.0))
        frames.append(arr)
        if i % 200 == 0 or i == total:
            elapsed = time.time() - t0
            eta     = elapsed / i * (total - i)
            print(f"  [STACK] {i}/{total} slices  "
                  f"elapsed {elapsed:.0f}s  ETA {eta:.0f}s",
                  flush=True)
    volume = np.stack(frames, axis=0)
    print(f"[INFO] Volume shape (Z,Y,X): {volume.shape}", flush=True)

    try:
        origin = [float(x) for x in ds0.ImagePositionPatient]
    except Exception:
        origin = [0.0, 0.0, 0.0]
    # Affine maps marching_cubes vertex order (Z, Y, X) → world (mm).
    # marching_cubes returns verts as (dim0=Z, dim1=Y, dim2=X) so
    # diagonal must be [slice_th, row_sp, col_sp] not [col_sp, row_sp, slice_th].
    affine = np.diag([slice_th, row_sp, col_sp, 1.0]).astype(np.float64)
    affine[:3, 3] = origin

    return volume, affine, float(np.mean(spacing_zyx))


# ─────────────────────────────────────────────────────────────
# Hessian + Frangi core (NumPy / SciPy — CPU)
# ─────────────────────────────────────────────────────────────

def _gaussian_hessian_cpu(volume, sigma):
    """Return the 6 upper-triangle Hessian components at scale sigma (CPU)."""
    from scipy.ndimage import gaussian_filter

    # Smooth once at this scale
    smoothed = gaussian_filter(volume.astype(np.float32), sigma=sigma)

    # Compute second derivatives via finite differences on the smoothed volume
    # (equivalent to convolving with second-derivative-of-Gaussian at scale sigma)
    Hxx = np.gradient(np.gradient(smoothed, axis=0), axis=0)
    Hyy = np.gradient(np.gradient(smoothed, axis=1), axis=1)
    Hzz = np.gradient(np.gradient(smoothed, axis=2), axis=2)
    Hxy = np.gradient(np.gradient(smoothed, axis=0), axis=1)
    Hxz = np.gradient(np.gradient(smoothed, axis=0), axis=2)
    Hyz = np.gradient(np.gradient(smoothed, axis=1), axis=2)

    # Scale normalisation: multiply by sigma^2 (Lindeberg's scale-normalisation)
    scale = sigma ** 2
    return Hxx * scale, Hyy * scale, Hzz * scale, \
           Hxy * scale, Hxz * scale, Hyz * scale


def _frangi_from_hessian(Hxx, Hyy, Hzz, Hxy, Hxz, Hyz,
                          alpha=0.5, beta=0.5, gamma=None):
    """
    Compute Frangi vesselness from Hessian components.
    Returns a float32 array in [0, 1].

    alpha: sensitivity to plate vs. tube  (default 0.5)
    beta:  sensitivity to blob vs. tube   (default 0.5)
    gamma: noise suppression              (default: half max Frobenius norm)
    """
    # Eigenvalues of symmetric 3x3 matrix at each voxel via characteristic poly
    # Using numpy's batch eigh is RAM-heavy for large volumes; we use analytical
    # approximation via sorted eigenvalue formulas for 3×3 symmetric matrices.

    # Pack into (..., 3, 3) — too RAM-intensive for full 1304^2 × 1495.
    # Instead compute eigenvalue statistics directly from matrix invariants.

    # Trace = sum of eigenvalues
    trace = Hxx + Hyy + Hzz

    # Frobenius norm squared
    frob2 = Hxx**2 + Hyy**2 + Hzz**2 + 2*(Hxy**2 + Hxz**2 + Hyz**2)

    # For vessels: we want eigenvalues sorted |λ1| ≤ |λ2| ≤ |λ3|
    # with λ2, λ3 large negative (dark tube on bright bg) or large positive.
    # We use a memory-efficient approximation: compute full eigenvalues per tile.

    # Stack into (N, 3, 3) just for the tile — caller is expected to call this
    # on tiles, not the full volume.
    N = Hxx.size
    M = np.zeros((N, 3, 3), dtype=np.float32)
    M[:, 0, 0] = Hxx.ravel()
    M[:, 1, 1] = Hyy.ravel()
    M[:, 2, 2] = Hzz.ravel()
    M[:, 0, 1] = M[:, 1, 0] = Hxy.ravel()
    M[:, 0, 2] = M[:, 2, 0] = Hxz.ravel()
    M[:, 1, 2] = M[:, 2, 1] = Hyz.ravel()

    eigvals = np.linalg.eigvalsh(M)  # shape (N, 3), ascending order
    # Sort by absolute value
    idx = np.argsort(np.abs(eigvals), axis=1)
    eigvals = np.take_along_axis(eigvals, idx, axis=1)

    l1 = eigvals[:, 0].reshape(Hxx.shape)
    l2 = eigvals[:, 1].reshape(Hxx.shape)
    l3 = eigvals[:, 2].reshape(Hxx.shape)

    del M, eigvals, idx

    # Frangi ratios
    # RA: plate vs tube
    Ra = np.abs(l2) / (np.abs(l3) + 1e-10)
    # RB: blob vs tube
    Rb = np.abs(l1) / (np.sqrt(np.abs(l2 * l3)) + 1e-10)
    # S: second-order structureness (Frobenius norm)
    S = np.sqrt(l1**2 + l2**2 + l3**2)

    if gamma is None:
        gamma = 0.5 * S.max()
    if gamma == 0:
        gamma = 1.0

    vesselness = (
        (1 - np.exp(-(Ra**2) / (2 * alpha**2))) *
        np.exp(-(Rb**2) / (2 * beta**2)) *
        (1 - np.exp(-(S**2) / (2 * gamma**2)))
    ).astype(np.float32)

    # Polarity enforcement:
    # Bright tube on dark background (contrast-enhanced micro-CT):
    #   l2 > 0 and l3 > 0  (intensity curves upward in cross-section)
    # Dark tube on bright background (standard clinical CT):
    #   l2 < 0 and l3 < 0
    # Suppress the wrong polarity to avoid noise from the other type.
    vesselness[l2 < 0] = 0   # suppress dark-tube responses (wrong polarity)
    vesselness[l3 < 0] = 0   # keep only bright-tube responses

    return vesselness


# ─────────────────────────────────────────────────────────────
# GPU-accelerated Hessian (PyTorch)
# ─────────────────────────────────────────────────────────────

def _gaussian_hessian_gpu(tile_np, sigma, device):
    """GPU Hessian using torch conv3d with Gaussian derivative kernels."""
    import torch
    import torch.nn.functional as F

    def gauss_kernel_1d(sigma, truncate=4.0):
        radius = int(truncate * sigma + 0.5)
        x = np.arange(-radius, radius + 1, dtype=np.float32)
        k = np.exp(-0.5 * (x / sigma) ** 2)
        k /= k.sum()
        return torch.tensor(k.astype(np.float32), device=device)

    def gauss_deriv2_kernel_1d(sigma, truncate=4.0):
        radius = int(truncate * sigma + 0.5)
        x = np.arange(-radius, radius + 1, dtype=np.float32)
        g = np.exp(-0.5 * (x / sigma) ** 2)
        g /= g.sum()
        g2 = (x**2 / sigma**4 - 1.0 / sigma**2) * g
        return torch.tensor(g2.astype(np.float32), device=device)

    t = torch.tensor(tile_np[np.newaxis, np.newaxis], dtype=torch.float32, device=device)

    def conv_sep(vol, kz, ky, kx):
        """Apply 3 separable 1D convolutions."""
        # z
        pad = kz.shape[0] // 2
        v = F.pad(vol, [0, 0, 0, 0, pad, pad], mode='replicate')
        v = F.conv3d(v, kz.view(1, 1, -1, 1, 1))
        # y
        pad = ky.shape[0] // 2
        v = F.pad(v, [0, 0, pad, pad, 0, 0], mode='replicate')
        v = F.conv3d(v, ky.view(1, 1, 1, -1, 1))
        # x
        pad = kx.shape[0] // 2
        v = F.pad(v, [pad, pad, 0, 0, 0, 0], mode='replicate')
        v = F.conv3d(v, kx.view(1, 1, 1, 1, -1))
        return v

    g = gauss_kernel_1d(sigma)
    d2 = gauss_deriv2_kernel_1d(sigma)
    scale = sigma ** 2

    Hzz = conv_sep(t, d2, g, g).squeeze().cpu().numpy() * scale
    Hyy = conv_sep(t, g, d2, g).squeeze().cpu().numpy() * scale
    Hxx = conv_sep(t, g, g, d2).squeeze().cpu().numpy() * scale

    # Mixed: apply deriv once along each axis
    def gauss_deriv1_kernel_1d(sigma, truncate=4.0):
        radius = int(truncate * sigma + 0.5)
        x = np.arange(-radius, radius + 1, dtype=np.float32)
        g = np.exp(-0.5 * (x / sigma) ** 2)
        g /= g.sum()
        d1 = (-x / sigma**2) * g
        return torch.tensor(d1.astype(np.float32), device=device)

    d1 = gauss_deriv1_kernel_1d(sigma)

    Hzy = conv_sep(t, d1, d1, g).squeeze().cpu().numpy() * scale
    Hzx = conv_sep(t, d1, g, d1).squeeze().cpu().numpy() * scale
    Hyx = conv_sep(t, g, d1, d1).squeeze().cpu().numpy() * scale

    del t
    torch.cuda.empty_cache()

    return Hxx, Hyy, Hzz, Hyx, Hzx, Hzy  # matches (Hxx,Hyy,Hzz,Hxy,Hxz,Hyz)


# ─────────────────────────────────────────────────────────────
# Tiled multi-scale Frangi
# ─────────────────────────────────────────────────────────────

def multiscale_frangi_tiled(volume, spacing_mm,
                             sigma_min_mm=0.05, sigma_max_mm=0.5,
                             num_scales=6, alpha=0.5, beta=0.5,
                             tile_size=256, overlap=32,
                             device='cpu'):
    """
    Run multiscale Frangi vesselness on a large 3D volume using tiled processing.

    Parameters
    ----------
    volume       : np.ndarray float32, shape (Z, Y, X) — or any axis order
    spacing_mm   : float, isotropic voxel spacing in mm
    sigma_min_mm : minimum vessel radius to detect (mm)
    sigma_max_mm : maximum vessel radius to detect (mm)
    num_scales   : number of log-spaced sigma values
    tile_size    : spatial size of each tile (voxels)
    overlap      : overlap between adjacent tiles (voxels) to avoid edge artifacts
    device       : 'cpu' or 'cuda' (or 'cuda:0' etc.)

    Returns
    -------
    vesselness : np.ndarray float32, same shape as volume, values in [0, 1]
    """
    use_gpu = device.startswith('cuda')
    if use_gpu:
        try:
            import torch
            if not torch.cuda.is_available():
                print("[WARN] CUDA not available, falling back to CPU.")
                use_gpu = False
                device = 'cpu'
        except ImportError:
            print("[WARN] PyTorch not installed, falling back to CPU.")
            use_gpu = False
            device = 'cpu'

    # Convert sigma from mm to voxels
    sigma_min_vox = sigma_min_mm / spacing_mm
    sigma_max_vox = sigma_max_mm / spacing_mm
    sigmas = np.logspace(np.log10(sigma_min_vox),
                         np.log10(sigma_max_vox),
                         num=num_scales)
    print(f"[INFO] Sigma range: {sigma_min_mm:.3f}–{sigma_max_mm:.3f} mm "
          f"({sigma_min_vox:.1f}–{sigma_max_vox:.1f} voxels), {num_scales} scales")
    print(f"[INFO] Sigmas (voxels): {np.round(sigmas, 2).tolist()}")

    Z, Y, X = volume.shape
    out = np.zeros(volume.shape, dtype=np.float32)

    # Tile coordinates (z, y, x)
    def tile_ranges(size, total, ov):
        starts = list(range(0, total, size - ov))
        if starts[-1] + size > total:
            starts[-1] = max(0, total - size)
        return starts

    z_starts = tile_ranges(tile_size, Z, overlap)
    y_starts = tile_ranges(tile_size, Y, overlap)
    x_starts = tile_ranges(tile_size, X, overlap)
    total_tiles = len(z_starts) * len(y_starts) * len(x_starts)
    print(f"[INFO] Volume: {Z}×{Y}×{X}, tile: {tile_size}³, overlap: {overlap}")
    print(f"[INFO] Total tiles: {total_tiles} × {num_scales} scales")

    t0 = time.time()
    tile_count = 0

    for z0 in z_starts:
        z1 = min(z0 + tile_size, Z)
        for y0 in y_starts:
            y1 = min(y0 + tile_size, Y)
            for x0 in x_starts:
                x1 = min(x0 + tile_size, X)
                tile = volume[z0:z1, y0:y1, x0:x1].astype(np.float32)
                vessel_tile = np.zeros_like(tile)

                for sigma in sigmas:
                    if use_gpu:
                        try:
                            Hxx, Hyy, Hzz, Hxy, Hxz, Hyz = \
                                _gaussian_hessian_gpu(tile, sigma, device)
                        except Exception as e:
                            print(f"[WARN] GPU Hessian failed ({e}), using CPU for this tile.")
                            Hxx, Hyy, Hzz, Hxy, Hxz, Hyz = \
                                _gaussian_hessian_cpu(tile, sigma)
                    else:
                        Hxx, Hyy, Hzz, Hxy, Hxz, Hyz = \
                            _gaussian_hessian_cpu(tile, sigma)

                    v = _frangi_from_hessian(Hxx, Hyy, Hzz, Hxy, Hxz, Hyz,
                                             alpha=alpha, beta=beta)
                    vessel_tile = np.maximum(vessel_tile, v)
                    del Hxx, Hyy, Hzz, Hxy, Hxz, Hyz, v

                # Blend overlap region to avoid seams
                iz0 = overlap // 2 if z0 > 0 else 0
                iy0 = overlap // 2 if y0 > 0 else 0
                ix0 = overlap // 2 if x0 > 0 else 0
                iz1 = (z1 - z0) - overlap // 2 if z1 < Z else (z1 - z0)
                iy1 = (y1 - y0) - overlap // 2 if y1 < Y else (y1 - y0)
                ix1 = (x1 - x0) - overlap // 2 if x1 < X else (x1 - x0)

                out[z0 + iz0: z0 + iz1,
                    y0 + iy0: y0 + iy1,
                    x0 + ix0: x0 + ix1] = vessel_tile[iz0:iz1, iy0:iy1, ix0:ix1]

                tile_count += 1
                elapsed = time.time() - t0
                eta = elapsed / tile_count * (total_tiles - tile_count)
                print(f"\r[INFO] Tile {tile_count}/{total_tiles} | "
                      f"elapsed {elapsed:.0f}s | ETA {eta:.0f}s    ", end='', flush=True)

    print()
    return out


# ─────────────────────────────────────────────────────────────
# STL export — chunked along Z to avoid full-volume RAM spike
# ─────────────────────────────────────────────────────────────

import struct as _struct

def _face_normals(verts, faces):
    v0 = verts[faces[:, 1]] - verts[faces[:, 0]]
    v1 = verts[faces[:, 2]] - verts[faces[:, 0]]
    n  = np.cross(v0, v1)
    norms = np.linalg.norm(n, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return n / norms


def _write_facets_binary(f, verts_world, faces, normals):
    """
    Write faces as binary STL records directly into an open binary file.
    Each record: 12 bytes normal + 36 bytes vertices + 2 bytes attr = 50 bytes.
    Uses a pre-allocated buffer per slab for speed.
    """
    n_faces = len(faces)
    buf = bytearray(n_faces * 50)
    off = 0
    for i, face in enumerate(faces):
        nx, ny, nz = normals[i]
        _struct.pack_into("<fff", buf, off, nx, ny, nz);      off += 12
        for vid in face:
            x, y, z = verts_world[vid]
            _struct.pack_into("<fff", buf, off, x, y, z);     off += 12
        _struct.pack_into("<H",   buf, off, 0);                off += 2
    f.write(buf)


# Binary STL header size + per-face size
_BINARY_HEADER = 80
_BINARY_FACE_BYTES = 50

# ─────────────────────────────────────────────────────────────
# CUDA kernel for GPU-accelerated binary STL loading
# ─────────────────────────────────────────────────────────────

_CUDA_READ_FACES = r"""
extern "C" __global__
void read_binary_stl_faces(
        const unsigned char* __restrict__ data,
        long long n_faces,
        float*    __restrict__ out)
{
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_faces) return;
    const unsigned char* rec = data + tid * 50LL + 12LL; // skip normal bytes
    float* dst = out + tid * 9LL;
    #pragma unroll
    for (int i = 0; i < 9; i++) {
        unsigned int bits = (unsigned int)rec[i*4]
                          | (unsigned int)rec[i*4+1] << 8
                          | (unsigned int)rec[i*4+2] << 16
                          | (unsigned int)rec[i*4+3] << 24;
        dst[i] = __uint_as_float(bits);
    }
}
"""


def mask_to_stl(mask, affine, spacing, stl_path, chunk_z=128):
    """
    Chunked marching cubes → binary STL.

    Binary STL is ~6x smaller than ASCII and loads in seconds not hours.
    Processes chunk_z Z-slices at a time with 1-voxel overlap so the
    isosurface is never cut at slab boundaries.
    Reduce chunk_z (e.g. 64) if still OOM.

    File layout:
        80-byte header  (blank)
        4-byte uint32   (total face count — written last via seek)
        N × 50-byte face records streamed slab by slab
    """
    import os
    Z = mask.shape[0]
    total_faces = 0
    # Pass spacing=(1,1,1) so marching_cubes returns verts in voxel coords.
    # The affine (diagonal=[slice_th, row_sp, col_sp]) then converts to mm.
    # This avoids the mismatch where z_lo (voxels) is added to verts already
    # scaled to mm by the spacing parameter.
    n_slabs = int(np.ceil(Z / chunk_z))

    print(f"[INFO] Chunked marching cubes -> binary STL: "
          f"Z={Z}, chunk_z={chunk_z}, ~{n_slabs} slabs ...", flush=True)

    with open(stl_path, "wb") as f:
        # Write placeholder header + face count (updated at the end)
        f.write(b"\x00" * _BINARY_HEADER)
        f.write(_struct.pack("<I", 0))   # placeholder — overwritten below

        for z0 in range(0, Z, chunk_z):
            z_lo = max(z0 - 1, 0)
            z_hi = min(z0 + chunk_z + 1, Z)
            slab = mask[z_lo:z_hi].astype(np.uint8)

            if not slab.any():
                continue

            try:
                verts, faces, _, _ = marching_cubes(
                    slab, level=0.5, spacing=(1.0, 1.0, 1.0))  # voxel coords
            except (ValueError, RuntimeError):
                continue

            if len(faces) == 0:
                continue

            verts[:, 0] += z_lo   # shift Z from slab-local to global voxel coords
            verts_world   = nib.affines.apply_affine(affine, verts)
            normals_world = _face_normals(verts_world, faces)

            _write_facets_binary(f, verts_world, faces, normals_world)
            total_faces += len(faces)

            size_mb = os.path.getsize(stl_path) / 1e6
            print(f"  slab z={z_lo}-{z_hi-1}  "
                  f"cumulative faces: {total_faces:,}  "
                  f"file: {size_mb:.1f} MB",
                  flush=True)

            del slab, verts, faces, verts_world, normals_world

        # Seek back and write the real face count into the header
        f.seek(_BINARY_HEADER)
        f.write(_struct.pack("<I", total_faces))

    final_mb = os.path.getsize(stl_path) / 1e6
    print(f"[INFO] Saved binary STL → {stl_path}  "
          f"({total_faces:,} faces, {final_mb:.1f} MB)", flush=True)


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def _in_jupyter() -> bool:
    try:
        return get_ipython().__class__.__name__ == "ZMQInteractiveShell"
    except NameError:
        return False


def export_html(stl_path: Path, html_path: Path, max_faces: int = 2_000_000):
    """
    Load binary STL on GPU, decimate, render with k3d, export HTML.
    Works as plain script (saves HTML) or inside Jupyter (displays widget).
    """
    try:
        import cupy as cp
        import k3d
    except ImportError as e:
        print(f"[WARN] HTML export skipped — missing package: {e}")
        print("       Install with: pip install cupy-cuda12x k3d")
        return

    cp.cuda.set_allocator(None)

    if not stl_path.exists():
        print(f"[WARN] STL not found, skipping HTML export: {stl_path}")
        return

    file_size = stl_path.stat().st_size
    with open(stl_path, "rb") as f:
        f.seek(80)
        import struct as _s
        n_faces = _s.unpack("<I", f.read(4))[0]

    HEADER     = 84
    FACE_BYTES = 50

    free_vram  = cp.cuda.runtime.memGetInfo()[0]
    need       = n_faces * (FACE_BYTES + 36)
    dev_name   = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
    print(f"[HTML] GPU        : {dev_name}", flush=True)
    print(f"[HTML] STL        : {file_size/1e6:.0f} MB  |  {n_faces:,} faces",
          flush=True)

    mod    = cp.RawModule(code=_CUDA_READ_FACES)
    kernel = mod.get_function("read_binary_stl_faces")
    BLOCK  = 256

    # Process in chunks if VRAM is tight
    chunk   = min(n_faces, int(free_vram * 0.40 / (FACE_BYTES + 36)))
    n_ch    = int(np.ceil(n_faces / chunk))
    all_v   = []

    print(f"[HTML] Loading {n_faces:,} faces in {n_ch} chunk(s) ...",
          flush=True)

    with open(stl_path, "rb") as f:
        f.seek(HEADER)
        for ci in range(n_ch):
            n   = min(chunk, n_faces - ci * chunk)
            raw = f.read(n * FACE_BYTES)
            gpu_data = cp.frombuffer(raw, dtype=cp.uint8).copy(); del raw
            gpu_out  = cp.empty(n * 9, dtype=cp.float32)
            kernel((int(np.ceil(n / BLOCK)),), (BLOCK,),
                   (gpu_data, np.int64(n), gpu_out))
            cp.cuda.runtime.deviceSynchronize()
            del gpu_data
            all_v.append(cp.asnumpy(gpu_out).reshape(n, 9))
            del gpu_out
            cp.cuda.runtime.deviceSynchronize()
            print(f"  chunk {ci+1}/{n_ch}  {(ci+1)/n_ch*100:.0f}%",
                  flush=True)

    verts = np.concatenate(all_v, axis=0); del all_v

    # Decimate
    if len(verts) > max_faces:
        step  = int(np.ceil(len(verts) / max_faces))
        verts = verts[::step]
        print(f"[HTML] Decimated  : {n_faces:,} -> {len(verts):,} faces",
              flush=True)

    # Build k3d arrays
    positions = verts.reshape(-1, 3).astype(np.float32)
    indices   = np.arange(len(verts) * 3,
                          dtype=np.uint32).reshape(len(verts), 3)

    # Normalise coords to [-0.5, 0.5] for stable k3d rendering
    center    = (positions.max(axis=0) + positions.min(axis=0)) / 2.0
    scale     = float((positions.max(axis=0) - positions.min(axis=0)).max())
    positions = ((positions - center) / (scale + 1e-8)).astype(np.float32)

    extents = (positions.max(axis=0) - positions.min(axis=0)) * scale
    print(f"[HTML] Extents    : {extents.tolist()} mm", flush=True)

    # Render
    plot = k3d.plot(background_color=0x0a0a0a,
                    camera_auto_fit=True, grid_visible=False)
    plot += k3d.mesh(vertices=positions, indices=indices,
                     color=0xE05050, opacity=0.85,
                     flat_shading=True, side="double",
                     name=stl_path.stem)

    if _in_jupyter():
        plot.display()
        print("[HTML] Widget displayed above.", flush=True)
    else:
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(plot.get_snapshot())
        mb = html_path.stat().st_size / 1e6
        print(f"[HTML] Saved      : {html_path}  ({mb:.1f} MB)", flush=True)
        print(f"[HTML] Copy       : scp yuxin@server:"
              f"{html_path.resolve()} ~/Desktop/", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Micro-CT vessel segmentation — reads hybrid_ultra/, outputs to ./")
    parser.add_argument('--sigma_min',  type=float, default=0.05,
                        help='Smallest vessel radius to detect (mm). Default: 0.05')
    parser.add_argument('--sigma_max',  type=float, default=0.5,
                        help='Largest vessel radius to detect (mm). Default: 0.5')
    parser.add_argument('--num_scales', type=int,   default=6,
                        help='Number of log-spaced sigma values. Default: 6')
    parser.add_argument('--threshold',  type=float, default=0.05,
                        help='Vesselness threshold for binary mask [0-1]. Default: 0.05')
    parser.add_argument('--alpha',      type=float, default=0.5,
                        help='Frangi alpha (plate sensitivity). Default: 0.5')
    parser.add_argument('--beta',       type=float, default=0.5,
                        help='Frangi beta (blob sensitivity). Default: 0.5')
    parser.add_argument('--tile_size',  type=int,   default=256,
                        help='Tile size (voxels per side). Reduce if OOM. Default: 256')
    parser.add_argument('--overlap',    type=int,   default=32,
                        help='Overlap between tiles (voxels). Default: 32')
    parser.add_argument('--device',     default='cuda',
                        help='Device: "cuda" or "cpu". Default: cuda')
    parser.add_argument('--spacing',    type=float, default=None,
                        help='Override voxel spacing (mm). Auto-read from DICOM header if omitted.')
    parser.add_argument('--no_stl',     action='store_true',
                        help='Skip STL export (saves time if only mask is needed)')
    parser.add_argument('--chunk_z',    type=int, default=128,
                        help='Z-slices per marching-cubes slab. Reduce if OOM. Default: 128')
    parser.add_argument('--resume',     action='store_true',
                        help='Skip Frangi+DICOM loading; read vesselness_map.nii.gz '
                             'from OUTPUT_DIR and go straight to STL export. '
                             'Use after a marching-cubes OOM crash.')
    parser.add_argument('--html',        action='store_true',
                        help='Export interactive 3D HTML after STL is written.')
    parser.add_argument('--max_faces',   type=int, default=2_000_000,
                        help='Max faces in HTML render (default: 2000000). '
                             'Reduce if browser is slow.')
    parser.add_argument('--invert',      action='store_true',
                        help='Invert volume before Frangi. Use if vessels are '
                             'DARK on bright background (standard CT). '
                             'Default: off (bright vessels on dark bg).')
    parser.add_argument('--smooth',      type=float, default=1.0,
                        help='Gaussian sigma (voxels) to smooth vesselness '
                             'before thresholding. 0=off, 1=mild, 2=strong. '
                             'Default: 1.0')
    parser.add_argument('--min_size',    type=int, default=500,
                        help='Remove isolated blobs smaller than this many '
                             'voxels before marching cubes. Default: 500')
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Output : {OUTPUT_DIR.resolve()}")

    vessel_map_path = OUTPUT_DIR / "vesselness_map.nii.gz"
    mask_path       = OUTPUT_DIR / "vessel_mask.nii.gz"
    stl_path        = OUTPUT_DIR / "vessels.stl"
    html_path       = OUTPUT_DIR / "vessels.html"

    # ── RESUME: load saved vesselness map, skip all Frangi work ──
    if args.resume:
        if not vessel_map_path.exists():
            sys.exit(f"[ERROR] --resume requested but {vessel_map_path} not found.")
        print(f"[INFO] --resume: loading saved vesselness map from {vessel_map_path}")
        img     = nib.load(str(vessel_map_path))
        vesselness = img.get_fdata(dtype=np.float32)
        affine  = img.affine
        zooms   = img.header.get_zooms()
        spacing = args.spacing if args.spacing is not None else float(np.mean(zooms[:3]))
        print(f"[INFO] Vesselness map shape: {vesselness.shape}, spacing: {spacing:.4f} mm")

    else:
        # ── Validate input folder ──────────────────────────────
        if not DICOM_INPUT_FOLDER.exists():
            sys.exit(f"[ERROR] DICOM input folder not found: {DICOM_INPUT_FOLDER.resolve()}")
        print(f"[INFO] Input  : {DICOM_INPUT_FOLDER.resolve()}")

        # ── Load DICOM series ──────────────────────────────────
        data, affine, spacing_auto = load_dicom_series(DICOM_INPUT_FOLDER)
        spacing = args.spacing if args.spacing is not None else spacing_auto
        print(f"[INFO] Using spacing: {spacing:.4f} mm")

        # Normalise to [0, 1]
        plo, phi = np.percentile(data, [1, 99])
        data = np.clip((data - plo) / (phi - plo + 1e-8), 0, 1).astype(np.float32)

        # Invert if vessels are bright (contrast-enhanced micro-CT)
        # Frangi's bright-tube polarity (l2>0, l3>0) works on normalised [0,1]
        # data where vessels are bright — no inversion needed.
        # If your vessels appear dark, add --invert to flip.
        if args.invert:
            print("[INFO] Inverting volume (--invert set) ...")
            data = 1.0 - data

        # ── Run Frangi ─────────────────────────────────────────
        print("[INFO] Running multiscale Frangi vesselness...")
        vesselness = multiscale_frangi_tiled(
            data,
            spacing_mm=spacing,
            sigma_min_mm=args.sigma_min,
            sigma_max_mm=args.sigma_max,
            num_scales=args.num_scales,
            alpha=args.alpha,
            beta=args.beta,
            tile_size=args.tile_size,
            overlap=args.overlap,
            device=args.device,
        )

        # ── Save vesselness map ────────────────────────────────
        nib.save(nib.Nifti1Image(vesselness, affine), str(vessel_map_path))
        print(f"[INFO] Saved vesselness map to {vessel_map_path}")
        print(f"[INFO] Vesselness stats: min={vesselness.min():.4f}  "
              f"max={vesselness.max():.4f}  mean={vesselness.mean():.4f}  "
              f"p95={np.percentile(vesselness, 95):.4f}")

    # ── Optional Gaussian smoothing before threshold ─────────
    if args.smooth > 0:
        from scipy.ndimage import gaussian_filter
        print(f"[INFO] Smoothing vesselness map "
              f"(sigma={args.smooth} voxels) ...", flush=True)
        vesselness = gaussian_filter(vesselness,
                                     sigma=args.smooth).astype(np.float32)

    # ── Threshold -> binary mask ─────────────────────────────
    print(f"[INFO] Thresholding at {args.threshold} ...")
    mask = (vesselness >= args.threshold).astype(np.uint8)
    del vesselness   # free ~10 GB before marching cubes
    nvox = int(mask.sum())
    print(f"[INFO] Vessel voxels after threshold: "
          f"{nvox:,} ({100*nvox/mask.size:.2f}% of volume)")

    # ── Remove small isolated blobs ───────────────────────────
    if args.min_size > 0 and nvox > 0:
        from scipy.ndimage import label
        print(f"[INFO] Removing components < {args.min_size} voxels ...",
              flush=True)
        labeled, n_comp = label(mask)
        print(f"[INFO]   {n_comp:,} components found", flush=True)
        comp_sizes = np.bincount(labeled.ravel())
        keep = np.where(comp_sizes >= args.min_size)[0]
        keep = keep[keep != 0]   # exclude background label 0
        clean = np.isin(labeled, keep).astype(np.uint8)
        del labeled, comp_sizes
        removed = nvox - int(clean.sum())
        mask  = clean
        nvox  = int(mask.sum())
        print(f"[INFO]   Kept {len(keep):,} components, "
              f"removed {removed:,} voxels", flush=True)

    print(f"[INFO] Final vessel voxels: "
          f"{nvox:,} ({100*nvox/mask.size:.2f}% of volume)")

    nib.save(nib.Nifti1Image(mask, affine), str(mask_path))
    print(f"[INFO] Saved binary mask to {mask_path}")

    # ── STL export ────────────────────────────────────────────
    if not args.no_stl:
        if nvox == 0:
            print("[WARN] Mask is empty — skipping STL. Try lowering --threshold.")
        else:
            mask_to_stl(mask, affine, spacing, str(stl_path),
                        chunk_z=args.chunk_z)

    # ── HTML export ──────────────────────────────────────────
    if args.html and not args.no_stl and nvox > 0:
        print("\n[INFO] Exporting interactive HTML ...", flush=True)
        export_html(stl_path, html_path, max_faces=args.max_faces)
    elif args.html and args.no_stl:
        print("[WARN] --html ignored because --no_stl was set.")

    # ── Summary ───────────────────────────────────────────────
    summary_path = OUTPUT_DIR / "summary.txt"
    with open(summary_path, "w") as f:
        if args.resume:
            f.write(f"Resumed from       : {vessel_map_path.resolve()}\n")
        else:
            f.write(f"Input DICOM folder : {DICOM_INPUT_FOLDER.resolve()}\n")
        f.write(f"Voxel spacing      : {spacing:.4f} mm\n")
        f.write(f"Threshold          : {args.threshold}\n")
        f.write(f"Vessel voxels      : {nvox:,}\n")
        f.write(f"Vesselness map     : {vessel_map_path.resolve()}\n")
        f.write(f"Binary mask        : {mask_path.resolve()}\n")
        if not args.no_stl:
            f.write(f"STL                : {stl_path.resolve()}\n")
            f.write(f"chunk_z            : {args.chunk_z}\n")
        if args.html and not args.no_stl:
            f.write(f"HTML               : {html_path.resolve()}\n")
    print(f"[INFO] Saved summary to {summary_path}")
    print("[DONE]")


if __name__ == '__main__':
    main()