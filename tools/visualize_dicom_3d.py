"""
visualize_dicom_3d.py
──────────────────────
Fast DICOM → interactive 3D visualization in Jupyter / browser.

Two render modes (set MODE below):
  "mip"   — Maximum Intensity Projection volume render. Shows all bright
             structures (vessels, bone) at once. Fastest, no threshold needed.
  "iso"   — Isosurface (marching cubes on GPU). Extracts a surface at a
             chosen intensity threshold. Good for isolating vessels.

Pipeline:
  1. Read N central DICOM slices (configurable)
  2. Upload volume to GPU → normalize on GPU
  3. MIP mode  : downsample on GPU → k3d volume render
     ISO mode  : GPU marching cubes (cupy) → k3d mesh
  4. Export HTML (script) or display widget (Jupyter)

Install:
    pip install pydicom cupy-cuda12x k3d numpy scikit-image
"""

import glob, os, struct, sys, time
from pathlib import Path

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — edit here
# ─────────────────────────────────────────────────────────────────────────────

DICOM_DIR    = Path("/home/yuxin/medical-segmentation/data/hybrid_ultra")   # folder with .dcm files
OUTPUT_HTML  = Path("/home/yuxin/medical-segmentation/data/processed/raw_3d.html")  # output when running as plain script

MODE         = "mip"    # "mip"  or  "iso"

# How many central slices to load (reduce for speed; None = all)
N_SLICES     = 300

# ISO mode: intensity threshold (0–1 after normalisation)
# Lower = more vessels but more noise. Start at 0.3, tune visually.
ISO_THRESHOLD = 0.35

# Downsample factor for MIP volume (higher = faster but coarser)
MIP_DOWNSAMPLE = 2

BACKGROUND   = 0x0a0a0a
VESSEL_COLOR = 0xE05050


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Load DICOM slices
# ─────────────────────────────────────────────────────────────────────────────

def load_dicom_slices(folder: Path, n_slices=None) -> tuple[np.ndarray, tuple]:
    """
    Load N central slices from a DICOM folder.
    Returns (volume float32 Z×Y×X, spacing_zyx tuple).
    """
    try:
        import pydicom
    except ImportError:
        sys.exit("pip install pydicom")

    files = sorted(folder.glob("*.dcm"))
    if not files:
        files = sorted(p for p in folder.iterdir() if p.is_file())
    if not files:
        sys.exit(f"No files in {folder}")

    total = len(files)
    if n_slices and n_slices < total:
        mid   = total // 2
        half  = n_slices // 2
        files = files[mid - half: mid + half]
    n = len(files)

    print(f"[DICOM] Loading {n}/{total} slices from {folder} ...", flush=True)
    t0 = time.time()

    slices = []
    for i, p in enumerate(files, 1):
        try:
            slices.append(pydicom.dcmread(str(p)))
        except Exception as e:
            print(f"  skip {p.name}: {e}", flush=True)
        if i % 50 == 0 or i == n:
            print(f"  {i}/{n}  {time.time()-t0:.0f}s", flush=True)

    def _zpos(ds):
        try:    return float(ds.ImagePositionPatient[2])
        except: 
            try:    return float(ds.InstanceNumber)
            except: return 0.0

    slices.sort(key=_zpos)
    ds0 = slices[0]

    try:
        row_sp, col_sp = float(ds0.PixelSpacing[0]), float(ds0.PixelSpacing[1])
    except:
        row_sp = col_sp = 0.0244

    if len(slices) > 1:
        try:
            slice_th = abs(float(slices[1].ImagePositionPatient[2]) -
                           float(slices[0].ImagePositionPatient[2]))
        except:
            slice_th = float(getattr(ds0, "SliceThickness", row_sp))
    else:
        slice_th = row_sp

    spacing = (slice_th, row_sp, col_sp)
    print(f"[DICOM] Spacing Z={slice_th:.4f} Y={row_sp:.4f} X={col_sp:.4f} mm",
          flush=True)

    frames = []
    for ds in slices:
        arr = ds.pixel_array.astype(np.float32)
        arr = arr * float(getattr(ds, "RescaleSlope",     1.0)) \
                  + float(getattr(ds, "RescaleIntercept", 0.0))
        frames.append(arr)

    volume = np.stack(frames, axis=0)   # (Z, Y, X)
    print(f"[DICOM] Volume shape: {volume.shape}  "
          f"range [{volume.min():.0f}, {volume.max():.0f}]", flush=True)
    return volume, spacing


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: GPU normalise + downsample
# ─────────────────────────────────────────────────────────────────────────────

def gpu_normalise(volume: np.ndarray) -> "cupy.ndarray":
    import cupy as cp
    cp.cuda.set_allocator(None)
    print("[GPU] Uploading volume ...", flush=True)
    t0  = time.time()
    vol = cp.asarray(volume)
    print(f"  {vol.nbytes/1e6:.0f} MB  {time.time()-t0:.1f}s", flush=True)

    # Percentile normalise on GPU (p1 – p99) → [0, 1]
    flat  = vol.ravel()
    p1    = float(cp.percentile(flat, 1))
    p99   = float(cp.percentile(flat, 99))
    vol   = cp.clip((vol - p1) / (p99 - p1 + 1e-8), 0, 1)
    print(f"[GPU] Normalised  p1={p1:.0f}  p99={p99:.0f}", flush=True)
    return vol


def gpu_downsample(vol_gpu, factor: int) -> "cupy.ndarray":
    if factor <= 1:
        return vol_gpu
    import cupy as cp
    s = factor
    out = vol_gpu[::s, ::s, ::s]
    print(f"[GPU] Downsampled: {vol_gpu.shape} → {out.shape}", flush=True)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Step 3a: MIP render
# ─────────────────────────────────────────────────────────────────────────────

def render_mip(vol_gpu, spacing) -> "k3d plot":
    import cupy as cp
    import k3d

    vol_ds  = gpu_downsample(vol_gpu, MIP_DOWNSAMPLE)
    vol_cpu = cp.asnumpy(vol_ds).astype(np.float32)
    del vol_ds

    Z, Y, X = vol_cpu.shape
    sz, sy, sx = [s * MIP_DOWNSAMPLE for s in spacing]

    print(f"[MIP] Volume shape {vol_cpu.shape}  "
          f"bounds {Z*sz:.1f} × {Y*sy:.1f} × {X*sx:.1f} mm", flush=True)

    plot = k3d.plot(background_color=BACKGROUND,
                    camera_auto_fit=True, grid_visible=False)

    # k3d volume: bounds = [xmin,xmax, ymin,ymax, zmin,zmax]
    vol_obj = k3d.volume(
        volume        = vol_cpu,
        color_map     = k3d.colormaps.matplotlib_color_maps.Bone,
        opacity_function = [
            0.0, 0.0,    # transparent at low intensity (background)
            0.2, 0.0,
            0.3, 0.05,   # faint soft tissue
            0.6, 0.3,    # vessels / bright structures
            1.0, 0.8,
        ],
        bounds        = [0, X*sx, 0, Y*sy, 0, Z*sz],
    )
    plot += vol_obj
    return plot


# ─────────────────────────────────────────────────────────────────────────────
# Step 3b: Isosurface render (GPU marching cubes)
# ─────────────────────────────────────────────────────────────────────────────

def render_iso(vol_gpu, spacing) -> "k3d plot":
    import cupy as cp
    import k3d
    from skimage.measure import marching_cubes

    threshold = ISO_THRESHOLD
    print(f"[ISO] Thresholding at {threshold} ...", flush=True)
    mask_gpu = (vol_gpu >= threshold)

    # Marching cubes on CPU (scikit-image); download mask first
    # GPU marching cubes requires cuml/cuda-based libs not always available
    print("[ISO] Downloading mask for marching cubes ...", flush=True)
    mask_cpu = cp.asnumpy(mask_gpu).astype(np.uint8)
    del mask_gpu, vol_gpu

    nvox = int(mask_cpu.sum())
    print(f"[ISO] Vessel voxels: {nvox:,} "
          f"({100*nvox/mask_cpu.size:.2f}%)", flush=True)

    if nvox == 0:
        sys.exit(f"[ISO] Empty mask — lower ISO_THRESHOLD (currently {threshold})")

    print("[ISO] Running marching cubes ...", flush=True)
    t0 = time.time()
    verts, faces, _, _ = marching_cubes(
        mask_cpu, level=0.5, spacing=spacing)
    print(f"[ISO] {len(faces):,} faces  {time.time()-t0:.0f}s", flush=True)

    # Normalise for rendering
    center    = (verts.max(axis=0) + verts.min(axis=0)) / 2
    scale     = float((verts.max(axis=0) - verts.min(axis=0)).max())
    verts_n   = ((verts - center) / (scale + 1e-8)).astype(np.float32)
    indices   = faces.astype(np.uint32)

    # Decimate if too many faces
    if len(faces) > 2_000_000:
        step    = int(np.ceil(len(faces) / 2_000_000))
        indices = indices[::step]
        print(f"[ISO] Decimated to {len(indices):,} faces", flush=True)

    plot = k3d.plot(background_color=BACKGROUND,
                    camera_auto_fit=True, grid_visible=False)
    plot += k3d.mesh(
        vertices     = verts_n,
        indices      = indices,
        color        = VESSEL_COLOR,
        opacity      = 0.85,
        flat_shading = True,
        side         = "double",
    )
    return plot


# ─────────────────────────────────────────────────────────────────────────────
# Step 4: Output
# ─────────────────────────────────────────────────────────────────────────────

def _in_jupyter() -> bool:
    try:
        return get_ipython().__class__.__name__ == "ZMQInteractiveShell"
    except NameError:
        return False


def show_or_export(plot, html_path: Path):
    if _in_jupyter():
        plot.display()
        print("Interactive 3D widget rendered above.")
    else:
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(plot.get_snapshot())
        mb = html_path.stat().st_size / 1e6
        print(f"\nSaved -> {html_path}  ({mb:.1f} MB)", flush=True)
        print(f"Copy:  scp yuxin@server:{html_path.resolve()} ~/Desktop/",
              flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__" or _in_jupyter():
    t0 = time.time()

    # 1. Load
    volume, spacing = load_dicom_slices(DICOM_DIR, N_SLICES)

    # 2. GPU normalise
    vol_gpu = gpu_normalise(volume)
    del volume   # free CPU copy

    # 3. Render
    print(f"\n[MODE] {MODE.upper()}", flush=True)
    if MODE == "mip":
        plot = render_mip(vol_gpu, spacing)
    elif MODE == "iso":
        plot = render_iso(vol_gpu, spacing)
    else:
        sys.exit(f"Unknown MODE '{MODE}' — use 'mip' or 'iso'")

    # 4. Output
    show_or_export(plot, OUTPUT_HTML)
    print(f"\nTotal: {time.time()-t0:.0f}s", flush=True)