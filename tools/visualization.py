"""
visualize_stl_gpu.ipynb-style script
──────────────────────────────────────
Run cell by cell in Jupyter, or as a plain script.

Pipeline:
  1. Read binary STL header → face count (CPU, negligible)
  2. mmap file → upload raw bytes to GPU in one transfer
  3. CUDA kernel: one thread per face, reads 12 floats from fixed offsets
  4. Decimate on GPU (keep every Nth face) to fit display budget
  5. Render with k3d (works headless over SSH in Jupyter)

Install:
    pip install cupy-cuda12x k3d jupyterlab numpy
    jupyter lab --no-browser --port=8888          # server
    ssh -L 8888:localhost:8888 user@server        # local tunnel
"""

# ── [Cell 1] Imports & config ─────────────────────────────────────────────────

import struct, os, sys, time, mmap
import numpy as np
from pathlib import Path

STL_PATH      = Path("../nnunet/output/vessels.stl")   # binary STL from segmentation script
DICOM_DIR     = Path("hybrid_ultra")         # DICOM folder for volume overlay
MAX_FACES     = 2_000_000    # faces sent to renderer; reduce if browser lags
N_SLICES      = 200          # central DICOM slices to load for volume (None = all)
MIP_DOWNSAMPLE = 2           # volume downsample factor (higher = faster/coarser)
SURFACE_COLOR = 0xE05050     # vessel mesh colour
BACKGROUND    = 0x0a0a0a
OPACITY       = 0.85
VOLUME_OPACITY = [           # intensity -> opacity transfer function
    0.0,  0.00,              # background: transparent
    0.15, 0.00,
    0.25, 0.03,              # faint soft tissue
    0.55, 0.15,              # vessels / dense structures
    1.0,  0.40,
]

# ── [Cell 2] CUDA kernel ──────────────────────────────────────────────────────

_CUDA_READ_FACES = r"""
/*
 * read_binary_stl_faces
 * ─────────────────────
 * One thread per face. Binary STL record (50 bytes):
 *   bytes  0-11  normal   (3 × float32)  — skipped for rendering
 *   bytes 12-23  vertex 0 (3 × float32)
 *   bytes 24-35  vertex 1 (3 × float32)
 *   bytes 36-47  vertex 2 (3 × float32)
 *   bytes 48-49  attr     (uint16)       — ignored
 *
 * out: (n_faces, 9) float32  [x0,y0,z0, x1,y1,z1, x2,y2,z2]
 */
extern "C" __global__
void read_binary_stl_faces(
        const unsigned char* __restrict__ data,
        long long n_faces,
        float*    __restrict__ out)
{
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_faces) return;

    const unsigned char* rec = data + tid * 50LL + 12LL; // skip normal
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

# ── [Cell 3] GPU loader ───────────────────────────────────────────────────────

def _validate_stl(path: Path) -> tuple[bool, int]:
    """
    Returns (is_binary, n_faces).
    Validates header, face count, and file size consistency.
    """
    file_size = path.stat().st_size
    with open(path, "rb") as f:
        header = f.read(80)
        raw_count = f.read(4)

    # ASCII STL starts with "solid"
    is_ascii = header[:5] == b"solid"

    if is_ascii:
        return False, 0

    if len(raw_count) < 4:
        return False, 0

    n_header   = struct.unpack("<I", raw_count)[0]
    n_from_size = (file_size - 84) // 50
    leftover    = (file_size - 84) % 50

    print(f"  Header face count : {n_header:,}")
    print(f"  File-size faces   : {n_from_size:,}  (leftover {leftover} bytes)")

    # Trust file size over header when header is 0 or wildly wrong
    if n_header == 0 or abs(n_header - n_from_size) > 100:
        print(f"  [WARN] Header count unreliable — using file-size count "
              f"({n_from_size:,})", flush=True)
        # Patch the header on disk
        with open(path, "r+b") as f:
            f.seek(80)
            f.write(struct.pack("<I", n_from_size))
        return True, n_from_size

    return True, n_header


def _convert_ascii_to_binary_cpu(src: Path) -> Path:
    """Convert ASCII STL to binary in-place. Returns path to binary file."""
    import re
    dst = src.with_suffix(".binary.stl")
    print(f"  Converting ASCII -> binary: {dst}", flush=True)
    faces = []
    t0 = time.time()
    size = src.stat().st_size
    bytes_read = 0
    with open(src, "r") as fin:
        normal, verts = None, []
        for line in fin:
            bytes_read += len(line.encode())
            line = line.strip()
            if line.startswith("facet normal"):
                p = line.split(); normal = (float(p[2]),float(p[3]),float(p[4])); verts=[]
            elif line.startswith("vertex"):
                p = line.split(); verts.append((float(p[1]),float(p[2]),float(p[3])))
            elif line.startswith("endfacet") and normal and len(verts)==3:
                faces.append((normal, verts))
            if len(faces) % 500_000 == 0 and faces:
                pct = bytes_read/size*100
                eta = (time.time()-t0)/max(pct,0.01)*(100-pct)
                print(f"  {len(faces):,} faces  {pct:.1f}%  ETA {eta:.0f}s",
                      flush=True)
    with open(dst, "wb") as f:
        f.write(b"\x00"*80)
        f.write(struct.pack("<I", len(faces)))
        for (nx,ny,nz),(v0,v1,v2) in faces:
            f.write(struct.pack("<ffffffffffffH",
                                nx,ny,nz,*v0,*v1,*v2,0))
    print(f"  Done: {len(faces):,} faces → {dst}", flush=True)
    return dst


def load_stl_gpu(path: Path) -> np.ndarray:
    """
    Load binary STL vertices using GPU.
    Automatically handles:
      - ASCII STL  → converts to binary first (CPU)
      - Wrong/zero face count in header → repairs from file size
      - File too large for VRAM → chunks automatically
    Returns float32 array of shape (N, 9):
        columns 0-2  vertex 0 / columns 3-5  vertex 1 / columns 6-8  vertex 2
    """
    import cupy as cp
    cp.cuda.set_allocator(None)

    path = Path(path)
    print(f"[STL]  Validating {path.name} ...", flush=True)

    is_binary, n_faces = _validate_stl(path)

    if not is_binary:
        print(f"[STL]  ASCII STL detected — converting to binary first ...",
              flush=True)
        path     = _convert_ascii_to_binary_cpu(path)
        is_binary, n_faces = _validate_stl(path)

    if n_faces == 0:
        sys.exit("[ERROR] STL has 0 faces after validation. File may be corrupt.")

    HEADER     = 84
    FACE_BYTES = 50
    file_size  = path.stat().st_size

    # Final safety clamp: never read past end of file
    max_faces_from_size = (file_size - HEADER) // FACE_BYTES
    if n_faces > max_faces_from_size:
        print(f"  [WARN] Clamping face count from {n_faces:,} to "
              f"{max_faces_from_size:,} (file boundary)", flush=True)
        n_faces = max_faces_from_size

    free_vram = cp.cuda.runtime.memGetInfo()[0]
    dev_name  = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
    print(f"GPU        : {dev_name}")
    print(f"Free VRAM  : {free_vram/1e9:.1f} GB")
    print(f"STL        : {file_size/1e6:.0f} MB  |  {n_faces:,} faces")

    need = n_faces * (FACE_BYTES + 36)
    if need > free_vram * 0.80:
        n_chunks = int(np.ceil(need / (free_vram * 0.80)))
        chunk_n  = int(np.ceil(n_faces / n_chunks))
        print(f"[INFO] Splitting into {n_chunks} chunks of {chunk_n:,} faces",
              flush=True)
        parts = []
        for i in range(n_chunks):
            start = i * chunk_n
            count = min(chunk_n, n_faces - start)
            parts.append(_load_chunk_gpu(path, HEADER, count, start, cp))
        verts = np.concatenate(parts, axis=0)
    else:
        verts = _load_chunk_gpu(path, HEADER, n_faces, 0, cp)

    print(f"Loaded     : {len(verts):,} faces")
    return verts


def _load_chunk_gpu(path, header_offset, n, face_offset, cp):
    import cupy as cp

    FACE_BYTES = 50
    mod    = cp.RawModule(code=_CUDA_READ_FACES)
    kernel = mod.get_function("read_binary_stl_faces")
    BLOCK  = 256

    # Read chunk from disk
    byte_offset = header_offset + face_offset * FACE_BYTES
    byte_count  = n * FACE_BYTES
    with open(path, "rb") as f:
        f.seek(byte_offset)
        raw = f.read(byte_count)

    # CRITICAL: f.read() may return fewer bytes than requested (file ends early).
    # Recompute n from actual bytes read — this is what the kernel must use.
    actual_bytes = len(raw)
    if actual_bytes < byte_count:
        print(f"  [WARN] Expected {byte_count} bytes, got {actual_bytes} — "
              f"truncating face count", flush=True)
    # Floor to complete 50-byte records only
    n = actual_bytes // FACE_BYTES
    if n == 0:
        del raw
        return np.empty((0, 9), dtype=np.float32)
    # Trim raw to exact multiple of FACE_BYTES to avoid partial record at end
    raw = raw[:n * FACE_BYTES]

    t0 = time.time()
    print(f"  uploading {len(raw)/1e6:.0f} MB  ({n:,} faces) to GPU ...",
          end=" ", flush=True)
    gpu_data = cp.frombuffer(raw, dtype=cp.uint8).copy()
    del raw
    cp.cuda.runtime.deviceSynchronize()
    print(f"{time.time()-t0:.1f}s", flush=True)

    # Verify buffer is exactly n * FACE_BYTES before launching kernel
    assert len(gpu_data) == n * FACE_BYTES,         f"Buffer size mismatch: {len(gpu_data)} != {n * FACE_BYTES}"

    # Run kernel — n is now guaranteed to match gpu_data size exactly
    gpu_out = cp.empty(n * 9, dtype=cp.float32)
    grid    = (int(np.ceil(n / BLOCK)),)
    t0 = time.time()
    print(f"  parsing {n:,} faces on GPU ...", end=" ", flush=True)
    kernel(grid, (BLOCK,), (gpu_data, np.int64(n), gpu_out))
    cp.cuda.runtime.deviceSynchronize()
    print(f"{time.time()-t0:.1f}s", flush=True)
    del gpu_data
    cp.cuda.runtime.deviceSynchronize()

    # Download
    t0 = time.time()
    print(f"  downloading to CPU ...", end=" ", flush=True)
    verts = cp.asnumpy(gpu_out).reshape(n, 9)
    del gpu_out
    cp.cuda.runtime.deviceSynchronize()
    print(f"{time.time()-t0:.1f}s", flush=True)

    return verts


# ── [Cell 4] Decimate ─────────────────────────────────────────────────────────

def decimate(verts: np.ndarray, max_faces: int) -> np.ndarray:
    """Uniform subsampling — keeps every Nth face."""
    n = len(verts)
    if n <= max_faces:
        print(f"No decimation needed ({n:,} faces ≤ {max_faces:,})")
        return verts
    step = int(np.ceil(n / max_faces))
    out  = verts[::step]
    print(f"Decimated  : {n:,} → {len(out):,} faces  (kept 1/{step})")
    return out


# ── [Cell 5] Build k3d mesh arrays ───────────────────────────────────────────

def to_k3d_arrays(verts: np.ndarray):
    """
    Convert (N, 9) face array to k3d vertices + indices.
    k3d expects:
        vertices : float32 (M, 3)  — vertex positions
        indices  : uint32  (N, 3)  — triangle indices into vertices
    """
    n = len(verts)

    if n == 0:
        sys.exit("[ERROR] No faces loaded from STL — file may be empty or corrupt.\n"
                 "        Re-run:  python -u segment_vessels_frangi.py "
                 "--resume --threshold 0.2 --chunk_z 128 --html")

    # verts shape (N, 9) -> (N*3, 3): each row is one vertex
    positions = verts.reshape(-1, 3).astype(np.float32)
    indices   = np.arange(n * 3, dtype=np.uint32).reshape(n, 3)

    # ── Diagnostics ───────────────────────────────────────────
    print(f"  positions shape : {positions.shape}")
    print(f"  indices shape   : {indices.shape}")
    print(f"  coordinate range:")
    print(f"    X: {positions[:,0].min():.3f}  to  {positions[:,0].max():.3f}")
    print(f"    Y: {positions[:,1].min():.3f}  to  {positions[:,1].max():.3f}")
    print(f"    Z: {positions[:,2].min():.3f}  to  {positions[:,2].max():.3f}")
    extents = positions.max(axis=0) - positions.min(axis=0)
    print(f"  extents (mm)    : {extents.tolist()}")

    if extents.max() < 1.0:
        print("  [WARN] Extents < 1 mm — coords may still be in voxel space!")
    if extents.max() > 1e5:
        print("  [WARN] Extents > 100 m — coords may have wrong scale.")

    return positions, indices


# ── [Cell 6] Render ───────────────────────────────────────────────────────────

def load_dicom_volume(folder: Path, n_slices=None):
    """
    Load N central DICOM slices → float32 volume (Z,Y,X) + spacing tuple.
    """
    try:
        import pydicom
    except ImportError:
        sys.exit("pip install pydicom")

    files = sorted(folder.glob("*.dcm"))
    if not files:
        files = sorted(p for p in folder.iterdir() if p.is_file())
    if not files:
        sys.exit(f"[ERROR] No DICOM files in {folder}")

    total = len(files)
    if n_slices and n_slices < total:
        mid   = total // 2
        half  = n_slices // 2
        files = files[mid - half: mid + half]

    print(f"[DICOM] Loading {len(files)}/{total} slices ...", flush=True)
    t0 = time.time()
    slices = []
    for i, p in enumerate(files, 1):
        try:
            slices.append(pydicom.dcmread(str(p)))
        except Exception as e:
            print(f"  skip {p.name}: {e}", flush=True)
        if i % 50 == 0 or i == len(files):
            print(f"  {i}/{len(files)}  {time.time()-t0:.0f}s", flush=True)

    def _zpos(ds):
        try:    return float(ds.ImagePositionPatient[2])
        except:
            try:    return float(ds.InstanceNumber)
            except: return 0.0

    slices.sort(key=_zpos)
    ds0 = slices[0]

    try:    row_sp, col_sp = float(ds0.PixelSpacing[0]), float(ds0.PixelSpacing[1])
    except: row_sp = col_sp = 0.0244

    if len(slices) > 1:
        try:    slice_th = abs(float(slices[1].ImagePositionPatient[2]) -
                               float(slices[0].ImagePositionPatient[2]))
        except: slice_th = float(getattr(ds0, "SliceThickness", row_sp))
    else:
        slice_th = row_sp

    print(f"[DICOM] Spacing Z={slice_th:.4f} Y={row_sp:.4f} X={col_sp:.4f} mm",
          flush=True)

    frames = []
    for ds in slices:
        arr = ds.pixel_array.astype(np.float32)
        arr = arr * float(getattr(ds, "RescaleSlope",     1.0))                   + float(getattr(ds, "RescaleIntercept", 0.0))
        frames.append(arr)

    volume = np.stack(frames, axis=0)
    print(f"[DICOM] Volume shape: {volume.shape}", flush=True)
    return volume, (slice_th, row_sp, col_sp)


def prepare_volume(volume: np.ndarray, spacing: tuple,
                   downsample: int = 2) -> tuple:
    """
    Normalise to [0,1], downsample, return (vol_float32, bounds_mm).
    bounds_mm = [xmin,xmax, ymin,ymax, zmin,zmax] for k3d.
    """
    # Percentile normalise
    p1  = float(np.percentile(volume,  1))
    p99 = float(np.percentile(volume, 99))
    vol = np.clip((volume - p1) / (p99 - p1 + 1e-8), 0, 1).astype(np.float32)

    # Downsample
    s   = max(1, downsample)
    vol = vol[::s, ::s, ::s]
    sz, sy, sx = spacing[0]*s, spacing[1]*s, spacing[2]*s
    Z, Y, X = vol.shape
    print(f"[VOL]  Downsampled : {volume.shape} -> {vol.shape}  "
          f"({Z*sz:.1f} x {Y*sy:.1f} x {X*sx:.1f} mm)", flush=True)

    bounds = [0, X*sx,  0, Y*sy,  0, Z*sz]   # k3d: [xmin,xmax,ymin,ymax,zmin,zmax]
    return vol, bounds


def _in_jupyter() -> bool:
    try:
        shell = get_ipython().__class__.__name__
        return shell == "ZMQInteractiveShell"
    except NameError:
        return False


def render(positions: np.ndarray, indices: np.ndarray,
           vol: np.ndarray = None, vol_bounds: list = None,
           html_out=None):
    """
    Render vessel mesh + optional DICOM volume overlay with k3d.

    vol        : float32 (Z,Y,X) normalised [0,1] volume — None = mesh only
    vol_bounds : [xmin,xmax,ymin,ymax,zmin,zmax] in mm — required if vol given
    html_out   : override output HTML path
    """
    try:
        import k3d
    except ImportError:
        sys.exit("k3d not found.  Install with:  pip install k3d")

    print(f"Rendering  : {len(indices):,} faces  |  "
          f"{len(positions):,} vertices", flush=True)

    # ── Normalise mesh coords to [-.5, .5] for stable k3d rendering ──────
    center = (positions.max(axis=0) + positions.min(axis=0)) / 2.0
    scale  = float((positions.max(axis=0) - positions.min(axis=0)).max())
    scale  = scale if scale > 0 else 1.0
    pos_n  = ((positions - center) / scale).astype(np.float32)

    print(f"  mesh extent  : "
          f"{((positions.max(axis=0)-positions.min(axis=0))).tolist()} mm",
          flush=True)
    print(f"  normalised   : {pos_n.min():.3f} to {pos_n.max():.3f}",
          flush=True)

    plot = k3d.plot(background_color=BACKGROUND,
                    camera_auto_fit=True, grid_visible=False)

    # ── Volume (DICOM) layer ──────────────────────────────────────────────
    if vol is not None and vol_bounds is not None:
        # Normalise volume bounds to same [-0.5, 0.5] space as mesh
        bx0, bx1, by0, by1, bz0, bz1 = vol_bounds
        vol_center = np.array([(bx0+bx1)/2, (by0+by1)/2, (bz0+bz1)/2])
        # use same scale as mesh so they align
        nb = [(bx0-vol_center[0])/scale, (bx1-vol_center[0])/scale,
              (by0-vol_center[1])/scale, (by1-vol_center[1])/scale,
              (bz0-vol_center[2])/scale, (bz1-vol_center[2])/scale]

        print(f"  volume shape : {vol.shape}  bounds(norm): {[round(x,3) for x in nb]}",
              flush=True)

        plot += k3d.volume(
            volume           = vol,
            color_map        = k3d.colormaps.matplotlib_color_maps.Bone,
            opacity_function = VOLUME_OPACITY,
            bounds           = nb,
            name             = "micro-CT volume",
        )

    # ── Vessel mesh layer ─────────────────────────────────────────────────
    plot += k3d.mesh(
        vertices     = pos_n,
        indices      = indices,
        color        = SURFACE_COLOR,
        opacity      = OPACITY,
        flat_shading = True,
        side         = "double",
        name         = STL_PATH.stem,
    )

    # ── Output ────────────────────────────────────────────────────────────
    if _in_jupyter():
        plot.display()
        print("Interactive 3D widget rendered above.", flush=True)
    else:
        out = Path(html_out) if html_out else STL_PATH.with_suffix(".html")
        with open(out, "w", encoding="utf-8") as f:
            f.write(plot.get_snapshot())
        size_mb = out.stat().st_size / 1e6
        print(f"\nSaved HTML -> {out}  ({size_mb:.1f} MB)", flush=True)
        print(f"Copy       : scp yuxin@server:{out.resolve()} ~/Desktop/",
              flush=True)

    return plot


# ── [Cell 7] Run everything ───────────────────────────────────────────────────

if __name__ == "__main__" or "get_ipython" in dir():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--no_volume", action="store_true",
                        help="Skip DICOM volume overlay, render mesh only")
    parser.add_argument("--n_slices",  type=int, default=N_SLICES,
                        help=f"Central DICOM slices to load (default {N_SLICES})")
    parser.add_argument("--downsample",type=int, default=MIP_DOWNSAMPLE,
                        help=f"Volume downsample factor (default {MIP_DOWNSAMPLE})")
    args = parser.parse_args()

    t_total = time.time()

    print("=" * 55)
    print("Step 1/4  Load binary STL via GPU")
    print("=" * 55)
    verts = load_stl_gpu(STL_PATH)

    print("\n" + "=" * 55)
    print("Step 2/4  Decimate for rendering")
    print("=" * 55)
    verts_dec = decimate(verts, MAX_FACES)
    del verts

    print("\n" + "=" * 55)
    print("Step 3/4  Build mesh arrays")
    print("=" * 55)
    positions, indices = to_k3d_arrays(verts_dec)

    print("\n" + "=" * 55)
    print("Step 4/4  Load DICOM volume" + (" (skipped)" if args.no_volume else ""))
    print("=" * 55)
    vol, vol_bounds = None, None
    if not args.no_volume:
        if DICOM_DIR.exists():
            raw_vol, spacing = load_dicom_volume(DICOM_DIR, args.n_slices)
            vol, vol_bounds  = prepare_volume(raw_vol, spacing, args.downsample)
            del raw_vol
        else:
            print(f"[WARN] DICOM_DIR not found: {DICOM_DIR} — rendering mesh only",
                  flush=True)

    print("\n" + "=" * 55)
    print("Step 5/5  Render in k3d")
    print("=" * 55)
    plot = render(positions, indices, vol=vol, vol_bounds=vol_bounds)

    print(f"\nTotal time : {time.time()-t_total:.1f}s")
    print("Done.")