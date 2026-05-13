"""
_to_binary_stl.py
─────────────────
GPU-accelerated ASCII → binary STL converter.

Strategy
────────
The file is read into CPU RAM in one shot (mmap), then transferred to GPU
memory as a raw byte array. A CUDA kernel runs one thread per byte to locate
all "facet normal" and "vertex" line starts in parallel, producing index
arrays. A second kernel parses the fixed-format numeric tokens in parallel.
The resulting float32 face array is transferred back and written as binary STL
with a single numpy tofile() call.

Why GPU works here
──────────────────
The original bottleneck is NOT disk I/O — it is the Python float() loop over
~1B tokens. That loop is sequential and GIL-bound. A CUDA kernel can run
millions of threads simultaneously, each parsing one token independently.
GPU RAM is the working buffer — the file is never fully expanded in CPU RAM
as a Python object.

Requirements
────────────
    pip install cupy-cuda12x        # match your CUDA version
    # or: cupy-cuda11x / cupy-cuda117 etc.

Usage
─────
    python _to_binary_stl.py
    python _to_binary_stl.py --src results_frangi/vessels.stl
    python _to_binary_stl.py --keep        # don't delete ASCII source
    python _to_binary_stl.py --cpu-fallback  # fall back to numpy if no GPU
"""

import argparse
import mmap
import struct
import os
import sys
import time
import numpy as np
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
SRC_DEFAULT = Path("results_frangi/vessels.stl")
DST_DEFAULT = Path("results_frangi/vessels_binary.stl")

HEADER_BYTES = 80
FACE_BYTES   = 50


# ─────────────────────────────────────────────────────────────────────────────
# CUDA kernel source
# ─────────────────────────────────────────────────────────────────────────────

_CUDA_FIND_LINES = r"""
/*
 * find_line_starts
 * ────────────────
 * One thread per byte. Marks positions where a target pattern starts.
 * out[i] = 1 if data[i:i+pat_len] matches pattern, else 0.
 */
extern "C" __global__
void find_line_starts(
        const unsigned char* __restrict__ data,
        long long data_len,
        const unsigned char* __restrict__ pattern,
        int pat_len,
        unsigned char* __restrict__ out)
{
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i + pat_len > data_len) { if (i < data_len) out[i] = 0; return; }

    for (int k = 0; k < pat_len; k++) {
        if (data[i + k] != pattern[k]) { out[i] = 0; return; }
    }
    out[i] = 1;
}
"""

_CUDA_PARSE_FLOATS = r"""
/*
 * parse_floats_from_positions
 * ────────────────────────────
 * Each thread handles one line start index.
 * It skips the prefix (prefix_skip bytes) then parses `n_floats` whitespace-
 * separated ASCII floats and writes them to out[thread_id * n_floats + j].
 *
 * Handles scientific notation (e.g. -1.234567e-03).
 */
extern "C" __global__
void parse_floats_from_positions(
        const unsigned char* __restrict__ data,
        long long data_len,
        const long long* __restrict__ positions,
        int n_positions,
        int prefix_skip,
        int n_floats,
        float* __restrict__ out)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_positions) return;

    long long pos = positions[tid] + prefix_skip;
    float* dst = out + (long long)tid * n_floats;

    // Skip leading whitespace after prefix
    while (pos < data_len && (data[pos] == ' ' || data[pos] == '\t')) pos++;

    for (int f = 0; f < n_floats; f++) {
        // Skip whitespace
        while (pos < data_len &&
               (data[pos] == ' ' || data[pos] == '\t' ||
                data[pos] == '\r' || data[pos] == '\n')) pos++;

        // Parse sign
        float sign = 1.0f;
        if (pos < data_len && data[pos] == '-') { sign = -1.0f; pos++; }
        else if (pos < data_len && data[pos] == '+') { pos++; }

        // Integer part
        float val = 0.0f;
        while (pos < data_len && data[pos] >= '0' && data[pos] <= '9') {
            val = val * 10.0f + (data[pos] - '0');
            pos++;
        }

        // Fractional part
        if (pos < data_len && data[pos] == '.') {
            pos++;
            float frac = 0.1f;
            while (pos < data_len && data[pos] >= '0' && data[pos] <= '9') {
                val += (data[pos] - '0') * frac;
                frac *= 0.1f;
                pos++;
            }
        }

        // Exponent
        if (pos < data_len && (data[pos] == 'e' || data[pos] == 'E')) {
            pos++;
            float esign = 1.0f;
            if (pos < data_len && data[pos] == '-') { esign = -1.0f; pos++; }
            else if (pos < data_len && data[pos] == '+') { pos++; }
            float exp = 0.0f;
            while (pos < data_len && data[pos] >= '0' && data[pos] <= '9') {
                exp = exp * 10.0f + (data[pos] - '0');
                pos++;
            }
            // pow via repeated multiply (faster than __powf for small exp)
            float mult = 1.0f;
            int iexp = (int)exp;
            for (int e = 0; e < iexp; e++) mult *= 10.0f;
            if (esign < 0) val /= mult; else val *= mult;
        }

        dst[f] = sign * val;

        // Skip to next whitespace (end of this token)
        while (pos < data_len &&
               data[pos] != ' ' && data[pos] != '\t' &&
               data[pos] != '\r' && data[pos] != '\n') pos++;
    }
}
"""


# ─────────────────────────────────────────────────────────────────────────────
# GPU conversion
# ─────────────────────────────────────────────────────────────────────────────

def _get_free_vram_bytes():
    import cupy as cp
    device = cp.cuda.Device()
    free, _ = cp.cuda.runtime.memGetInfo()
    return free


def _convert_gpu(src: Path, dst: Path, tmp: Path):
    """
    Chunked GPU conversion. Reads the source file in VRAM-sized chunks
    directly from disk — never loads the whole file into RAM or VRAM.
    """
    import cupy as cp

    file_size = src.stat().st_size

    # Query free VRAM and decide chunk size
    free_vram   = cp.cuda.runtime.memGetInfo()[0]
    usable_vram = int(free_vram * 0.25)   # 25%: leaves 75% for flag arrays + float outputs
    n_chunks    = max(1, int(np.ceil(file_size / usable_vram)))
    chunk_size  = int(np.ceil(file_size / n_chunks))

    dev_name = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
    print(f"[INFO] GPU        : {dev_name}", flush=True)
    print(f"[INFO] File size  : {file_size/1e9:.2f} GB", flush=True)
    print(f"[INFO] Free VRAM  : {free_vram/1e9:.2f} GB  "
          f"(usable {usable_vram/1e9:.2f} GB)", flush=True)
    print(f"[INFO] Chunks     : {n_chunks} × ~{chunk_size/1e9:.2f} GB",
          flush=True)

    # ── Disable CuPy memory pool so del/free actually releases to CUDA ──────
    # Default pool caches blocks internally; nvidia-smi shows low usage but
    # CUDA allocator still sees them as reserved → OOM on next large alloc.
    cp.cuda.set_allocator(None)           # raw cudaMalloc / cudaFree
    cp.cuda.set_pinned_memory_allocator(None)

    # ── Compile CUDA kernels once ─────────────────────────────────────────
    mod_find  = cp.RawModule(code=_CUDA_FIND_LINES)
    mod_parse = cp.RawModule(code=_CUDA_PARSE_FLOATS)
    k_find    = mod_find.get_function("find_line_starts")
    k_parse   = mod_parse.get_function("parse_floats_from_positions")
    BLOCK     = 256

    def find_pattern_gpu(gpu_data, pattern_str):
        n   = len(gpu_data)
        pat = cp.frombuffer(pattern_str.encode("ascii"), dtype=cp.uint8)
        # Use uint8 flag array — 1 byte per position, minimal footprint
        out = cp.zeros(n, dtype=cp.uint8)
        grid = (int(np.ceil(n / BLOCK)),)
        k_find(grid, (BLOCK,),
               (gpu_data, np.int64(n), pat, np.int32(len(pat)), out))
        # Convert to bool view and use nonzero via cumsum to avoid
        # cp.where allocating a second ~chunk-sized array internally.
        # cumsum on uint8 is compact; final index array is tiny.
        nz = cp.flatnonzero(out).astype(cp.int64)
        del out
        return nz

    def parse_at_gpu(gpu_data, positions, prefix_skip, n_floats):
        m   = len(positions)
        if m == 0:
            return cp.empty((0, n_floats), dtype=cp.float32)
        out  = cp.zeros(m * n_floats, dtype=cp.float32)
        grid = (int(np.ceil(m / BLOCK)),)
        k_parse(grid, (BLOCK,),
                (gpu_data, np.int64(len(gpu_data)),
                 positions, np.int32(m),
                 np.int32(prefix_skip), np.int32(n_floats),
                 out))
        return out.reshape(m, n_floats)

    dtype_face = np.dtype([
        ("normal", np.float32, (3,)),
        ("v0",     np.float32, (3,)),
        ("v1",     np.float32, (3,)),
        ("v2",     np.float32, (3,)),
        ("attr",   np.uint16),
    ])

    # ── Find chunk boundaries aligned to "facet normal" on disk ──────────
    # Read small windows to find "facet normal" near each boundary point
    pattern     = b"facet normal"
    pat_len     = len(pattern)
    search_win  = 4096   # bytes to search around each boundary

    boundaries = [0]
    with open(src, "rb") as fh:
        for i in range(1, n_chunks):
            approx = i * chunk_size
            if approx >= file_size:
                break
            # Read a small window around the approximate boundary
            win_start = max(0, approx - search_win)
            fh.seek(win_start)
            window = fh.read(search_win * 2)
            idx    = window.find(pattern)
            if idx == -1:
                boundaries.append(approx)   # fallback: use approx position
            else:
                boundaries.append(win_start + idx)
    boundaries.append(file_size)

    actual_n = len(boundaries) - 1
    print(f"[INFO] Boundary-aligned chunks: {actual_n}", flush=True)

    # ── Process each chunk ────────────────────────────────────────────────
    t0          = time.time()
    total_faces = 0

    with open(src, "rb") as fin, open(tmp, "wb") as fout:
        # Write binary STL header + placeholder face count
        fout.write(b"\x00" * HEADER_BYTES)
        fout.write(struct.pack("<I", 0))

        for ci in range(actual_n):
            b_start = boundaries[ci]
            b_end   = boundaries[ci + 1]
            csize   = b_end - b_start

            ela = time.time() - t0
            pct = b_start / file_size * 100
            eta = (ela / max(pct, 0.1)) * (100 - pct) if pct > 0 else 0
            print(f"\n  [chunk {ci+1}/{actual_n}]  {pct:.1f}%  "
                  f"size {csize/1e9:.2f} GB  "
                  f"elapsed {ela:.0f}s  ETA {eta:.0f}s",
                  flush=True)

            # Read chunk from disk into CPU RAM
            fin.seek(b_start)
            chunk_bytes = fin.read(csize)

            # ── Step A: Upload to GPU ─────────────────────────────────────────
            print(f"    uploading to GPU ...", flush=True)
            gpu_data = cp.frombuffer(chunk_bytes, dtype=cp.uint8).copy()
            del chunk_bytes
            cp.cuda.runtime.deviceSynchronize()

            # ── Step B: Find normal positions; flag array freed inside fn ─────
            print(f"    finding facet normal lines ...", flush=True)
            normal_pos = find_pattern_gpu(gpu_data, "facet normal")
            cp.cuda.runtime.deviceSynchronize()

            # ── Step C: Find vertex positions; flag array freed inside fn ─────
            print(f"    finding vertex lines ...", flush=True)
            vertex_pos = find_pattern_gpu(gpu_data, "      vertex")
            cp.cuda.runtime.deviceSynchronize()

            n_f = int(len(normal_pos))
            n_v = int(len(vertex_pos))
            print(f"    normals: {n_f:,}  vertices: {n_v:,}", flush=True)

            # ── Step D: Free chunk — parse kernel re-reads from a fresh upload ─
            del gpu_data
            cp.cuda.runtime.deviceSynchronize()

            if n_f == 0:
                del normal_pos, vertex_pos
                continue

            use = min(n_f, n_v // 3)
            if use < n_f:
                normal_pos = normal_pos[:use]
                vertex_pos = vertex_pos[:use * 3]
            if use == 0:
                del normal_pos, vertex_pos
                continue

            # ── Step E: Re-upload chunk for float parsing ─────────────────────
            # Parse kernel reads raw bytes to extract float values, so we need
            # the chunk on GPU again. Re-upload now that find arrays are freed.
            print(f"    re-uploading for float parse ...", flush=True)
            fin.seek(b_start)
            gpu_data2 = cp.frombuffer(fin.read(csize), dtype=cp.uint8).copy()
            cp.cuda.runtime.deviceSynchronize()

            # ── Step F: Parse normals; free position array after ──────────────
            print(f"    parsing {use:,} faces on GPU ...", flush=True)
            normals = parse_at_gpu(gpu_data2, normal_pos,
                                   prefix_skip=13, n_floats=3)
            del normal_pos
            cp.cuda.runtime.deviceSynchronize()

            # ── Step G: Parse vertices; free everything else ──────────────────
            vertices = parse_at_gpu(gpu_data2, vertex_pos,
                                    prefix_skip=13, n_floats=3)
            del gpu_data2, vertex_pos
            cp.cuda.runtime.deviceSynchronize()

            # Assemble face records on GPU then download
            vr        = vertices.reshape(use, 3, 3)
            faces_gpu = cp.concatenate(
                [normals, vr[:, 0, :], vr[:, 1, :], vr[:, 2, :]], axis=1)
            del normals, vertices, vr

            print(f"    downloading {use:,} faces to CPU ...", flush=True)
            faces_cpu = cp.asnumpy(faces_gpu)
            del faces_gpu

            # Write binary records
            records           = np.zeros(use, dtype=dtype_face)
            records["normal"] = faces_cpu[:, 0:3]
            records["v0"]     = faces_cpu[:, 3:6]
            records["v1"]     = faces_cpu[:, 6:9]
            records["v2"]     = faces_cpu[:, 9:12]
            records.tofile(fout)
            del faces_cpu, records

            total_faces += use
            elapsed = time.time() - t0
            print(f"    wrote {use:,} faces  "
                  f"cumulative: {total_faces:,}  "
                  f"file so far: {fout.tell()/1e6:.0f} MB",
                  flush=True)

        # Write real face count into header
        fout.seek(HEADER_BYTES)
        fout.write(struct.pack("<I", total_faces))

    os.replace(tmp, dst)
    total_time = time.time() - t0
    final_mb   = dst.stat().st_size / 1e6
    print(f"\n[DONE] {dst}  "
          f"({total_faces:,} faces  {final_mb:.0f} MB)  "
          f"total {total_time:.0f}s", flush=True)


def _convert_cpu(raw: bytes, dst: Path, tmp: Path):
    import multiprocessing as mp

    n_workers = min(mp.cpu_count(), 16)
    print(f"[INFO] GPU not available — using CPU fallback "
          f"({n_workers} workers)", flush=True)

    chunk_size = max(len(raw) // (n_workers * 4), 64 * 1024 * 1024)

    # Split at "facet normal" boundaries
    pattern = b"facet normal"
    boundaries = [0]
    pos = chunk_size
    while pos < len(raw):
        idx = raw.find(pattern, pos)
        if idx == -1:
            break
        boundaries.append(idx)
        pos = idx + chunk_size
    boundaries.append(len(raw))

    chunks = [(raw[boundaries[i]: boundaries[i+1]], i)
              for i in range(len(boundaries) - 1)]

    t0 = time.time()
    results = [None] * len(chunks)
    with mp.Pool(n_workers) as pool:
        for done, (idx, arr) in enumerate(
                pool.imap_unordered(_parse_chunk_cpu, chunks), 1):
            results[idx] = arr
            print(f"  chunk {done}/{len(chunks)}  "
                  f"{done/len(chunks)*100:.0f}%  "
                  f"elapsed {time.time()-t0:.0f}s",
                  flush=True)

    faces_cpu = np.concatenate([r for r in results if r is not None and len(r)])
    n_faces   = len(faces_cpu)
    _write_binary_stl(faces_cpu, n_faces, dst, tmp)
    print(f"\n[DONE] {dst}  ({n_faces:,} faces)  "
          f"total {time.time()-t0:.0f}s", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Shared binary writer
# ─────────────────────────────────────────────────────────────────────────────

def _write_binary_stl(faces_cpu: np.ndarray, n_faces: int,
                      dst: Path, tmp: Path):
    print(f"[INFO] Writing binary STL ({n_faces:,} faces) → {tmp} ...",
          flush=True)
    t1 = time.time()

    dtype_face = np.dtype([
        ("normal", np.float32, (3,)),
        ("v0",     np.float32, (3,)),
        ("v1",     np.float32, (3,)),
        ("v2",     np.float32, (3,)),
        ("attr",   np.uint16),
    ])
    records           = np.zeros(n_faces, dtype=dtype_face)
    records["normal"] = faces_cpu[:, 0:3]
    records["v0"]     = faces_cpu[:, 3:6]
    records["v1"]     = faces_cpu[:, 6:9]
    records["v2"]     = faces_cpu[:, 9:12]

    with open(tmp, "wb") as f:
        f.write(b"\x00" * HEADER_BYTES)
        f.write(struct.pack("<I", n_faces))
        records.tofile(f)

    os.replace(tmp, dst)
    print(f"  written {dst.stat().st_size/1e6:.0f} MB  "
          f"({time.time()-t1:.1f}s)", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src",         default=str(SRC_DEFAULT))
    parser.add_argument("--dst",         default=str(DST_DEFAULT))
    parser.add_argument("--keep",        action="store_true",
                        help="Keep ASCII source after conversion")
    parser.add_argument("--cpu-fallback", action="store_true",
                        help="Force CPU even if GPU is available")
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    tmp = dst.with_suffix(".tmp")

    if not src.exists():
        sys.exit(f"[ERROR] Not found: {src}")

    size = src.stat().st_size
    stat = os.statvfs(src.parent)
    free = stat.f_bavail * stat.f_frsize
    need = size // 6

    print(f"[INFO] Source     : {src}  ({size/1e9:.2f} GB)", flush=True)
    print(f"[INFO] Free space : {free/1e9:.2f} GB  "
          f"(need ~{need/1e9:.2f} GB)", flush=True)

    if free < need * 1.1:
        sys.exit("[ERROR] Not enough disk space.")

    # Choose GPU or CPU path — neither loads the whole file into RAM
    use_gpu = not args.cpu_fallback
    if use_gpu:
        try:
            import cupy as cp
            cp.cuda.runtime.getDeviceCount()
        except Exception as e:
            print(f"[WARN] GPU not available ({e}) — falling back to CPU",
                  flush=True)
            use_gpu = False

    if use_gpu:
        _convert_gpu(src, dst, tmp)
    else:
        # CPU fallback still needs full file in RAM for multiprocessing
        print("[INFO] Reading file into CPU RAM for CPU path ...", flush=True)
        t0 = time.time()
        with open(src, "rb") as fh:
            mm  = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
            raw = bytes(mm)
            mm.close()
        print(f"  {len(raw)/1e9:.2f} GB read  ({time.time()-t0:.1f}s)",
              flush=True)
        _convert_cpu(raw, dst, tmp)

    if not args.keep:
        print(f"[INFO] Deleting ASCII source ({src.stat().st_size/1e9:.2f} GB) ...",
              flush=True)
        src.unlink()
        print("  freed.", flush=True)

if __name__ == "__main__":
    main()