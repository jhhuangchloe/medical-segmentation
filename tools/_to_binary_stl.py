# convert_inplace.py  — deletes ASCII as soon as binary is written
import struct, os, time

src = "results_frangi/vessels.stl"
dst = "results_frangi/vessels_binary.stl"
tmp = "results_frangi/vessels_binary.tmp"   # write to .tmp first, then rename

size = os.path.getsize(src)
print(f"ASCII STL size: {size/1e9:.2f} GB", flush=True)
print(f"Binary will need: ~{size/6/1e9:.2f} GB "
      f"(binary is ~6x smaller than ASCII)", flush=True)

# Check available space
stat = os.statvfs(".")
free = stat.f_bavail * stat.f_frsize
print(f"Free space: {free/1e9:.2f} GB", flush=True)

if free < size / 6 * 1.2:
    print("WARNING: may still be tight — consider option 1 (different partition)")

faces = []
t0 = time.time()
bytes_read = 0

print("Parsing ...", flush=True)
with open(src, "r") as f:
    normal, verts = None, []
    for line in f:
        bytes_read += len(line.encode())
        line = line.strip()
        if line.startswith("facet normal"):
            parts  = line.split()
            normal = (float(parts[2]), float(parts[3]), float(parts[4]))
            verts  = []
        elif line.startswith("vertex"):
            parts = line.split()
            verts.append((float(parts[1]), float(parts[2]), float(parts[3])))
        elif line.startswith("endfacet") and normal and len(verts) == 3:
            faces.append((normal, verts))
        if len(faces) % 500_000 == 0 and len(faces) > 0:
            pct = bytes_read / size * 100
            ela = time.time() - t0
            eta = ela / (bytes_read / size) - ela
            print(f"  {len(faces):,} faces  {pct:.1f}%  "
                  f"elapsed {ela:.0f}s  ETA {eta:.0f}s", flush=True)

print(f"Writing binary to {tmp} ...", flush=True)
with open(tmp, "wb") as f:
    f.write(b"\0" * 80)
    f.write(struct.pack("<I", len(faces)))
    for (nx, ny, nz), (v0, v1, v2) in faces:
        f.write(struct.pack("<fff", nx, ny, nz))
        f.write(struct.pack("<fff", *v0))
        f.write(struct.pack("<fff", *v1))
        f.write(struct.pack("<fff", *v2))
        f.write(struct.pack("<H", 0))

os.rename(tmp, dst)
print(f"Deleting ASCII STL ({size/1e9:.2f} GB) ...", flush=True)
os.remove(src)
print(f"DONE → {dst}  ({os.path.getsize(dst)/1e6:.0f} MB free'd "
      f"{size/1e9:.2f} GB)", flush=True)