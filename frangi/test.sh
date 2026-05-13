python3 - << 'EOF'
import pydicom, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import label

files = sorted(glob.glob("../data/hybrid_ultra/*.dcm"))
ds    = pydicom.dcmread(files[149])
raw   = ds.pixel_array.astype(np.float32)
raw   = raw * float(getattr(ds,"RescaleSlope",1)) \
           + float(getattr(ds,"RescaleIntercept",0))

# 只看亮结构（血管）
p99  = np.percentile(raw, 99)
mask = raw > p99

labeled, n = label(mask)
sizes = np.bincount(labeled.ravel())[1:]  # exclude background
sizes = np.sort(sizes)[::-1]

print(f"p99 threshold : {p99:.0f}")
print(f"Number of bright blobs : {n}")
print(f"Top 10 blob sizes (voxels) :")
for i, s in enumerate(sizes[:10]):
    diameter_vox = np.sqrt(s / np.pi) * 2   # assume circular cross-section
    diameter_mm  = diameter_vox * 0.0244
    print(f"  blob {i+1:2d}: {s:6d} voxels  "
          f"~{diameter_vox:.1f} vox diameter  "
          f"~{diameter_mm:.2f} mm")
EOF