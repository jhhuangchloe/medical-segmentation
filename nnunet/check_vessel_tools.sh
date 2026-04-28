#!/usr/bin/env bash
set -euo pipefail

# ---- USER EDIT (only if needed) ----
ENV_PREFIX="$HOME/miniconda3/envs/sam2"
# ------------------------------------

echo "=== System ==="
uname -a
echo "Arch: $(uname -m)"
echo

if [[ ! -d "$ENV_PREFIX" ]]; then
  echo "[ERROR] ENV_PREFIX not found: $ENV_PREFIX"
  exit 1
fi

PY="$ENV_PREFIX/bin/python"
TS="$ENV_PREFIX/bin/TotalSegmentator"

echo "=== Env prefix ==="
echo "$ENV_PREFIX"
echo

echo "=== Python (inside env) ==="
"$PY" - <<'PY'
import platform, sys
print("Python:", sys.version.replace("\n"," "))
print("Platform:", platform.platform())
print("Machine:", platform.machine())
PY
echo

echo "=== Core imports (inside env) ==="
"$PY" - <<'PY'
mods = ["numpy","scipy","skimage","nibabel","SimpleITK","torch","totalsegmentator"]
for m in mods:
    try:
        mod = __import__(m)
        ver = getattr(mod, "__version__", "unknown")
        print(f"[OK]   import {m} (version={ver})")
    except Exception as e:
        print(f"[MISS] import {m} -> {e}")

try:
    import torch
    print(f"[INFO] torch MPS available: {torch.backends.mps.is_available()}")
except Exception:
    pass
PY
echo

echo "=== TotalSegmentator CLI discovery ==="
if [[ -x "$TS" ]]; then
  echo "[OK]   Using: $TS"
else
  echo "[ERROR] TotalSegmentator binary not found at: $TS"
  echo "        (If it's installed, locate it with: find \"$ENV_PREFIX/bin\" -maxdepth 1 -iname 'TotalSegmentator*')"
  exit 1
fi
echo

echo "=== TotalSegmentator help header ==="
"$TS" -h | head -n 25 || true
echo

echo "=== Does task 'liver_vessels' exist? ==="
if "$TS" -h | grep -q "liver_vessels"; then
  echo "[OK]   liver_vessels appears available in this install."
else
  echo "[WARN] liver_vessels not visible in help output."
  echo "       You may need a different TotalSegmentator version/weights/license depending on your setup."
fi
echo

echo "=== Notes ==="
echo "- Your env may be x86_64 (Rosetta) even on M1; that's OK for CPU runs, but slower."
echo "- We'll run segmentation with --device cpu for reliability on macOS."
echo "Done."

