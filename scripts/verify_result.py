import sys
import numpy as np

out = np.fromfile(sys.argv[1], dtype=np.float16).astype(np.float32)
gold = np.fromfile(sys.argv[2], dtype=np.float16).astype(np.float32)

if out.size != gold.size:
    print(f"size mismatch: {out.size} vs {gold.size}")
    sys.exit(1)

fro = np.linalg.norm(out - gold) / max(np.linalg.norm(gold), 1e-12)
cos = float(out @ gold / (np.linalg.norm(out) * np.linalg.norm(gold) + 1e-12))

print(f"fro_err={fro:.3e}  cos={cos:.8f}")
ok = fro < 1e-3 and cos > 0.9999
print("test pass" if ok else "result error")
sys.exit(0 if ok else 1)