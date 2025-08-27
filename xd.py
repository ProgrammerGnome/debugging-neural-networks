import numpy as np
import glob
import os

run_dir = "runs/20250827-222130"
files = sorted(glob.glob(os.path.join(run_dir, "step_*.npz")))

if not files:
    print("Nincsenek npz fájlok.")
else:
    data = np.load(files[0], allow_pickle=True)
    print("Első npz fájl kulcsai:")
    for k in data.files:
        print(k)
