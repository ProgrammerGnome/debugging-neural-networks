import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from scipy.linalg import svd
import argparse

def load_npz_files(run_dir):
    files = sorted(glob.glob(os.path.join(run_dir, "step_*.npz")))
    data = []
    for f in files:
        data.append(np.load(f, allow_pickle=True))
    return files, data

def analyze_all_weights_svd(run_dir, top_k=50, out_prefix="svd"):
    _, data = load_npz_files(run_dir)
    if not data:
        print("Nincsenek npz fájlok a megadott könyvtárban.")
        return

    # vegyük az első fájlt, listázzuk az összes param__ kulcsot
    first_file_keys = data[0].files
    #param_keys = [k for k in first_file_keys if k.startswith("param__")]
    param_keys = [k for k in first_file_keys if k.startswith("param__") and "weight" in k]
    if not param_keys:
        print("Nincs 'param__' kulcs az npz fájlokban.")
        return

    for param in param_keys:
        print(f"Elemzés: {param}")
        sv_spectra = []
        steps = []
        for i, d in enumerate(data):
            if param not in d.files:
                continue
            W = d[param]
            try:
                s = svd(W, compute_uv=False)
            except Exception as e:
                print(f"SVD hiba {param} step {i}: {e}")
                continue
            sv_spectra.append(s)
            steps.append(i)

        if not sv_spectra:
            print(f"Nincs találat a {param} kulcsra.")
            continue

        sv_spectra = np.array([s[:top_k] if len(s)>=top_k else np.pad(s,(0,top_k-len(s))) for s in sv_spectra])
        plt.figure(figsize=(8,5))
        for k in range(min(top_k, sv_spectra.shape[1])):
            plt.plot(steps, sv_spectra[:,k], alpha=0.6)
        plt.yscale('log')
        plt.title(f"Singular values evolution for {param}")
        plt.xlabel("saved step")
        plt.ylabel("singular value (log scale)")
        plt.tight_layout()
        out_dir = "analysis_outputs"
        os.makedirs(out_dir, exist_ok=True)
        fn = os.path.join(out_dir, f"{out_prefix}_{param.replace('.', '_')}.png")
        plt.savefig(fn)
        plt.close()
        #plt.show()

def analyze_all_activations(run_dir, out_prefix="act_hist"):
    _, data = load_npz_files(run_dir)
    if not data:
        print("Nincsenek npz fájlok a megadott könyvtárban.")
        return

    first_file_keys = data[0].files
    act_keys = [k for k in first_file_keys if k.startswith("act__")]

    if not act_keys:
        print("Nincs 'act__' kulcs az npz fájlokban.")
        return

    idxs = list(range(0, len(data), max(1,len(data)//6)))  # max 6 checkpoint
    for act in act_keys:
        print(f"Elemzés: {act}")
        for ii in idxs:
            d = data[ii]
            if act not in d.files:
                continue
            vals = d[act].ravel()
            plt.figure(figsize=(8,5))
            plt.hist(vals, bins=120, alpha=0.8)
            plt.title(f"Activation distribution: {act}, checkpoint {ii}")
            plt.xlabel("activation value")
            plt.ylabel("count")
            plt.tight_layout()
            out_dir = "analysis_outputs"
            os.makedirs(out_dir, exist_ok=True)
            fn = os.path.join(out_dir, f"{out_prefix}_{act}_checkpoint_{ii}.png")
            plt.savefig(fn)
            plt.close()
            #plt.show()

def get_latest_run(base_dir="runs"):
    if not os.path.exists(base_dir):
        raise ValueError(f"Nincs '{base_dir}' mappa.")
    subdirs = [os.path.join(base_dir, d) for d in os.listdir(base_dir)
               if os.path.isdir(os.path.join(base_dir, d))]
    if not subdirs:
        raise ValueError(f"Nincs elérhető run a '{base_dir}' alatt.")
    latest = max(subdirs, key=os.path.getmtime)
    return latest

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze NN training logs (npz files).")
    parser.add_argument("--run_dir", default=os.path.join("runs", "20250827-222130"),
                        help="Path to run directory (default: runs/20250827-222130).")
    parser.add_argument("--mode", choices=["svd", "activations", "all"], default="all",
                        help="Analysis mode: 'svd', 'activations', or 'all'.")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k singular values to plot (for SVD).")
    args = parser.parse_args()

    run_dir = os.path.join(os.path.dirname(__file__), "..", "runs", "20250827-222130")
    run_dir = os.path.abspath(run_dir)

    print(f"Elemzés a következő run könyvtáron: {run_dir}")

    if args.mode == "svd":
        analyze_all_weights_svd(run_dir, top_k=args.top_k)
    elif args.mode == "activations":
        analyze_all_activations(run_dir)
    else:  # all
        analyze_all_weights_svd(run_dir, top_k=args.top_k)
        analyze_all_activations(run_dir)
