import os
import time
import numpy as np
import torch
import h5py
from typing import Dict

def make_run_dir(base="runs"):
    t = time.strftime("%Y%m%d-%H%M%S")
    d = os.path.join(base, t)
    os.makedirs(d, exist_ok=True)
    return d

def save_step_data(out_dir, step_idx, model: torch.nn.Module, loss, activations: Dict[str, np.ndarray], grads: Dict[str, np.ndarray]=None):
    """
    Mentés .npz formátumban: súlyok, biasok, gradiensek, activations, loss
    (kis projekthez .npz kényelmes; nagyobb loghoz h5py ajánlott)
    """
    state = {}
    for name, p in model.named_parameters():
        state[f"param__{name}"] = p.detach().cpu().numpy()
    if grads:
        for k,v in grads.items():
            state[f"grad__{k}"] = np.array(v)
    for aname, aval in activations.items():
        state[f"act__{aname}"] = np.array(aval)
    state["loss"] = float(loss)
    np.savez_compressed(os.path.join(out_dir, f"step_{step_idx:06d}.npz"), **state)

