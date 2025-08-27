import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import yaml
import os
from models import SimpleAutoencoder
from utils import make_run_dir, save_step_data
from tqdm import tqdm

# --- egyszerű synthetic dataset generálása (tetszőleges helyett cserélhető)
def make_synthetic(n_samples=256, input_dim=16, seed=0):
    rng = np.random.RandomState(seed)
    # például alacsony-rangsú gauss + kis zavar
    latent = rng.randn(n_samples, 4)
    W = rng.randn(4, input_dim)
    X = latent.dot(W) + 0.05 * rng.randn(n_samples, input_dim)
    return X.astype(np.float32)

# activation hook regisztrálás (layer név -> utolsó aktiváció)
def register_activation_hooks(model, activations):
    hooks = []
    for name, module in model.named_modules():
        # Linear után vesszük az aktivációt: ha module egy aktivációs függvény, akkor hook-oljuk
        if isinstance(module, (nn.ReLU, nn.Tanh, nn.Sigmoid, nn.LeakyReLU)):
            def get_hook(n):
                def hook(m, inp, out):
                    # out lehet tensor
                    activations[n] = out.detach().cpu().numpy()
                return hook
            hooks.append(module.register_forward_hook(get_hook(name)))
    return hooks

def compute_param_grads(model):
    out = {}
    for name, p in model.named_parameters():
        if p.grad is not None:
            out[name] = p.grad.detach().cpu().numpy().copy()
        else:
            out[name] = None
    return out

def main(config_path="experiments/config.yaml"):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    torch.manual_seed(cfg.get("seed", 0))
    device = torch.device(cfg.get("device", "cpu"))

    X = make_synthetic(cfg["dataset"]["n_samples"], cfg["dataset"]["input_dim"], cfg.get("seed",0))
    ds = TensorDataset(torch.from_numpy(X))
    dl = DataLoader(ds, batch_size=cfg["training"]["batch_size"], shuffle=True)

    model = SimpleAutoencoder(input_dim=cfg["dataset"]["input_dim"],
                              hidden_dims=cfg["model"]["hidden_dims"],
                              activation=cfg["model"]["activation"],
                              dropout=cfg["model"]["dropout"]).to(device)

    optimiz = optim.Adam(model.parameters(), lr=cfg["training"]["lr"])
    criterion = nn.MSELoss()

    out_dir = make_run_dir(cfg["save"]["out_dir"])
    step = 0
    hooks_activations = {}
    hooks = register_activation_hooks(model, hooks_activations)

    for epoch in range(cfg["training"]["epochs"]):
        pbar = tqdm(dl, desc=f"Epoch {epoch+1}/{cfg['training']['epochs']}")
        for batch in pbar:
            x = batch[0].to(device)
            optimiz.zero_grad()
            recon, z = model(x)
            loss = criterion(recon, x)
            loss.backward()

            # collect grads and activations
            grads = compute_param_grads(model)
            # activations are in hooks_activations; add latent z also
            activs = dict(hooks_activations)
            activs["latent_z"] = z.detach().cpu().numpy()

            # mentés konfigurálható gyakorisággal
            if step % cfg["training"].get("log_every_steps", 10) == 0:
                save_step_data(out_dir, step, model, float(loss.cpu().item()), activs, grads)

            optimiz.step()
            step += 1
            pbar.set_postfix({"loss": float(loss.cpu().item())})

    # cleanup hooks
    for h in hooks:
        h.remove()

    print("Végzett. Mentett run:", out_dir)

if __name__ == "__main__":
    main()

