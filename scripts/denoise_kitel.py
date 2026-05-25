import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR / "notebooks" / "denoiser" / "lit"))

import torch
import ase
import ase.io
from sklearn.preprocessing import LabelEncoder
from tqdm import trange

from modules import LitEquivariantNoiseNet
from graphite.nn import periodic_radius_graph

# -----------------------------
# Checkpoint path
# -----------------------------
# Option 1: set the exact checkpoint manually
# CKPT_PATH = Path.home() / "graphite" / "lit_logs" / "equivariant-noise-net-kitel" / "version_0" / "checkpoints" / "last.ckpt"

# Option 2: auto-find the newest checkpoint
ckpt_dir = Path.home() / "graphite" / "lit_logs" / "equivariant-noise-net-kitel"
ckpt_candidates = sorted(ckpt_dir.rglob("*.ckpt"), key=lambda p: p.stat().st_mtime)

if not ckpt_candidates:
    raise FileNotFoundError(f"No checkpoint files found under {ckpt_dir}")

CKPT_PATH = Path("/volume/NFS/pb685/graphite/lit_logs/equivariant-noise-net-kitel/version_8/checkpoints/kitel-epoch=68-step=50000.ckpt")
print(f"Using checkpoint: {CKPT_PATH}", flush=True)

# -----------------------------
# Load trained model
# -----------------------------
noise_net = LitEquivariantNoiseNet.load_from_checkpoint(str(CKPT_PATH))
noise_net.eval()
noise_net.to("cpu")

# -----------------------------
# Denoising helper
# -----------------------------
@torch.no_grad()
def denoise_snapshot(atoms, model, scale=1.0, steps=8):
    x = LabelEncoder().fit_transform(atoms.numbers)
    species = torch.tensor(x, dtype=torch.long)
    pos = torch.tensor(atoms.positions, dtype=torch.float)
    cell = torch.tensor(atoms.cell.tolist(), dtype=torch.float)

    pos *= scale
    cell *= scale

    pos_traj = [atoms.positions.copy()]
    for _ in trange(steps):
        edge_index, edge_vec = periodic_radius_graph(pos, r=3.2, cell=cell)
        edge_len = torch.linalg.norm(edge_vec, dim=1, keepdim=True)
        _, disp = model(species, edge_index, edge_attr=edge_len, edge_vec=edge_vec)
        pos -= disp
        pos_traj.append(pos.clone().numpy() / scale)

    return pos_traj

# -----------------------------
# Synthetic input / output dirs
# -----------------------------
SYNTH_ROOT = Path.home() / "KiteL_synthetic_0K"
OUT_ROOT = SYNTH_ROOT / "denoised"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

print("Synthetic root:", SYNTH_ROOT, flush=True)
print("Output root:", OUT_ROOT, flush=True)

for subset in ["kite", "L"]:
    in_dir = SYNTH_ROOT / subset
    out_dir = OUT_ROOT / subset
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted([p for p in in_dir.rglob("*") if p.is_file()])
    print(f"\nProcessing subset={subset}, num_files={len(files)}", flush=True)

    for path in files:
        try:
            noisy_atoms = ase.io.read(str(path))

            pos_traj = denoise_snapshot(
                noisy_atoms,
                noise_net.model,
                scale=1.0,
                steps=8
            )

            denoising_traj = [
                ase.Atoms(
                    symbols=noisy_atoms.get_chemical_symbols(),
                    positions=pos,
                    cell=noisy_atoms.cell,
                    pbc=True
                )
                for pos in pos_traj
            ]

            for atoms in denoising_traj:
                atoms.wrap()

            out_path = out_dir / f"{path.stem}.denoised_.extxyz"
            ase.io.write(str(out_path), denoising_traj)
            print(f"Saved {out_path}", flush=True)

        except Exception as e:
            print(f"Failed on {path}: {e}", flush=True)

print("Finished denoising.", flush=True)
