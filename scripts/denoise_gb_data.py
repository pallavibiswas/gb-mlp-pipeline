import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import sys
from pathlib import Path
import sys

GRAPHITE_ROOT = Path.home() / "graphite"
sys.path.insert(0, str(GRAPHITE_ROOT))
sys.path.insert(0, str(GRAPHITE_ROOT / "notebooks" / "denoiser" / "lit"))


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
# Generic GB data denoising
# -----------------------------
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input-root", required=True)
parser.add_argument("--output-root", required=True)
parser.add_argument("--pattern", default="*")
parser.add_argument("--force-symbol", default="H")
parser.add_argument("--steps", type=int, default=8)
parser.add_argument("--overwrite", action="store_true")
args = parser.parse_args()

in_root = Path(args.input_root).expanduser()
out_root = Path(args.output_root).expanduser()
out_root.mkdir(parents=True, exist_ok=True)

print("Input root:", in_root, flush=True)
print("Output root:", out_root, flush=True)
print("Pattern:", args.pattern, flush=True)
print("Force symbol:", args.force_symbol, flush=True)

files = sorted([p for p in in_root.rglob(args.pattern) if p.is_file()])
print("Found files:", len(files), flush=True)

def read_atoms_any(path):
    try:
        return ase.io.read(str(path), index=0)
    except Exception as e1:
        try:
            return ase.io.read(str(path), format="lammps-dump-text", index=0)
        except Exception as e2:
            raise RuntimeError(f"ASE read failed normally ({e1}) and as lammps-dump-text ({e2})")

for path in files:
    rel = path.relative_to(in_root)
    out_dir = out_root / rel.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # Use full filename to avoid dump.gb.0, dump.gb.1 all collapsing to dump.gb
    out_path = out_dir / f"{path.name}.denoised_.extxyz"

    if out_path.exists() and not args.overwrite:
        print(f"Skipping existing {out_path}", flush=True)
        continue

    try:
        noisy_atoms = read_atoms_any(path)

        if args.force_symbol:
            noisy_atoms.set_chemical_symbols([args.force_symbol] * len(noisy_atoms))

        pos_traj = denoise_snapshot(
            noisy_atoms,
            noise_net.model,
            scale=1.0,
            steps=args.steps
        )

        symbols = noisy_atoms.get_chemical_symbols()

        denoising_traj = [
            ase.Atoms(
                symbols=symbols,
                positions=pos,
                cell=noisy_atoms.cell,
                pbc=True
            )
            for pos in pos_traj
        ]

        for atoms in denoising_traj:
            atoms.wrap()

        ase.io.write(str(out_path), denoising_traj)
        print(f"Saved {out_path}", flush=True)

    except Exception as e:
        print(f"Failed on {path}: {e}", flush=True)

print("Finished denoising.", flush=True)
