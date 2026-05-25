import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR / "notebooks" / "denoiser" / "lit"))

import torch
from datamodules import PeriodicStructureDataModule
from modules import LitEquivariantNoiseNet

import lightning as L
from lightning.pytorch.callbacks import TQDMProgressBar, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

# -----------------------------
# Training file list
# -----------------------------
TRAIN_DIR = Path.home() / "Kite_L"
file_list = [str(TRAIN_DIR / f"dump.gb.{i}") for i in range(51)]

print(f"Training on {len(file_list)} files", flush=True)
print("First few files:", file_list[:5], flush=True)

# -----------------------------
# Datamodule
# -----------------------------
datamodule = PeriodicStructureDataModule(
    file_list=file_list,
    large_cutoff=5.0,
    duplicate=16,
    batch_size=1,
    num_workers=0,
)

# -----------------------------
# Model
# -----------------------------
noise_net = LitEquivariantNoiseNet(
    num_species=1,
    node_dim=64,
    ff_dim=64,
    init_edge_dim=1,
    edge_dim=64,
    num_heads=4,
    num_layers=4,
    sigma_max=0.3,
    cutoff=3.2,
    learn_rate=1e-4,
)

# -----------------------------
# Logging / checkpoints
# -----------------------------
logger = TensorBoardLogger(
    save_dir="./lit_logs/",
    name="equivariant-noise-net-kitel"
)

ckpt_callback = ModelCheckpoint(
    dirpath=logger.log_dir + "/checkpoints",
    filename="kitel-{epoch:02d}-{step}",
    save_top_k=1,
    monitor=None,
    save_last=True,
)

# -----------------------------
# Trainer
# -----------------------------
trainer = L.Trainer(
    max_steps=500,
    logger=logger,
    callbacks=[TQDMProgressBar(refresh_rate=10), ckpt_callback],
    accelerator="cpu",
    devices=1,
)

print("About to start trainer.fit()", flush=True)
trainer.fit(noise_net, datamodule)
print("Finished trainer.fit()", flush=True)

print(f"Best checkpoint: {ckpt_callback.best_model_path}", flush=True)
print(f"Last checkpoint should be in: {logger.log_dir}/checkpoints", flush=True)
