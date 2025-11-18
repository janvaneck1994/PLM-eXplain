#!/usr/bin/env python3
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Tuple
import subprocess
from torch.optim.lr_scheduler import LambdaLR


SEED = 0
BATCH_SIZE = int(128)
NUM_EPOCHS = 10


LAMBDA_LIST: List[float] = [0,1,2,5,10,20]

GPU_IDS: List[int] = [1,2,3,4,5,6]  
MAX_JOBS_PER_GPU = 3

EMB_BASE_DIR = "./data/output/embeddings"

MODEL_SPECS = [
    {"name": "ProtBert-BFD", "stub": "prot_bert_bfd"}, 
    {"name": "ESM2-35M",  "stub": "esm2_t12_35M_UR50D"},
    {"name": "ESM2-650M", "stub": "esm2_t33_650M_UR50D"},
]

LOG_DIR = "launcher_logs"
TRAINED_MODELS_DIR = "trained_models"

import math
import torch
import numpy as np
import random
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import f1_score, r2_score

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from PartitionedEmbeddingModel import PartitionedEmbeddingModel
from ProteinDataset import ProteinDataset


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_class_weights_from_csv(csv_path, column_name, num_classes, mapping_dict):
    df = pd.read_csv(csv_path)
    labels = df[column_name].astype(str)
    labels = labels.map(lambda x: mapping_dict.get(x, -1))
    labels = labels[labels != -1].astype(int).values
    counts = np.bincount(labels, minlength=num_classes)
    weights = 1.0 / (counts + 1e-6)
    weights = weights / weights.sum() * num_classes
    return torch.tensor(weights, dtype=torch.float)


def pick_embeddings_from_batch(batch):
    """
    Return the 2D float tensor for embeddings, tolerant to different keys.
    """
    candidates = ["ESM2Embedding", "CARPEmbedding", "Embedding", "embeddings"]
    for k in candidates:
        if k in batch:
            return batch[k]
    for v in batch.values():
        if torch.is_tensor(v) and v.dim() == 2 and v.dtype.is_floating_point:
            return v
    raise KeyError("Embedding tensor not found in batch.")


def build_paths(emb_base_dir: str, model_stub: str, split: str):
    d = os.path.join(emb_base_dir, model_stub)
    return (
        os.path.join(d, f"values_{split}.csv"),
        os.path.join(d, f"embeddings_{split}.npy"),
    )

def collate_fn(batch):
    collated = {}
    for key in batch[0]:
        arrs = [sample[key] for sample in batch]
        collated[key] = torch.from_numpy(np.stack(arrs))
    return collated



def make_loaders(emb_base_dir: str, model_stub: str, batch_size: int):
    tr_vals, tr_emb = build_paths(emb_base_dir, model_stub, "train")
    va_vals, va_emb = build_paths(emb_base_dir, model_stub, "val")

    train_ds = ProteinDataset(tr_vals, tr_emb, chunksize=500)
    val_ds   = ProteinDataset(va_vals, va_emb, chunksize=500)

    train_loader = train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        collate_fn=collate_fn
    )
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, collate_fn=collate_fn)
    return train_ds, val_ds, train_loader, val_loader


@torch.no_grad()
def evaluate_on_val(model, loader, device, rec_loss_fn):
    model.eval()
    all_amino_preds, all_amino_labels = [], []
    all_ss3_preds,   all_ss3_labels   = [], []
    all_ss8_preds,   all_ss8_labels   = [], []
    all_arom_preds,  all_arom_labels  = [], []
    all_asa_preds,   all_asa_labels   = [], []
    all_gravy_preds, all_gravy_labels = [], []

    total_rec, batches = 0.0, 0

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        embeddings = pick_embeddings_from_batch(batch)

        amino_lab = batch['AminoAcid']
        if amino_lab.dim() > 1 and amino_lab.size(-1) > 1:
            amino_lab = amino_lab.argmax(dim=1)

        ss3_lab = batch['SS3']
        if ss3_lab.dim() > 1 and ss3_lab.size(-1) > 1:
            ss3_lab = ss3_lab.argmax(dim=1)

        ss8_lab = batch['SS8']
        if ss8_lab.dim() > 1 and ss8_lab.size(-1) > 1:
            ss8_lab = ss8_lab.argmax(dim=1)

        asa   = batch['ASA'].squeeze()
        gravy = batch['GRAVY'].squeeze()
        arom  = batch['AROM']

        known, unknown, recon, combined, _ = model(embeddings, lambda_grl=1.0)
        total_rec += rec_loss_fn(recon, embeddings).item()
        batches += 1

        pred_amino = known[:, :20].argmax(dim=1)
        pred_ss3   = known[:, 20:23].argmax(dim=1)
        pred_ss8   = known[:, 23:31].argmax(dim=1)
        pred_asa   = known[:, 31]
        pred_gravy = known[:, 32]
        pred_arom  = (((known[:, 33] + 1) / 2) > 0.5).long()

        all_amino_preds.append(pred_amino.cpu().numpy())
        all_amino_labels.append(amino_lab.cpu().numpy())
        all_ss3_preds.append(pred_ss3.cpu().numpy())
        all_ss3_labels.append(ss3_lab.cpu().numpy())
        all_ss8_preds.append(pred_ss8.cpu().numpy())
        all_ss8_labels.append(ss8_lab.cpu().numpy())
        all_arom_preds.append(pred_arom.cpu().numpy())
        all_arom_labels.append(arom.cpu().numpy())
        all_asa_preds.append(pred_asa.cpu().numpy())
        all_asa_labels.append(asa.cpu().numpy())
        all_gravy_preds.append(pred_gravy.cpu().numpy())
        all_gravy_labels.append(gravy.cpu().numpy())

    if batches == 0:
        return dict(rec=np.nan, f1_amino=np.nan, f1_ss3=np.nan, f1_ss8=np.nan, f1_arom=np.nan, r2_asa=np.nan, r2_gravy=np.nan)

    all_amino_preds = np.concatenate(all_amino_preds)
    all_amino_labels = np.concatenate(all_amino_labels)
    all_ss3_preds = np.concatenate(all_ss3_preds)
    all_ss3_labels = np.concatenate(all_ss3_labels)
    all_ss8_preds = np.concatenate(all_ss8_preds)
    all_ss8_labels = np.concatenate(all_ss8_labels)
    all_arom_preds = np.concatenate(all_arom_preds)
    all_arom_labels = np.concatenate(all_arom_labels)
    all_asa_preds = np.concatenate(all_asa_preds)
    all_asa_labels = np.concatenate(all_asa_labels)
    all_gravy_preds = np.concatenate(all_gravy_preds)
    all_gravy_labels = np.concatenate(all_gravy_labels)

    return dict(
        rec = total_rec / batches,
        f1_amino = f1_score(all_amino_labels, all_amino_preds, average="weighted"),
        f1_ss3   = f1_score(all_ss3_labels, all_ss3_preds,   average="weighted"),
        f1_ss8   = f1_score(all_ss8_labels, all_ss8_preds,   average="weighted"),
        f1_arom  = f1_score(all_arom_labels, all_arom_preds, average="weighted"),
        r2_asa   = r2_score(all_asa_labels, all_asa_preds),
        r2_gravy = r2_score(all_gravy_labels, all_gravy_preds),
    )


def train_worker(emb_base_dir: str, model_name: str, model_stub: str, lambda_adv: float):
    """
    One job: (model_stub, lambda_adv) on the SINGLE GPU visible via CUDA_VISIBLE_DEVICES.
    Validation-only evaluation each epoch. Saves metrics + checkpoints.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(SEED)

    # label mappings
    amino_acid_dict = {
        'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7,
        'K': 8, 'L': 9, 'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14,
        'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19
    }
    ss3_dict = {'H': 0, 'E': 1, 'C': 2}
    ss8_dict = {'H': 0, 'E': 1, 'G': 2, 'I': 3, 'B': 4, 'T': 5, 'S': 6, '-': 7}

    train_ds, val_ds, train_loader, val_loader = make_loaders(emb_base_dir, model_stub, BATCH_SIZE)

    def lr_lambda(step):
        warmup_steps = 1000
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        return 1.0

    # weights from TRAIN
    train_csv_path = os.path.join(emb_base_dir, model_stub, "values_train.csv")
    amino_weights = compute_class_weights_from_csv(train_csv_path, 'AminoAcid', 20, amino_acid_dict).to(device)
    ss3_weights   = compute_class_weights_from_csv(train_csv_path, 'SS3', 3, ss3_dict).to(device)
    ss8_weights   = compute_class_weights_from_csv(train_csv_path, 'SS8', 8, ss8_dict).to(device)

    rec_criterion   = nn.L1Loss()
    class_criterion = nn.CrossEntropyLoss(weight=amino_weights)
    ss3_criterion   = nn.CrossEntropyLoss(weight=ss3_weights)
    ss8_criterion   = nn.CrossEntropyLoss(weight=ss8_weights)
    asa_criterion   = nn.L1Loss()
    gravy_criterion = nn.L1Loss()
    arom_criterion  = nn.BCELoss()

    input_dim  = train_ds.embeddings.shape[1] if hasattr(train_ds, "embeddings") else 1024
    known_dim  = 34
    hidden_dim = input_dim

    base_dir = os.path.join(TRAINED_MODELS_DIR, model_stub)
    os.makedirs(base_dir, exist_ok=True)
    lambda_dir = os.path.join(base_dir, f"lambda_{lambda_adv}")
    os.makedirs(lambda_dir, exist_ok=True)

    model = PartitionedEmbeddingModel(input_dim, known_dim, hidden_dim, adv_output_dim=34).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.00003)
    scheduler = LambdaLR(optimizer, lr_lambda)

    epoch_metrics = []
    current_iteration = 0

    for epoch in range(NUM_EPOCHS):
        model.train()
        train_total_loss = 0.0
        train_rec_loss_total = 0.0
        train_batches = 0

        for batch in tqdm(train_loader, desc=f"[{model_stub} | λ={lambda_adv}] Epoch {epoch+1}/{NUM_EPOCHS} (Train)"):
            current_iteration += 1
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            embeddings = pick_embeddings_from_batch(batch)

            # labels (allow one-hot)
            amino_acid = batch['AminoAcid']
            if amino_acid.dim() > 1 and amino_acid.size(-1) > 1:
                amino_acid = amino_acid.argmax(dim=1)
            ss3_labels = batch['SS3']
            if ss3_labels.dim() > 1 and ss3_labels.size(-1) > 1:
                ss3_labels = ss3_labels.argmax(dim=1)
            ss8_labels = batch['SS8']
            if ss8_labels.dim() > 1 and ss8_labels.size(-1) > 1:
                ss8_labels = ss8_labels.argmax(dim=1)

            asa   = batch['ASA'].squeeze()
            gravy = batch['GRAVY'].squeeze()
            arom  = batch['AROM']

            optimizer.zero_grad()
            known, unknown, reconstructed, combined, adv_pred = model(embeddings, lambda_grl=1.0)

            rec_loss   = rec_criterion(reconstructed, embeddings)
            class_loss = class_criterion(known[:, :20], amino_acid)
            ss3_loss   = ss3_criterion(known[:, 20:23], ss3_labels)
            ss8_loss   = ss8_criterion(known[:, 23:31], ss8_labels)
            asa_loss   = asa_criterion(known[:, 31], asa)
            gravy_loss = gravy_criterion(known[:, 32], gravy)
            arom_prob  = (known[:, 33] + 1) / 2
            arom_loss  = arom_criterion(arom_prob, arom.float().squeeze())

            adv_loss_amino = class_criterion(adv_pred[:, :20], amino_acid)
            adv_loss_ss3   = ss3_criterion(adv_pred[:, 20:23], ss3_labels)
            adv_loss_ss8   = ss8_criterion(adv_pred[:, 23:31], ss8_labels)
            adv_loss_asa   = asa_criterion(adv_pred[:, 31], asa)
            adv_loss_gravy = gravy_criterion(adv_pred[:, 32], gravy)
            adv_loss_arom  = arom_criterion(torch.sigmoid(adv_pred[:, 33]), arom.float().squeeze())

            adv_loss_total = (adv_loss_amino + adv_loss_ss3 + adv_loss_ss8 +
                              adv_loss_asa + adv_loss_gravy + adv_loss_arom)

            total_loss = (100.0 * rec_loss +
                          class_loss + ss3_loss + ss8_loss +
                          asa_loss + gravy_loss + arom_loss +
                          float(lambda_adv) * adv_loss_total)

            total_loss.backward()
            optimizer.step()
            scheduler.step()

            train_total_loss += float(total_loss.item())
            train_rec_loss_total += float(rec_loss.item())
            train_batches += 1

        avg_train_loss = train_total_loss / max(1, train_batches)
        avg_train_rec_loss = train_rec_loss_total / max(1, train_batches)

        val_metrics = evaluate_on_val(model, val_loader, device, nn.L1Loss())

        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}]  λ={lambda_adv}  model={model_stub}")
        print(f"  Train Loss: {avg_train_loss:.6f} (rec: {avg_train_rec_loss:.6f})")
        print(f"  Val   -> F1(Amino): {val_metrics['f1_amino']:.6f}, F1(SS3): {val_metrics['f1_ss3']:.6f}, "
              f"F1(SS8): {val_metrics['f1_ss8']:.6f}, F1(AROM): {val_metrics['f1_arom']:.6f}, "
              f"R2(ASA): {val_metrics['r2_asa']:.6f}, R2(GRAVY): {val_metrics['r2_gravy']:.6f}, "
              f"Rec: {val_metrics['rec']:.6f}")

        epoch_row = {
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "train_rec_loss": avg_train_rec_loss,
            "val_f1_amino": val_metrics["f1_amino"],
            "val_f1_ss3":   val_metrics["f1_ss3"],
            "val_f1_ss8":   val_metrics["f1_ss8"],
            "val_f1_arom":  val_metrics["f1_arom"],
            "val_r2_asa":   val_metrics["r2_asa"],
            "val_r2_gravy": val_metrics["r2_gravy"],
            "val_rec":      val_metrics["rec"],
        }
        pd.DataFrame([epoch_row]).to_csv(os.path.join(lambda_dir, f"epoch_{epoch+1}_metrics.csv"), index=False)
        torch.save(model.state_dict(), os.path.join(lambda_dir, f"epoch_{epoch+1}.pt"))

def launch():
    Path(LOG_DIR).mkdir(parents=True, exist_ok=True)

    jobs: List[Tuple[float, str, str]] = []
    for l in LAMBDA_LIST:
        for spec in MODEL_SPECS:
            jobs.append((l, spec["name"], spec["stub"]))

    lambda_to_gpu: Dict[float, int] = {}
    for i, l in enumerate(LAMBDA_LIST):
        lambda_to_gpu[l] = GPU_IDS[i % len(GPU_IDS)]

    queues: Dict[int, List[Tuple[float, str, str]]] = {gid: [] for gid in GPU_IDS}
    running: Dict[int, List[Tuple[subprocess.Popen, str]]] = {gid: [] for gid in GPU_IDS}

    for l, mname, mstub in jobs:
        gid = lambda_to_gpu[l]
        queues[gid].append((l, mname, mstub))

    print("GPU assignment per λ:")
    for l in LAMBDA_LIST:
        print(f"  λ={l} -> GPU {lambda_to_gpu[l]}")
    print(f"Max concurrent jobs per GPU: {MAX_JOBS_PER_GPU}")

    def start_jobs_on_gpu(gpu_id: int):
        while queues[gpu_id] and len(running[gpu_id]) < MAX_JOBS_PER_GPU:
            l, mname, mstub = queues[gpu_id].pop(0)

            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            env["WORKER_MODE"] = "1"
            env["JOB_EMB_BASE_DIR"] = EMB_BASE_DIR
            env["JOB_MODEL_NAME"] = mname
            env["JOB_MODEL_STUB"] = mstub
            env["JOB_LAMBDA"] = str(l)

            log_path = Path(LOG_DIR) / f"gpu{gpu_id}_lambda{l}_{mstub}.log"
            lf = open(log_path, "w")

            cmd = [sys.executable, sys.argv[0]]
            proc = subprocess.Popen(cmd, stdout=lf, stderr=lf, env=env)
            running[gpu_id].append((proc, str(log_path)))

    for gid in GPU_IDS:
        start_jobs_on_gpu(gid)

    while True:
        any_running = False
        for gid in GPU_IDS:
            survivors = []
            for proc, log in running[gid]:
                ret = proc.poll()
                if ret is None:
                    any_running = True
                    survivors.append((proc, log))
                else:
                    pass
            running[gpu_id := gid] = survivors

            start_jobs_on_gpu(gid)

        if not any_running and all(len(queues[g]) == 0 for g in GPU_IDS):
            break
        time.sleep(1)

    print("All jobs completed.")


if __name__ == "__main__":
    if os.getenv("WORKER_MODE", "") == "1":
        emb_base_dir = os.getenv("JOB_EMB_BASE_DIR", EMB_BASE_DIR)
        model_name   = os.getenv("JOB_MODEL_NAME", "MODEL")
        model_stub   = os.getenv("JOB_MODEL_STUB", "")
        lambda_adv   = float(os.getenv("JOB_LAMBDA", "0"))
        train_worker(emb_base_dir, model_name, model_stub, lambda_adv)
    else:
        launch()
