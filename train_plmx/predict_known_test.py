#!/usr/bin/env python3
import os
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

SEED = 42
BATCH_SIZE = 256

LAMBDA_LIST: List[float] = [1.0]
EMB_BASE_DIR = "./data/output/embeddings"
TRAINED_MODELS_DIR = "trained_models"
OUTPUT_DIR = "predictions_test"

MODEL_SPECS = [
    {"name": "ProtBert-BFD", "stub": "prot_bert_bfd"},
    {"name": "ESM2-35M",     "stub": "esm2_t12_35M_UR50D"},
    {"name": "ESM2-650M",    "stub": "esm2_t33_650M_UR50D"},
]

from PartitionedEmbeddingModel import PartitionedEmbeddingModel
from ProteinDataset import ProteinDataset

def set_seed(seed: int):
    import random
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def build_paths(emb_base_dir: str, model_stub: str, split: str):
    d = os.path.join(emb_base_dir, model_stub)
    return (
        os.path.join(d, f"values_{split}.csv"),
        os.path.join(d, f"embeddings_{split}.npz"),
    )

@torch.no_grad()
def run_predictions(
    ckpt_path: str,
    values_csv: str,
    emb_npz: str,
    out_csv_path: str,
    out_metrics_csv_path: str,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = ProteinDataset(values_csv, emb_npz)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)

    input_dim  = ds.embeddings.shape[1] if hasattr(ds, "embeddings") else 1024
    known_dim  = 34
    hidden_dim = input_dim

    model = PartitionedEmbeddingModel(input_dim, known_dim, hidden_dim, adv_output_dim=34).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()

    ids = getattr(ds, "row_ids", [str(i) for i in range(len(ds))])

    def pick_embeddings(batch):
        for k in ["ESM2Embedding", "CARPEmbedding", "Embedding", "embeddings"]:
            if k in batch:
                return batch[k]
        for v in batch.values():
            if torch.is_tensor(v) and v.dim() == 2 and v.dtype.is_floating_point:
                return v
        raise KeyError("Embedding tensor not found in batch.")

    pred_rows = []

    aa_preds,  aa_trues  = [], []
    ss3_preds, ss3_trues = [], []
    ss8_preds, ss8_trues = [], []
    asa_preds, asa_trues = [], []
    gravy_preds, gravy_trues = [], []
    arom_preds, arom_trues = [], []

    recon_mae_list = []

    row_cursor = 0
    for batch in tqdm(loader, desc="Predicting"):
        emb = pick_embeddings(batch).to(device)

        known, unknown, recon, combined, _ = model(emb, lambda_grl=1.0)

        batch_mae = torch.mean(torch.abs(recon - emb), dim=1)
        recon_mae_list.extend(batch_mae.detach().cpu().numpy().tolist())

        # Heads
        logits_aa  = known[:, :20]
        logits_ss3 = known[:, 20:23]
        logits_ss8 = known[:, 23:31]
        pred_asa   = known[:, 31]
        pred_gravy = known[:, 32]
        arom_prob  = (known[:, 33] + 1) / 2
        arom_pred  = (arom_prob > 0.5).long()

        # To numpy
        aa_idx   = torch.argmax(logits_aa,  dim=1).cpu().numpy()
        ss3_idx  = torch.argmax(logits_ss3, dim=1).cpu().numpy()
        ss8_idx  = torch.argmax(logits_ss8, dim=1).cpu().numpy()
        asa_np   = pred_asa.cpu().numpy()
        gravy_np = pred_gravy.cpu().numpy()
        arom_np  = arom_pred.cpu().numpy()
        arom_p   = arom_prob.cpu().numpy()

        bs = aa_idx.shape[0]
        id_slice = ids[row_cursor: row_cursor + bs]

        for i in range(bs):
            pred_rows.append({
                "ID": id_slice[i],
                "pred_amino_idx": int(aa_idx[i]),
                "pred_ss3_idx":   int(ss3_idx[i]),
                "pred_ss8_idx":   int(ss8_idx[i]),
                "pred_asa":       float(asa_np[i]),
                "pred_gravy":     float(gravy_np[i]),
                "pred_arom":      int(arom_np[i]),
                "pred_arom_prob": float(arom_p[i]),
            })

        def get(name):
            return batch.get(name, None)

        if get('AminoAcid') is not None:
            aa_true = torch.argmax(batch['AminoAcid'], dim=1).cpu().numpy()
            aa_trues.append(aa_true); aa_preds.append(aa_idx)

        if get('SS3') is not None:
            ss3_true = torch.argmax(batch['SS3'], dim=1).cpu().numpy()
            ss3_trues.append(ss3_true); ss3_preds.append(ss3_idx)

        if get('SS8') is not None:
            ss8_true = torch.argmax(batch['SS8'], dim=1).cpu().numpy()
            ss8_trues.append(ss8_true); ss8_preds.append(ss8_idx)

        if get('ASA') is not None:
            asa_true = batch['ASA'].squeeze(1).cpu().numpy()
            asa_trues.append(asa_true); asa_preds.append(asa_np)
        elif get('AccessibleSurfaceArea') is not None:
            asa_true = batch['AccessibleSurfaceArea'].squeeze(1).cpu().numpy()
            asa_trues.append(asa_true); asa_preds.append(asa_np)

        if get('GRAVY') is not None:
            gravy_true = batch['GRAVY'].squeeze(1).cpu().numpy()
            gravy_trues.append(gravy_true); gravy_preds.append(gravy_np)

        if get('AROM') is not None:
            arom_true = np.rint(batch['AROM'].squeeze(1).cpu().numpy()).astype(int)
            arom_trues.append(arom_true); arom_preds.append(arom_np)
        elif get('Aromaticity') is not None:
            arom_true = np.rint(batch['Aromaticity'].squeeze(1).cpu().numpy()).astype(int)
            arom_trues.append(arom_true); arom_preds.append(arom_np)

        row_cursor += bs

    Path(out_csv_path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(pred_rows).to_csv(out_csv_path, index=False)

    from sklearn.metrics import f1_score, r2_score

    def cat_metric(y_true_list, y_pred_list):
        if not y_true_list: return np.nan
        yt = np.concatenate(y_true_list).astype(int)
        yp = np.concatenate(y_pred_list).astype(int)
        return f1_score(yt, yp, average="weighted")

    def reg_metric(y_true_list, y_pred_list):
        if not y_true_list: return np.nan
        yt = np.concatenate(y_true_list)
        yp = np.concatenate(y_pred_list)
        return r2_score(yt, yp)

    mae_recon = float(np.mean(recon_mae_list)) if len(recon_mae_list) else np.nan

    metrics = {
        "f1_amino": float(cat_metric(aa_trues, aa_preds)),
        "f1_ss3":   float(cat_metric(ss3_trues, ss3_preds)),
        "f1_ss8":   float(cat_metric(ss8_trues, ss8_preds)),
        "f1_arom":  float(cat_metric(arom_trues, arom_preds)),
        "r2_asa":   float(reg_metric(asa_trues, asa_preds)),
        "r2_gravy": float(reg_metric(gravy_trues, gravy_preds)),
        "mae_recon": mae_recon,  # NEW
    }
    pd.DataFrame([metrics]).to_csv(out_metrics_csv_path, index=False)

def main():
    set_seed(SEED)

    any_ran = False
    for spec in MODEL_SPECS:
        stub = spec["stub"]
        values_test, emb_test = build_paths(EMB_BASE_DIR, stub, "test")

        if not (Path(values_test).exists() and Path(emb_test).exists()):
            print(f"[SKIP] No test split for {stub}")
            continue

        for lmbda in LAMBDA_LIST:
            ckpt = Path(TRAINED_MODELS_DIR) / stub / f"lambda_{lmbda}" / "epoch_10.pt"
            if not ckpt.exists():
                print(f"[SKIP] Missing checkpoint: {ckpt}")
                continue

            out_csv     = Path(OUTPUT_DIR) / stub / f"lambda_{lmbda}" / "predictions_test.csv"
            out_metrics = Path(OUTPUT_DIR) / stub / f"lambda_{lmbda}" / "metrics.csv"

            print(f"[RUN] {stub} λ={lmbda}")
            run_predictions(
                ckpt_path=str(ckpt),
                values_csv=values_test,
                emb_npz=emb_test,
                out_csv_path=str(out_csv),
                out_metrics_csv_path=str(out_metrics),
            )
            any_ran = True

    if not any_ran:
        print("No predictions ran. Check test files and checkpoints.")

if __name__ == "__main__":
    main()
