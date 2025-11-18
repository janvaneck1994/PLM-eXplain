#!/usr/bin/env python3
import os
import re
import glob
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_squared_error

from PartitionedEmbeddingModel import PartitionedEmbeddingModel
from ProteinDataset import ProteinDataset

GPU_IDS: List[int] = [0, 1, 2, 3, 4, 5] 
MAX_WORKERS: int = len(GPU_IDS)          
DATALOADER_NUM_WORKERS: int = 0         
PIN_MEMORY: bool = True                

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

EMB_BASE_DIR = "./data/output/embeddings"
TRAINED_MODELS_DIR = "trained_models"
LAMBDA_LIST: List[float] = [0, 1, 2, 5, 10, 20]

BATCH_SIZE_EXTRACT = 4096

PROBE_EPOCHS = 10
PROBE_BATCH_SIZE = 512
PROBE_LR = 1e-3
PROBE_HIDDEN = (200, 50) 
REPEATS = 5 
T_CRIT_095_DF9 = 2.262 

EXTRACT_DIR = "extracted_embeddings"
RESULTS_DIR = "informed_feature_scores"
os.makedirs(EXTRACT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

device = torch.device("cpu")
print(f"[Info] Parent using device: {device}")

def list_model_stubs(emb_base_dir: str) -> List[str]:

    stubs = []
    if not os.path.isdir(emb_base_dir):
        return stubs

    def looks_like_model_dir(name: str) -> bool:
        return (
            name.startswith("esm2_")
            or name == "prot_bert_bfd"
            or name.startswith("prot_") 
        )

    for name in os.listdir(emb_base_dir):
        full = os.path.join(emb_base_dir, name)
        if not (os.path.isdir(full) and looks_like_model_dir(name)):
            continue
        have_train = (
            os.path.exists(os.path.join(full, "values_train_holdout.csv")) and
            os.path.exists(os.path.join(full, "embeddings_train_holdout.npz"))
        )
        have_val = (
            os.path.exists(os.path.join(full, "values_test.csv")) and
            os.path.exists(os.path.join(full, "embeddings_test.npz"))
        )
        if have_train and have_val:
            stubs.append(name)
    return sorted(stubs)


def build_paths(emb_base_dir: str, model_stub: str, split: str) -> Tuple[str, str]:
    d = os.path.join(emb_base_dir, model_stub)
    return (
        os.path.join(d, f"values_{split}.csv"),
        os.path.join(d, f"embeddings_{split}.npz"),
    )

def make_loaders(emb_base_dir: str, model_stub: str, batch_size: int,
                 num_workers: int = 0, pin_memory: bool = False):
    tr_vals, tr_emb = build_paths(emb_base_dir, model_stub, "train_holdout")
    va_vals, va_emb = build_paths(emb_base_dir, model_stub, "test")
    train_ds = ProteinDataset(tr_vals, tr_emb)
    val_ds   = ProteinDataset(va_vals, va_emb)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )
    val_loader   = DataLoader(
        val_ds,   batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )
    return train_ds, val_ds, train_loader, val_loader

def pick_embeddings_from_batch(batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    candidates = ["ESM2Embedding", "CARPEmbedding", "Embedding", "embeddings"]
    for k in candidates:
        if k in batch:
            return batch[k]
    for v in batch.values():
        if torch.is_tensor(v) and v.dim() == 2 and v.dtype.is_floating_point:
            return v
    raise KeyError("Embedding tensor not found in batch.")

def latest_checkpoint_path(trained_root: str, model_stub: str, lam: float) -> str:

    candidates = [
        os.path.join(trained_root, model_stub, f"lambda_{lam}"),
        os.path.join(trained_root, model_stub, f"lambda_{lam}.0"),
    ]
    paths = []
    for lam_dir in candidates:
        if os.path.isdir(lam_dir):
            paths.extend(glob.glob(os.path.join(lam_dir, "epoch_*.pt")))
    if not paths:
        return ""
    def epoch_num(p):
        m = re.search(r"epoch_(\d+)\.pt$", os.path.basename(p))
        return int(m.group(1)) if m else -1
    paths.sort(key=epoch_num)
    return os.path.join(lam_dir, "epoch_10.pt") #paths[-1]

def compute_class_weights_from_csv(csv_path, column_name, num_classes, mapping_dict):
    df = pd.read_csv(csv_path)
    labels = df[column_name].astype(str)
    labels = labels.map(lambda x: mapping_dict.get(x, -1))
    labels = labels[labels != -1].astype(int).values
    counts = np.bincount(labels, minlength=num_classes)
    weights = 1.0 / (counts + 1e-6)
    weights = weights / weights.sum() * num_classes
    return weights  # numpy array

@torch.no_grad()
def extract_embeddings_and_targets(model, dataloader, device_local):
    unk_list, kn_list, orig_list = [], [], []
    asa_list, gravy_list, arom_list = [], [], []
    amino_list, ss3_list, ss8_list = [], [], []

    model.eval()
    for batch in dataloader:
        batch = {k: v.to(device_local) for k, v in batch.items()}
        embeddings = pick_embeddings_from_batch(batch)

        amino = batch['AminoAcid']
        if amino.dim() > 1 and amino.size(-1) > 1:
            amino = amino.argmax(dim=1)
        ss3 = batch['SS3']
        if ss3.dim() > 1 and ss3.size(-1) > 1:
            ss3 = ss3.argmax(dim=1)
        ss8 = batch['SS8']
        if ss8.dim() > 1 and ss8.size(-1) > 1:
            ss8 = ss8.argmax(dim=1)

        asa   = batch['ASA'].squeeze()
        gravy = batch['GRAVY'].squeeze()
        arom  = batch['AROM'].squeeze()

        known, unknown, _, _, _ = model(embeddings)

        unk_list.append(unknown.detach().cpu())
        kn_list.append(known.detach().cpu())
        orig_list.append(embeddings.detach().cpu())

        asa_list.append(asa.detach().cpu())
        gravy_list.append(gravy.detach().cpu())
        arom_list.append((arom.detach().cpu().float() > 0.5).long())

        amino_list.append(amino.detach().cpu())
        ss3_list.append(ss3.detach().cpu())
        ss8_list.append(ss8.detach().cpu())

    return {
        "unknown":  torch.cat(unk_list).numpy(),
        "known":    torch.cat(kn_list).numpy(),
        "original": torch.cat(orig_list).numpy(),
        "ASA":      torch.cat(asa_list).numpy(),
        "GRAVY":    torch.cat(gravy_list).numpy(),
        "AROM":     torch.cat(arom_list).numpy(),
        "AminoAcid":torch.cat(amino_list).numpy(),
        "SS3":      torch.cat(ss3_list).numpy(),
        "SS8":      torch.cat(ss8_list).numpy(),
    }

def ensure_extracted_or_extract(model_stub: str, lam: float, input_dim: int,
                                train_loader, val_loader, device_local):

    out_dir = os.path.join(EXTRACT_DIR, model_stub)
    os.makedirs(out_dir, exist_ok=True)
    tr_path = os.path.join(out_dir, f"lambda_{lam}_train_holdout.npz")
    va_path = os.path.join(out_dir, f"lambda_{lam}_test.npz")

    load_train = os.path.exists(tr_path)
    load_val   = os.path.exists(va_path)

    if load_train and load_val:
        print(f"[Use] Reusing extracted arrays: {tr_path} & {va_path}")
        train_data = dict(np.load(tr_path, allow_pickle=True))
        val_data   = dict(np.load(va_path, allow_pickle=True))
        return train_data, val_data

    # We moeten (opnieuw) extraheren
    ckpt = latest_checkpoint_path(TRAINED_MODELS_DIR, model_stub, lam)
    if not ckpt:
        raise FileNotFoundError(f"Geen checkpoint gevonden voor λ={lam} in {TRAINED_MODELS_DIR}/{model_stub}")

    known_dim, hidden_dim = 34, input_dim
    model = PartitionedEmbeddingModel(input_dim, known_dim, hidden_dim, adv_output_dim=34).to(device_local)
    state = torch.load(ckpt, map_location=device_local)
    model.load_state_dict(state)
    model.eval()

    if not load_train:
        train_data = extract_embeddings_and_targets(model, train_loader, device_local)
        np.savez(tr_path, **train_data)
        print(f"[Save] Extracted train -> {tr_path}")
    else:
        train_data = dict(np.load(tr_path, allow_pickle=True))

    if not load_val:
        val_data = extract_embeddings_and_targets(model, val_loader, device_local)
        np.savez(va_path, **val_data)
        print(f"[Save] Extracted val   -> {va_path}")
    else:
        val_data = dict(np.load(va_path, allow_pickle=True))

    return train_data, val_data

class TorchMLP(torch.nn.Module):
    def __init__(self, in_dim: int, out_dim: int, is_regression: bool = False):
        super().__init__()
        self.is_regression = is_regression
        h1, h2 = PROBE_HIDDEN
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, h1),
            torch.nn.ReLU(),
            torch.nn.Linear(h1, h2),
            torch.nn.ReLU(),
            torch.nn.Linear(h2, out_dim),
        )

    def forward(self, x):
        return self.net(x)

def to_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    X_t = torch.from_numpy(X).float()
    if y.dtype.kind in "iu":
        y_t = torch.from_numpy(y.astype(np.int64))
    else:
        y_t = torch.from_numpy(y.astype(np.float32))
    ds = TensorDataset(X_t, y_t)
    return DataLoader(ds, batch_size=min(batch_size, len(ds)), shuffle=shuffle)

def train_torch_classifier(X_train, y_train, n_classes: int, class_weights=None, device_local=None):
    model = TorchMLP(X_train.shape[1], n_classes, is_regression=False).to(device_local)
    if class_weights is not None:
        cw_tensor = torch.tensor(class_weights, dtype=torch.float, device=device_local)
        criterion = torch.nn.CrossEntropyLoss(weight=cw_tensor)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    optim = torch.optim.Adam(model.parameters(), lr=PROBE_LR)
    loader = to_loader(X_train, y_train, PROBE_BATCH_SIZE, shuffle=True)

    model.train()
    for _ in range(PROBE_EPOCHS):
        for xb, yb in tqdm(loader, leave=False):
            xb = xb.to(device_local)
            yb = yb.to(device_local)
            logits = model(xb)
            loss = criterion(logits, yb)
            optim.zero_grad()
            loss.backward()
            optim.step()
    return model

@torch.no_grad()
def eval_torch_classifier(model, X_test, device_local=None):
    model.eval()
    X_t = torch.from_numpy(X_test).float().to(device_local)
    logits = []
    B = 65536
    for i in range(0, X_t.size(0), B):
        logits.append(model(X_t[i:i+B]).detach().cpu())
    logits = torch.cat(logits, dim=0)
    y_pred = logits.argmax(dim=1).numpy()
    return y_pred

def train_torch_regressor(X_train, y_train, device_local=None):
    model = TorchMLP(X_train.shape[1], 1, is_regression=True).to(device_local)
    criterion = torch.nn.MSELoss()
    optim = torch.optim.Adam(model.parameters(), lr=PROBE_LR)
    loader = to_loader(X_train, y_train.reshape(-1, 1), PROBE_BATCH_SIZE, shuffle=True)

    model.train()
    for _ in range(PROBE_EPOCHS):
        for xb, yb in tqdm(loader, leave=False):
            xb = xb.to(device_local)
            yb = yb.to(device_local).float()
            pred = model(xb)
            loss = criterion(pred, yb)
            optim.zero_grad()
            loss.backward()
            optim.step()
    return model

@torch.no_grad()
def eval_torch_regressor(model, X_test, device_local=None):
    model.eval()
    X_t = torch.from_numpy(X_test).float().to(device_local)
    preds = []
    B = 65536
    for i in range(0, X_t.size(0), B):
        preds.append(model(X_t[i:i+B]).detach().cpu().numpy())
    y_pred = np.concatenate(preds, axis=0).squeeze()
    return y_pred

def ci_stats(values: List[float], t_mult: float = T_CRIT_095_DF9):
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if n > 1 else 0.0
    half_width = t_mult * std / math.sqrt(n) if n > 1 else 0.0
    return mean, std, mean - half_width, mean + half_width

def _setup_device_for_worker(gpu_id: int) -> torch.device:
    global device
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
        device = torch.device(f"cuda:{gpu_id}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"[Worker {os.getpid()}] Using device: {device}")
    return device

def _run_task(model_stub: str, lam: float, gpu_id: int) -> Dict[str, float]:
    torch.manual_seed(SEED + int(lam) + os.getpid())
    np.random.seed(SEED + int(lam) + os.getpid())

    device_local = _setup_device_for_worker(gpu_id)

    train_ds, val_ds, train_loader, val_loader = make_loaders(
        EMB_BASE_DIR, model_stub, BATCH_SIZE_EXTRACT,
        num_workers=DATALOADER_NUM_WORKERS, pin_memory=PIN_MEMORY and (device_local.type == "cuda")
    )
    input_dim = train_ds.embeddings.shape[1] if hasattr(train_ds, "embeddings") else 1024

    train_csv = os.path.join(EMB_BASE_DIR, model_stub, "values_train_holdout.csv")
    amino_acid_dict = {
        'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7,
        'K': 8, 'L': 9, 'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14,
        'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19
    }
    ss3_dict = {'H': 0, 'E': 1, 'C': 2}
    ss8_dict = {'H': 0, 'E': 1, 'G': 2, 'I': 3, 'B': 4, 'T': 5, 'S': 6, '-': 7}

    amino_weights = compute_class_weights_from_csv(train_csv, 'AminoAcid', 20, amino_acid_dict)
    ss3_weights   = compute_class_weights_from_csv(train_csv, 'SS3', 3, ss3_dict)
    ss8_weights   = compute_class_weights_from_csv(train_csv, 'SS8', 8, ss8_dict)

    try:
        train_data, val_data = ensure_extracted_or_extract(
            model_stub, lam, input_dim, train_loader, val_loader, device_local
        )
    except FileNotFoundError as e:
        print(f"[Skip] {e}")
        return {"model_stub": model_stub, "lambda": lam, "skipped": 1, "gpu": gpu_id}

    X_tr = train_data["unknown"]
    X_te = val_data["unknown"]

    raw_rows = []  
    probe_order = ['SS8', 'ASA', 'GRAVY', 'AROM', 'AminoAcid', 'SS3']

    for rep in range(REPEATS):
        rep_seed = SEED + int(lam * 10) + rep + 1000 * (os.getpid() % 997)
        torch.manual_seed(rep_seed)
        np.random.seed(rep_seed)

        rep_row = {"model_stub": model_stub, "lambda": lam, "gpu": gpu_id, "repeat": rep}

        for target in probe_order:
            y_tr = train_data[target]
            y_te = val_data[target]

            if target in ['ASA', 'GRAVY']:
                reg = train_torch_regressor(X_tr, y_tr.astype(np.float32), device_local=device_local)
                y_pred = eval_torch_regressor(reg, X_te, device_local=device_local)
                rep_row[f"{target}_R2"]  = r2_score(y_te, y_pred)
                rep_row[f"{target}_MSE"] = mean_squared_error(y_te, y_pred)
                print(f"[{model_stub} | λ={lam} | GPU {gpu_id} | rep {rep}] {target}: R2={rep_row[f'{target}_R2']:.4f}, MSE={rep_row[f'{target}_MSE']:.4f}")
            else:
                if target == 'AminoAcid':
                    n_classes = 20; cw = amino_weights
                elif target == 'SS3':
                    n_classes = 3; cw = ss3_weights
                elif target == 'SS8':
                    n_classes = 8; cw = ss8_weights
                elif target == 'AROM':
                    n_classes = 2; cw = None
                else:
                    raise ValueError(f"Unknown target {target}")

                clf = train_torch_classifier(X_tr, y_tr.astype(np.int64), n_classes, class_weights=cw, device_local=device_local)
                y_pred = eval_torch_classifier(clf, X_te, device_local=device_local)
                rep_row[f"{target}_ACC"] = accuracy_score(y_te, y_pred)
                rep_row[f"{target}_F1"]  = f1_score(y_te, y_pred, average="weighted")
                print(f"[{model_stub} | λ={lam} | GPU {gpu_id} | rep {rep}] {target}: ACC={rep_row[f'{target}_ACC']:.4f}, F1={rep_row[f'{target}_F1']:.4f}")

        raw_rows.append(rep_row)

    results_summary: Dict[str, float] = {"model_stub": model_stub, "lambda": lam, "gpu": gpu_id}
    raw_df = pd.DataFrame(raw_rows)

    def summarize_metric(prefix: str, metric: str):
        vals = raw_df[f"{prefix}_{metric}"].dropna().tolist() if f"{prefix}_{metric}" in raw_df else []
        if not vals:
            return
        mean, std, low, high = ci_stats(vals)
        results_summary[f"{prefix}_{metric}_mean"] = mean
        results_summary[f"{prefix}_{metric}_std"]  = std
        results_summary[f"{prefix}_{metric}_ci95_low"]  = low
        results_summary[f"{prefix}_{metric}_ci95_high"] = high

    for tgt in ['AROM', 'AminoAcid', 'SS3', 'SS8']:
        summarize_metric(tgt, "ACC")
        summarize_metric(tgt, "F1")

    for tgt in ['ASA', 'GRAVY']:
        summarize_metric(tgt, "R2")
        summarize_metric(tgt, "MSE")

    per_model_dir = os.path.join(RESULTS_DIR, model_stub)
    os.makedirs(per_model_dir, exist_ok=True)

    repeats_csv = os.path.join(per_model_dir, f"results_lambda_{lam}_repeats.csv")
    raw_df.to_csv(repeats_csv, index=False)

    summary_csv = os.path.join(per_model_dir, f"results_lambda_{lam}.csv")
    pd.DataFrame([results_summary]).to_csv(summary_csv, index=False)

    print(f"[Save] Repeats -> {repeats_csv}")
    print(f"[Save] Summary -> {summary_csv}")

    return results_summary

def main():
    model_stubs = list_model_stubs(EMB_BASE_DIR)
    if not model_stubs:
        print(f"[Warn] Geen ESM2-modelstubs gevonden in {EMB_BASE_DIR}. Stop.")
        return

    tasks = []
    for model_stub in model_stubs:
        for i, lam in enumerate(LAMBDA_LIST):
            gpu_id = GPU_IDS[i % len(GPU_IDS)]
            tasks.append((model_stub, lam, gpu_id))

    print(f"[Parallel] Scheduling {len(tasks)} tasks over GPUs {GPU_IDS} with max_workers={MAX_WORKERS}")

    all_rows: List[Dict] = []
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(_run_task, ms, lam, gid): (ms, lam, gid) for (ms, lam, gid) in tasks}
        for fut in as_completed(futs):
            ms, lam, gid = futs[fut]
            try:
                res = fut.result()
                all_rows.append(res)
            except Exception as e:
                print(f"[Error] {ms} λ={lam} (GPU {gid}): {e}")
                all_rows.append({"model_stub": ms, "lambda": lam, "gpu": gid, "error": str(e)})

    if all_rows:
        agg = pd.DataFrame(all_rows)
        agg_path = os.path.join(RESULTS_DIR, "all_results.csv")
        agg.to_csv(agg_path, index=False)
        print(f"\n[Done] Geaggregeerde resultaten (means/std/CIs) -> {agg_path}")
    else:
        print("\n[Info] Geen resultaten geschreven (ontbrekende checkpoints/embeddings?).")

if __name__ == "__main__":
    main()
