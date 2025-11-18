import os
import re
import copy
import random
import glob
from pathlib import Path
from typing import Dict, List

import warnings
warnings.filterwarnings("ignore")  # suppress all warnings as requested

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler, RandomSampler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from math import sqrt
import numpy as np

BASE_EMB_DIR = "partitioned_embeddings"
SEQS_DIR     = "downstream_task_data"
OUTPUT_ROOT  = "output_cnn_ci"
GLOBAL_OUT   = "metrics_summary_with_ci_cnn.csv"

REPEATS      = 10 

TASKS    = ["AGG", "TM", "EV"]
VARIANTS = ["adapted", "original", "baseline", "informed"]

CNN_PARAMS = {
    "TM":  {"batch_size": 16, "epochs": 20, "learning_rate": 1e-4,
            "kernel_size": 8, "dropout_rate": 0.2, "num_filters": 50,
            "use_weighted_sampler": False},
    "EV":  {"batch_size": 16, "epochs": 5, "learning_rate": 1e-4,
            "kernel_size": 8, "dropout_rate": 0.2, "num_filters": 50,
            "use_weighted_sampler": False},
    "AGG": {"batch_size": 16, "epochs": 20, "learning_rate": 1e-4,
            "kernel_size": 6, "dropout_rate": 0.2, "num_filters": 50,
            "use_weighted_sampler": True},
}

VARIANT_MAP = {
    "adapted":  {"suffix": "latent",            "embedding_type": "Partitioned"},
    "original": {"suffix": "valid",             "embedding_type": "ESM2"},
    "baseline": {"suffix": "binary",            "embedding_type": "Baseline"},
    "informed": {"suffix": "truncated_latent",  "embedding_type": "Informed"},
}

SEQ_SUFFIX = {"EV": "_sequences_EV", "TM": "_sequences_TM", "AGG": "_sequences_AGG"}

DETERMINISTIC = False
ACCUM_STEPS = 1

MAX_PARALLEL = 12
PER_GPU      = 2

CI_MODE  = "normal"
CI_ALPHA = 0.05

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = DETERMINISTIC
    torch.backends.cudnn.benchmark = not DETERMINISTIC

os.environ.setdefault("PYTHONHASHSEED", "42")
set_seed(42)

def discover_model_dirs():
    return sorted({Path(p).parts[1] for p in glob.glob(os.path.join(BASE_EMB_DIR, "*", "train_*_embeddings_full.npy"))})

def percentile_ci(arr, alpha=0.05):
    return np.percentile(arr, 100*alpha/2), np.percentile(arr, 100*(1-alpha/2))

def t_ci_of_mean(arr, alpha=0.05):
    arr = np.asarray(arr, float)
    n = arr.size
    mean = float(np.mean(arr))
    if n <= 1:
        return mean, mean
    std = float(np.std(arr, ddof=1))
    sem = std / sqrt(n)
    try:
        from scipy.stats import t
        t_crit = t.ppf(1 - alpha/2, df=n-1)
    except Exception:
        t_crit = 2.262 
    half = t_crit * sem
    return mean - half, mean + half

def compute_ci(arr, alpha=0.05, mode="bootstrap"):
    if mode == "normal":
        return t_ci_of_mean(arr, alpha)
    return percentile_ci(arr, alpha)

def compute_metrics(y_true, y_logits, threshold=0.5):
    y_prob = 1 / (1 + np.exp(-y_logits))
    y_pred = (y_prob > threshold).astype(int)
    return {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "f1":        f1_score(y_true, y_pred, zero_division=0),
        "roc_auc":   roc_auc_score(y_true, y_prob),
    }

def effective_num_workers(default: int = 8) -> int:
    cpu = os.cpu_count() or 8
    return max(0, min(default, cpu - 1))

def make_loader(dataset, batch_size, sampler=None, shuffle=False, use_cuda=True):
    num_workers = effective_num_workers()
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(shuffle and sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=use_cuda,
        persistent_workers=(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else None,
        drop_last=False,
    )
class SequenceDataset(Dataset):
    def __init__(self, sequences: np.ndarray, labels: np.ndarray):
        if sequences.dtype != np.float16:
            sequences = sequences.astype(np.float16, copy=False)
        self.X = torch.from_numpy(sequences)  # (N, L, F), float16
        self.y = torch.from_numpy(labels.astype(np.float32, copy=False))  # (N,)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class SequenceCNN(nn.Module):
    def __init__(self, input_features: int, sequence_length: int,
                 kernel_size: int, num_filters: int, dropout_rate: float = 0.0):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout_rate)

        # One conv + pool
        self.conv = nn.Conv1d(input_features, num_filters, kernel_size,
                              padding=kernel_size // 2)
        self.pool = nn.MaxPool1d(2)

        # figure out output size dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, input_features, sequence_length)
            out = self._forward_features(dummy)
            fc_in = out.view(1, -1).size(1)

        self.fc = nn.Linear(fc_in, 1)

    def _forward_features(self, x):
        x = self.relu(self.conv(x))
        x = self.pool(x)
        x = self.dropout(x)
        return x

    def forward(self, x):
        x = x.permute(0, 2, 1).contiguous()
        x = self._forward_features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x.squeeze(1)

def make_pos_weight(labels: np.ndarray, device: torch.device):
    pos = float(labels.sum())
    neg = float(len(labels) - pos)
    if pos <= 0:
        return torch.tensor([1.0], dtype=torch.float32, device=device)
    return torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32, device=device)

def train_one_epoch(model, loader, criterion, optimizer, scaler, device: torch.device, amp_enabled: bool):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    for step, (X, y) in enumerate(loader, start=1):
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.amp.autocast(device_type="cuda", enabled=amp_enabled):
            logits = model(X)
            loss = criterion(logits, y) / ACCUM_STEPS
        if amp_enabled:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        if step % ACCUM_STEPS == 0:
            if amp_enabled:
                scaler.step(optimizer); scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

@torch.no_grad()
def evaluate(model, loader, device: torch.device, amp_enabled: bool):
    model.eval()
    all_y, all_logits = [], []
    for X, y in loader:
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.amp.autocast(device_type="cuda", enabled=amp_enabled):
            logits = model(X)
        all_y.append(y.cpu().numpy())
        all_logits.append(logits.float().cpu().numpy())
    y_true = np.concatenate(all_y)
    y_logits = np.concatenate(all_logits)
    return compute_metrics(y_true, y_logits)

def run_combo(model_dir: str, task: str, variant: str, gpu_index: int | None) -> Dict[str, float]:
    if torch.cuda.is_available() and gpu_index is not None:
        torch.cuda.set_device(gpu_index)
        device = torch.device(f"cuda:{gpu_index}")
        amp_enabled = True
    else:
        device = torch.device("cpu")
        amp_enabled = False

    suf = VARIANT_MAP[variant]["suffix"]
    train_np = f"{BASE_EMB_DIR}/{model_dir}/train_{task}_{suf}_embeddings_full.npy"
    test_np  = f"{BASE_EMB_DIR}/{model_dir}/test_{task}_{suf}_embeddings_full.npy"
    train_csv = f"{SEQS_DIR}/train{SEQ_SUFFIX[task]}.csv"
    test_csv  = f"{SEQS_DIR}/test{SEQ_SUFFIX[task]}.csv"

    if not (os.path.exists(train_np) and os.path.exists(test_np)):
        raise FileNotFoundError(f"missing npy for {model_dir}-{task}-{variant}")

    X_train = np.load(train_np).astype(np.float16, copy=False)
    X_test  = np.load(test_np).astype(np.float16, copy=False)
    y_train = pd.read_csv(train_csv)["label"].to_numpy()
    y_test  = pd.read_csv(test_csv)["label"].to_numpy()

    params = CNN_PARAMS[task]
    test_loader = make_loader(SequenceDataset(X_test, y_test),
                              batch_size=params["batch_size"], shuffle=False,
                              use_cuda=(device.type=="cuda"))

    runs = []
    train_ds = SequenceDataset(X_train, y_train)

    for _ in range(REPEATS):
        if params["use_weighted_sampler"]:
            pos = float(y_train.sum()); neg = float(len(y_train) - pos)
            w_pos = 0.5 / max(pos, 1.0); w_neg = 0.5 / max(neg, 1.0)
            sample_weights = torch.empty(len(train_ds), dtype=torch.float32)
            sample_weights[y_train > 0] = w_pos
            sample_weights[y_train <= 0] = w_neg
            sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(train_ds), replacement=True)
        else:
            sampler = RandomSampler(train_ds, replacement=True, num_samples=len(train_ds))

        train_loader = make_loader(train_ds, batch_size=params["batch_size"],
                                   sampler=sampler, shuffle=False,
                                   use_cuda=(device.type=="cuda"))

        model = SequenceCNN(
            input_features=X_train.shape[-1],
            sequence_length=X_train.shape[1],
            kernel_size=params["kernel_size"],
            num_filters=params["num_filters"],
            dropout_rate=params["dropout_rate"]
        ).to(device)

        if not params["use_weighted_sampler"]:
            pos_weight = make_pos_weight(y_train, device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            criterion = nn.BCEWithLogitsLoss()

        optimizer = optim.Adam(model.parameters(), lr=params["learning_rate"])
        try:
            scaler = torch.amp.GradScaler(device_type="cuda", enabled=amp_enabled)
        except TypeError:
            scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)  # older PyTorch

        for _ in range(params["epochs"]):
            train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, amp_enabled)

        metrics = evaluate(model, test_loader, device, amp_enabled)
        runs.append(metrics)

    df = pd.DataFrame(runs)
    summary = {}
    for m in df.columns:
        arr = df[m].values.astype(float)
        summary[f"{m}_mean"] = float(np.mean(arr))
        lo, hi = compute_ci(arr, alpha=CI_ALPHA, mode=CI_MODE)
        summary[f"{m}_ci_low"], summary[f"{m}_ci_high"] = float(lo), float(hi)

    out_dir = Path(OUTPUT_ROOT) / "per_combo_summaries"
    out_dir.mkdir(parents=True, exist_ok=True)
    row = {
        "Model Subfolder": model_dir,
        "Task": task,
        "Embedding Type": VARIANT_MAP[variant]["embedding_type"],
        "Variant": variant,
        "CI_Mode": CI_MODE,
        "CI_Alpha": CI_ALPHA,
    }
    row.update({k.replace("accuracy","Accuracy").replace("precision","Precision")
                  .replace("recall","Recall").replace("f1","F1").replace("roc_auc","ROC_AUC"): v
                for k,v in summary.items()})
    out_path = out_dir / f"{model_dir}__{task}__{variant}.csv"
    pd.DataFrame([row]).to_csv(out_path, index=False)

    return row

def run_with_gpu_reservation(gpu_queue, model_dir, task, variant):
    """Reserve a GPU slot, run the combo, then release the slot."""
    gpu_index = gpu_queue.get() 
    try:
        return run_combo(model_dir, task, variant, gpu_index)
    finally:
        gpu_queue.put(gpu_index)

def main():
    Path(OUTPUT_ROOT).mkdir(parents=True, exist_ok=True)

    model_dirs = discover_model_dirs()
    combos = [(m, t, v) for m in model_dirs for t in TASKS for v in VARIANTS]

    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if num_gpus == 0:
        print("[WARN] No CUDA detected; running serial on CPU.")
        max_workers = 1
    else:
        max_workers = min(MAX_PARALLEL, num_gpus * PER_GPU)
        print(f"[INFO] GPUs detected: {num_gpus} | PER_GPU={PER_GPU} | "
              f"MAX_PARALLEL={MAX_PARALLEL} -> using up to {max_workers} workers.")
        print(f"[INFO] CI mode: {CI_MODE} | alpha={CI_ALPHA}")

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    manager = mp.Manager()
    gpu_queue = manager.Queue()
    if num_gpus == 0:
        gpu_queue.put(None)
    else:
        for gid in range(num_gpus):
            for _ in range(PER_GPU):
                gpu_queue.put(gid)

    results: List[Dict[str, float]] = []
    futures = []

    with ProcessPoolExecutor(max_workers=max_workers, mp_context=mp.get_context("spawn")) as ex:
        for (m, t, v) in combos:
            fut = ex.submit(run_with_gpu_reservation, gpu_queue, m, t, v)
            futures.append(fut)

        for fut in as_completed(futures):
            try:
                res = fut.result()
                results.append(res)
                print(f"[OK] {res['Model Subfolder']} | {res['Task']} | {res['Variant']}")
            except FileNotFoundError as e:
                print(f"[SKIP] {e}")
            except Exception as e:
                print(f"[ERROR] {e}")

    if results:
        df = pd.DataFrame(results)
        df.to_csv(f"{OUTPUT_ROOT}/{GLOBAL_OUT}", index=False)
        print(f"[OK] saved {OUTPUT_ROOT}/{GLOBAL_OUT}")
    else:
        print("[WARN] No results to save.")

if __name__ == "__main__":
    main()
