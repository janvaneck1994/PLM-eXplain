import os
import re
import glob
from pathlib import Path
from typing import Tuple, List, Dict
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

import shap

BASE_EMB_DIR = "partitioned_embeddings"
SEQS_DIR = "downstream_task_data"
OUTPUT_ROOT = "output_gradient_boost"
REPEATS = 10
RANDOM_SEED = 42

TOPK_IMPORTANT = 10

EMBEDDING_ORDER = ["Partitioned", "Informed", "Baseline", "ESM2"]

COLOR_HANDCRAFTED = "#1f77b4"  # blue
COLOR_NON_TASK    = "#d62728"  # red

task_labels = {
    0: 'A', 1: 'C', 2: 'D', 3: 'E', 4: 'F', 5: 'G', 6: 'H', 7: 'I', 8: 'K',
    9: 'L', 10: 'M', 11: 'N', 12: 'P', 13: 'Q', 14: 'R', 15: 'S', 16: 'T',
    17: 'V', 18: 'W', 19: 'Y',
    20: 'SS3_H', 21: 'SS3_E', 22: 'SS3_C',
    23: 'SS8_H', 24: 'SS8_E', 25: 'SS8_G', 26: 'SS8_I', 27: 'SS8_B',
    28: 'SS8_T', 29: 'SS8_S', 30: 'SS8_-',
    31: 'ASA', 32: 'GRAVY', 33: 'AROM'
}
HANDCRAFTED_SET = set(task_labels.values())

from math import sqrt

def t_ci_of_mean(x: np.ndarray, alpha: float = 0.05):
    x = np.asarray(x, float)
    n = x.size
    mean = float(np.mean(x)) if n else float("nan")
    if n <= 1:
        return mean, mean
    std = float(np.std(x, ddof=1))
    sem = std / sqrt(n)
    try:
        from scipy.stats import t
        t_crit = t.ppf(1 - alpha/2, df=n-1)
    except Exception:
        t_crit = 2.262
    half = t_crit * sem
    return mean - half, mean + half

def ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)

def percentile_ci(x: np.ndarray, alpha: float = 0.05):
    return float(np.percentile(x, 100 * (alpha / 2.0))), float(np.percentile(x, 100 * (1 - alpha / 2.0)))

def safe_metrics(y_true, y_pred, y_prob) -> Dict[str, float]:
    out = {"Accuracy": accuracy_score(y_true, y_pred)}
    if len(np.unique(y_true)) == 2:
        out["Precision"] = precision_score(y_true, y_pred)
        out["Recall"] = recall_score(y_true, y_pred)
        out["F1"] = f1_score(y_true, y_pred)
        try:
            out["ROC_AUC"] = roc_auc_score(y_true, y_prob)
        except Exception:
            out["ROC_AUC"] = np.nan
    else:
        out["Precision"] = precision_score(y_true, y_pred, average="macro")
        out["Recall"] = recall_score(y_true, y_pred, average="macro")
        out["F1"] = f1_score(y_true, y_pred, average="macro")
        out["ROC_AUC"] = np.nan
    return out

def plot_ci_bars(df_group, metric: str, title: str, out_path: str):
    labels = df_group["Embedding Type"].tolist()
    means  = df_group["mean"].to_numpy()
    lows   = df_group["ci_low"].to_numpy()
    highs  = df_group["ci_high"].to_numpy()
    yerr   = np.vstack([means - lows, highs - means])

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(labels, means, yerr=yerr, capsize=6)
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_title(title)
    ax.set_ylim(0.0, 1.0)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

def make_feature_names(n_features: int, adapted: bool) -> List[str]:
    if adapted:
        return [task_labels.get(i, f"feature_{i}") for i in range(n_features)]
    return [f"feature_{i}" for i in range(n_features)]

def human_task_name(task_short: str) -> str:
    return {"AGG": "AGG", "EV": "EV", "TM": "TM"}.get(task_short, task_short)

def seq_suffix_for_task(task_short: str) -> str:
    return {"AGG": "_sequences_AGG", "EV": "_sequences_EV", "TM": "_sequences_TM"}.get(task_short, f"_sequences_{task_short}")

def infer_task_and_title_bits(embedding_key: str) -> Tuple[str, str, bool]:
    if embedding_key.startswith("AGG"):
        task_short = "AGG"
    elif embedding_key.startswith("EV"):
        task_short = "EV"
    elif embedding_key.startswith("TM"):
        task_short = "TM"
    else:
        m = re.match(r"([A-Z]+)_", embedding_key)
        task_short = m.group(1) if m else "TASK"

    if "valid" in embedding_key:
        title_variant = "Original Embeddings"
        adapted = False
    elif "truncated_latent" in embedding_key:
        title_variant = "Informed Embeddings"
        adapted = True
    elif "binary" in embedding_key:
        title_variant = "Baseline Embeddings"
        adapted = True
    else:
        title_variant = "Partitioned Embeddings"
        adapted = True
    return task_short, title_variant, adapted

def title_to_label(title_variant: str) -> str:
    return {
        "Original Embeddings": "ESM2",
        "Partitioned Embeddings": "Partitioned",
        "Informed Embeddings": "Informed",
        "Baseline Embeddings": "Baseline",
    }.get(title_variant, title_variant)

def compute_corr_signs(X_orig: np.ndarray, shap_vals: np.ndarray) -> np.ndarray:
    n_features = X_orig.shape[1]
    signs = np.zeros(n_features, dtype=float)
    for j in range(n_features):
        x = X_orig[:, j]
        s = shap_vals[:, j]
        if x.size < 2 or np.std(x) == 0 or np.std(s) == 0:
            signs[j] = 0.0
            continue
        r = np.corrcoef(x, s)[0, 1]
        if np.isfinite(r):
            signs[j] = 1.0 if r > 0 else (-1.0 if r < 0 else 0.0)
        else:
            signs[j] = 0.0
    return signs

def discover_train_test_pairs(base_dir: str) -> List[Tuple[str, str, str, str]]:
    pairs = []
    pattern = os.path.join(base_dir, "*", "train_*_embeddings.csv")
    for train_csv in glob.glob(pattern):
        parts = Path(train_csv).parts
        base_idx = parts.index(BASE_EMB_DIR) if BASE_EMB_DIR in parts else 1
        model_dir = parts[base_idx + 1] if base_idx + 1 < len(parts) else parts[1]
        m = re.match(r"train_(.+)_embeddings\.csv$", Path(train_csv).name)
        if not m:
            continue
        embedding_key = m.group(1)
        test_csv = os.path.join(base_dir, model_dir, f"test_{embedding_key}_embeddings.csv")
        if os.path.exists(test_csv):
            pairs.append((model_dir, embedding_key, train_csv, test_csv))
        else:
            print(f"[WARN] Missing test CSV for {train_csv}; expected {test_csv}")
    return pairs

def save_shap_outputs_for_run(gb,
                              X_test_std: np.ndarray,
                              X_test_orig: np.ndarray,
                              adapted: bool,
                              out_dir: str,
                              run_idx: int):
    
    explainer = shap.TreeExplainer(gb)
    shap_values = explainer.shap_values(X_test_std)
    shap_vals = shap_values[1] if isinstance(shap_values, list) and len(shap_values) > 1 else shap_values

    feature_names = make_feature_names(X_test_std.shape[1], adapted)

    plt.figure(figsize=(6, 16))
    shap.summary_plot(shap_vals, features=X_test_std, feature_names=feature_names, max_display=10, show=False)
    plt.savefig(os.path.join(out_dir, f"shap_summary_run{run_idx}.png"), bbox_inches="tight")
    plt.close()

    mean_abs   = np.mean(np.abs(shap_vals), axis=0)
    corr_signs = compute_corr_signs(X_test_orig, shap_vals)
    directional = corr_signs * mean_abs

    pd.DataFrame({
        "Index": range(len(feature_names)),
        "Feature": feature_names,
        "CorrSign": corr_signs,
        "MeanAbsSHAP": mean_abs,
        "DirectionalEffect": directional
    }).sort_values("MeanAbsSHAP", ascending=False).to_csv(
        os.path.join(out_dir, f"feature_importance_run{run_idx}.csv"), index=False
    )

    order_idx = np.argsort(-mean_abs)[:10]
    top_feats = [feature_names[i] for i in order_idx]
    top_vals  = directional[order_idx]
    colors    = [COLOR_HANDCRAFTED if f in HANDCRAFTED_SET else COLOR_NON_TASK for f in top_feats]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(top_feats, top_vals, color=colors)
    ax.set_xlabel("Directional effect = sign(corr(feature, SHAP)) × mean(|SHAP|)")
    ax.set_title(f"Directional Top-10 (run {run_idx})\n(left = negative corr sign, right = positive)")
    ax.axvline(0.0, linestyle="--", linewidth=1)
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"feature_importance_run{run_idx}.png"), bbox_inches="tight")
    plt.close()

    return feature_names, mean_abs, directional, corr_signs

def bootstrap_runs_for_pair(train_csv: str,
                            test_csv: str,
                            seq_suffix: str,
                            pretty_task: str,
                            title_variant: str,
                            adapted: bool,
                            out_dir: str) -> pd.DataFrame:
    ensure_dir(out_dir)

    X_train_full = pd.read_csv(train_csv).to_numpy()
    X_test       = pd.read_csv(test_csv).to_numpy()
    y_train_full = pd.read_csv(os.path.join(SEQS_DIR, f"train{seq_suffix}.csv"))["label"].to_numpy()
    y_test       = pd.read_csv(os.path.join(SEQS_DIR, f"test{seq_suffix}.csv"))["label"].to_numpy()

    le = LabelEncoder()
    y_train_full_enc = le.fit_transform(y_train_full)
    y_test_enc       = le.transform(y_test)

    rng_base = np.random.RandomState(RANDOM_SEED)

    per_run_rows = []
    topk_counter = Counter()
    pos_counter  = Counter()
    neg_counter  = Counter()

    directional_accumulator = None
    runs_count = 0
    last_feature_names = None

    for run in range(REPEATS):
        rng = np.random.RandomState(rng_base.randint(0, 1_000_000))
        n = X_train_full.shape[0]
        idx = rng.randint(0, n, size=n)

        Xb = X_train_full[idx]
        yb = y_train_full_enc[idx]

        scaler = StandardScaler()
        Xb_std     = scaler.fit_transform(Xb)
        X_test_std = scaler.transform(X_test)

        gb = HistGradientBoostingClassifier(
            max_iter=100, max_depth=4, learning_rate=0.05,
            random_state=int(rng.randint(0, 2**31 - 1))
        )
        gb.fit(Xb_std, yb)

        y_pred = gb.predict(X_test_std)
        try:
            proba = gb.predict_proba(X_test_std)
            y_prob = proba[:, 1] if proba.shape[1] == 2 else proba.max(axis=1)
        except Exception:
            y_prob = np.zeros(len(X_test_std))
        m = safe_metrics(y_test_enc, y_pred, y_prob)
        m["run"] = run + 1
        per_run_rows.append(m)

        try:
            feats_this_run, mean_abs, directional, corr_signs = save_shap_outputs_for_run(
                gb, X_test_std, X_test, adapted, out_dir, run_idx=run + 1
            )
            last_feature_names = feats_this_run

            if directional_accumulator is None:
                directional_accumulator = np.zeros_like(directional, dtype=float)
            directional_accumulator += directional
            runs_count += 1

            top_indices = np.argsort(-mean_abs)[:TOPK_IMPORTANT]
            for i in top_indices:
                fname = feats_this_run[i]
                topk_counter[fname] += 1
                if corr_signs[i] > 0:
                    pos_counter[fname] += 1
                elif corr_signs[i] < 0:
                    neg_counter[fname] += 1
        except Exception as e:
            print(f"[WARN] SHAP failed on run {run + 1} ({pretty_task}): {e}")

    runs_df = pd.DataFrame(per_run_rows)
    runs_df.to_csv(os.path.join(out_dir, "per_run_metrics.csv"), index=False)

    agg = {}
    for metric in ["Accuracy", "F1", "ROC_AUC", "Precision", "Recall"]:
        if metric in runs_df.columns:
            vals = runs_df[metric].to_numpy()
            mean = float(np.mean(vals))
            low, high = t_ci_of_mean(vals, alpha=0.05)
            agg[f"{metric}_mean"]   = mean
            agg[f"{metric}_ci_low"] = low
            agg[f"{metric}_ci_high"]= high
    pd.DataFrame([agg]).to_csv(os.path.join(out_dir, "summary_metrics.csv"), index=False)

    if last_feature_names is not None and (len(pos_counter) > 0 or len(neg_counter) > 0):
        feats = set(pos_counter.keys()) | set(neg_counter.keys())
        rows = [(f, pos_counter.get(f, 0), neg_counter.get(f, 0),
                pos_counter.get(f, 0) + neg_counter.get(f, 0)) for f in feats]
        counts_df = pd.DataFrame(rows, columns=["Feature", "PosCount", "NegCount", "TotalCount"])

        counts_df["DominantCount"] = counts_df[["PosCount", "NegCount"]].max(axis=1)
        counts_df = counts_df.sort_values(
            ["DominantCount", "TotalCount", "Feature"],
            ascending=[False, False, True]
        )

        counts_df.to_csv(os.path.join(out_dir, f"feature_top{TOPK_IMPORTANT}_appearance_counts.csv"), index=False)

        top10 = counts_df.head(10).copy()
        top10["Color"] = top10["Feature"].apply(lambda f: COLOR_HANDCRAFTED if f in HANDCRAFTED_SET else COLOR_NON_TASK)

        fig, ax = plt.subplots(figsize=(8, 5))
        for _, row in top10.iterrows():
            f, p, n, c = row["Feature"], int(row["PosCount"]), int(row["NegCount"]), row["Color"]
            if p >= n:
                ax.barh(f,  p, color=c)
            else:
                ax.barh(f, -n, color=c)

        ax.axvline(0.0, linestyle="--", linewidth=1)
        ax.set_xlabel(f"Top-{TOPK_IMPORTANT} appearances across runs (left = negative, right = positive)")
        ax.set_title("")
        ax.invert_yaxis()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "top10_feature_appearance_counts.png"), bbox_inches="tight")
        plt.close()

    if (directional_accumulator is not None) and (runs_count > 0):
        dir_mean_across_runs = directional_accumulator / float(runs_count)

        order_idx = np.argsort(-np.abs(dir_mean_across_runs))[:10]
        top_feats = [last_feature_names[i] for i in order_idx]
        top_vals  = dir_mean_across_runs[order_idx]
        colors    = [COLOR_HANDCRAFTED if f in HANDCRAFTED_SET else COLOR_NON_TASK for f in top_feats]

        pd.DataFrame({
            "Feature": last_feature_names,
            "DirectionalMean_across_runs": dir_mean_across_runs
        }).sort_values("DirectionalMean_across_runs", key=lambda s: np.abs(s), ascending=False)\
         .to_csv(os.path.join(out_dir, "feature_importance_signed_aggregate.csv"), index=False)

        fig, ax = plt.subplots(figsize=(10, 7))
        ax.barh(top_feats, top_vals, color=colors)
        ax.set_xlabel("Directional effect across runs (sign = corr(feature, SHAP), magnitude = mean|SHAP|)")
        ax.set_title("Final Directional Feature Importance (Top-10)\n(left = negative corr sign, right = positive)")
        ax.axvline(0.0, linestyle="--", linewidth=1)
        ax.invert_yaxis()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "feature_importance_signed_aggregate_top10.png"), bbox_inches="tight")
        plt.close()

    return runs_df

def build_per_task_bar_charts(global_summary_df: pd.DataFrame):
    if global_summary_df.empty:
        return
    for (model_dir, task), grp in global_summary_df.groupby(["Model Subfolder", "Task"]):
        grp = grp.set_index("Embedding Type").reindex(EMBEDDING_ORDER).reset_index()

        acc_df = grp[["Embedding Type", "Accuracy_mean", "Accuracy_ci_low", "Accuracy_ci_high"]].dropna()
        acc_df.columns = ["Embedding Type", "mean", "ci_low", "ci_high"]
        if not acc_df.empty:
            plot_ci_bars(acc_df, "Accuracy", f"{model_dir} — {task}: Accuracy (mean ± 95% CI)",
                         os.path.join(OUTPUT_ROOT, model_dir, f"{task}_accuracy_bars.png"))

        f1_df = grp[["Embedding Type", "F1_mean", "F1_ci_low", "F1_ci_high"]].dropna()
        f1_df.columns = ["Embedding Type", "mean", "ci_low", "ci_high"]
        if not f1_df.empty:
            plot_ci_bars(f1_df, "F1", f"{model_dir} — {task}: F1 (mean ± 95% CI)",
                         os.path.join(OUTPUT_ROOT, model_dir, f"{task}_f1_bars.png"))

def run_all():
    if not os.path.isdir(BASE_EMB_DIR):
        raise FileNotFoundError(f"Directory '{BASE_EMB_DIR}' not found. Please place your model outputs under this path.")

    pairs = discover_train_test_pairs(BASE_EMB_DIR)
    if not pairs:
        print("[WARN] No train/test embedding pairs found.")
        return

    global_rows = []

    for model_dir, embedding_key, train_csv, test_csv in pairs:
        task_short, title_variant, adapted = infer_task_and_title_bits(embedding_key)
        pretty_task = human_task_name(task_short)
        seq_suffix  = seq_suffix_for_task(task_short)

        out_dir = os.path.join(OUTPUT_ROOT, model_dir, f"{pretty_task}_{embedding_key}")
        ensure_dir(out_dir)

        try:
            runs_df = bootstrap_runs_for_pair(
                train_csv, test_csv, seq_suffix, pretty_task, title_variant, adapted, out_dir
            )
        except FileNotFoundError as e:
            print(f"[WARN] Skipping {embedding_key}: {e}")
            continue
        except Exception as e:
            print(f"[ERROR] Failed on {embedding_key}: {e}")
            continue

        emb_type = title_to_label(title_variant)
        row = {"Model Subfolder": model_dir, "Task": pretty_task, "Embedding Type": emb_type}

        for metric in ["Accuracy", "F1", "ROC_AUC", "Precision", "Recall"]:
            if metric in runs_df.columns:
                vals = runs_df[metric].to_numpy()
                mean = float(np.mean(vals))
                low, high = t_ci_of_mean(vals, alpha=0.05)
                row[f"{metric}_mean"]   = mean
                row[f"{metric}_ci_low"] = low
                row[f"{metric}_ci_high"]= high

        global_rows.append(row)

    if not global_rows:
        print("[INFO] No results to summarize.")
        return

    global_df = pd.DataFrame(global_rows)
    type_cat = pd.Categorical(global_df["Embedding Type"], categories=EMBEDDING_ORDER, ordered=True)
    global_df = global_df.assign(**{"Embedding Type": type_cat}).sort_values(
        ["Model Subfolder", "Task", "Embedding Type"]
    )
    global_df["Embedding Type"] = global_df["Embedding Type"].astype(str)

    ensure_dir(OUTPUT_ROOT)
    out_csv = os.path.join(OUTPUT_ROOT, "metrics_summary_with_ci.csv")
    global_df.to_csv(out_csv, index=False)
    print(f"[INFO] Saved summary with CI to {out_csv}")

    build_per_task_bar_charts(global_df)

if __name__ == "__main__":
    run_all()
