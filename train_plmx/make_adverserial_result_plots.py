#!/usr/bin/env python3
import os
import re
import glob
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

RESULTS_DIR = "informed_feature_scores"
OUT_DIR     = "probe_plots"
MAE_DIR     = "predictions_test"
MODE        = "combined"

TASKS = ['SS8','SS3','AminoAcid','AROM','ASA','GRAVY']
METRIC_KEY = {
    'SS8': 'ACC',
    'SS3': 'ACC',
    'AminoAcid': 'ACC',
    'AROM': 'ACC',
    'ASA': 'R2',
    'GRAVY': 'R2',
}

def find_model_stubs(results_dir: str):
    if not os.path.isdir(results_dir):
        return []
    return sorted([d for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d))])

def read_lambda_rows(model_dir: str):
    rows = []
    for path in glob.glob(os.path.join(model_dir, "results_lambda_*.csv")):
        m = re.search(r"results_lambda_([0-9.]+)\.csv$", os.path.basename(path))
        if not m:
            continue
        lam = float(m.group(1))
        try:
            df = pd.read_csv(path)
            if df.empty: continue
            r = df.iloc[0].to_dict()
            r['lambda'] = lam
            rows.append(r)
        except Exception as e:
            print(f"[Warn] Could not read {path}: {e}")
    return pd.DataFrame(rows).sort_values('lambda') if rows else pd.DataFrame()

def collect_series(df: pd.DataFrame, task: str):
    metric = METRIC_KEY[task]
    base = f"{task}_{metric}"
    mean_col = f"{base}_mean"
    low_col  = f"{base}_ci95_low"
    high_col = f"{base}_ci95_high"

    if not all(c in df.columns for c in [mean_col, low_col, high_col]):
        if mean_col not in df.columns:
            return None
        y = df[mean_col].values
        return df['lambda'].values, y, np.zeros_like(y), np.zeros_like(y)

    y = df[mean_col].values
    low = df[low_col].values
    high = df[high_col].values
    lo = np.clip(np.nan_to_num(y - low, nan=0.0), a_min=0.0, a_max=None)
    hi = np.clip(np.nan_to_num(high - y, nan=0.0), a_min=0.0, a_max=None)
    return df['lambda'].values, y, lo, hi

def _style_axes_for_task(ax, task, y, lo, hi):
    ax.set_ylabel("Accuracy" if METRIC_KEY[task] == 'ACC' else "R²")

    if y is None or len(y) == 0:
        return

    lo = np.zeros_like(y) if lo is None else lo
    hi = np.zeros_like(y) if hi is None else hi

    ymin = np.nanmin(y - lo)
    ymax = np.nanmax(y + hi)

    if math.isfinite(ymin) and math.isfinite(ymax) and ymin < ymax:
        pad = 0.05 * (ymax - ymin)
        ax.set_ylim(ymin - pad, ymax + pad)


def make_figure_for_model(model_stub: str, df: pd.DataFrame, out_path: str):
    fig = plt.figure(figsize=(14, 10), constrained_layout=True)
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 0.9])

    task_axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(3)]
    ax_mae = fig.add_subplot(gs[2, :])

    for i, task in enumerate(TASKS):
        ax = task_axes[i]
        series = collect_series(df, task)
        ax.set_title(f"{task} ({METRIC_KEY[task]})", fontsize=11)
        ax.set_xlabel("λ"); ax.grid(True, linestyle="--", alpha=0.4)
        if series is None:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            continue
        x, y, lo, hi = series
        ax.errorbar(x, y, yerr=[lo, hi], fmt='o', capsize=3, linewidth=1)
        ax.plot(x, y, linestyle='-', linewidth=1)
        _style_axes_for_task(ax, task, y, lo, hi)

    x_mae, y_mae = read_mae_series(model_stub)
    ax_mae.set_title(f"{model_stub} — Reconstruction MAE vs λ", fontsize=12)
    ax_mae.set_xlabel("λ"); ax_mae.set_ylabel("MAE"); ax_mae.grid(True, linestyle="--", alpha=0.4)
    if x_mae is None:
        ax_mae.text(0.5, 0.5, "No MAE data", ha="center", va="center", transform=ax_mae.transAxes)
    else:
        ax_mae.plot(x_mae, y_mae, marker='o', linestyle='-', linewidth=1)

    fig.suptitle(f"{model_stub} — Residual space known feature performance vs λ (95% CI)", fontsize=13)
    fig.savefig(out_path, dpi=200, bbox_inches="tight"); plt.close(fig)


def make_figure_for_all_models(stubs, dfs, out_path: str):
    fig = plt.figure(figsize=(14, 10), constrained_layout=True)
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 0.9])

    task_axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(3)]
    ax_mae = fig.add_subplot(gs[2, :])

    cmap = plt.cm.get_cmap("tab10")
    color_for = lambda i: cmap(i % 10)

    per_task_y = {t: [] for t in TASKS}
    per_task_lo = {t: [] for t in TASKS}
    per_task_hi = {t: [] for t in TASKS}

    for i, task in enumerate(TASKS):
        ax = task_axes[i]
        ax.set_title(f"{task} ({METRIC_KEY[task]})", fontsize=11)
        ax.set_xlabel("λ"); ax.grid(True, linestyle="--", alpha=0.4)

        for j, (stub, df) in enumerate(zip(stubs, dfs)):
            series = collect_series(df, task)
            if series is None:
                continue
            x, y, lo, hi = series
            ax.errorbar(x, y, yerr=[lo, hi], fmt='o', capsize=3, linewidth=1, color=color_for(j))
            ax.plot(x, y, linestyle='-', linewidth=1, color=color_for(j), label=stub if i == 0 else None)
            per_task_y[task].append(y)
            per_task_lo[task].append(lo)
            per_task_hi[task].append(hi)

        if per_task_y[task]:
            ycat  = np.concatenate(per_task_y[task])
            locat = np.concatenate(per_task_lo[task])
            hicat = np.concatenate(per_task_hi[task])
            _style_axes_for_task(ax, task, ycat, locat, hicat)

    handles, labels = task_axes[0].get_legend_handles_labels()
    if labels:
        fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 5),
                   frameon=False, bbox_to_anchor=(0.5, -0.02))

    ax_mae.set_title("Reconstruction MAE vs λ", fontsize=12)
    ax_mae.set_xlabel("λ"); ax_mae.set_ylabel("MAE"); ax_mae.grid(True, linestyle="--", alpha=0.4)

    any_series = False
    for i, stub in enumerate(stubs):
        x_mae, y_mae = read_mae_series(stub)
        if x_mae is None:
            continue
        any_series = True
        ax_mae.plot(x_mae, y_mae, marker='o', linestyle='-', linewidth=1, color=color_for(i), label=stub)

    if not any_series:
        ax_mae.text(0.5, 0.5, "No MAE data", ha="center", va="center", transform=ax_mae.transAxes)
        
    fig.suptitle("Residual space known feature performance vs λ (95% CI) — All models", fontsize=13)
    fig.savefig(out_path, dpi=200, bbox_inches="tight"); plt.close(fig)

def read_mae_series(model_stub: str):
    model_dir=os.path.join(MAE_DIR,model_stub)
    lambdas,maes=[],[]
    for path in glob.glob(os.path.join(model_dir,"lambda_*","metrics.csv")):
        m=re.search(r"lambda_([0-9.]+)",path)
        if not m: continue
        lam=float(m.group(1))
        try:
            df=pd.read_csv(path)
            if "mae_recon" in df.columns and not df.empty:
                lambdas.append(lam); maes.append(float(df.iloc[0]["mae_recon"]))
        except Exception as e: print(f"[Warn] Could not read {path}: {e}")
    if not lambdas: return None,None
    order=np.argsort(lambdas)
    return np.array(lambdas)[order], np.array(maes)[order]

def make_mae_figure_all(stubs,out_path:str):
    fig,ax=plt.subplots(figsize=(8,5),constrained_layout=True)
    cmap=plt.cm.get_cmap("tab10"); color_for=lambda i:cmap(i%10)
    any_series=False
    for i,stub in enumerate(stubs):
        x,y=read_mae_series(stub)
        if x is None: continue
        any_series=True
        ax.plot(x,y,marker='o',linestyle='-',linewidth=1,color=color_for(i),label=stub)
    ax.set_title("Reconstruction MAE vs λ",fontsize=12); ax.set_xlabel("λ"); ax.set_ylabel("MAE")
    ax.grid(True,linestyle="--",alpha=0.4)
    if any_series:
        ax.legend(loc="lower center",ncol=min(len(stubs),5),frameon=False,bbox_to_anchor=(0.5,-0.25))
        fig.subplots_adjust(bottom=0.25)
    else:
        ax.text(0.5,0.5,"No MAE data",ha="center",va="center",transform=ax.transAxes)
    fig.savefig(out_path,dpi=200,bbox_inches="tight"); plt.close(fig)

def make_mae_figure_model(stub:str,out_path:str):
    x,y=read_mae_series(stub)
    fig,ax=plt.subplots(figsize=(7,4),constrained_layout=True)
    ax.set_title(f"{stub} — Reconstruction MAE vs λ",fontsize=12)
    ax.set_xlabel("λ"); ax.set_ylabel("MAE"); ax.grid(True,linestyle="--",alpha=0.4)
    if x is None: ax.text(0.5,0.5,"No MAE data",ha="center",va="center",transform=ax.transAxes)
    else: ax.plot(x,y,marker='o',linestyle='-',linewidth=1)
    fig.savefig(out_path,dpi=200,bbox_inches="tight"); plt.close(fig)

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    stubs = find_model_stubs(RESULTS_DIR)
    if not stubs:
        return
    dfs = []; valid = []
    for stub in stubs:
        df = read_lambda_rows(os.path.join(RESULTS_DIR, stub))
        if not df.empty:
            dfs.append(df); valid.append(stub)
    if not dfs:
        return

    if MODE == "combined":
        make_figure_for_all_models(valid, dfs, os.path.join(OUT_DIR, "all_models_probes_plus_mae.png"))
    else:
        for stub, df in zip(valid, dfs):
            make_figure_for_model(stub, df, os.path.join(OUT_DIR, f"{stub}_probes_plus_mae.png"))

if __name__=="__main__":
    main()
