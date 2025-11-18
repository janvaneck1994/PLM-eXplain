#!/usr/bin/env python3
import os, re, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_DIR = "informed_feature_scores" 
MAE_DIR     = "predictions_test"
OUT_DIR     = "probe_plots_simple"
LAMBDA_MAX  = 5.0

def find_model_stubs(results_dir: str):
    if not os.path.isdir(results_dir):
        return []
    return sorted([d for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d))])

def read_lambda_rows(model_dir: str) -> pd.DataFrame:
    """Read rows from results_lambda_*.csv inside a single model dir; add a float 'lambda' column."""
    rows = []
    for path in glob.glob(os.path.join(model_dir, "results_lambda_*.csv")):
        m = re.search(r"results_lambda_([0-9.]+)\.csv$", os.path.basename(path))
        if not m:
            continue
        lam = float(m.group(1))
        try:
            df = pd.read_csv(path)
            if df.empty:
                continue
            r = df.iloc[0].to_dict()
            r['lambda'] = lam
            rows.append(r)
        except Exception as e:
            print(f"[Warn] Could not read {path}: {e}")
    return pd.DataFrame(rows).sort_values('lambda') if rows else pd.DataFrame()

def read_mae_series(model_stub: str):
    """Scan MAE files: <MAE_DIR>/<stub>/lambda_*/metrics.csv -> (lambda, mae)."""
    model_dir = os.path.join(MAE_DIR, model_stub)
    lambdas, maes = [], []
    for path in glob.glob(os.path.join(model_dir, "lambda_*", "metrics.csv")):
        m = re.search(r"lambda_([0-9.]+)", path)
        if not m:
            continue
        lam = float(m.group(1))
        try:
            df = pd.read_csv(path)
            if not df.empty:
                for col in ("mae_recon", "recon_mae", "MAE", "mae"):
                    if col in df.columns:
                        lambdas.append(lam)
                        maes.append(float(df.iloc[0][col]))
                        break
        except Exception as e:
            print(f"[Warn] Could not read {path}: {e}")
    if not lambdas:
        return None, None
    order = np.argsort(lambdas)
    return np.array(lambdas)[order], np.array(maes)[order]

def get_ss8_acc_series(df: pd.DataFrame):
    """Return (lambda, y, yerr_low, yerr_high) for SS8 ACC if present; else (lambda, y, 0, 0)."""
    if df.empty:
        return None, None, None, None
    mean_col = "SS8_ACC_mean"
    low_col  = "SS8_ACC_ci95_low"
    high_col = "SS8_ACC_ci95_high"

    if mean_col in df.columns:
        x = df["lambda"].to_numpy()
        y = df[mean_col].to_numpy()
        if low_col in df.columns and high_col in df.columns:
            lo = np.clip(y - df[low_col].to_numpy(), a_min=0.0, a_max=None)
            hi = np.clip(df[high_col].to_numpy() - y, a_min=0.0, a_max=None)
        else:
            lo = np.zeros_like(y); hi = np.zeros_like(y)
        return x, y, lo, hi

    for c in ("SS8_ACC", "SS8_acc", "ACC_SS8"):
        if c in df.columns:
            x = df["lambda"].to_numpy()
            y = df[c].to_numpy()
            return x, y, np.zeros_like(y), np.zeros_like(y)

    return None, None, None, None

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    stubs = find_model_stubs(RESULTS_DIR)
    if not stubs:
        print("[Info] No model folders found.")
        return

    series_acc = [] 
    series_mae = []
    for stub in stubs:
        df = read_lambda_rows(os.path.join(RESULTS_DIR, stub))
        if df.empty:
            continue

        x, y, lo, hi = get_ss8_acc_series(df)
        if x is not None:
            m = (x <= LAMBDA_MAX)
            series_acc.append((stub, x[m], y[m], lo[m], hi[m]))

        x_mae, y_mae = read_mae_series(stub)
        if x_mae is not None:
            m2 = (x_mae <= LAMBDA_MAX)
            series_mae.append((stub, x_mae[m2], y_mae[m2]))

    if not series_acc and not series_mae:
        print("[Info] Nothing to plot.")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 5), constrained_layout=True, sharex=True)

    cmap = plt.cm.get_cmap("tab10")
    color_for = {}
    for i, (stub, *_rest) in enumerate(series_acc or series_mae):
        color_for.setdefault(stub, cmap(len(color_for) % 10))

    any_acc = False
    for stub, x, y, lo, hi in series_acc:
        if len(x) == 0:
            continue
        any_acc = True
        c = color_for[stub]
        if (lo is not None and hi is not None) and (np.any(lo > 0) or np.any(hi > 0)):
            ax1.errorbar(x, y, yerr=[lo, hi], fmt='o', capsize=3, linewidth=1, color=c)
        else:
            ax1.plot(x, y, marker='o', linestyle='none', color=c)
        ax1.plot(x, y, linestyle='-', linewidth=1, color=c, label=stub)
    ax1.set_ylabel("SS8 Accuracy")
    ax1.set_title(f"SS8 Accuracy and Mean Absolute Error vs λ")
    ax1.grid(True, linestyle="--", alpha=0.4)
    if any_acc:
        ax1.legend(loc="upper right", frameon=False, ncol=1)

    any_mae = False
    for stub, x, y in series_mae:
        if len(x) == 0:
            continue
        any_mae = True
        c = color_for.get(stub, cmap(len(color_for) % 10))
        ax2.plot(x, y, marker='o', linestyle='-', linewidth=1, color=c, label=stub)
    ax2.set_xlabel("λ")
    ax2.set_ylabel("MAE")
    ax2.grid(True, linestyle="--", alpha=0.4)
    if not any_mae:
        ax2.text(0.5, 0.5, "No MAE data", ha="center", va="center", transform=ax2.transAxes)

    out_path = os.path.join(OUT_DIR, f"ALL_MODELS_simple_ss8_acc_plus_mae_le_{int(LAMBDA_MAX)}.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"[OK] Saved {out_path}")

if __name__ == "__main__":
    main()
