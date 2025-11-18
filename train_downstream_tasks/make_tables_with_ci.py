import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

CSV_PATH = "output_gradient_boost/metrics_summary_with_ci.csv"
OUTDIR = Path("ci_tables_png")
OUTDIR.mkdir(parents=True, exist_ok=True)

TASK_FRIENDLY = {
    "AGG": "Aggregation propensity",
    "EV": "EV association",
    "TM": "Transmembrane helix",
}
EMBED_FRIENDLY = {
    "Partitioned": "Partitioned (PLM-X)",
    "ESM2": "Original (ESM2)",
    "Baseline": "Crafted only (Baseline)",
    "Informed": "Crafted only",
}
EMBED_ORDER = ["Partitioned", "ESM2", "Baseline", "Informed"]

MODEL_ALIASES = {
    "35M":     ["35m", "t12_35m"],
    "650M":    ["650m", "t33_650m"],
    "protbert":["protbert", "prot_bert"],
}

def pick_models(df, alias):
    keys = MODEL_ALIASES[alias]
    mask = df["Model Subfolder"].str.lower().str.contains("|".join(keys))
    return df[mask].copy()

def format_pm(mean, low, high):
    if pd.isna(mean) or pd.isna(low) or pd.isna(high):
        return ""
    hw = (high - low) / 2.0
    return f"{mean:.2f} ± {hw:.2f}"

def build_table_df(df):
    rows = []
    for task_key in ["AGG", "EV", "TM"]:
        dft = df[df["Task"] == task_key]
        for emb in EMBED_ORDER:
            row = dft[dft["Embedding Type"] == emb]
            if row.empty:
                rows.append([TASK_FRIENDLY[task_key], EMBED_FRIENDLY[emb], "", "", ""])
                continue
            r = row.iloc[0]
            roc = format_pm(r["ROC_AUC_mean"], r["ROC_AUC_ci_low"], r["ROC_AUC_ci_high"])
            acc = format_pm(r["Accuracy_mean"], r["Accuracy_ci_low"], r["Accuracy_ci_high"])
            f1  = format_pm(r["F1_mean"],       r["F1_ci_low"],       r["F1_ci_high"])
            rows.append([TASK_FRIENDLY[task_key], EMBED_FRIENDLY[emb], roc, acc, f1])
    return pd.DataFrame(rows, columns=["Prediction task", "Embeddings", "ROC-AUC", "Accuracy", "F1"])

def render_table_image(tbl, out_path, title=""):
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.axis("off")

    table = ax.table(
        cellText=tbl.values,
        colLabels=tbl.columns,
        loc="center",
        cellLoc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)

    plt.title(title, fontsize=12, pad=10)
    plt.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close()

df = pd.read_csv(CSV_PATH)

for alias in ["35M", "650M", "protbert"]:
    sub = pick_models(df, alias)
    if sub.empty:
        print(f"[WARN] No rows for {alias}")
        continue
    tbl = build_table_df(sub)
    render_table_image(tbl, OUTDIR / f"table_{alias}.png", title=f"{alias} Backbone")
    print(f"[OK] Saved {OUTDIR}/table_{alias}.png")
