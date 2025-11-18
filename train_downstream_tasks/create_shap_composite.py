import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

root_dir = Path("output_gradient_boost")
models   = ["esm2_t12_35M_UR50D", "esm2_t33_650M_UR50D", "prot_bert_bfd"]
tasks    = ["AGG_AGG_", "EV_EV_", "TM_TM_"]
variants = ["valid", "latent", "binary"]

col_headers = [
    "Original Embedding",
    "Partitioned Embedding",
    "Crafted Only",
]
task_names = {
    "AGG_AGG_": "Aggregation Propensity",
    "EV_EV_":  "Extracellular Vesicle",
    "TM_TM_":  "α-Helical Transmembrane Protein",
}
out_dir = root_dir / "_composites_per_task"
out_dir.mkdir(parents=True, exist_ok=True)

def load_image_or_placeholder(p: Path, placeholder_text="Missing"):
    if p.exists():
        try:
            return np.asarray(Image.open(p).convert("RGB"))
        except Exception:
            pass
    fig = plt.figure(figsize=(6, 4), dpi=120)
    plt.axis("off")
    plt.text(0.5, 0.5, placeholder_text, ha="center", va="center",
             fontsize=16, weight="bold", transform=plt.gca().transAxes)
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
    plt.close(fig)
    return buf

def make_composite_for_task(task: str):
    fig, axes = plt.subplots(
        nrows=len(models), ncols=len(variants),
        figsize=(14, 10), dpi=140
    )

    for j, header in enumerate(col_headers):
        axes[0, j].set_title(header, fontsize=14, pad=10)

    for i, model in enumerate(models):
        axes[i, 0].set_ylabel(model, fontsize=11)
        for j, variant in enumerate(variants):
            ax = axes[i, j]
            path = root_dir / model / f"{task}{variant}" / "top10_feature_appearance_counts.png"
            img = load_image_or_placeholder(path, placeholder_text="Missing")
            ax.imshow(img)
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.965])

    out_path = out_dir / f"composite_{task.strip('_')}.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path

def main():
    for task in tasks:
        out = make_composite_for_task(task)
        print(f"Saved: {out}")

if __name__ == "__main__":
    main()
