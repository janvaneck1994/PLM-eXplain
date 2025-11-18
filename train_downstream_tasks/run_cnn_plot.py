import os
import copy
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime

import matplotlib.pyplot as plt
from matplotlib import cm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report
)

def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(0)

task_labels = {
    0: 'A', 1: 'C', 2: 'D', 3: 'E', 4: 'F', 5: 'G', 6: 'H', 7: 'I', 8: 'K',
    9: 'L', 10: 'M', 11: 'N', 12: 'P', 13: 'Q', 14: 'R', 15: 'S', 16: 'T',
    17: 'V', 18: 'W', 19: 'Y',
    20: 'SS3_H', 21: 'SS3_E', 22: 'SS3_C',
    23: 'SS8_H', 24: 'SS8_E', 25: 'SS8_G', 26: 'SS8_I', 27: 'SS8_B',
    28: 'SS8_T', 29: 'SS8_S', 30: 'SS8_-',
    31: 'ASA',
    32: 'GRAVY',
    33: 'AROM'
}

device = torch.device(
    "mps" if torch.backends.mps.is_built() else (
        "cuda:1" if torch.cuda.is_available() else "cpu"
    )
)

class SequenceDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

class SequenceCNN(nn.Module):
    def __init__(self, input_features: int, sequence_length: int,
                 kernel_size: int, num_filters: int, dropout_rate: float = 0.3):
        super().__init__()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

        self.conv1 = nn.Conv1d(input_features, num_filters, kernel_size, padding=kernel_size // 2)
        self.pool1 = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(num_filters, num_filters * 2, kernel_size, padding=kernel_size // 2)
        self.pool2 = nn.MaxPool1d(2)

        dummy = torch.zeros(1, input_features, sequence_length)
        out = self._forward_features(dummy)
        fc_in = out.view(1, -1).size(1)

        self.fc = nn.Linear(fc_in, 1)

    def _forward_features(self, x):
        x = self.dropout(self.pool1(self.relu(self.conv1(x))))
        x = self.dropout(self.pool2(self.relu(self.conv2(x))))
        return x

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self._forward_features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def train_one_epoch(model, dataloader, criterion, optimizer):
    model.train()
    running = 0.0
    for sequences, labels in tqdm(dataloader, desc="Training", leave=False):
        sequences, labels = sequences.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(sequences).squeeze(1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running += loss.item()
    return running / len(dataloader)

@torch.no_grad()
def evaluate(model, dataloader, threshold=0.5, mask=False):
    model.eval()
    all_labels, all_outputs = [], []
    for sequences, labels in tqdm(dataloader, desc="Evaluating", leave=False):
        sequences, labels = sequences.to(device), labels.to(device)
        if mask:
            sequences[:, :, :34] = 0
        outputs = model(sequences).squeeze(1)
        all_labels.append(labels.cpu().numpy())
        all_outputs.append(outputs.cpu().numpy())
    all_labels = np.concatenate(all_labels)
    all_outputs = np.concatenate(all_outputs)
    preds = (all_outputs > threshold).astype(int)
    metrics = {
        'accuracy': accuracy_score(all_labels, preds),
        'precision': precision_score(all_labels, preds),
        'recall': recall_score(all_labels, preds),
        'f1': f1_score(all_labels, preds),
        'roc_auc': roc_auc_score(all_labels, all_outputs)
    }
    return metrics, all_outputs, all_labels

def train_model(model, train_loader, val_loader, criterion, optimizer,
                epochs=10, save_dir="checkpoints", early_stopping_patience=None):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_f1 = 0.0
    patience_counter = 0
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(epochs):
        print(f"\nEpoch [{epoch+1}/{epochs}]")
        tr_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        print(f"  Training Loss: {tr_loss:.4f}")
        metrics, _, _ = evaluate(model, val_loader)
        print(f"  Val Acc: {metrics['accuracy']:.4f} | "
              f"Prec: {metrics['precision']:.4f} | Rec: {metrics['recall']:.4f} | "
              f"F1: {metrics['f1']:.4f} | AUC: {metrics['roc_auc']:.4f}")
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pt"))
            print("  Best model updated.")
            patience_counter = 0
        else:
            patience_counter += 1
        if early_stopping_patience is not None and patience_counter >= early_stopping_patience:
            print("Early stopping triggered.")
            break

    model.load_state_dict(best_model_wts)
    return model

def extract_motifs(model, dataloader, original_sequences, kernel_size=15, top_k=5):
    model.eval()
    conv_outputs = []
    seq_storage = []
    def hook_fn(module, input, output):
        conv_outputs.append(output.detach().cpu().numpy())
    hook_handle = model.conv1.register_forward_hook(hook_fn)
    with torch.no_grad():
        for (sequences, _), seq_text in tqdm(
            zip(dataloader, original_sequences),
            desc="Extracting motifs", total=len(dataloader), leave=False
        ):
            sequences = sequences.to(device)
            _ = model(sequences)
            seq_storage.append(seq_text)
    hook_handle.remove()
    conv_outputs = np.concatenate(conv_outputs, axis=0)
    assert len(seq_storage) == conv_outputs.shape[0]
    num_filters = conv_outputs.shape[1]
    conv_length = conv_outputs.shape[2]
    all_filter_motifs = []
    for filter_idx in range(num_filters):
        print(f"Analyzing Filter {filter_idx}...")
        filter_acts = conv_outputs[:, filter_idx, :]
        flat_indices = np.argsort(filter_acts, axis=None)[-top_k:]
        motifs = []
        for flat_idx in flat_indices:
            sample_idx, pos_idx = divmod(flat_idx, conv_length)
            motif_start = pos_idx
            motif_end   = pos_idx + kernel_size
            seq_str = seq_storage[sample_idx]
            motif_str = seq_str[motif_start:motif_end]
            motifs.append((sample_idx, motif_start, motif_end, motif_str))
            print(motif_str)
        all_filter_motifs.append(motifs)
    return all_filter_motifs

class SingleKernelSequenceCNN(nn.Module):
    def __init__(self, model, idx: int):
        super().__init__()
        self.filter_weights = model.conv1.weight[idx].detach().clone()
        self.filter_bias = model.conv1.bias[idx].detach().clone() if model.conv1.bias is not None else torch.tensor(0.)

    def forward(self, x):
        # x: (B, L, F) -> scalar per sample
        x = x.permute(0, 2, 1)
        return ((x * self.filter_weights).sum(dim=(1, 2)) + self.filter_bias).unsqueeze(-1)

import py3Dmol
def save_colored_protein_html(pdb_file: str, activations: np.ndarray, out_html_path: str, cmap_name: str = "coolwarm"):
    activations = np.asarray(activations)
    if activations.max() == activations.min():
        norm_activations = np.zeros_like(activations)
    else:
        norm_activations = (activations - activations.min()) / (activations.max() - activations.min())
    cmap = cm.get_cmap(cmap_name)
    colors = [cmap(val) for val in norm_activations]
    hex_colors = [f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}" for r, g, b, _ in colors]
    viewer = py3Dmol.view(width=800, height=600)
    with open(pdb_file, "r") as f:
        pdb_content = f.read()
    viewer.addModel(pdb_content, "pdb")
    viewer.setStyle({"cartoon": {"color": "white"}})
    for i, color in enumerate(hex_colors):
        viewer.setStyle({"resi": [i + 1]}, {"cartoon": {"color": color}})
    viewer.zoomTo()
    html_content = viewer._make_html()
    with open(out_html_path, "w", encoding="utf-8") as f:
        f.write(html_content)

def _ensure_array(sv):
    if isinstance(sv, (list, tuple)):
        sv = sv[0]
    return sv

def _to_BLF(arr):
    """
    Convert arr to shape (B, L, F):
      B = batch/examples,
      L = window-length (product of all middle dims),
      F = features (last dim).
    Works for shapes: (B,F), (B,L,F), (B,L1,L2,F), (B,L,F,1), etc.
    """
    arr = np.asarray(arr)
    arr = np.squeeze(arr) 
    if arr.ndim < 2:
        arr = arr.reshape(1, 1, -1)
        return arr
    B = arr.shape[0]
    F = arr.shape[-1]
    if arr.ndim == 2:
        arr = arr.reshape(B, 1, F)
    else:
        L = int(np.prod(arr.shape[1:-1])) if arr.ndim > 3 else arr.shape[1]
        arr = arr.reshape(B, L, F)
    return arr


if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_DIR = os.path.join("outputs", f"run_{timestamp}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Saving figures to: {OUTPUT_DIR}")

    batch_size = 16 
    epochs = 20
    learning_rate = 0.0001
    kernel_size = 8
    dropout_rate = 0.2
    num_filters = 50
    use_weighted_sampler = False

    X_train = np.load("partitioned_embeddings/esm2_t12_35M_UR50D/train_TM_latent_embeddings_full.npy")
    train_df = pd.read_csv("downstream_task_data/train_sequences_TM.csv")
    y_train = train_df["label"].values
    train_sequences = train_df["Sequence"].tolist()

    X_test = np.load("partitioned_embeddings/esm2_t12_35M_UR50D/test_TM_latent_embeddings_full.npy")
    test_df = pd.read_csv("downstream_task_data/test_sequences_TM.csv")
    y_test = test_df["label"].values
    test_sequences = test_df["Sequence"].tolist()

    X_leptin = np.load("partitioned_embeddings/esm2_t12_35M_UR50D/leptin_latent_embeddings_full.npy")
    leptin_sequence = pd.read_csv("downstream_task_data/leptin_sequence.csv")["Sequence"].to_list()

    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42, stratify=y_train
    )

    train_loader = DataLoader(SequenceDataset(X_train_split, y_train_split), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(SequenceDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(SequenceDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

    model = SequenceCNN(
        input_features=X_train.shape[-1],
        sequence_length=X_train.shape[1],
        kernel_size=kernel_size,
        num_filters=num_filters, 
        dropout_rate=dropout_rate
    ).to(device)

    if not use_weighted_sampler:
        positive_count = int(sum(y_train))
        negative_count = len(y_train) - positive_count
        weight_ratio = negative_count / positive_count if positive_count > 0 else 1.0
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([weight_ratio], dtype=torch.float32).to(device)
        )
    else:
        criterion = nn.BCEWithLogitsLoss()

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model = train_model(
        model, train_loader, val_loader, criterion, optimizer,
        epochs=epochs, save_dir="checkpoints", early_stopping_patience=None
    )

    metrics, all_outputs, all_labels = evaluate(model, test_loader, mask=False)
    print("\nFinal Test Metrics:")
    for k, v in metrics.items():
        print(f"{k.capitalize()}: {v:.4f}")
    print("\nClassification Report (threshold=0.5):")
    preds = (all_outputs > 0.5).astype(int)
    print(classification_report(all_labels, preds))

    import shap

    motif_size = kernel_size
    num_background_motifs = 100
    random_range = 160

    background_motifs = []
    for seq in X_train:
        max_start = max(1, min(len(seq) - motif_size + 1, random_range))
        random_starts = np.random.randint(0, max_start, size=num_background_motifs)
        for start in random_starts:
            motif = seq[start:start + motif_size]
            background_motifs.append(motif)

    selected_indices = np.random.choice(len(background_motifs), size=num_background_motifs, replace=False)
    background_array = np.array([background_motifs[i] for i in selected_indices])
    background_data = torch.tensor(background_array, dtype=torch.float32).to(torch.device('cpu'))
    print(f"Background data shape: {background_data.shape}")
    leptin_motifs = []
    pad = motif_size // 2
    padded_sequence = np.pad(X_leptin[0], ((pad, pad), (0, 0)), mode='constant')
    seq_len_target = len(leptin_sequence[0])
    for i in range(seq_len_target):
        motif = padded_sequence[i:i + motif_size]
        leptin_motifs.append(motif)
    test_data = torch.tensor(np.array(leptin_motifs), dtype=torch.float32).to(torch.device('cpu'))
    print(f"Test data shape: {test_data.shape}")

    kernel_activations = {}
    model_cpu = model.to(torch.device('cpu')).eval()
    filter_importance = []

    def compute_sv(explainer, data_tensor):
        sv_raw = explainer.shap_values(data_tensor)
        sv = _to_BLF(_ensure_array(sv_raw))
        return sv 

    for kernel_idx in range(num_filters):
        masked_model = SingleKernelSequenceCNN(model_cpu, kernel_idx).eval()
        with torch.no_grad():
            kernel_activations[kernel_idx] = list(masked_model(test_data).squeeze().numpy())

        explainer_masked = shap.DeepExplainer(masked_model, background_data)
        sv = compute_sv(explainer_masked, test_data) 
        mean_abs_shap_filter = np.abs(sv).sum() 
        filter_importance.append((kernel_idx, mean_abs_shap_filter))

    filter_importance.sort(key=lambda x: x[1], reverse=True)
    print("\nTop 5 Most Important Filters:")
    for rank, (kernel_idx, importance) in enumerate(filter_importance[:5], start=1):
        print(f"  {rank}. Filter {kernel_idx} - Sum|SHAP|: {importance:.4f}")

    most_important_filter_idx = filter_importance[0][0]
    print(f"\nMost Important Filter: {most_important_filter_idx}")

    top_masked_model = SingleKernelSequenceCNN(model_cpu, most_important_filter_idx).eval()
    top_explainer = shap.DeepExplainer(top_masked_model, background_data)
    top_sv = compute_sv(top_explainer, test_data)

    test_data_np = _to_BLF(test_data.numpy())

    B_test, Lw, F = top_sv.shape

    shap_for_summary = top_sv.reshape(B_test * Lw, F)
    features_for_summary = test_data_np.reshape(B_test * Lw, F)
    feature_names = [task_labels.get(idx, f"Feature {idx}") for idx in range(F)]

    plt.figure()
    shap.summary_plot(
        shap_values=shap_for_summary,
        features=features_for_summary,
        feature_names=feature_names,
        max_display=10,
        cmap=cm.get_cmap("coolwarm"),
        show=False
    )
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "shap_summary.png"), dpi=200, bbox_inches="tight")
    plt.close()

    per_residue_attr = top_sv.mean(axis=1)
    sum_abs_shap = np.abs(per_residue_attr).sum(axis=0)
    top_features = np.argsort(sum_abs_shap)[::-1][:5]
    top_features_names = [task_labels.get(idx, f"Feature {idx}") for idx in top_features]

    heatmap_data_top_features = per_residue_attr[:, top_features]
    activation_values = kernel_activations[most_important_filter_idx] 

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.15,
        row_heights=[0.7, 0.3],
        subplot_titles=(
            f"Top 5 Features for Filter {most_important_filter_idx}",
            f"Activations for Filter {most_important_filter_idx}"
        ),
    )

    fig.add_trace(
        go.Heatmap(
            z=heatmap_data_top_features.T,
            x=list(range(heatmap_data_top_features.shape[0])),
            y=top_features_names,
            colorscale="RdBu_r",
            zmid=0,
            colorbar=dict(title="SHAP Value")
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(x=list(range(len(activation_values))), y=activation_values, mode='lines', name='Activation'),
        row=2, col=1
    )

    fig.update_layout(width=1600, height=500, showlegend=False)
    fig.update_xaxes(title_text="Sequence Position", row=2, col=1)
    fig.update_yaxes(title_text="Activation", row=2, col=1)

    pio.write_html(
        fig,
        file=os.path.join(OUTPUT_DIR, f"filter_{most_important_filter_idx}_heatmap_activation.html"),
        auto_open=False,
        include_plotlyjs="cdn",
    )

    pdb_file = "downstream_task_data/leptin_alphafold.pdb"
    protein_html_path = os.path.join(OUTPUT_DIR, "leptin_colored_by_activation.html")
    save_colored_protein_html(pdb_file, np.array(activation_values), protein_html_path)

    print(f"\nSaved files in: {OUTPUT_DIR}")
