import os
import copy
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset

# ----------------------
# Reproducibility
# ----------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # <-- Add this
    torch.backends.cudnn.benchmark = False     # <-- And this

os.environ["PYTHONHASHSEED"] = "42"

set_seed(42)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(42)

device = torch.device(
    "mps" if torch.backends.mps.is_built() else (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
)

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

class SequenceDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

class SequenceCNN(nn.Module):
    def __init__(self, input_features: int, sequence_length: int, kernel_size: int, num_filters: int, dropout_rate: float = 0.3):
        super(SequenceCNN, self).__init__()

        # Calculate padding to help maintain shape (you can modify this as needed)
        self.padding = 5
        self.conv1 = nn.Conv1d(in_channels=input_features, out_channels=num_filters, kernel_size=kernel_size, padding=self.padding)
        self.relu = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)  

        # Determine the flattened size after convolution/pooling
        dummy_input = torch.zeros((1, input_features, sequence_length))
        out = self._forward_features(dummy_input)
        flattened_size = out.view(1, -1).size(1)

        self.fc = nn.Linear(flattened_size, 1)

    def _forward_features(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool1(x)
        return x

    def forward(self, x):
        # x: (batch_size, sequence_length, input_features)
        x = x.permute(0, 2, 1)  # reshape to (batch_size, input_features, sequence_length)
        x = self._forward_features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def train_one_epoch(model, dataloader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    for sequences, labels in tqdm(dataloader, desc="Training", leave=False):
        sequences, labels = sequences.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(sequences).squeeze(1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(dataloader)

@torch.no_grad()
def evaluate(model, dataloader, threshold=0.5, mask=False):
    model.eval()
    all_labels = []
    all_outputs = []
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

def train_model(model, train_loader, test_loader, criterion, optimizer, epochs=10, save_dir="output_cnn", early_stopping_patience=None):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_f1 = 0.0
    patience_counter = 0
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for epoch in range(epochs):
        print(f"\nEpoch [{epoch+1}/{epochs}]")
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        print(f"  Training Loss: {train_loss:.4f}")
        metrics, _, _ = evaluate(model, test_loader)
        print(f"  Test Accuracy: {metrics['accuracy']:.4f}, Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, F1: {metrics['f1']:.4f}, ROC AUC: {metrics['roc_auc']:.4f}")
        current_f1 = metrics['f1']
        if current_f1 > best_f1:
            best_f1 = current_f1
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

def run_cnn_experiment(task, variant, cnn_params):

    # Mapping from our variant name to file naming abbreviation
    variant_mapping = {"adapted": "latent", "original": "valid", "binary_baseline": "binary", "baseline": "truncated_latent"}
    # Mapping for sequence CSV suffix (empty string means no suffix)
    sequence_mapping = {"EV": "_sequences_EV", "TM": "_sequences_TM", "AGG": "_sequences_AGG"}
    
    train_embeddings_path = f"downstream_task_data/new_embedding/train_{task}_{variant_mapping[variant]}_embeddings_full.npy"
    test_embeddings_path  = f"downstream_task_data/new_embedding/test_{task}_{variant_mapping[variant]}_embeddings_full.npy"
    
    # For sequences, use the suffix if provided; otherwise assume default file names.
    if sequence_mapping[task]:
        train_csv = f"downstream_task_data/train{sequence_mapping[task]}.csv"
        test_csv = f"downstream_task_data/test{sequence_mapping[task]}.csv"
    
    print(f"\nLoading embeddings from {train_embeddings_path} and {test_embeddings_path}")
    X_train = np.load(train_embeddings_path)
    X_test  = np.load(test_embeddings_path)
    train_df = pd.read_csv(train_csv)
    test_df  = pd.read_csv(test_csv)
    
    y_train = train_df["label"].values
    train_sequences = train_df["Sequence"].tolist()
    y_test  = test_df["label"].values
    test_sequences = test_df["Sequence"].tolist()
    
    # Create a validation split from the training data
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42, stratify=y_train
    )
    train_dataset = SequenceDataset(X_train_split, y_train_split)
    val_dataset = SequenceDataset(X_val, y_val)
    test_dataset = SequenceDataset(X_test, y_test)
    
    batch_size = cnn_params[task]["batch_size"]
    train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True,
    worker_init_fn=seed_worker, generator=g, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        worker_init_fn=seed_worker, generator=g, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        worker_init_fn=seed_worker, generator=g, num_workers=0
    )
    
    input_features = X_train.shape[-1]
    sequence_length = X_train.shape[1]
    model = SequenceCNN(
        input_features=input_features,
        sequence_length=sequence_length,
        kernel_size=cnn_params[task]["kernel_size"],
        num_filters=cnn_params[task]["num_filters"],
        dropout_rate=cnn_params[task]["dropout_rate"]
    ).to(device)
    
    # Loss and optimizer
    if not cnn_params[task]["use_weighted_sampler"]:
        positive_count = sum(y_train)
        negative_count = len(y_train) - positive_count
        weight_ratio = negative_count / positive_count if positive_count > 0 else 1.0
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([weight_ratio], dtype=torch.float32).to(device))
    else:
        criterion = nn.BCEWithLogitsLoss()
    
    optimizer = optim.Adam(model.parameters(), lr=cnn_params[task]["learning_rate"])
    
    # Use the output_cnn folder to store each experiment's outputs
    save_dir = f"output_cnn/{task}_{variant}"
    model = train_model(
        model, 
        train_loader, 
        val_loader,
        criterion, 
        optimizer,
        epochs=cnn_params[task]["epochs"],
        save_dir=save_dir,
        early_stopping_patience=None  # Adjust if you wish to use early stopping
    )
    
    # Final Evaluation
    metrics, all_outputs, all_labels = evaluate(model, test_loader, mask=False)
    print(f"\nFinal Test Metrics for {task} {variant}:")
    for k, v in metrics.items():
        print(f"  {k.capitalize()}: {v:.4f}")
    
    # Optionally, you can store additional outputs (like predictions) in the save_dir
    metrics_file = os.path.join(save_dir, "metrics.txt")
    with open(metrics_file, "w") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v:.4f}\n")
    
    return model, metrics

if __name__ == "__main__":
    # Define tasks and embedding variants
    tasks = ["AGG", "EV", "TM"]
    variants = ["adapted", "original", "baseline"]
    
    # Define CNN hyperparameters per task (modify as needed)
    cnn_params = {
        "TM": {
             "batch_size": 16,
             "epochs": 10,
             "learning_rate": 0.001,
             "kernel_size": 8,
             "dropout_rate": 0,
             "num_filters": 50,
             "use_weighted_sampler": False,
        },
        "EV": {
             "batch_size": 16,
             "epochs": 5,
             "learning_rate": 0.001,
             "kernel_size": 8,
             "dropout_rate": 0,
             "num_filters": 50,
             "use_weighted_sampler": False,
        },
        "AGG": {
             "batch_size": 16,
             "epochs": 15,
             "learning_rate": 0.001,
             "kernel_size": 6,
             "dropout_rate": 0,
             "num_filters": 50,
             "use_weighted_sampler": True,
        },
    }
    
    # Ensure the main output folder exists
    output_folder = "output_cnn"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    results = []
    for task in tasks:
        for variant in variants:
            print(f"\n=== Running CNN Experiment for Task: {task}, Variant: {variant} ===")
            model, metrics = run_cnn_experiment(task, variant, cnn_params)
            metrics_record = {"task": task, "variant": variant}
            metrics_record.update(metrics)
            results.append(metrics_record)
    
    # Write all experiment results to a CSV file in the output_cnn folder
    results_df = pd.DataFrame(results)
    results_csv = os.path.join(output_folder, "experiment_results.csv")
    results_df.to_csv(results_csv, index=False)
    print(f"\nAll experiment results have been stored in {results_csv}")
