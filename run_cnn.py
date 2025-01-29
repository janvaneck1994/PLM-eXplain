import tempfile
import threading
import webbrowser
from matplotlib import cm
from matplotlib.colors import TwoSlopeNorm
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, classification_report)
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import plotly.graph_objects as go
import os
import random
import copy

# ----------------------
# Reproducibility
# ----------------------
def set_seed(seed: int = 42):
    """
    Fix all possible randomness for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# ----------------------
# Task labels (if needed)
# ----------------------
task_labels = {
    0: 'AccessibleSurfaceArea', 
    1: 'SS8_H', 2: 'SS8_E', 3: 'SS8_G', 4: 'SS8_I', 5: 'SS8_B', 6: 'SS8_T', 7: 'SS8_S', 8: 'SS8_-',
    9: 'SS3_H', 10: 'SS3_E', 11: 'SS3_C',
    12: 'A', 13: 'C', 14: 'D', 15: 'E', 16: 'F', 17: 'G', 18: 'H', 19: 'I', 20: 'K', 
    21: 'L', 22: 'M', 23: 'N', 24: 'P', 25: 'Q', 26: 'R', 
    27: 'S', 28: 'T', 29: 'V', 30: 'W', 31: 'Y', 
    32: 'GRAVY', 
    33: "AROM"
}

# ----------------------
# Device configuration
# ----------------------
device = torch.device(
    "mps" if torch.backends.mps.is_built() else (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
)

# ----------------------
# Dataset & DataLoader
# ----------------------
class SequenceDataset(Dataset):
    """
    A basic PyTorch Dataset for sequences (X) and labels (y).
    """
    def __init__(self, sequences, labels):
        """
        sequences: np.ndarray or torch.Tensor with shape (num_sequences, seq_len, num_features)
        labels:    np.ndarray or torch.Tensor with shape (num_sequences,)
        """
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

# ----------------------
# Model Definition
# ----------------------
class SequenceCNN(nn.Module):
    """
    A CNN model for sequence classification/regression tasks.
    """
    def __init__(
        self,
        input_features: int = 768, 
        sequence_length: int = 1024,
        kernel_size: int = 3,
        num_filters: int = 15, 
        dropout_rate: float = 0.3
    ):
        """
        input_features: Number of input channels/features.
        sequence_length: Length of the input sequences.
        kernel_size: Convolution kernel size.
        num_filters: Number of filters in the convolutional layer(s).
        dropout_rate: Dropout rate.
        """
        super(SequenceCNN, self).__init__()

        # Calculate padding to maintain shape
        self.padding = 5
        # Convolutional layer
        self.conv1 = nn.Conv1d(
            in_channels=input_features, 
            out_channels=num_filters, 
            kernel_size=kernel_size,
            padding=self.padding
        )

        # Batch normalization layer
        self.relu = nn.ReLU()

        # Max pooling layer
        self.pool1 = nn.MaxPool1d(kernel_size=2)  

        # Dynamically compute the flattened size after these layers
        dummy_input = torch.zeros((1, input_features, sequence_length))
        out = self._forward_features(dummy_input)
        flattened_size = out.view(1, -1).size(1)

        # Classifier head
        # self.dropout = nn.Dropout(p=dropout_rate)
        self.fc = nn.Linear(flattened_size, 1)

    def _forward_features(self, x):
        """Forward pass through the CNN layers (feature extraction part)."""
        x = self.conv1(x)
        # x = self.bn1(x)
        x = self.relu(x)
        x = self.pool1(x)
        return x

    def forward(self, x):
        """
        x shape: (batch_size, sequence_length, input_features)
        We need to permute to (batch_size, input_features, sequence_length) for PyTorch conv1d.
        """
        x = x.permute(0, 2, 1)  # Permute to (batch_size, input_features, sequence_length)
        x = self._forward_features(x)
        x = x.view(x.size(0), -1)  # Flatten
        # x = self.dropout(x)
        x = self.fc(x)
        return x



# ----------------------
# Training & Evaluation
# ----------------------
def train_one_epoch(model, dataloader, criterion, optimizer):
    """
    Perform one epoch of training.
    Returns the average loss over the epoch.
    """
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
    """
    Evaluate the model on a given dataloader.
    Returns a dictionary of metrics: accuracy, precision, recall, f1, roc_auc.
    """
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

    # Compute metrics
    metrics = {}
    metrics['accuracy'] = accuracy_score(all_labels, preds)
    metrics['precision'] = precision_score(all_labels, preds)
    metrics['recall'] = recall_score(all_labels, preds)
    metrics['f1'] = f1_score(all_labels, preds)
    metrics['roc_auc'] = roc_auc_score(all_labels, all_outputs)

    return metrics, all_outputs, all_labels


def train_model(
    model, 
    train_loader, 
    test_loader, 
    criterion, 
    optimizer, 
    epochs=10, 
    save_dir="checkpoints", 
    early_stopping_patience=None
):
    """
    Train the model for a specified number of epochs. Optionally use early stopping.
    Saves best model checkpoints in `save_dir`.
    """
    best_model_wts = copy.deepcopy(model.state_dict())
    best_f1 = 0.0
    patience_counter = 0

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for epoch in range(epochs):
        print(f"\nEpoch [{epoch+1}/{epochs}]")
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        print(f"  Training Loss: {train_loss:.4f}")

        # Evaluate on the test set
        metrics, _, _ = evaluate(model, test_loader)
        print(f"  Test Accuracy: {metrics['accuracy']:.4f}, "
              f"Precision: {metrics['precision']:.4f}, "
              f"Recall: {metrics['recall']:.4f}, "
              f"F1: {metrics['f1']:.4f}, "
              f"ROC AUC: {metrics['roc_auc']:.4f}")

        # Checkpointing
        current_f1 = metrics['f1']
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pt"))
            print("  Best model updated.")
            patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping
        if early_stopping_patience is not None:
            if patience_counter >= early_stopping_patience:
                print("Early stopping triggered.")
                break

    # Load best weights
    model.load_state_dict(best_model_wts)
    return model

# ----------------------
# Motif Extraction
# ----------------------
def extract_motifs(model, dataloader, original_sequences, kernel_size=15, top_k=5):
    """
    Extract motifs from the first convolutional layer by analyzing the top_k 
    activations per filter.

    Args:
        model: Trained CNN model
        dataloader: DataLoader for the dataset (preferably training data)
        original_sequences: A list of original string sequences. 
                           Must align with the dataloader order.
        kernel_size: The convolution kernel size. Must match model's conv1.
        top_k: Number of top activations to extract per filter.

    Returns:
        A list of lists of extracted motifs (one list per filter).
    """
    model.eval()
    conv_outputs = []
    seq_storage = []

    # Hook to capture the outputs of the first convolution
    def hook_fn(module, input, output):
        conv_outputs.append(output.detach().cpu().numpy())

    hook_handle = model.conv1.register_forward_hook(hook_fn)

    with torch.no_grad():
        for (sequences, _), seq_text in tqdm(zip(dataloader, original_sequences), 
                                             desc="Extracting motifs", 
                                             total=len(dataloader),
                                             leave=False):
            # sequences shape: (batch_size, seq_len, input_features)
            sequences = sequences.to(device)
            _ = model(sequences)
            seq_storage.append(seq_text)

    hook_handle.remove()

    # conv_outputs will be a list of arrays (one per batch). Concatenate them.
    conv_outputs = np.concatenate(conv_outputs, axis=0)  # (num_samples, num_filters, conv_len)

    # For simplicity, ensure we have as many original_sequences as conv_outputs
    # in the same order. If your dataloader is shuffled, you'll need to handle that carefully.
    assert len(seq_storage) == conv_outputs.shape[0], (
        f"Mismatch: {len(seq_storage)} vs {conv_outputs.shape[0]}"
    )

    num_filters = conv_outputs.shape[1]
    conv_length = conv_outputs.shape[2]
    all_filter_motifs = []

    # For each filter, find the top activations
    for filter_idx in range(num_filters):
        print(f"Analyzing Filter {filter_idx}...")
        filter_activations = conv_outputs[:, filter_idx, :]  # (num_samples, conv_length)

        # Flatten and take top_k indices
        flat_indices = np.argsort(filter_activations, axis=None)[-top_k:]
        motifs = []

        for flat_idx in flat_indices:
            sample_idx, pos_idx = divmod(flat_idx, conv_length)

            # We assume stride=1 and the kernel covers [pos_idx, pos_idx+kernel_size]
            motif_start = pos_idx
            motif_end   = pos_idx + kernel_size

            # Retrieve the actual subsequence from original text
            seq_str = seq_storage[sample_idx]
            # Safely handle edges
            motif_str = seq_str[motif_start:motif_end]
            motifs.append((sample_idx, motif_start, motif_end, motif_str))

            # print(f"  Sample {sample_idx}, Position {motif_start}-{motif_end}: {motif_str}")
            print(motif_str)

        all_filter_motifs.append(motifs)

    return all_filter_motifs

# ----------------------
# Main Execution Flow
# ----------------------
if __name__ == "__main__":
    # Hyperparameters
    batch_size = 16 
    epochs = 40
    learning_rate = 0.001
    kernel_size = 3
    dropout_rate = 0.3
    sequence_length = 6
    num_filters = 40
    use_weighted_sampler = False

    #### TM
    # batch_size = 16 
    # epochs = 20
    # learning_rate = 0.0001
    # kernel_size = 11
    # dropout_rate = 0.3
    # sequence_length = 6
    # num_filters = 20
    # use_weighted_sampler = False

    # EV
    # batch_size = 16 
    # epochs = 20
    # learning_rate = 0.00001
    # kernel_size = 31
    # dropout_rate = 0.3
    # sequence_length = 512
    # num_filters = 30
    # use_weighted_sampler = False




    X_train = np.load("data/embeddings_avg/train_latent_embeddings_full.npy")
    train_df = pd.read_csv("data/train.csv")
    y_train = train_df["label"].values
    train_sequences = train_df["Sequence"].tolist()

    X_test = np.load("data/embeddings_avg/test_latent_embeddings_full.npy")
    test_df = pd.read_csv("data/test.csv")
    y_test = test_df["label"].values
    test_sequences = test_df["Sequence"].tolist()

    X_leptin = np.load("data/embeddings_avg/test_TM_leptin_latent_embeddings_full.npy")
    leptin_sequence = pd.read_csv("data/leptin_sequence.csv")["Sequence"].to_list()

    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42, stratify=y_train
    )

    train_dataset = SequenceDataset(X_train_split, y_train_split)
    val_dataset = SequenceDataset(X_val, y_val)
    test_dataset = SequenceDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # ----------------------
    # Initialize Model
    # ----------------------
    model = SequenceCNN(
        input_features=X_train.shape[-1],
        sequence_length=X_train.shape[1],
        kernel_size=kernel_size,
        num_filters=num_filters, 
        dropout_rate=dropout_rate
    ).to(device)

    # ----------------------
    # Loss Function
    # ----------------------
    # Option A: If not using WeightedRandomSampler, we can use pos_weight for class imbalance
    if not use_weighted_sampler:
        positive_count = sum(y_train)
        negative_count = len(y_train) - positive_count
        weight_ratio = negative_count / positive_count if positive_count > 0 else 1.0
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([weight_ratio], dtype=torch.float32).to(device))
    else:
        criterion = nn.BCEWithLogitsLoss()

    # ----------------------
    # Optimizer
    # ----------------------
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # ----------------------
    # Train the Model
    # ----------------------
    model = train_model(
        model, 
        train_loader, 
        val_loader,
        criterion, 
        optimizer,
        epochs=epochs,
        save_dir="checkpoints",
        early_stopping_patience=None  # or None to disable
    )

    # ----------------------
    # Final Evaluation
    # ----------------------
    metrics, all_outputs, all_labels = evaluate(model, test_loader, mask=False)
    print("\nFinal Test Metrics:")
    for k, v in metrics.items():
        print(f"{k.capitalize()}: {v:.4f}")

    # Optionally, print a classification report
    print("\nClassification Report (threshold=0.5):")
    preds = (all_outputs > 0.5).astype(int)
    print(classification_report(all_labels, preds))
    model.eval()
    import shap
    exit()
    import torch.nn as nn

    class SingleKernelSequenceCNN(nn.Module):
        def __init__(self, model, idx: int):
            super().__init__()
            self.filter_weights = model.conv1.weight[idx]
            self.filter_bias = model.conv1.bias[idx] if model.conv1.bias is not None else 0

        def forward(self, x):
            x = x.permute(0, 2, 1)
            return ((x * self.filter_weights).sum(dim=(1, 2)) + self.filter_bias).unsqueeze(-1)

    # Parameters
    motif_size = kernel_size  # Size of the motifs
    num_background_motifs = 100  # Number of motifs for the background
    random_range = 160  # Random start point range

    # Create motifs for the background data
    background_motifs = []
    for seq in X_train:  # Iterate through training sequences
        # Randomly choose starting points within the range
        random_starts = np.random.randint(0, min(len(seq) - motif_size + 1, random_range), size=num_background_motifs)
        for start in random_starts:
            motif = seq[start:start + motif_size]  # Extract motif of size `motif_size`
            background_motifs.append(motif)

    # Randomly select a subset of motifs
    selected_indices = np.random.choice(len(background_motifs), size=num_background_motifs, replace=False)
    background_array = np.array([background_motifs[i] for i in selected_indices])

    # Convert to PyTorch tensor
    background_data = torch.tensor(background_array, dtype=torch.float32).to(torch.device('cpu'))
    print(f"Background data shape: {background_data.shape}")

    # Create motifs for the test data using sliding window
    leptin_motifs = []

    padded_sequence = np.pad(X_leptin[0], ((5, 5), (0, 0)), mode='constant')
    # padded_sequence = X_leptin[0]

    for i in range(len(X_test[0])):
        motif = padded_sequence[i:i + motif_size]  # Extract motif of size `motif_size` from the first test sequence
        leptin_motifs.append(motif)
    print(leptin_sequence)
    leptin_array = np.array(leptin_motifs[:len(leptin_sequence[0])])#[:27])
    test_data = torch.tensor(leptin_array, dtype=torch.float32).to(torch.device('cpu'))
    print(f"Test data shape: {test_data.shape}")
    
    filter_importance = []
    model.eval()
    
    kernel_activations = {}
    kernel_activations = {}

    for kernel_idx in range(num_filters):
        # Initialize single-kernel model
        masked_model = SingleKernelSequenceCNN(model.to(torch.device('cpu')), kernel_idx)
        masked_model.eval()

        with torch.no_grad():
            kernel_activations[kernel_idx] = list(masked_model(test_data).squeeze().numpy())

        # Compute SHAP values
        explainer_masked = shap.DeepExplainer(masked_model.to(torch.device('cpu')), background_data.to(torch.device('cpu')))
        shap_values_masked = explainer_masked.shap_values(test_data)
        print(shap_values_masked.shape)
        #exit()
        print("shap_values_masked.shape:", shap_values_masked.shape)
        print("Baseline (expected value):", explainer_masked.expected_value)

        # Squeeze and average SHAP values
        shap_values_masked_squeezed = shap_values_masked.squeeze(-1)  # Remove extra dimension if necessary
        heatmap_data = shap_values_masked_squeezed.mean(axis=0)       # Mean over the batch -> shape: (seq_len, features)

        # Compute mean absolute SHAP value for the entire filter
        mean_abs_shap_filter = np.abs(heatmap_data).sum()  # Scalar: Overall importance of the filter
        filter_importance.append((kernel_idx, mean_abs_shap_filter))  # Store filter index and its importance

        # Optional: Compute mean absolute SHAP values for each feature
        sum_abs_shap = np.abs(heatmap_data).sum(axis=0)  # shape: (features,)

        # Get top features for this filter
        top_features = np.argsort(sum_abs_shap)[::-1]  # Indices of features sorted by importance (descending)
        top_features_values = sum_abs_shap[top_features]  # Corresponding SHAP values

        print(f"\nTop Features for Kernel {kernel_idx}:")
        for i in range(5):  # Change 5 to any number to display more/less top features
            feature_idx = top_features[i]
            feature_name = task_labels.get(feature_idx, f"Feature {feature_idx}")
            print(f"  {feature_name}: {top_features_values[i]:.4f}")

    # Sort filters by overall importance
    filter_importance = sorted(filter_importance, key=lambda x: x[1], reverse=True)

    # Print top 5 most important filters
    print("\nTop 5 Most Important Filters:")
    for rank, (kernel_idx, importance) in enumerate(filter_importance[:5], start=1):
        print(f"  {rank}. Filter {kernel_idx} - Mean Absolute SHAP Value: {importance:.4f}")

    # Select the most important filter
    most_important_filter_idx = filter_importance[0][0]  # Index van de meest belangrijke filter
    print(f"\nMost Important Filter: {most_important_filter_idx}")

    # Recompute SHAP values voor de meest belangrijke filter
    masked_model = SingleKernelSequenceCNN(model, most_important_filter_idx)
    masked_model.eval()

    explainer_masked = shap.DeepExplainer(masked_model, background_data)
    shap_values_masked = explainer_masked.shap_values(test_data)

    # Squeeze en gemiddelde SHAP waarden
    shap_values_masked_squeezed = shap_values_masked.squeeze(-1)  # Verwijder extra dimensie indien nodig
    heatmap_data = np.sum(shap_values_masked_squeezed, axis=1)
    print("Heatmap data shape:", heatmap_data.shape)

    # Converteer feature indices naar namen voor plotting
    feature_names = [task_labels.get(idx, f"Feature {idx}") for idx in range(shap_values_masked_squeezed.shape[2])]

    cmap = cm.get_cmap("coolwarm")


    # SHAP summary plot (optioneel, kan worden verwijderd als je alleen de heatmap en lijnplot wilt)
    shap.summary_plot(
        shap_values=shap_values_masked_squeezed.reshape(-1, shap_values_masked_squeezed.shape[-1]),
        features=test_data.cpu().numpy().reshape(-1, test_data.shape[-1]),
        feature_names=feature_names,
        max_display=10,
        cmap=cmap,  # Pass the colormap
    )

    sum_abs_shap = np.abs(shap_values_masked_squeezed).sum(axis=0).sum(axis=0)  # vorm: (features,)
    top_features = np.argsort(sum_abs_shap)[::-1][:5]  # Indices van de top 5 features
    top_features_names = [task_labels.get(idx, f"Feature {idx}") for idx in top_features]

    # Extract SHAP waarden voor de top 5 features over de sequentie
    heatmap_data_top_features = heatmap_data[:, top_features]  # vorm: (seq_len, 5)

    activation_values = kernel_activations[most_important_filter_idx] # Vorm: (1, seq_len, ...)

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.15,
        row_heights=[0.7, 0.3],  # Pas verhoudingen aan indien nodig
        subplot_titles=(
            f"Top 5 Features for Filter {most_important_filter_idx}",
            f"Activations for Filter {most_important_filter_idx}"
        ),
    )

    heatmap = go.Heatmap(
        z=heatmap_data_top_features.T, 
        x=list(range(heatmap_data_top_features.shape[0])), 
        y=top_features_names, 
        colorscale="rdbu_r",
        zmid=0,
        colorbar=dict(title="SHAP Value")
    )
    fig.add_trace(heatmap, row=1, col=1)
    
    # Voeg de lijnplot toe in de tweede rij
    activation_plot = go.Scatter(
        x=list(range(len(activation_values))), 
        y=activation_values, 
        mode='lines',
        name='Activatie',
        line=dict(color='black')
    )
    fig.add_trace(activation_plot, row=2, col=1)

    # Update layout van de figuur
    fig.update_layout(
        title_text=f"",
        width=1200,
        height=400,
        showlegend=False
    )

    # Werk de as-titels bij
    fig.update_xaxes(title_text="", row=1, col=1)
    fig.update_yaxes(title_text="", row=1, col=1)
    fig.update_xaxes(title_text="Sequence Position", row=2, col=1)
    fig.update_yaxes(title_text="Activation", row=2, col=1)

    # Toon de figuur
    fig.show()

    import py3Dmol
    import numpy as np
    from matplotlib import cm

    def show_colored_protein(pdb_file: str, activations: np.ndarray, cmap_name: str = "coolwarm"):
        """
        Visualize a protein ribbon with amino acids colored based on activations directly in the browser.

        Args:
            pdb_file (str): Path to the PDB file of the protein structure.
            activations (np.ndarray): 1D array of activation values for each residue.
            cmap_name (str): Name of the colormap to use (e.g., "viridis", "plasma").
        """
        # Normalize activations to a range of [0, 1]
        norm_activations = (activations - activations.min()) / (activations.max() - activations.min())
        
        # Generate colors based on activations using the chosen colormap
        cmap = cm.get_cmap(cmap_name)
        colors = [cmap(val) for val in norm_activations]

        # Convert to hex color strings
        hex_colors = [
            f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"
            for r, g, b, _ in colors
        ]

        # Create a Py3Dmol viewer
        viewer = py3Dmol.view(width=800, height=600)

        # Read PDB file content
        with open(pdb_file, "r") as f:
            pdb_content = f.read()

        # Load the protein structure
        viewer.addModel(pdb_content, "pdb")

        # Style the protein as a cartoon ribbon
        viewer.setStyle({"cartoon": {"color": "white"}})

        # Apply color to each residue based on activations
        print(len(hex_colors))
        for i, color in enumerate(hex_colors):
            viewer.setStyle({'resi': [i+1]}, {'cartoon': {'color': color}})

        # Zoom to the structure
        viewer.zoomTo()

        # Get the HTML content
        html_content = viewer._make_html()

        # Serve the HTML content
        with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as temp_file:
            temp_file.write(html_content.encode("utf-8"))
            temp_file_path = temp_file.name

        # Open in default web browser
        webbrowser.open(f"file://{temp_file_path}")

        # Clean up the temporary file after some time
        def clean_up():
            os.remove(temp_file_path)

        threading.Timer(60, clean_up).start() 

    # Example usage
    pdb_file = "/Users/Eck00018/Documents/PhD/ECCB/plots/leptin_alphafold.pdb"  # Replace with the path to your PDB file

    show_colored_protein(pdb_file, np.array(activation_values))

    # # ----------------------
    # # Extract Motifs
    # # ----------------------
    # # NOTE: If train_loader is shuffled or has multiple batches, 
    # # you need a consistent way to match each batch to the original sequence. 
    # # For demonstration, we'll pass the entire train data as single-batch or 
    # # just be aware that you might need to adjust how sequences are gathered.

    # # One approach is to rebuild a "single-batch" dataloader for motif extraction:
    # motif_dataset = SequenceDataset(X_train, y_train) 
    # motif_loader = DataLoader(motif_dataset, batch_size=1, shuffle=False)
    # activated_motifs = extract_motifs(
    #     model=model,
    #     dataloader=motif_loader,
    #     original_sequences=train_sequences,  # must align with motif_loader
    #     kernel_size=kernel_size,
    #     top_k=50  # number of motifs to extract per filter
    # )

    # all_features = X_train[:, :, :]
    # global_mean = all_features.mean(axis=(0, 1))    
    # filter_weights = model.conv1.weight.detach().cpu().numpy()  # shape: (num_filters, input_features, kernel_size)
    
    # for i in range(num_filters):
    #     motifs = activated_motifs[i]  # top-k motifs for this filter
    #     top_k = len(motifs)
        
    #     # Get filter-specific weights
    #     filter_weights_i = filter_weights[i, :, :]

    #     # ---- Compute the percentage of positives for these top-k motifs ----
    #     pos_labels = [y_train[sample_idx] for (sample_idx, _, _, _) in motifs]
    #     pos_rate = np.mean(pos_labels) * 100.0

    #     # ---- Collect embeddings and compute average motif ----
    #     all_motifs_embs = []
    #     for (sample_idx, motif_start, motif_end, motif_str) in motifs:
    #         motif_embs = X_train[sample_idx, motif_start:motif_end, :] #* filter_weights_i.T
    #         all_motifs_embs.append(motif_embs)

    #     all_motifs_embs = np.stack(all_motifs_embs, axis=0)  
    #     avg_motif_embs = all_motifs_embs.mean(axis=0)        # shape: (kernel_size, features)

    #     # Multiply by filter weights if needed
    #     heatmap_data = avg_motif_embs.T # shape: (features, kernel_size)

    #     # Create a Plotly heatmap
    #     fig = go.Figure(
    #         data=go.Heatmap(
    #             z=heatmap_data,
    #             x=list(str(x) for x in range(1, kernel_size + 1)),               # X-axis tick labels
    #             y=[str(task_labels.get(j, j)) for j in range(480)],           # Y-axis tick labels
    #             colorscale="RdBu",      # Divergent colormap
    #             colorbar=dict(title="Value")                     # or your preferred label
    #         )
    #     )

    #     fig.update_layout(
    #         title=(
    #             f"Filter {i} - Avg Features for Top-K Motifs<br>"
    #             f"(Positive labels among top motifs: {pos_rate:.2f}%)"
    #         ),
    #         xaxis_title="Position along motif",
    #         yaxis_title="Crafted Features",
    #         width=700,
    #         height=700
    #     )

    #     fig.show()
