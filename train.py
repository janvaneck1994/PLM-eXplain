import random
import torch
import pickle
import os
from torch.utils.data import DataLoader, Subset
import warnings
from sklearn.exceptions import ConvergenceWarning
from tqdm import tqdm
from sklearn.metrics import accuracy_score, r2_score
import torch.nn as nn
import torch.optim as optim
from models import SplitEmbeddingModel

from torch.autograd import Function

class FeatureConfig:
    def __init__(self, name, start_index, num_features, loss_fn, scaling_weight, task_type, loss_weight=1):
        """
        Initialize a feature configuration.

        Args:
            name (str): Name of the task.
            start_index (int): Start index of the feature in the tensor.
            num_features (int): Number of features for the task.
            loss_fn (nn.Module): Loss function for the task.
            scaling_weight (float): Initial scaling weight for the task.
            task_type (str): Type of task ('regression', 'binary', 'multiclass').
        """
        self.name = name
        self.start_index = start_index
        self.num_features = num_features
        self.loss_fn = loss_fn
        self.scaling_weight = nn.Parameter(torch.tensor(scaling_weight, requires_grad=True))  # Trainable weight
        self.task_type = task_type
        self.loss_weight = loss_weight

def evaluate(model, dataloader, feature_configs, reconstruction_loss_fn):
    """
    Evaluate the model on the given dataloader for multiple tasks including reconstruction.

    Args:
        model (nn.Module): The trained model.
        dataloader (DataLoader): DataLoader for the evaluation set.
        feature_configs (list): List of FeatureConfig objects.
        reconstruction_loss_fn (nn.Module): Loss function for reconstruction.

    Returns:
        dict: Loss, accuracy, and R² for each task and reconstruction.
    """
    model.eval()
    task_results = {config.name: {"loss": 0.0, "accuracy": 0.0, "r2": 0.0, "count": 0} for config in feature_configs}
    reconstruction_loss_total = 0.0

    with torch.no_grad():
        for batch in tqdm(dataloader):
            esm2_embedding = batch['ESM2Embedding'].to(device)

            # Forward pass
            known_subspace, _, reconstructed, _, _= model(esm2_embedding)

            # Reconstruction loss
            reconstruction_loss = reconstruction_loss_fn(reconstructed, esm2_embedding)
            reconstruction_loss_total += reconstruction_loss.item()

            for config in feature_configs:
                start_idx = config.start_index
                end_idx = start_idx + config.num_features
                prediction = known_subspace[:, start_idx:end_idx] * config.scaling_weight
                if config.task_type in ['regression', 'binary']:
                    prediction = prediction.squeeze()
                target = batch[config.name].to(device)

                # Compute task loss
                loss = config.loss_fn(prediction, target)
                task_results[config.name]["loss"] += loss.item()

                # Calculate accuracy and R²
                if config.task_type == "multiclass":
                    preds = torch.argmax(prediction, dim=1).cpu().numpy()
                    target = torch.argmax(target, dim=1).cpu().numpy()
                    task_results[config.name]["accuracy"] += (preds == target).sum()
                    task_results[config.name]["count"] += target.size
                elif config.task_type == "regression":
                    target = target.cpu().numpy()
                    prediction = prediction.cpu().numpy()
                    r2 = r2_score(target, prediction)
                    task_results[config.name]["r2"] += r2
                elif config.task_type == "binary":
                    preds = (torch.sigmoid(prediction) > 0.5).cpu().numpy()
                    target = target.cpu().numpy()
                    task_results[config.name]["accuracy"] += (preds == target).sum()
                    task_results[config.name]["count"] += target.size

    # Average metrics
    for config in feature_configs:
        task_results[config.name]["loss"] /= len(dataloader)
        if config.task_type == "multiclass" or config.task_type == "binary":
            task_results[config.name]["accuracy"] /= task_results[config.name]["count"]
        elif config.task_type == "regression":
            task_results[config.name]["r2"] /= len(dataloader)

    avg_reconstruction_loss = reconstruction_loss_total / len(dataloader)
    task_results["Reconstruction"] = {"loss": avg_reconstruction_loss}

    return task_results

def calculate_task_loss(known_subspace, batch, feature_config):
    start_idx = feature_config.start_index
    end_idx = start_idx + feature_config.num_features
    target = batch[feature_config.name].to(device)
    prediction = known_subspace[:, start_idx:end_idx] * feature_config.scaling_weight
    if feature_config.task_type in ['regression', 'binary']:
        prediction = prediction.squeeze()
    task_loss = feature_config.loss_fn(prediction, target)
    return task_loss

    

warnings.filterwarnings("ignore", category=ConvergenceWarning)
device = torch.device("cpu")
print(device)

# Constants
EMBEDDING_DIM = 480
EPOCHS = 20
LEARNING_RATE = 0.001
BATCH_SIZE = 64
SPLIT_DIR = 'generated_datasets_keep'

# Load datasets
train_dataset = pickle.load(open(os.path.join(SPLIT_DIR, 'train_dataset_20.pkl'), 'rb'))
val_dataset = pickle.load(open(os.path.join(SPLIT_DIR, 'test_dataset.pkl'), 'rb'))

data_subset = 0.1  # Use a smaller subset of the dataset
subset_size = int(len(train_dataset) * data_subset)
subset_indices = random.sample(range(len(train_dataset)), subset_size)
short_train_dataset = Subset(train_dataset, subset_indices)

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Initialize the model
model = SplitEmbeddingModel(
    input_dim=EMBEDDING_DIM,
    known_dim=34,  # SS8 (8) + SS3 (3) + AminoAcid (20) + AccessibleSurfaceArea (1) + AROM (1)
    hidden_dim=480
).to(device)

# Define feature configurations
feature_configs = [
    FeatureConfig("AccessibleSurfaceArea", start_index=0, num_features=1, loss_fn=nn.L1Loss(), scaling_weight=1.0, task_type="regression", loss_weight=10),
    FeatureConfig("SS8", start_index=1, num_features=8, loss_fn=nn.CrossEntropyLoss(), scaling_weight=1.0, task_type="multiclass", loss_weight=3),
    FeatureConfig("SS3", start_index=9, num_features=3, loss_fn=nn.CrossEntropyLoss(), scaling_weight=1.0, task_type="multiclass", loss_weight=2),
    FeatureConfig("AminoAcid", start_index=12, num_features=20, loss_fn=nn.CrossEntropyLoss(), scaling_weight=1.0, task_type="multiclass"),
    FeatureConfig("GRAVY", start_index=32, num_features=1, loss_fn=nn.L1Loss(), scaling_weight=1.0, task_type="regression", loss_weight=5),
    FeatureConfig("Aromaticity", start_index=33, num_features=1, loss_fn=nn.BCEWithLogitsLoss(), scaling_weight=1.0, task_type="binary"),
]

# Loss function for reconstruction
reconstruction_loss_fn = nn.L1Loss()

# Optimizer
optimizer = optim.Adam(list(model.parameters()) + [config.scaling_weight for config in feature_configs], lr=LEARNING_RATE)

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0.0
    reconstruction_loss_total = 0.0
    task_loss_logs = {config.name: 0.0 for config in feature_configs}
    adv_loss_logs = {config.name: 0.0 for config in feature_configs}

    for batch in tqdm(train_loader):
        esm2_embedding = batch['ESM2Embedding'].to(device)

        optimizer.zero_grad()

        # Forward pass
        known_subspace, _, reconstructed, _, discriminator_output = model(esm2_embedding)

        # Reconstruction loss
        reconstruction_loss = reconstruction_loss_fn(reconstructed, esm2_embedding)

        # Compute loss for each task
        total_loss = 1000 * reconstruction_loss
        for config in feature_configs:
            task_loss = calculate_task_loss(known_subspace, batch, config) * config.loss_weight
            adversarial_loss = calculate_task_loss(discriminator_output, batch, config) * config.loss_weight
            total_loss += task_loss + 1.5 * adversarial_loss
            task_loss_logs[config.name] += task_loss.item()
            adv_loss_logs[config.name] += adversarial_loss.item()

        # Backpropagation and optimization
        total_loss.backward()
        optimizer.step()

        epoch_loss += total_loss.item()
        reconstruction_loss_total += reconstruction_loss.item()

    avg_epoch_loss = epoch_loss / len(train_loader)
    avg_reconstruction_loss = reconstruction_loss_total / len(train_loader)
    print(f"Epoch {epoch + 1}/{EPOCHS}, Total Loss: {avg_epoch_loss:.4f}, Reconstruction Loss: {avg_reconstruction_loss:.4f}")
    for config in feature_configs:
        avg_task_loss = task_loss_logs[config.name] / len(train_loader)
        avg_adv_loss = adv_loss_logs[config.name] / len(train_loader)
        print(f"  {config.name} Task Loss: {avg_task_loss:.4f}, Adversarial Loss: {avg_adv_loss:.4f}")

    # Validation
    val_results = evaluate(model, val_loader, feature_configs, reconstruction_loss_fn)
    for config in feature_configs:
        print(f"Validation {config.name}: Loss: {val_results[config.name]['loss']:.4f}")
        if config.task_type == "multiclass" or config.task_type == "binary":
            print(f"Validation {config.name} Accuracy: {val_results[config.name]['accuracy']:.4f}")
        elif config.task_type == "regression":
            print(f"Validation {config.name} R²: {val_results[config.name]['r2']:.4f}")
    print(f"Validation Reconstruction Loss: {val_results['Reconstruction']['loss']:.4f}")


    model_save_path = "models/split_embedding_model_weights_1.5_real.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'scaling_weights': {config.name: config.scaling_weight.data.clone().cpu() for config in feature_configs}
    }, model_save_path)
    print(f"Model and scaling weights saved to {model_save_path}")