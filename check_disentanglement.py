import csv
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor, MLPClassifier
import pickle
import os
from torch.utils.data import DataLoader
from history.models_reverse import AdversaryPrediction, TaskPredictorOneToOne
from models import SplitEmbeddingModel

# Constants
CRAFTED_LATENT = 34
NUMBER_OF_LATENTS = 768
EMBEDDING_DIM = 480
LEARNING_RATE = 0.001
BATCH_SIZE = 64
SPLIT_DIR = 'generated_datasets'
MODEL_WEIGHTS = [
    # 'split_embedding_model_weights_0.pth',
    'split_embedding_model_weights_real.pth',
    # 'split_embedding_model_weights_2.pth',
    # 'split_embedding_model_weights_3.pth',
]
CSV_FILE = 'evaluation_results.csv'

# Initialize CSV file
with open(CSV_FILE, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow([
        'Model', 'Task', 'Train_Accuracy/R2', 'Test_Accuracy/R2', 
        'Adversary_Train_Accuracy/R2', 'Adversary_Test_Accuracy/R2', 'Reconstruction_Error'
    ])

# Load test dataset
test_dataset_path = os.path.join(SPLIT_DIR, 'test_dataset.pkl')
with open(test_dataset_path, 'rb') as f:
    test_dataset = pickle.load(f)

test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Device setup
device = torch.device("cpu")

# ModelConfig class
class ModelConfig:
    def __init__(self, name, predictor_class, adversary_class, latent_index, learning_rate, weight, criterion, is_classification, latents):
        self.name = name
        self.latent_index = latent_index
        self.weight = weight
        self.is_classification = is_classification
        self.latents = latents

        self.predictor = predictor_class
        self.adversary = adversary_class

# Define configurations
configs = [
    ModelConfig("AccessibleSurfaceArea", TaskPredictorOneToOne(1).to(device), AdversaryPrediction(NUMBER_OF_LATENTS - CRAFTED_LATENT, 1).to(device), 0, LEARNING_RATE, 5, nn.L1Loss(), False, 1),
    ModelConfig("SS8", TaskPredictorOneToOne(8).to(device), AdversaryPrediction(NUMBER_OF_LATENTS - CRAFTED_LATENT, 8).to(device), 1, LEARNING_RATE, 1, nn.CrossEntropyLoss(), True, 8),
    ModelConfig("SS3", TaskPredictorOneToOne(3).to(device), AdversaryPrediction(NUMBER_OF_LATENTS - CRAFTED_LATENT, 3).to(device), 9, LEARNING_RATE, 1, nn.CrossEntropyLoss(), True, 3),
    ModelConfig("AminoAcid", TaskPredictorOneToOne(20).to(device), AdversaryPrediction(NUMBER_OF_LATENTS - CRAFTED_LATENT, 20).to(device), 12, LEARNING_RATE, 10, nn.CrossEntropyLoss(), True, 20),
    ModelConfig("GRAVY", TaskPredictorOneToOne(1).to(device), AdversaryPrediction(NUMBER_OF_LATENTS - CRAFTED_LATENT, 1).to(device), 32, LEARNING_RATE, 2, nn.L1Loss(), False, 1),
    ModelConfig("Aromaticity", TaskPredictorOneToOne(1).to(device), AdversaryPrediction(NUMBER_OF_LATENTS - CRAFTED_LATENT, 1).to(device), 33, LEARNING_RATE, 10, nn.BCEWithLogitsLoss(), True, 1)
]

# Evaluate model
l1_loss_fn = nn.L1Loss()
def evaluate_model(model_path):
    ae = SplitEmbeddingModel(input_dim=EMBEDDING_DIM, known_dim=CRAFTED_LATENT, hidden_dim=EMBEDDING_DIM)
    ae.load_state_dict(torch.load(model_path)['model_state_dict'])
    ae.eval()

    embeddings, labels_dict = [], {config.name: [] for config in configs}
    recon_losses = []

    with torch.no_grad():
        for batch in test_loader:
            inputs = batch["ESM2Embedding"].to(device).float()
            labels = {config.name: batch[config.name].to(device).float() for config in configs}

            _, _, reconstructed, z, _ = ae(inputs)
            recon_loss = l1_loss_fn(reconstructed, inputs)
            recon_losses.append(recon_loss.item())

            embeddings.append(z.cpu())
            for config in configs:
                labels_dict[config.name].append(labels[config.name].cpu())

        embeddings = torch.cat(embeddings, dim=0).numpy()
        labels_dict = {name: torch.cat(labels_dict[name], dim=0).numpy() for name in labels_dict}

    avg_recon_loss = np.mean(recon_losses)
    print(f"Model: {model_path}, Average Reconstruction Loss: {avg_recon_loss:.4f}")

    # Task evaluation
    for config in configs:
        task_emb = embeddings[:, config.latent_index:config.latent_index + config.latents]
        adv_emb = embeddings[:, CRAFTED_LATENT:]

        train_emb, test_emb, train_lbls, test_lbls = train_test_split(task_emb, labels_dict[config.name], test_size=0.2, random_state=0)
        adv_train_emb, adv_test_emb, adv_train_lbls, adv_test_lbls = train_test_split(adv_emb, labels_dict[config.name], test_size=0.2, random_state=0)

        if config.is_classification:
            mlp = MLPClassifier(hidden_layer_sizes=(16), max_iter=100, random_state=42)
            mlp.fit(train_emb, train_lbls.squeeze().astype(int))
            train_acc = accuracy_score(train_lbls, mlp.predict(train_emb))
            test_acc = accuracy_score(test_lbls, mlp.predict(test_emb))

            mlp_adv = MLPClassifier(hidden_layer_sizes=(128, 32, 16), max_iter=100, random_state=42)
            mlp_adv.fit(adv_train_emb, adv_train_lbls.squeeze().astype(int))
            adv_train_acc = accuracy_score(adv_train_lbls, mlp_adv.predict(adv_train_emb))
            adv_test_acc = accuracy_score(adv_test_lbls, mlp_adv.predict(adv_test_emb))
        else:
            mlp = MLPRegressor(hidden_layer_sizes=(16), max_iter=100, random_state=42)
            mlp.fit(train_emb, train_lbls)
            train_r2 = r2_score(train_lbls, mlp.predict(train_emb))
            test_r2 = r2_score(test_lbls, mlp.predict(test_emb))

            mlp_adv = MLPRegressor(hidden_layer_sizes=(128, 32, 16), max_iter=100, random_state=42)
            mlp_adv.fit(adv_train_emb, adv_train_lbls)
            adv_train_r2 = r2_score(adv_train_lbls, mlp_adv.predict(adv_train_emb))
            adv_test_r2 = r2_score(adv_test_lbls, mlp_adv.predict(adv_test_emb))

        # Store results in CSV
        with open(CSV_FILE, 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            if config.is_classification:
                csvwriter.writerow([
                    model_path, config.name, train_acc, test_acc, adv_train_acc, adv_test_acc, avg_recon_loss
                ])
            else:
                csvwriter.writerow([
                    model_path, config.name, train_r2, test_r2, adv_train_r2, adv_test_r2, avg_recon_loss
                ])

# Run evaluation for all models
for model in MODEL_WEIGHTS:
    evaluate_model(os.path.join('models', model))
