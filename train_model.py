import os
import torch
import numpy as np
import random
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import f1_score, r2_score

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from PartitionedEmbeddingModel import PartitionedEmbeddingModel
from ProteinDataset import ProteinDataset

# -----------------------------
# Utility Functions & Seed Setup
# -----------------------------
def map_sequence(sequence, mapping_dict, unknown_value=-1):
    return [mapping_dict.get(char, unknown_value) for char in sequence]

def compute_class_weights_from_csv(csv_path, column_name, num_classes, mapping_dict):
    df = pd.read_csv(csv_path)
    labels = df[column_name].astype(str)
    labels = labels.map(lambda x: mapping_dict.get(x, -1))
    labels = labels[labels != -1].astype(int).values
    counts = np.bincount(labels, minlength=num_classes)
    weights = 1.0 / (counts + 1e-6)
    weights = weights / weights.sum() * num_classes
    return torch.tensor(weights, dtype=torch.float)

# Set seed for reproducibility
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

amino_acid_dict = {
    'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7,
    'K': 8, 'L': 9, 'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14,
    'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19
}
ss3_dict = {'H': 0, 'E': 1, 'C': 2}
ss8_dict = {'H': 0, 'E': 1, 'G': 2, 'I': 3, 'B': 4, 'T': 5, 'S': 6, '-': 7}

train_csv = 'data/combined_filtered_protein_values_swissprot_25_all_train.csv'
amino_weights = compute_class_weights_from_csv(train_csv, 'AminoAcid', 20, amino_acid_dict).to(device)
ss3_weights   = compute_class_weights_from_csv(train_csv, 'SS3', 3, ss3_dict).to(device)
ss8_weights   = compute_class_weights_from_csv(train_csv, 'SS8', 8, ss8_dict).to(device)

print("Amino weights:", amino_weights)
print("SS3 weights:", ss3_weights)
print("SS8 weights:", ss8_weights)


train_dataset = ProteinDataset('data/combined_filtered_protein_values_swissprot_25_all_train.csv',
                               'data/combined_filtered_protein_embeddings_swissprot_25_all_train.npz')
test_dataset  = ProteinDataset('data/combined_filtered_protein_values_swissprot_25_all_test.csv',
                               'data/combined_filtered_protein_embeddings_swissprot_25_all_test.npz')

batch_size = 256
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

rec_criterion   = nn.L1Loss()
class_criterion = nn.CrossEntropyLoss(weight=amino_weights)
class_criterion = nn.CrossEntropyLoss()
ss3_criterion   = nn.CrossEntropyLoss(weight=ss3_weights)
ss3_criterion   = nn.CrossEntropyLoss()
ss8_criterion   = nn.CrossEntropyLoss(weight=ss8_weights)
ss8_criterion   = nn.CrossEntropyLoss()
asa_criterion   = nn.L1Loss()
gravy_criterion = nn.L1Loss()
arom_criterion  = nn.BCELoss()

for lambda_adv in [2.5]:
    lambda_amino = 1.0
    lambda_ss3   = 1.0
    lambda_ss8   = 1.0
    lambda_asa   = 1.0
    lambda_gravy = 1.0
    lambda_arom  = 1.0
    lambda_rec   = 100.0

    input_dim = test_dataset.embeddings.shape[1] if hasattr(test_dataset, 'embeddings') else 1024
    known_dim = 34  
    hidden_dim = 1200

    base_dir = "trained_models"
    os.makedirs(base_dir, exist_ok=True)
    lambda_dir = os.path.join(base_dir, f"lambda_{lambda_adv}")
    os.makedirs(lambda_dir, exist_ok=True)

    model = PartitionedEmbeddingModel(input_dim, known_dim, hidden_dim, adv_output_dim=34).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.00025)

    num_epochs = 1
    epoch_metrics = []
    current_iteration=0

    for epoch in range(num_epochs):
        model.train()
        train_total_loss = 0.0
        train_rec_loss_total = 0.0
        train_batches = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Train)"):
            current_iteration += 1
            batch = {key: value.to(device) for key, value in batch.items()}
            embeddings = batch['ESM2Embedding']
            amino_acid = batch['AminoAcid']
            ss3_labels = batch['SS3']
            ss8_labels = batch['SS8']
            asa = batch['ASA']
            gravy = batch['GRAVY']
            arom  = batch['AROM']
            
            optimizer.zero_grad()
            known_subspace, unknown_subspace, reconstructed, combined, adv_pred = model(embeddings, lambda_grl=1.0)
            
            rec_loss   = rec_criterion(reconstructed, embeddings)
            class_loss = class_criterion(known_subspace[:, :20], amino_acid)
            ss3_loss   = ss3_criterion(known_subspace[:, 20:23], ss3_labels)
            ss8_loss   = ss8_criterion(known_subspace[:, 23:31], ss8_labels) * lambda_ss8
            asa_loss   = asa_criterion(known_subspace[:, 31], asa.squeeze()) * lambda_asa
            gravy_loss = gravy_criterion(known_subspace[:, 32], gravy.squeeze()) * 3
            arom_prob  = (known_subspace[:, 33] + 1) / 2
            arom_loss  = arom_criterion(arom_prob, arom.float().squeeze())
            
            adv_loss_amino = class_criterion(adv_pred[:, :20], amino_acid)
            adv_loss_ss3   = ss3_criterion(adv_pred[:, 20:23], ss3_labels)
            adv_loss_ss8   = ss8_criterion(adv_pred[:, 23:31], ss8_labels)
            adv_loss_asa   = asa_criterion(adv_pred[:, 31], asa.squeeze())
            adv_loss_gravy = gravy_criterion(adv_pred[:, 32], gravy.squeeze())
            adv_loss_arom  = arom_criterion(torch.sigmoid(adv_pred[:, 33]), arom.float().squeeze())
            
            adv_loss_total = (lambda_amino * adv_loss_amino +
                            lambda_ss3   * adv_loss_ss3 +
                            lambda_ss8   * adv_loss_ss8 +
                            lambda_asa   * adv_loss_asa +
                            lambda_gravy * adv_loss_gravy +
                            lambda_arom  * adv_loss_arom)

            rampup_iterations = 30000

            current_lambda_adv = min(lambda_adv * current_iteration / rampup_iterations, lambda_adv)
            
            total_loss = (lambda_rec * rec_loss +
                        class_loss + ss3_loss + ss8_loss +
                        asa_loss + gravy_loss + arom_loss +
                        lambda_adv * adv_loss_total)
            
            total_loss.backward()
            optimizer.step()
            
            train_total_loss    += total_loss.item()
            train_rec_loss_total += rec_loss.item()
            train_batches += 1
        
        avg_train_loss = train_total_loss / train_batches
        avg_train_rec_loss = train_rec_loss_total / train_batches
        
        model.eval()
        all_amino_preds, all_amino_labels = [], []
        all_ss3_preds, all_ss3_labels = [], []
        all_ss8_preds, all_ss8_labels = [], []
        all_arom_preds, all_arom_labels = [], []
        all_asa_preds, all_asa_labels = [], []
        all_gravy_preds, all_gravy_labels = [], []
        
        test_rec_loss_total = 0.0
        test_batches = 0
        
        with torch.no_grad():
            for batch in test_loader:
                batch = {key: value.to(device) for key, value in batch.items()}
                embeddings = batch['ESM2Embedding']
                # For evaluation, use real labels appropriately
                amino_acid = torch.argmax(batch['AminoAcid'], dim=1) if batch['AminoAcid'].dim() > 0 else batch['AminoAcid']
                ss3_labels   = torch.argmax(batch['SS3'], dim=1) if batch['SS3'].dim() > 0 else batch['SS3']
                ss8_labels   = torch.argmax(batch['SS8'], dim=1) if batch['SS8'].dim() > 0 else batch['SS8']
                asa   = batch['ASA'].squeeze()
                gravy = batch['GRAVY'].squeeze()
                arom  = batch['AROM']
                
                known_subspace, unknown_subspace, reconstructed, combined, _ = model(embeddings, lambda_grl=1.0)
                
                rec_loss_batch = rec_criterion(reconstructed, embeddings)
                test_rec_loss_total += rec_loss_batch.item()
                test_batches += 1
                
                pred_amino = torch.argmax(known_subspace[:, :20], dim=1)
                pred_ss3   = torch.argmax(known_subspace[:, 20:23], dim=1)
                pred_ss8   = torch.argmax(known_subspace[:, 23:31], dim=1)
                pred_asa   = known_subspace[:, 31]
                pred_gravy = known_subspace[:, 32]
                pred_arom  = (((known_subspace[:, 33] + 1) / 2) > 0.5).long()
                
                all_amino_preds.append(pred_amino.cpu().numpy())
                all_amino_labels.append(amino_acid.cpu().numpy())
                all_ss3_preds.append(pred_ss3.cpu().numpy())
                all_ss3_labels.append(ss3_labels.cpu().numpy())
                all_ss8_preds.append(pred_ss8.cpu().numpy())
                all_ss8_labels.append(ss8_labels.cpu().numpy())
                all_arom_preds.append(pred_arom.cpu().numpy())
                all_arom_labels.append(arom.cpu().numpy())
                all_asa_preds.append(pred_asa.cpu().numpy())
                all_asa_labels.append(asa.cpu().numpy())
                all_gravy_preds.append(pred_gravy.cpu().numpy())
                all_gravy_labels.append(gravy.cpu().numpy())
        
        avg_test_rec_loss = test_rec_loss_total / test_batches
        
        all_amino_preds = np.concatenate(all_amino_preds)
        all_amino_labels = np.concatenate(all_amino_labels)
        all_ss3_preds = np.concatenate(all_ss3_preds)
        all_ss3_labels = np.concatenate(all_ss3_labels)
        all_ss8_preds = np.concatenate(all_ss8_preds)
        all_ss8_labels = np.concatenate(all_ss8_labels)
        all_arom_preds = np.concatenate(all_arom_preds)
        all_arom_labels = np.concatenate(all_arom_labels)
        all_asa_preds = np.concatenate(all_asa_preds)
        all_asa_labels = np.concatenate(all_asa_labels)
        all_gravy_preds = np.concatenate(all_gravy_preds)
        all_gravy_labels = np.concatenate(all_gravy_labels)
        
        f1_amino = f1_score(all_amino_labels, all_amino_preds, average="weighted")
        f1_ss3   = f1_score(all_ss3_labels, all_ss3_preds, average="weighted")
        f1_ss8   = f1_score(all_ss8_labels, all_ss8_preds, average="weighted")
        f1_arom  = f1_score(all_arom_labels, all_arom_preds, average="weighted")
        r2_asa   = r2_score(all_asa_labels, all_asa_preds)
        r2_gravy = r2_score(all_gravy_labels, all_gravy_preds)
        
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"  Train Loss: {avg_train_loss:.6f} (rec: {avg_train_rec_loss:.6f})")
        print(f"  Test F1 Amino: {f1_amino:.6f}, F1 SS3: {f1_ss3:.6f}, F1 SS8: {f1_ss8:.6f}, F1 AROM: {f1_arom:.6f}")
        print(f"  R2 ASA: {r2_asa:.6f}, R2 GRAVY: {r2_gravy:.6f}")
        print(f"  Test Rec Loss: {avg_test_rec_loss:.6f}")
        
        epoch_data = {
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "train_rec_loss": avg_train_rec_loss,
            "test_rec_loss": avg_test_rec_loss,
            "f1_amino": f1_amino,
            "f1_ss3": f1_ss3,
            "f1_ss8": f1_ss8,
            "f1_arom": f1_arom,
            "r2_asa": r2_asa,
            "r2_gravy": r2_gravy
        }
        epoch_metrics.append(epoch_data)
        
        epoch_df = pd.DataFrame([epoch_data])
        epoch_csv_path = os.path.join(lambda_dir, f"epoch_{epoch+1}_metrics.csv")
        epoch_df.to_csv(epoch_csv_path, index=False)
        print(f"Epoch {epoch+1} metrics saved to {epoch_csv_path}")
        
        checkpoint_path = os.path.join(lambda_dir, f"epoch_{epoch+1}.pt")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

    metrics_df = pd.DataFrame(epoch_metrics)
    csv_path = os.path.join(lambda_dir, "metrics.csv")
    metrics_df.to_csv(csv_path, index=False)
    print(f"Aggregated metrics saved to {csv_path}")
