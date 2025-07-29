from PartitionedEmbeddingModel import PartitionedEmbeddingModel
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import pandas as pd
from tqdm import tqdm
import numpy as np

CRAFTED_FEATURES = 34

class ESM2VAEModel(nn.Module):
    def __init__(self, esm2_model_name, vae_weights_path):
        super(ESM2VAEModel, self).__init__()

        # Load ESM2
        self.tokenizer = AutoTokenizer.from_pretrained(esm2_model_name)
        self.esm2 = AutoModel.from_pretrained(esm2_model_name)

        # Freeze ESM2
        for param in self.esm2.parameters():
            param.requires_grad = False

        # Load pre-trained VAE
        input_dim = self.esm2.config.hidden_size
        self.ae = PartitionedEmbeddingModel(input_dim, CRAFTED_FEATURES, 1200)

        self.ae.load_state_dict(torch.load(vae_weights_path))
        self.ae.eval()

    def forward(self, sequence):
        tokenized = self.tokenizer(sequence, return_tensors="pt", padding=True, truncation=True)
        input_ids, attention_mask = tokenized["input_ids"], tokenized["attention_mask"]

        with torch.no_grad():
            esm2_outputs = self.esm2(input_ids=input_ids, attention_mask=attention_mask)
            embeddings = esm2_outputs.last_hidden_state

        embeddings = embeddings.squeeze(0)
        attention_mask = attention_mask.squeeze(0)
        valid_embeddings = embeddings[attention_mask == 1][1:-1]

        with torch.no_grad():
            _, _, _, z, _ = self.ae(valid_embeddings)

        # Binary one-hot encoding
        binary_encoded = z.clone()
        for i in range(valid_embeddings.size(0)):
            row = z[i]
            ss8_index = torch.argmax(row[23:31]).item()
            binary_encoded[i, 23:31] = 0
            binary_encoded[i, 23 + ss8_index] = 1

            ss3_index = torch.argmax(row[20:23]).item()
            binary_encoded[i, 20:23] = 0
            binary_encoded[i, 20 + ss3_index] = 1

            aa_index = torch.argmax(row[:20]).item()
            binary_encoded[i, :20] = 0
            binary_encoded[i, aa_index] = 1

        return (
            valid_embeddings,
            z,
            z[:, :CRAFTED_FEATURES],
            binary_encoded[:, :CRAFTED_FEATURES]
        )

def generate_and_pad_embeddings(emb, max_length=512, feature_dim=480):
    """
    Generates embeddings and ensures they are padded or truncated.

    Args:
        model (nn.Module): The model used to generate embeddings.
        sequences (list): List of sequences to embed.
        max_length (int): Maximum sequence length.
        feature_dim (int): Dimensionality of each embedding.

    Returns:
        numpy array: Padded or truncated embeddings.
    """
    if emb.shape[0] > max_length:
        return emb[:max_length]
    else:
        padding = np.zeros((max_length - emb.shape[0], feature_dim))
        return np.vstack((emb, padding))


def generate_and_save_embeddings(model, sequences, output_prefix):
    valids, latents, truncated_latents, binaries, valid_means, latent_means, truncated_latent_means, binary_means = [], [], [], [], [], [], [], []

    max_length = 256
    for seq in tqdm(sequences):
        valid, latent, truncated_latent, binary = model(seq)
        valid_means.append(valid.mean(dim=0).numpy())
        latent_means.append(latent.mean(dim=0).numpy())
        truncated_latent_means.append(truncated_latent.mean(dim=0).numpy())
        binary_means.append(binary.mean(dim=0).numpy())

        valid = generate_and_pad_embeddings(valid, max_length=max_length, feature_dim=embedding_dim)
        latent = generate_and_pad_embeddings(latent, max_length=max_length, feature_dim=embedding_dim)
        truncated_latent = generate_and_pad_embeddings(truncated_latent, max_length=max_length, feature_dim=CRAFTED_FEATURES)
        binary = generate_and_pad_embeddings(binary, max_length=max_length, feature_dim=CRAFTED_FEATURES)
        valids.append(valid)
        latents.append(latent)
        truncated_latents.append(truncated_latent)
        binaries.append(binary)

    pd.DataFrame(valid_means).to_csv(f"{output_prefix}_valid_embeddings.csv", index=False)
    pd.DataFrame(latent_means).to_csv(f"{output_prefix}_latent_embeddings.csv", index=False)
    pd.DataFrame(truncated_latent_means).to_csv(f"{output_prefix}_truncated_latent_embeddings.csv", index=False)
    pd.DataFrame(binary_means).to_csv(f"{output_prefix}_binary_embeddings.csv", index=False)
    np.save(f"{output_prefix}_valid_embeddings_full.npy", np.array(valids))
    np.save(f"{output_prefix}_latent_embeddings_full.npy", np.array(latents))
    np.save(f"{output_prefix}_truncated_latent_embeddings_full.npy", np.array(truncated_latents))
    np.save(f"{output_prefix}_binary_embeddings_full.npy", np.array(binaries))


if __name__ == "__main__":
    esm2_model_name = "facebook/esm2_t12_35M_UR50D"
    vae_weights_path = "trained_models/lambda_1.5/epoch_1.pt"
    embedding_dim = 480

    esm2_vae_model = ESM2VAEModel(
        esm2_model_name=esm2_model_name,
        vae_weights_path=vae_weights_path,
        embedding_dim=embedding_dim,
    )

    train_df = pd.read_csv("downstream_task_data/train_sequences_AGG.csv")
    test_df = pd.read_csv("downstream_task_data/test_sequences_AGG.csv")
    # leptin_df = pd.read_csv("data2/leptin_sequence.csv")

    train_sequences = train_df["Sequence"].tolist()
    test_sequences = test_df["Sequence"].tolist()
    # leptin_sequences = leptin_df["Sequence"].tolist()

    print("Generating train embeddings...")
    generate_and_save_embeddings(esm2_vae_model, train_sequences, "downstream_task_data/new_embedding/train_AGG")

    print("Generating test embeddings...")
    generate_and_save_embeddings(esm2_vae_model, test_sequences, "downstream_task_data/new_embedding/test_AGG")
    # generate_and_save_embeddings(esm2_vae_model, leptin_sequences, "data2/embeddings_avg/leptin")

