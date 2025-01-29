import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoModel, AutoTokenizer
import os
from tqdm import tqdm

class ProteinDataset(Dataset):
    def __init__(self, csv_paths, esm_model_name="facebook/esm2_t12_35M_UR50D"):
        """
        Initialize the dataset by reading multiple CSV files, preprocessing the data,
        and setting up the ESM2 model for embedding generation on the MPS device.
        """
        self.amino_acid_dict = {
            'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7,
            'K': 8, 'L': 9, 'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14,
            'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19
        }
        self.ss3_dict = {'H': 0, 'E': 1, 'C': 2}
        self.ss8_dict = {'H': 0, 'E': 1, 'G': 2, 'I': 3, 'B': 4, 'T': 5, 'S': 6, '-': 7}

        # Choose MPS device if available, otherwise fallback to CPU
        self.device = torch.device("cpu" if torch.backends.mps.is_available() else "cpu")

        self.data = []

        self.tokenizer = AutoTokenizer.from_pretrained(esm_model_name)
        self.model = AutoModel.from_pretrained(esm_model_name).to(self.device)
        self.model.eval()

        for path in tqdm(csv_paths):
            protein_name = os.path.basename(path).split('.')[0] 
            df = pd.read_csv(path)
            sequence = "".join(df["AminoAcid"])

            # Generate embeddings using the model on MPS
            esm2_embedding = self._generate_esm2_embedding(sequence)

            # Move embeddings to CPU and store as numpy arrays for DataFrame compatibility
            df["ESM2Embedding"] = list(esm2_embedding.cpu().numpy())

            df['AminoAcidEncoded'] = df['AminoAcid'].map(
                lambda x: self._one_hot_encode(x, self.amino_acid_dict, len(self.amino_acid_dict))
            )
            df['SS3Encoded'] = df['SS3'].map(
                lambda x: self._one_hot_encode(x, self.ss3_dict, len(self.ss3_dict))
            )
            df['SS8Encoded'] = df['SS8'].map(
                lambda x: self._one_hot_encode(x, self.ss8_dict, len(self.ss8_dict))
            )

            df['ProteinName'] = protein_name
            self.data.append(df)

        self.data = pd.concat(self.data, ignore_index=True)
        self.data.dropna(inplace=True)

    def _one_hot_encode(self, value, encoding_dict, vector_size):
        """
        One-hot encode a value using a predefined encoding dictionary.
        """
        vector = np.zeros(vector_size, dtype=np.float32)
        if value in encoding_dict:
            vector[encoding_dict[value]] = 1.0
        return vector

    def _generate_esm2_embedding(self, sequence):
        """
        Generate ESM2 embeddings for a given amino acid sequence on the specified device.
        """
        # Tokenize and move inputs to the MPS device
        inputs = self.tokenizer(sequence, return_tensors="pt", add_special_tokens=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Extract embeddings, move them back to CPU for numpy conversion if needed
        embeddings = outputs.last_hidden_state[0, 1:-1, :]
        return embeddings

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """Retrieve a single sample as a dictionary of features on the specified device."""
        row = self.data.iloc[idx]

        # Create tensors directly on the specified device
        esm2_embedding = torch.tensor(row['ESM2Embedding'], dtype=torch.float32, device=self.device)

        features = {
            'AminoAcid': torch.tensor(row['AminoAcidEncoded'], dtype=torch.float32, device=self.device),
            'SS3': torch.tensor(row['SS3Encoded'], dtype=torch.float32, device=self.device),
            'SS8': torch.tensor(row['SS8Encoded'], dtype=torch.float32, device=self.device),
            'AccessibleSurfaceArea': torch.tensor(row['AccessibleSurfaceArea'], dtype=torch.float32, device=self.device),
            'Phi': torch.tensor(row['Phi'], dtype=torch.float32, device=self.device),
            'Psi': torch.tensor(row['Psi'], dtype=torch.float32, device=self.device),
            'GRAVY': torch.tensor(row['GRAVY'], dtype=torch.float32, device=self.device),
            'Aromaticity': torch.tensor(row['Aromaticity'], dtype=torch.float32, device=self.device),
            'ESM2Embedding': esm2_embedding
        }

        return features
