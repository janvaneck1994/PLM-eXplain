import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

class ProteinDataset(Dataset):
    def __init__(self, csv_path, npz_path):
        """
        Initialize the dataset by reading a CSV file with features and an NPZ file with embeddings.
        """
        self.device = torch.device("cpu" if torch.backends.mps.is_available() else "cpu")

        self.amino_acid_dict = {
            'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7,
            'K': 8, 'L': 9, 'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14,
            'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19
        }
        self.ss3_dict = {'H': 0, 'E': 1, 'C': 2}
        self.ss8_dict = {'H': 0, 'E': 1, 'G': 2, 'I': 3, 'B': 4, 'T': 5, 'S': 6, '-': 7}

        self.data = pd.read_csv(csv_path)
        
        self.embeddings = np.load(npz_path)["embeddings"]
        
        assert len(self.data) == self.embeddings.shape[0], "Mismatch between CSV rows and NPZ embeddings"

        valid_indices = self.data.dropna().index.to_numpy()

        self.data = self.data.loc[valid_indices].reset_index(drop=True)
        self.embeddings = self.embeddings[valid_indices]

        self.valid_indices = np.arange(len(self.data))

        self.data['AminoAcidEncoded'] = self.data['AminoAcid'].map(
            lambda x: self._one_hot_encode(x, self.amino_acid_dict, len(self.amino_acid_dict)) if x in self.amino_acid_dict else np.zeros(len(self.amino_acid_dict))
        )
        self.data['SS3Encoded'] = self.data['SS3'].map(
            lambda x: self._one_hot_encode(x, self.ss3_dict, len(self.ss3_dict)) if x in self.ss3_dict else np.zeros(len(self.ss3_dict))
        )
        self.data['SS8Encoded'] = self.data['SS8'].map(
            lambda x: self._one_hot_encode(x, self.ss8_dict, len(self.ss8_dict)) if x in self.ss8_dict else np.zeros(len(self.ss8_dict))
        )
    
    def _one_hot_encode(self, value, encoding_dict, vector_size):
        """
        One-hot encode a value using a predefined encoding dictionary.
        """
        vector = np.zeros(vector_size, dtype=np.float32)
        if value in encoding_dict:
            vector[encoding_dict[value]] = 1.0
        return vector
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """Retrieve a single sample as a dictionary of features."""
        idx = int(idx)

        row = self.data.iloc[idx]
        esm2_embedding = torch.tensor(self.embeddings[idx], dtype=torch.float32, device=self.device)
        
        features = {
            'AminoAcid': torch.tensor(row['AminoAcidEncoded'], dtype=torch.float32, device=self.device),
            'SS3': torch.tensor(row['SS3Encoded'], dtype=torch.float32, device=self.device),
            'SS8': torch.tensor(row['SS8Encoded'], dtype=torch.float32, device=self.device),
            'ASA': torch.tensor([row['AccessibleSurfaceArea']], dtype=torch.float32, device=self.device),
            'Phi': torch.tensor([row['Phi']], dtype=torch.float32, device=self.device),
            'Psi': torch.tensor([row['Psi']], dtype=torch.float32, device=self.device),
            'GRAVY': torch.tensor([row['GRAVY']], dtype=torch.float32, device=self.device),
            'AROM': torch.tensor([row['Aromaticity']], dtype=torch.float32, device=self.device),
            'ESM2Embedding': esm2_embedding
        }
        
        return features
