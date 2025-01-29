import os
import random
import pickle
from dataset import ProteinDataset

random.seed(1)

path = '/Users/Eck00018/Documents/PhD/feature_app/output/UP000005640_9606_HUMAN_v4/residue'
csv_paths = [os.path.join(path, x) for x in os.listdir(path) if x.endswith('.csv')]

# Shuffle paths for random splitting
random.shuffle(csv_paths)

# Calculate split sizes
total_size = len(csv_paths)
train_size = int(0.2 * total_size)
val_size = int(0.05 * total_size)
test_size = int(0.05 * total_size)

# Split paths
train_paths = csv_paths[:train_size]
val_paths = csv_paths[train_size:train_size + val_size]
test_paths = csv_paths[train_size + val_size:train_size + val_size + test_size]

# Create separate datasets
train_dataset = ProteinDataset(train_paths)

# Save datasets
output_dir = 'generated_datasets'
os.makedirs(output_dir, exist_ok=True)

with open(os.path.join(output_dir, 'train_dataset.pkl'), 'wb') as f:
    pickle.dump(train_dataset, f)

val_dataset = ProteinDataset(val_paths)
with open(os.path.join(output_dir, 'val_dataset.pkl'), 'wb') as f:
    pickle.dump(val_dataset, f)

test_dataset = ProteinDataset(test_paths)
with open(os.path.join(output_dir, 'test_dataset.pkl'), 'wb') as f:
    pickle.dump(test_dataset, f)

print(f"Dataset splits saved to {output_dir}")
