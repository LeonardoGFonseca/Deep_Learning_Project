import numpy as np
import h5py
import torch
from torch.utils.data import Dataset, DataLoader


_CACHE = {}

def load_data_cache(h5_path):
    if h5_path not in _CACHE:
        import time
        print(f"[CACHE] Loading {h5_path} into memory cache...", flush=True)
        t0 = time.time()
        with h5py.File(h5_path, 'r') as f:
            x = f['X'][:]
            y = f['Y'][:]
            strains = [s.decode('utf-8') if isinstance(s, bytes) else s for s in f['strain_id'][:]]
            genes = [g.decode('utf-8') if isinstance(g, bytes) else g for g in f['gene_target'][:]]
        _CACHE[h5_path] = {'x': x, 'y': y, 'strains': strains, 'genes': genes}
        print(f"[CACHE] Loaded in {time.time() - t0:.2f} seconds. Shape: {x.shape}, Memory: {x.nbytes / 1e9:.2f} GB", flush=True)
    return _CACHE[h5_path]


class NanoSquiggleDataset(Dataset):
    """HDF5 feature store reader with LOSO strain filtering + multiclass support."""
    def __init__(self, h5_path, exclude_strain=None, include_only=None, task='binary'):
        cache = load_data_cache(h5_path)
        self.x = cache['x']
        self.y = cache['y']
        self.strains = cache['strains']
        self.genes = cache['genes']
        self.task = task

        self.gene_to_class = {
            "ENA|HEE1644226|HEE1644226.1": 0,
            "ENA|MH733892|MH733892.1": 1,
            "ENA|MZ092836|MZ092836.1": 2,
        }

        # Filter indices based on task
        if self.task == 'multiclass':
            all_indices = [
                i for i in range(self.x.shape[0])
                if self.genes[i] in self.gene_to_class
            ]
        else:
            all_indices = list(range(self.x.shape[0]))

        # Filter indices based on strain
        if exclude_strain is not None:
            all_indices = [
                i for i in all_indices
                if self.strains[i] != exclude_strain
            ]
        elif include_only is not None:
            all_indices = [
                i for i in all_indices
                if self.strains[i] == include_only
            ]

        self.indices = all_indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        signal = torch.from_numpy(self.x[real_idx])
        
        if self.task == 'binary':
            label = torch.tensor(self.y[real_idx], dtype=torch.float32)
        elif self.task == 'multiclass':
            gene_name = self.genes[real_idx]
            class_idx = self.gene_to_class[gene_name]
            label = torch.tensor(class_idx, dtype=torch.long)
            
        return signal, label

    def close(self):
        pass


def balanced_undersample_indices(dataset, seed=None):
    """Return balanced (50/50) subset of indices from dataset."""
    if seed is not None:
        np.random.seed(seed)

    y_subset = [dataset.y[real_idx] for real_idx in dataset.indices]
    pos_idx = [i for i, val in enumerate(y_subset) if val == 1]
    neg_idx = [i for i, val in enumerate(y_subset) if val == 0]

    n_pos = len(pos_idx)
    if n_pos == 0:
        return list(range(len(dataset)))

    neg_sampled = np.random.choice(neg_idx, size=n_pos, replace=False).tolist()
    return pos_idx + neg_sampled


def make_loso_loaders(h5_path, test_strain, batch_size=32, seed=42, balanced=True, task='binary'):
    """Create train + test DataLoaders for one LOSO fold."""
    train_ds = NanoSquiggleDataset(h5_path, exclude_strain=test_strain, task=task)
    if balanced and task == 'binary':
        train_idx = balanced_undersample_indices(train_ds, seed=seed)
        train_sampler = torch.utils.data.Subset(train_ds, train_idx)
        train_loader = DataLoader(train_sampler, batch_size=batch_size, shuffle=True)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    test_ds = NanoSquiggleDataset(h5_path, include_only=test_strain, task=task)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
