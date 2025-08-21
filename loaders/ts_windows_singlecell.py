import xarray as xr
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import os
import json


class SingleCellWindowDataset(Dataset):
    """
    Dataset for extracting sliding windows of time series
    from a single (xh,yh) grid cell in an xarray dataset.

    Parameters
    ----------
    ds : xr.Dataset
        Xarray dataset containing variables.
    input_vars : list of str
        List of variable names to use as input features.
    target_vars : list of str
        List of variable names to predict.
    window_size : int
        Number of timesteps per input sequence.
    stride : int
        Step size between windows.
    cell_idx : tuple(int,int)
        Indices (yh, xh) of the grid cell to extract.
    transform : callable, optional
        Transform applied to each (x, y) sample.
    """

    def __init__(self, ds, input_vars, target_vars,
                 window_size=12, stride=1,
                 cell_idx=(0, 0), transform=None):
        self.ds = ds
        self.input_vars = input_vars
        self.target_vars = target_vars
        self.window_size = window_size
        self.stride = stride
        self.cell_idx = cell_idx
        self.transform = transform

        # Extract time length
        self.time_len = ds.dims["time"]

        # Precompute windows
        self.indices = []
        for start in range(0, self.time_len - window_size, stride):
            end = start + window_size
            self.indices.append((start, end))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        start, end = self.indices[idx]
        yh, xh = self.cell_idx

        # Extract input and target slices
        x = []
        for var in self.input_vars:
            arr = self.ds[var].isel(yh=yh, xh=xh).values[start:end]
            x.append(arr)
        x = np.stack(x, axis=0)  # shape: [channels, window_size]

        y = []
        for var in self.target_vars:
            arr = self.ds[var].isel(yh=yh, xh=xh).values[start:end]
            y.append(arr)
        y = np.stack(y, axis=0)

        sample = (torch.tensor(x, dtype=torch.float32),
                  torch.tensor(y, dtype=torch.float32))

        if self.transform:
            sample = self.transform(sample)
        return sample


def resolve_dataset_path(paths_config, exp_cfg):
    """
    Resolve dataset path from experiment config and paths_config.json.
    """
    # Case 1: absolute path directly given
    if "dataset_path" in exp_cfg.get("dataloader", {}):
        return exp_cfg["dataloader"]["dataset_path"]

    # Case 2: reference to key in paths_config.json
    if "dataset_path_key" in exp_cfg.get("dataloader", {}):
        key = exp_cfg["dataloader"]["dataset_path_key"]
        if key in paths_config:
            return paths_config[key]
        else:
            raise KeyError(f"dataset_path_key '{key}' not found in paths_config.json")

    # Case 3: training path inside dataset block
    if "dataset" in exp_cfg and "train_path" in exp_cfg["dataset"]:
        return exp_cfg["dataset"]["train_path"]

    # If nothing works, raise error
    raise KeyError(
        "Could not resolve dataset path. Provide either:\n"
        "  • dataloader.dataset_path (absolute), or\n"
        "  • dataloader.dataset_path_key (must exist in paths_config.json), or\n"
        "  • dataset.train_path in the experiment config."
    )


def make_dataloaders(paths_config_path, exp_cfg):
    """
    Build PyTorch DataLoaders for training and validation.

    Parameters
    ----------
    paths_config_path : str
        Path to paths_config.json file.
    exp_cfg : dict
        Experiment configuration dictionary.

    Returns
    -------
    train_loader, val_loader, stats
    """

    # Load paths config
    with open(paths_config_path, "r") as f:
        paths_config = json.load(f)

    # Resolve dataset path
    ds_path = resolve_dataset_path(paths_config, exp_cfg)

    if not os.path.exists(ds_path):
        raise FileNotFoundError(f"Dataset file not found: {ds_path}")

    # Load dataset lazily
    ds = xr.open_dataset(ds_path, chunks={"time": 128})

    input_vars = exp_cfg["dataloader"]["input_vars"]
    target_vars = exp_cfg["dataloader"]["target_vars"]
    window_size = exp_cfg["dataloader"].get("window_size", 12)
    stride = exp_cfg["dataloader"].get("stride", 1)
    cell_idx = tuple(exp_cfg["dataloader"].get("cell_idx", (0, 0)))

    full_dataset = SingleCellWindowDataset(
        ds, input_vars, target_vars,
        window_size=window_size, stride=stride,
        cell_idx=cell_idx
    )

    # Train/val split
    val_split = exp_cfg["dataloader"].get("val_split", 0.2)
    n_val = int(len(full_dataset) * val_split)
    n_train = len(full_dataset) - n_val
    train_ds, val_ds = random_split(full_dataset, [n_train, n_val])

    # Build dataloaders
    num_workers = exp_cfg["dataloader"].get("num_workers", 2)
    batch_size = exp_cfg["dataloader"].get("batch_size", 16)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers)

    stats = {"n_train": n_train, "n_val": n_val}
    return train_loader, val_loader, stats
