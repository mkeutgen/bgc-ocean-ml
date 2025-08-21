# Goal

Train a tiny **SeqCNN** baseline that learns the **seasonal cycle** of a surface tracer (default **CT**) from the lightest dataset tier **`nano_surf_one_year`**.

* **Inputs** (surface, monthly): `SST, SSS, Qnet, taux, tauy` + time encoding (sin/cos of month).
* **Target**: `CT` at the surface (`time, yh, xh`).
* **Split**: spatial split (train/val by grid cells) to avoid trivial temporal leakage with only one year.
* **I/O**: open lazily with xarray/dask using schema chunk hints; load minibatches on the fly.

---

## 1) `config/paths_config.json`

```json
{
  "data_root": "./data",                            
  "nano_surf_one_year": "./data/ds_nano_surf_one_year.nc",
  "results_root": "./results",
  "num_workers": 2
}
```

> For HPC, point `nano_surf_one_year` to the Tiger3 processed subset path. Keep a local symlink for dev.

---

## 2) `config/experiments/baseline_seqcnn.yaml`

```yaml
# Minimal experiment config
seed: 42

# data
dataset_path_key: nano_surf_one_year
inputs: [SST, SSS, Qnet, taux, tauy]
target: CT
normalize:
  method: per-feature-zscore          # or "minmax"
  stats: null                         # if null, computed on-the-fly from train split
split:
  type: spatial_holdout               # spatial 80/20 split by (yh, xh) tiles
  val_fraction: 0.2
  tile_size: 8                        # 8x8 tiles to reduce spatial leakage
loader:
  batch_size: 8                       # batches of spatial tiles
  shuffle: true
  num_workers: 2
  prefetch_factor: 2

# model
model:
  name: SeasonalSeqCNN
  in_channels: 7                       # 5 forcings + 2 time encodings
  hidden_channels: 32
  num_layers: 3
  kernel_size: 3
  dropout: 0.1

# optimization
optim:
  lr: 1e-3
  weight_decay: 1e-5
  epochs: 30
  grad_clip: 1.0
  device: cuda                        # or "cpu"

# logging
log_every: 50
save_every: 1
```

---

## 3) `loaders/surface_seasonal_dataset.py`

```python
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import xarray as xr

# ---------- helpers ----------

def load_paths(paths_config_path: str) -> Dict:
    with open(paths_config_path, 'r') as f:
        return json.load(f)

def month_sin_cos(month_indices: np.ndarray) -> np.ndarray:
    # month indices expected 0..11
    theta = 2 * np.pi * (month_indices / 12.0)
    return np.stack([np.sin(theta), np.cos(theta)], axis=0)  # [2, T]


def compute_feature_stats(ds: xr.Dataset, varnames: List[str]) -> Dict[str, Tuple[float, float]]:
    stats = {}
    for v in varnames:
        # robust stats with dask-aware operations
        mu = float(ds[v].mean().compute())
        sigma = float(ds[v].std().compute() + 1e-8)
        stats[v] = (mu, sigma)
    return stats


def apply_norm(arr: xr.DataArray, mu: float, sigma: float) -> xr.DataArray:
    return (arr - mu) / sigma


# ---------- Dataset ----------

class SurfaceSeasonalTiles(Dataset):
    """
    Produces (X, y) for tiles of shape [T, Ht, Wt].
    X channels = [SST, SSS, Qnet, taux, tauy, sin(month), cos(month)]
    y = CT
    """
    def __init__(self,
                 ds_path: str,
                 inputs: List[str],
                 target: str,
                 tile_size: int = 8,
                 split: str = 'train',
                 val_fraction: float = 0.2,
                 normalize: Dict = None,
                 suggested_chunks: Dict = None):
        assert split in {'train', 'val'}
        self.inputs = inputs
        self.target = target
        self.tile = tile_size

        # open lazily with chunks; fall back to schema hints if given
        chunks = suggested_chunks or {"time": 128, "yh": 9, "xh": 9}
        ds = xr.open_dataset(ds_path, chunks=chunks)

        # ensure required vars exist
        for v in inputs + [target]:
            if v not in ds:
                raise ValueError(f"Missing variable {v} in {ds_path}")

        # derive month encodings
        time_index = ds["time"].dt.month - 1  # 0..11
        months = time_index.values
        t_enc = month_sin_cos(months)  # [2, T]

        # compute/collect normalization stats
        if normalize is None:
            normalize = {"method": "per-feature-zscore", "stats": None}
        if normalize.get("stats") is None:
            stats = compute_feature_stats(ds, inputs + [target])
        else:
            stats = normalize["stats"]

        # normalize features lazily
        feats = []
        for v in inputs:
            mu, sigma = stats[v]
            feats.append(apply_norm(ds[v], mu, sigma))  # [T, Y, X]
        # stack features -> [C_f, T, Y, X]
        X = xr.concat(feats, dim="channel").transpose("channel", "time", "yh", "xh")
        # add time encodings
        t_sin = xr.DataArray(np.broadcast_to(t_enc[0, :, None, None], X.shape[1:]), dims=("time", "yh", "xh"))
        t_cos = xr.DataArray(np.broadcast_to(t_enc[1, :, None, None], X.shape[1:]), dims=("time", "yh", "xh"))
        X = xr.concat([X, t_sin.expand_dims({"channel": ["sin_m"]}), t_cos.expand_dims({"channel": ["cos_m"]})], dim="channel")

        # normalize target
        y_mu, y_sigma = stats[target]
        y = apply_norm(ds[target], y_mu, y_sigma).transpose("time", "yh", "xh")

        self.X = X
        self.y = y
        self.y_mu = y_mu
        self.y_sigma = y_sigma

        # build spatial tiles list and split
        H = int(ds.dims["yh"]) if "yh" in ds.dims else int(ds.sizes["yh"])  # robust
        W = int(ds.dims["xh"]) if "xh" in ds.dims else int(ds.sizes["xh"])  
        tiles = []
        for y0 in range(0, H, tile_size):
            for x0 in range(0, W, tile_size):
                y1, x1 = min(y0 + tile_size, H), min(x0 + tile_size, W)
                tiles.append((slice(y0, y1), slice(x0, x1)))
        rng = np.random.default_rng(42)
        rng.shuffle(tiles)
        n_val = int(len(tiles) * val_fraction)
        self.tiles = tiles[:n_val] if split == 'val' else tiles[n_val:]

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        ys, xs = self.tiles[idx]
        # load small tile into memory
        X_tile = self.X.isel(yh=ys, xh=xs).data  # dask->numpy when indexed
        y_tile = self.y.isel(yh=ys, xh=xs).data
        # torch expects [B, C, T, H, W] later; return without batch
        return (
            torch.as_tensor(np.asarray(X_tile), dtype=torch.float32),   # [C, T, Ht, Wt]
            torch.as_tensor(np.asarray(y_tile), dtype=torch.float32)    # [T, Ht, Wt]
        )


def make_dataloaders(paths_config: str,
                      exp_cfg: Dict) -> Tuple[DataLoader, DataLoader, Dict]:
    paths = load_paths(paths_config)
    ds_path = paths[exp_cfg["dataset_path_key"]]

    # suggested chunks for nano_surf_one_year (from schema):
    suggested_chunks = {"time": 128, "yh": 9, "xh": 9}

    train_set = SurfaceSeasonalTiles(ds_path,
                                     inputs=exp_cfg["inputs"],
                                     target=exp_cfg["target"],
                                     tile_size=exp_cfg["split"]["tile_size"],
                                     split='train',
                                     val_fraction=exp_cfg["split"]["val_fraction"],
                                     normalize=exp_cfg.get("normalize"),
                                     suggested_chunks=suggested_chunks)

    val_set = SurfaceSeasonalTiles(ds_path,
                                   inputs=exp_cfg["inputs"],
                                   target=exp_cfg["target"],
                                   tile_size=exp_cfg["split"]["tile_size"],
                                   split='val',
                                   val_fraction=exp_cfg["split"]["val_fraction"],
                                   normalize=exp_cfg.get("normalize"),
                                   suggested_chunks=suggested_chunks)

    dl_kwargs = dict(batch_size=exp_cfg["loader"]["batch_size"],
                     shuffle=exp_cfg["loader"]["shuffle"],
                     num_workers=exp_cfg["loader"].get("num_workers", 0),
                     pin_memory=True,
                     prefetch_factor=exp_cfg["loader"].get("prefetch_factor", 2))

    return (DataLoader(train_set, **dl_kwargs),
            DataLoader(val_set,  **{**dl_kwargs, "shuffle": False}),
            {"y_mu": train_set.y_mu, "y_sigma": train_set.y_sigma})
```

---

## 4) `models/seqcnn.py`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalBlock(nn.Module):
    def __init__(self, c_in, c_out, k=3, p=1, d=1, dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv1d(c_in, c_out, kernel_size=k, padding=p, dilation=d)
        self.bn1 = nn.BatchNorm1d(c_out)
        self.conv2 = nn.Conv1d(c_out, c_out, kernel_size=k, padding=p, dilation=d)
        self.bn2 = nn.BatchNorm1d(c_out)
        self.dropout = nn.Dropout(dropout)
        self.res = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x):  # x: [B, C, T]
        y = self.dropout(F.gelu(self.bn1(self.conv1(x))))
        y = self.dropout(F.gelu(self.bn2(self.conv2(y))))
        return y + self.res(x)

class SeasonalSeqCNN(nn.Module):
    """
    Per-pixel temporal CNN that maps T-step input features to T-step target.
    Expects input shaped [B, C_in, T, H, W]. Runs 1D Conv over time per pixel.
    """
    def __init__(self, in_channels=7, hidden=32, num_layers=3, kernel_size=3, dropout=0.1):
        super().__init__()
        self.spatial_pool = nn.Identity()  # future: add shallow 2D conv if needed
        layers = []
        c_in = in_channels
        dilation = 1
        for _ in range(num_layers):
            layers.append(TemporalBlock(c_in, hidden, k=kernel_size, p=kernel_size//2, d=dilation, dropout=dropout))
            c_in = hidden
            dilation *= 2  # exponentially growing RF over months
        self.tcn = nn.Sequential(*layers)
        self.head = nn.Conv1d(hidden, 1, kernel_size=1)  # predict target per time step

    def forward(self, x):
        # x: [B, C_in, T, H, W] -> merge spatial dims, run Conv1d over T, then un-merge
        B, C, T, H, W = x.shape
        x = x.permute(0, 3, 4, 1, 2).contiguous().view(B*H*W, C, T)
        y = self.tcn(x)                     # [B*H*W, hidden, T]
        y = self.head(y)                    # [B*H*W, 1, T]
        y = y.view(B, H, W, T).permute(0, 3, 1, 2).contiguous()  # [B, T, H, W]
        return y
```

---

## 5) `metrics/skill_scores.py`

```python
import torch

def rmse(yhat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean((yhat - y) ** 2))

@torch.no_grad()
def pearson_corr(yhat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # compute correlation over (B,T,H,W)
    yhat_c = yhat - yhat.mean()
    y_c = y - y.mean()
    denom = torch.sqrt((yhat_c ** 2).mean() * (y_c ** 2).mean()) + 1e-8
    return (yhat_c * y_c).mean() / denom
```

---

## 6) `scripts/train.py`

```python
import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import yaml

from loaders.surface_seasonal_dataset import make_dataloaders
from models.seqcnn import SeasonalSeqCNN
from metrics.skill_scores import rmse, pearson_corr


def load_cfg(exp_yaml: str):
    with open(exp_yaml, 'r') as f:
        return yaml.safe_load(f)


def main(args):
    exp = load_cfg(args.experiment)
    torch.manual_seed(exp.get('seed', 42))

    train_dl, val_dl, norm = make_dataloaders(args.paths_config, exp)

    model = SeasonalSeqCNN(in_channels=exp['model']['in_channels'],
                           hidden=exp['model']['hidden_channels'],
                           num_layers=exp['model']['num_layers'],
                           kernel_size=exp['model']['kernel_size'],
                           dropout=exp['model']['dropout'])
    device = torch.device(exp['optim'].get('device', 'cpu'))
    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=exp['optim']['lr'], weight_decay=exp['optim']['weight_decay'])
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))

    outdir = Path(json.load(open(args.paths_config))['results_root']) / 'baseline_seqcnn'
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / 'config_snapshot.yaml').write_text(open(args.experiment).read())

    best_val = float('inf')

    for epoch in range(exp['optim']['epochs']):
        model.train()
        running = 0.0
        for step, (X, y) in enumerate(train_dl):
            X = X.to(device)                # [B, C, T, H, W]
            y = y.to(device)                # [B, T, H, W]
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type=='cuda')):
                yhat = model(X)
                loss = nn.functional.mse_loss(yhat, y)
            scaler.scale(loss).backward()
            if exp['optim'].get('grad_clip', 0) > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), exp['optim']['grad_clip'])
            scaler.step(optimizer)
            scaler.update()

            running += loss.item()
            if (step + 1) % exp.get('log_every', 50) == 0:
                print(f"epoch {epoch} step {step+1} train_mse={running/exp.get('log_every',50):.4f}")
                running = 0.0

        # validation
        model.eval()
        with torch.no_grad():
            v_losses, v_corrs = [], []
            for X, y in val_dl:
                X, y = X.to(device), y.to(device)
                yhat = model(X)
                v_losses.append(nn.functional.mse_loss(yhat, y).item())
                v_corrs.append(float(pearson_corr(yhat, y).cpu()))
            val_mse = sum(v_losses) / len(v_losses)
            val_r = sum(v_corrs) / len(v_corrs)
        print(f"epoch {epoch} VAL mse={val_mse:.4f} r={val_r:.3f}")

        # checkpoint
        if val_mse < best_val:
            best_val = val_mse
            ckpt = {
                'model_state': model.state_dict(),
                'norm': norm,
                'config': exp
            }
            torch.save(ckpt, outdir / 'best.pt')

    # final save
    torch.save({'model_state': model.state_dict(), 'norm': norm, 'config': exp}, outdir / 'last.pt')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--paths_config', default='config/paths_config.json')
    p.add_argument('--experiment', default='config/experiments/baseline_seqcnn.yaml')
    args = p.parse_args()
    main(args)
```

---

## 7) `scripts/evaluate.py` (optional quicklook)

```python
import argparse, json
from pathlib import Path
import torch, yaml
from loaders.surface_seasonal_dataset import make_dataloaders
from models.seqcnn import SeasonalSeqCNN
from metrics.skill_scores import rmse, pearson_corr


def main(args):
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    exp = ckpt.get('config')
    train_dl, val_dl, _ = make_dataloaders(args.paths_config, exp)

    model = SeasonalSeqCNN(in_channels=exp['model']['in_channels'],
                           hidden=exp['model']['hidden_channels'],
                           num_layers=exp['model']['num_layers'],
                           kernel_size=exp['model']['kernel_size'],
                           dropout=exp['model']['dropout'])
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    # evaluate on val
    tot_mse, tot_r, n = 0.0, 0.0, 0
    with torch.no_grad():
        for X, y in val_dl:
            yhat = model(X)
            tot_mse += rmse(yhat, y).item()
            tot_r += pearson_corr(yhat, y).item()
            n += 1
    print({"rmse": tot_mse/n, "corr": tot_r/n})

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--paths_config', default='config/paths_config.json')
    ap.add_argument('--checkpoint', default='results/baseline_seqcnn/best.pt')
    args = ap.parse_args()
    main(args)
```

---

## 8) Usage

```bash
# 0) prepare paths
mkdir -p config/experiments results
# edit config/paths_config.json to point to your ds_nano_surf_one_year.nc

# 1) run training (CPU or CUDA)
python scripts/train.py \
  --paths_config config/paths_config.json \
  --experiment config/experiments/baseline_seqcnn.yaml

# 2) evaluate quicklook
python scripts/evaluate.py --paths_config config/paths_config.json --checkpoint results/baseline_seqcnn/best.pt
```

---

## Notes & next steps

* This baseline **only uses 1D temporal convs per pixel**—good for seasonal cycle; extend with shallow 2D spatial conv for mesoscale signals.
* Add a **climatology baseline** (predict monthly mean) and report skill vs. it.
* When moving to deeper tiers (e.g., `mini`, with depth `z_l`), wrap the model with ConvLSTM or 3D convs and add vertical encodings.
* For HPC SLURM: create `slurm_jobs/train_seqcnn.sbatch` that loads your env, sets `CUDA_VISIBLE_DEVICES`, and calls the same CLI.

---

# V2 – Multi‑target daily SeqCNN on `ds_nano_surf_three_years`

**Task**: jointly predict surface **CT, dic, o2** from **Qnet, taux, tauy** at **daily** resolution, using a **single 1° grid cell** (timeseries only). We add **day‑of‑year sin/cos** encodings and learn all three targets with a shared temporal CNN.

## 1) `config/paths_config.json` (add the 3‑year nano surf key)

```json
{
  "data_root": "./data",
  "nano_surf_three_years": "./data/ds_nano_surf_three_years.nc",
  "results_root": "./results",
  "num_workers": 2
}
```

## 2) `config/experiments/baseline_seqcnn_multitarget.yaml`

```yaml
seed: 42

# data
dataset_path_key: nano_surf_three_years
inputs: [Qnet, taux, tauy]           # forcings at surface
targets: [CT, dic, o2]               # predict 3 tracers jointly
normalize:
  method: per-feature-zscore
  stats: null                        # computed from TRAIN portion only

split:
  type: time_holdout                 # first ~2y train, last ~1y val
  val_fraction_time: 0.33            # on ~3y -> ~1y val

windows:
  length: 64                         # days per sample
  stride: 1                          # dense windows

loader:
  batch_size: 64
  shuffle: true
  num_workers: 2
  prefetch_factor: 2

model:
  name: SeasonalSeqCNN_MTL
  in_channels: 5                      # 3 forcings + 2 time encodings (sin/cos DOY)
  hidden_channels: 64
  num_layers: 4
  kernel_size: 5
  dropout: 0.1
  out_channels: 3                     # CT, dic, o2

optim:
  lr: 3e-4
  weight_decay: 1e-5
  epochs: 50
  grad_clip: 1.0
  device: cuda

log_every: 100
save_every: 1
```

## 3) `loaders/ts_windows_singlecell.py`

```python
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import xarray as xr

# ---------- helpers ----------

def load_paths(paths_config_path: str) -> Dict:
    with open(paths_config_path, 'r') as f:
        return json.load(f)

def doy_sin_cos(time: xr.DataArray) -> np.ndarray:
    # robust 365.25-day normalization to absorb leap years
    doy = (time.dt.dayofyear - 1).astype('float64')
    theta = 2 * np.pi * (doy / 365.25)
    return np.stack([np.sin(theta), np.cos(theta)], axis=0)  # [2, T]


def collapse_singlecell(da: xr.DataArray) -> xr.DataArray:
    # Reduce spatial/staggered dims to scalar for the 1° cell
    for d in ("yh", "xh", "yq", "xq"):
        if d in da.dims:
            da = da.mean(dim=d)
    return da


def compute_stats(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = arr.mean(axis=1, keepdims=True)
    sig = arr.std(axis=1, keepdims=True) + 1e-8
    return mu, sig


# ---------- Dataset ----------

class SingleCellDailyWindows(Dataset):
    """Timeseries windows for a single grid cell.
    Inputs: [C_in, T]; Targets: [C_out, T]. Returns windows of length L.
    """
    def __init__(self,
                 ds_path: str,
                 inputs: List[str],
                 targets: List[str],
                 window: int = 64,
                 stride: int = 1,
                 split: str = 'train',
                 val_fraction_time: float = 0.33,
                 suggested_chunks: Dict = None,
                 normalize: Dict = None):
        assert split in {"train", "val"}
        chunks = suggested_chunks or {"time": 128}
        ds = xr.open_dataset(ds_path, chunks=chunks)

        # collect inputs
        x_list = []
        for v in inputs:
            if v not in ds:
                raise ValueError(f"Missing input {v}")
            x_list.append(collapse_singlecell(ds[v]))  # -> [time]
        X = xr.concat(x_list, dim="channel")  # [C_in, time]

        # time encodings
        sincos = doy_sin_cos(ds["time"])  # [2, T]
        sin_da = xr.DataArray(sincos[0], dims=("time"))
        cos_da = xr.DataArray(sincos[1], dims=("time"))
        X = xr.concat([X, sin_da.expand_dims({"channel": ["sin_doy"]}),
                          cos_da.expand_dims({"channel": ["cos_doy"]})], dim="channel")

        # collect targets
        y_list = []
        for v in targets:
            if v not in ds:
                raise ValueError(f"Missing target {v}")
            y_list.append(collapse_singlecell(ds[v]))
        Y = xr.concat(y_list, dim="target")  # [C_out, time]

        # to numpy
        X = np.asarray(X.transpose("channel", "time"))
        Y = np.asarray(Y.transpose("target", "time"))

        T = X.shape[1]
        split_idx = int(T * (1.0 - val_fraction_time))

        # compute normalization on TRAIN portion only
        if normalize is None or normalize.get("stats") is None:
            x_mu, x_sig = compute_stats(X[:, :split_idx])
            y_mu, y_sig = compute_stats(Y[:, :split_idx])
        else:
            x_mu, x_sig = normalize["stats"]["x_mu"], normalize["stats"]["x_sig"]
            y_mu, y_sig = normalize["stats"]["y_mu"], normalize["stats"]["y_sig"]

        # store normalized arrays
        self.X = ((X - x_mu) / x_sig).astype(np.float32)
        self.Y = ((Y - y_mu) / y_sig).astype(np.float32)
        self.stats = {"x_mu": x_mu, "x_sig": x_sig, "y_mu": y_mu, "y_sig": y_sig}

        # window indices
        if split == 'train':
            start, stop = 0, split_idx - window + 1
        else:
            # include context that reaches into val region
            start, stop = max(0, split_idx - window), T - window + 1
        self.starts = list(range(start, stop, stride))
        self.window = window

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, i):
        s = self.starts[i]
        e = s + self.window
        Xw = self.X[:, s:e]  # [C_in, L]
        Yw = self.Y[:, s:e]  # [C_out, L]
        return (
            torch.from_numpy(Xw),
            torch.from_numpy(Yw)
        )


def make_dataloaders(paths_config: str, exp_cfg: Dict) -> Tuple[DataLoader, DataLoader, Dict]:
    paths = load_paths(paths_config)
    ds_path = paths[exp_cfg["dataset_path_key"]]
    suggested_chunks = {"time": 128}

    train_set = SingleCellDailyWindows(ds_path,
                                       inputs=exp_cfg["inputs"],
                                       targets=exp_cfg["targets"],
                                       window=exp_cfg["windows"]["length"],
                                       stride=exp_cfg["windows"]["stride"],
                                       split='train',
                                       val_fraction_time=exp_cfg["split"]["val_fraction_time"],
                                       suggested_chunks=suggested_chunks,
                                       normalize=exp_cfg.get("normalize"))

    val_set = SingleCellDailyWindows(ds_path,
                                     inputs=exp_cfg["inputs"],
                                     targets=exp_cfg["targets"],
                                     window=exp_cfg["windows"]["length"],
                                     stride=exp_cfg["windows"]["stride"],
                                     split='val',
                                     val_fraction_time=exp_cfg["split"]["val_fraction_time"],
                                     suggested_chunks=suggested_chunks,
                                     normalize={"stats": train_set.stats})

    dl_kwargs = dict(batch_size=exp_cfg["loader"]["batch_size"],
                     shuffle=exp_cfg["loader"]["shuffle"],
                     num_workers=exp_cfg["loader"].get("num_workers", 0),
                     pin_memory=True,
                     prefetch_factor=exp_cfg["loader"].get("prefetch_factor", 2))

    return (DataLoader(train_set, **dl_kwargs),
            DataLoader(val_set,  **{**dl_kwargs, "shuffle": False}),
            train_set.stats)
```

## 4) `models/seqcnn_multitask.py`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalBlock(nn.Module):
    def __init__(self, c_in, c_out, k=5, p=2, d=1, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(c_in, c_out, kernel_size=k, padding=p, dilation=d)
        self.bn1 = nn.BatchNorm1d(c_out)
        self.conv2 = nn.Conv1d(c_out, c_out, kernel_size=k, padding=p, dilation=d)
        self.bn2 = nn.BatchNorm1d(c_out)
        self.dropout = nn.Dropout(dropout)
        self.res = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x):
        y = self.dropout(F.gelu(self.bn1(self.conv1(x))))
        y = self.dropout(F.gelu(self.bn2(self.conv2(y))))
        return y + self.res(x)

class SeasonalSeqCNN_MTL(nn.Module):
    """Multi‑task temporal CNN for per‑cell daily series.
    Input: [B, C_in, L] -> Output: [B, C_out, L]
    """
    def __init__(self, in_channels=5, hidden=64, num_layers=4, kernel_size=5, dropout=0.1, out_channels=3):
        super().__init__()
        layers = []
        c_in = in_channels
        dilation = 1
        pad = kernel_size // 2
        for _ in range(num_layers):
            layers.append(TemporalBlock(c_in, hidden, k=kernel_size, p=pad, d=dilation, dropout=dropout))
            c_in = hidden
            dilation *= 2
        self.tcn = nn.Sequential(*layers)
        self.head = nn.Conv1d(hidden, out_channels, kernel_size=1)

    def forward(self, x):  # x: [B, C_in, L]
        y = self.tcn(x)
        y = self.head(y)
        return y
```

## 5) `scripts/train_multitarget.py`

```python
import argparse, json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import yaml

from loaders.ts_windows_singlecell import make_dataloaders
from models.seqcnn_multitask import SeasonalSeqCNN_MTL


def load_cfg(exp_yaml: str):
    with open(exp_yaml, 'r') as f:
        return yaml.safe_load(f)


def rmse(yhat, y):
    return torch.sqrt(torch.mean((yhat - y) ** 2))

@torch.no_grad()
def pearson_corr_per_channel(yhat, y):
    # yhat, y: [B, C, L]
    B, C, L = yhat.shape
    rs = []
    for c in range(C):
        yh = yhat[:, c, :].reshape(-1)
        yt = y[:, c, :].reshape(-1)
        yh_c = yh - yh.mean()
        yt_c = yt - yt.mean()
        denom = torch.sqrt((yh_c**2).mean() * (yt_c**2).mean()) + 1e-8
        rs.append(((yh_c * yt_c).mean() / denom).item())
    return rs


def main(args):
    exp = load_cfg(args.experiment)
    torch.manual_seed(exp.get('seed', 42))

    train_dl, val_dl, stats = make_dataloaders(args.paths_config, exp)

    model = SeasonalSeqCNN_MTL(in_channels=exp['model']['in_channels'],
                               hidden=exp['model']['hidden_channels'],
                               num_layers=exp['model']['num_layers'],
                               kernel_size=exp['model']['kernel_size'],
                               dropout=exp['model']['dropout'],
                               out_channels=exp['model']['out_channels'])
    device = torch.device(exp['optim'].get('device', 'cpu'))
    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=exp['optim']['lr'], weight_decay=exp['optim']['weight_decay'])
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))

    outdir = Path(json.load(open(args.paths_config))['results_root']) / 'baseline_seqcnn_multitarget'
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / 'config_snapshot.yaml').write_text(open(args.experiment).read())

    best_val = float('inf')

    for epoch in range(exp['optim']['epochs']):
        model.train()
        running = 0.0
        for step, (X, Y) in enumerate(train_dl):
            X, Y = X.to(device), Y.to(device)           # [B, C_in, L], [B, C_out, L]
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type=='cuda')):
                Yhat = model(X)
                loss = nn.functional.mse_loss(Yhat, Y)
            scaler.scale(loss).backward()
            if exp['optim'].get('grad_clip', 0) > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), exp['optim']['grad_clip'])
            scaler.step(optimizer)
            scaler.update()

            running += loss.item()
            if (step + 1) % exp.get('log_every', 100) == 0:
                print(f"epoch {epoch} step {step+1} train_mse={running/exp.get('log_every',100):.4f}")
                running = 0.0

        # validation
        model.eval()
        with torch.no_grad():
            v_losses = []
            v_rs = []
            for X, Y in val_dl:
                X, Y = X.to(device), Y.to(device)
                Yhat = model(X)
                v_losses.append(nn.functional.mse_loss(Yhat, Y).item())
                v_rs.append(pearson_corr_per_channel(Yhat.cpu(), Y.cpu()))
            val_mse = sum(v_losses)/len(v_losses)
            # average corr per channel over batches
            val_r = [sum(r[c] for r in v_rs)/len(v_rs) for c in range(exp['model']['out_channels'])]
        print(f"epoch {epoch} VAL mse={val_mse:.5f} r_CT={val_r[0]:.3f} r_dic={val_r[1]:.3f} r_o2={val_r[2]:.3f}")

        if val_mse < best_val:
            best_val = val_mse
            ckpt = {'model_state': model.state_dict(), 'stats': stats, 'config': exp}
            torch.save(ckpt, outdir / 'best.pt')

    torch.save({'model_state': model.state_dict(), 'stats': stats, 'config': exp}, outdir / 'last.pt')

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--paths_config', default='config/paths_config.json')
    ap.add_argument('--experiment', default='config/experiments/baseline_seqcnn_multitarget.yaml')
    args = ap.parse_args()
    main(args)
```

## 6) Usage

```bash
mkdir -p config/experiments results
# 1) point paths to your ds_nano_surf_three_years.nc
#    (can be a symlink to your Tiger3 file if preferred)

# 2) run training
python scripts/train_multitarget.py \
  --paths_config config/paths_config.json \
  --experiment config/experiments/baseline_seqcnn_multitarget.yaml
```

## Notes

* **Staggered stresses** (`taux`, `tauy`) are averaged over their staggered dims and the single cell (yh/xh) to align with the scalar cell forcing.
* **Normalization is per‑feature/target** and computed on the **train** portion only; stored in the checkpoint for reproducibility.
* **Causal vs. same‑length**: the current head returns same‑length windows (no look‑ahead). Switch to causal padding if you later want strict forecasting.
* **Window length** 64d captures intra‑seasonal variability; try 32–128 for sensitivity.
* Add a simple **climatology**/persistence baseline for reference (optional).
