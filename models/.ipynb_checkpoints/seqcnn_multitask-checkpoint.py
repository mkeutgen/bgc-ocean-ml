"""
A VERY SIMPLE, WELL-COMMENTED SPATIO–TEMPORAL BASELINE
=====================================================

Goal (plain English):
  • We want a tiny neural net that looks at a short time window for each grid cell (time regression)
    AND also peeks at the 4 immediate neighbors (north, south, east, west) in space (2D convolution-ish).
  • It predicts 3 targets per time step (CT, DIC, O2) at the same spatial resolution.

What goes in / out:
  • Input  X has shape [B, C_in, T, H, W]
       B  = batch size (how many training samples at once)
       C_in = number of input features (e.g., Qnet, taux, tauy, plus optional time encodings)
       T  = length of the time window (e.g., 64 days)
       H,W = spatial grid size (height/width) — can be as small as 1×1 or larger

  • Output Y has shape [B, C_out, T, H, W]
       C_out = number of predicted variables (e.g., 3 => CT, DIC, O2)
       Same T,H,W as input: we predict a value for every time step and grid cell in the window.

Design choices (kept intentionally simple):
  1) Temporal processing (per grid cell, over time):
     - Use small 1D convolutions along the TIME axis only.
     - We flatten the spatial dims into the batch temporarily, so we can run nn.Conv1d easily.

  2) Spatial processing (per time step, over space):
     - We compute the average of the 4 neighbors (N, S, E, W) using a fixed, non-trainable convolution.
     - At borders and corners, there are fewer neighbors; we automatically divide by the number of
       available neighbors (so we don’t unfairly count missing neighbors as zeros).

  3) Head:
     - A simple 1×1 convolution mixes channels to produce the 3 outputs per time step.

This file is written for learning purposes — it trades performance for readability.
If you want to scale up, replace the fixed spatial mix with a trainable Conv2d.
"""
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalBlock(nn.Module):
    """A tiny 1D convolution block that processes SEQUENCES over time.

    Input/Output shape inside this block: [B_flat, C, T]
      • B_flat is actually B*H*W from the original 5D tensor. We collapse space into batch so we
        can apply Conv1d along time easily.
      • Padding is set so the time length T stays the SAME after convolution.

    Why BatchNorm + GELU + residual?:
      • BatchNorm keeps activations well-scaled.
      • GELU is a smooth nonlinearity that generally works well.
      • The residual (skip) connection helps gradients flow through deeper stacks.
    """
    def __init__(self, c_in: int, c_out: int, kernel_size: int = 3, dilation: int = 1, dropout: float = 0.0):
        super().__init__()
        # For SAME-length output when using dilation d, padding should be d*(k-1)//2 (k odd)
        pad = dilation * (kernel_size - 1) // 2
        self.conv1 = nn.Conv1d(c_in,  c_out, kernel_size=kernel_size, padding=pad, dilation=dilation)
        self.bn1   = nn.BatchNorm1d(c_out)
        self.conv2 = nn.Conv1d(c_out, c_out, kernel_size=kernel_size, padding=pad, dilation=dilation)
        self.bn2   = nn.BatchNorm1d(c_out)
        self.dropout = nn.Dropout(dropout)
        self.residual = nn.Conv1d(c_in, c_out, kernel_size=1) if c_in != c_out else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B_flat, C_in, T]
        y = self.conv1(x)               # -> [B_flat, C_out, T]
        y = self.bn1(y)
        y = F.gelu(y)
        y = self.dropout(y)
        y = self.conv2(y)               # -> [B_flat, C_out, T]
        y = self.bn2(y)
        y = F.gelu(y)
        y = self.dropout(y)
        return y + self.residual(x)     # residual add keeps shapes identical


class SpatialFourNeighborMix(nn.Module):
    """A fixed (non-trainable) operator that averages the 4 neighbors (N,S,E,W).

    • Operates on a single TIME SLICE at a time: input [B, C, H, W] -> output [B, C, H, W]
    • We use grouped Conv2d with a fixed 3×3 kernel per channel:
        [ [0, 1, 0],
          [1, 0, 1],   (center weight is 0, only 4-neighborhood is 1)
          [0, 1, 0] ]
      This gives the SUM of neighbors.

    • To divide by the number of VALID neighbors at borders, we convolve a mask of ones with the
      SAME kernel to get a COUNT of neighbors for each pixel. Then we do sum / count.

    • IMPORTANT: This is not learnable; it’s just a clear conceptual demo of "use nearby cells".
      You can replace it with a trainable nn.Conv2d for better performance later.
    """
    def __init__(self, channels: int):
        super().__init__()
        # Build the 3x3 kernel that picks N,S,E,W (one kernel per channel, grouped conv)
        k = torch.tensor([[0., 1., 0.],
                          [1., 0., 1.],
                          [0., 1., 0.]], dtype=torch.float32)  # shape [3,3]
        # Weight shape for grouped conv: [C_out, 1, 3, 3] with groups=C (so one per channel)
        w = k.view(1, 1, 3, 3).repeat(channels, 1, 1, 1)       # [C,1,3,3]
        self.register_buffer('weight', w, persistent=False)

        # A matching kernel for counting neighbors: same as above
        self.register_buffer('count_weight', w.clone(), persistent=False)

        # We will use padding=1 so output H,W matches input H,W.
        self.padding = 1
        self.groups = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        # Sum over N,S,E,W using our fixed kernel (per-channel grouped conv)
        neighbor_sum = F.conv2d(x, self.weight, bias=None, stride=1, padding=self.padding, groups=self.groups)

        # Count how many neighbors each pixel actually has (borders have fewer than 4)
        ones = torch.ones_like(x)
        neighbor_count = F.conv2d(ones, self.count_weight, bias=None, stride=1, padding=self.padding, groups=self.groups)

        # Avoid division by zero (it can’t happen here, but be safe for weird shapes)
        neighbor_count = torch.clamp(neighbor_count, min=1.0)

        # Average of existing neighbors
        neighbor_avg = neighbor_sum / neighbor_count
        return neighbor_avg


class SimpleSpatioTemporalCNN(nn.Module):
    """A minimal model that first looks over TIME, then mixes in 4-neighbor spatial info.

    Steps in forward():
      1) TEMPORAL: For each (H,W) location, we run a small temporal CNN over the T axis.
         This finds patterns like daily/weekly cycles or responses to forcing.

      2) SPATIAL: For each time step independently, we average the 4 neighbors to inject
         immediate spatial context. Borders naturally use fewer neighbors.

      3) HEAD: A 1×1 conv maps the mixed features to the 3 desired outputs per time step.

    Arguments:
      in_channels  – how many input variables (e.g., 3 if using [Qnet, taux, tauy]; add 2 for sin/cos time)
      hidden       – size of the internal temporal features
      out_channels – how many targets to predict (3 for CT, DIC, O2)
      kernel_size  – Conv1d kernel size for temporal blocks (odd number recommended)
      num_tblocks  – how many temporal blocks to stack (2 is fine for a start)
    """
    def __init__(
        self,
        in_channels: int = 5,     # e.g., [Qnet, taux, tauy, sin_doy, cos_doy]
        hidden: int = 32,
        out_channels: int = 3,    # [CT, DIC, O2]
        kernel_size: int = 3,
        num_tblocks: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()

        # ---- 1) TEMPORAL STACK (works on [B*H*W, C_in, T]) ----
        tblocks = []
        c_in = in_channels
        dilation = 1
        for _ in range(num_tblocks):
            tblocks.append(TemporalBlock(c_in, hidden, kernel_size=kernel_size, dilation=dilation, dropout=dropout))
            c_in = hidden
            # (Optionally increase dilation to see farther in time; we keep it 1 for simplicity)
        self.temporal = nn.Sequential(*tblocks)

        # ---- 2) SPATIAL MIX (fixed 4-neighbor average, per time slice) ----
        self.spatial_mix = SpatialFourNeighborMix(channels=hidden)

        # After spatial mixing, we combine original temporal features and neighbor-avg by concatenation
        # and squeeze them back to `hidden` channels using a 1x1 conv (per time step, per pixel).
        self.combine = nn.Conv2d(in_channels=hidden * 2, out_channels=hidden, kernel_size=1)

        # ---- 3) HEAD: project to the 3 output variables per time step ----
        # We apply this with Conv3d(kernel_size=(1,1,1)) by reshaping, but an easier route is:
        #   For each time step independently, use a 1x1 Conv2d across channels.
        self.head = nn.Conv2d(in_channels=hidden, out_channels=out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C_in, T, H, W]
        returns: [B, C_out, T, H, W]
        """
        B, C_in, T, H, W = x.shape

        # -------- TEMPORAL STEP --------
        # Reshape to merge spatial dims into batch: [B*H*W, C_in, T]
        x_flat = x.permute(0, 3, 4, 1, 2).contiguous().view(B * H * W, C_in, T)
        tfeat = self.temporal(x_flat)                       # [B*H*W, hidden, T]

        # Restore to [B, hidden, T, H, W]
        tfeat = tfeat.view(B, H, W, -1, T).permute(0, 3, 4, 1, 2).contiguous()
        # Now tfeat is the per-time, per-pixel feature map we will spatially mix.

        # -------- SPATIAL STEP (for each time step independently) --------
        mixed = []
        for t in range(T):
            # Take the t-th time slice: [B, hidden, H, W]
            ft = tfeat[:, :, t, :, :]
            # Compute 4-neighbor average: [B, hidden, H, W]
            neigh = self.spatial_mix(ft)
            # Concatenate original and neighbor features along channels: [B, 2*hidden, H, W]
            cat = torch.cat([ft, neigh], dim=1)
            # Compress back to `hidden` channels with a 1x1 conv: [B, hidden, H, W]
            fused = self.combine(cat)
            mixed.append(fused)
        # Stack time back: list of T tensors -> [B, hidden, T, H, W]
        mixed = torch.stack(mixed, dim=2)

        # -------- HEAD: map to outputs per time step --------
        # Apply the 1x1 spatial head at each time slice
        outs = []
        for t in range(T):
            outs.append(self.head(mixed[:, :, t, :, :]))   # [B, C_out, H, W]
        y = torch.stack(outs, dim=2)                        # [B, C_out, T, H, W]
        return y


if __name__ == "__main__":
    # A tiny self-test so you can run: python models/simple_spatiotemporal_cnn.py
    B, C_in, T, H, W = 2, 5, 8, 3, 4
    model = SimpleSpatioTemporalCNN(in_channels=C_in, hidden=16, out_channels=3, kernel_size=3, num_tblocks=2)
    X = torch.randn(B, C_in, T, H, W)
    Y = model(X)
    print("Input shape:", X.shape)
    print("Output shape:", Y.shape)  # should be [2, 3, 8, 3, 4]
