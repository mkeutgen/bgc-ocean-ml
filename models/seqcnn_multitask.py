import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv2dBNReLU(nn.Module):
    """
    Small helper block: 2D conv + batchnorm + ReLU.
    Implemented as Conv3d with kernel (1, k, k) so we apply it to all timesteps at once.
    """
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=(1, kernel_size, kernel_size),
            padding=(0, padding, padding),
            bias=False,
        )
        self.bn = nn.BatchNorm3d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        # x: [B, C, T, H, W]
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class SpatialEncoder(nn.Module):
    """
    Lightweight spatial feature extractor shared across timesteps.
    Uses Conv3d with kernel (1,k,k), so operations are purely spatial.
    """
    def __init__(self, in_channels, feat_channels, num_blocks=2, kernel_size=3):
        super().__init__()
        pad = kernel_size // 2
        layers = [Conv2dBNReLU(in_channels, feat_channels, kernel_size, pad)]
        for _ in range(num_blocks - 1):
            layers.append(Conv2dBNReLU(feat_channels, feat_channels, kernel_size, pad))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x: [B, C_in, T, H, W] -> [B, C_feat, T, H, W]
        return self.net(x)


class TemporalAR8(nn.Module):
    """
    Depthwise temporal AR(8) component:
      - nn.Conv3d with kernel_size=(8,1,1), groups=C_feat
      - Causal: we pad on the left (past) only and slice to avoid future leakage.

    Input:  [B, C_feat, T, H, W]
    Output: [B, C_feat, T-7, H, W]  (each time uses 8 past days)
    """
    def __init__(self, channels):
        super().__init__()
        # depthwise temporal conv (per-channel, per-(H,W) location)
        self.temporal = nn.Conv3d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=(8, 1, 1),
            padding=(0, 0, 0),   # we'll pad manually to keep causal padding on the left
            groups=channels,
            bias=True,
        )

    def forward(self, x):
        # x: [B, C, T, H, W]
        B, C, T, H, W = x.shape
        if T < 8:
            raise ValueError(f"TemporalAR8 requires at least 8 timesteps, got T={T}")

        # Causal left padding: pad (time) on the left by 7, nothing on spatial dims.
        # F.pad pads last dims first; for 5D: (W_left, W_right, H_left, H_right, T_left, T_right)
        x_padded = F.pad(x, pad=(0, 0, 0, 0, 7, 0))  # add 7 past frames of zeros on the left

        # Now a plain valid conv3d (no padding) is causal; output length becomes T
        y = self.temporal(x_padded)  # [B, C, T, H, W]

        # We only want the outputs that have seen 8 *real* inputs → indices starting at t=7
        # This is already ensured by left padding; the first valid AR output is aligned to input index 0.
        # To expose strictly "uses 8 actual days" outputs, we can drop the first 7 outputs:
        y = y[:, :, 7:, :, :]       # -> [B, C, T-7, H, W]
        return y


class SimpleSequentialCNN_AR8(nn.Module):
    """
    Simple sequential CNN:
      1) Spatial 2D CNN applied across all timesteps (Conv3d with (1,k,k))
      2) Autoregressive temporal component with 8-day memory (depthwise Conv3d with (8,1,1))
      3) 1x1x1 projection to target channels

    Args
    ----
    in_channels:   number of input variables (e.g., Qnet, tau_x, tau_y, etc.)
    feat_channels: internal feature width after spatial encoder (e.g., 32)
    out_channels:  number of target variables (e.g., 3 for [O2, DIC, CT])
    num_spatial_blocks: how many Conv2dBNReLU blocks in the spatial encoder
    predict_sequence: if True, returns the full sequence of length (T-7);
                      if False, returns the last step only (shape [B, C_out, 1, H, W])

    Input
    -----
    x: [B, C_in, T, H, W]

    Output
    ------
    If predict_sequence=False: [B, C_out, 1, H, W] (next-day prediction aligned to last valid step)
    If predict_sequence=True:  [B, C_out, T-7, H, W] (per-time predictions from t=7..T-1)
    """
    def __init__(self,
                 in_channels: int,
                 feat_channels: int,
                 out_channels: int,
                 num_spatial_blocks: int = 2,
                 predict_sequence: bool = False):
        super().__init__()
        self.predict_sequence = predict_sequence

        # 1) Spatial encoder (shared over time)
        self.spatial = SpatialEncoder(
            in_channels=in_channels,
            feat_channels=feat_channels,
            num_blocks=num_spatial_blocks,
            kernel_size=3
        )

        # 2) Temporal AR(8) over encoded features
        self.temporal = TemporalAR8(channels=feat_channels)

        # 3) 1x1x1 projection to targets
        self.head = nn.Conv3d(
            in_channels=feat_channels,
            out_channels=out_channels,
            kernel_size=(1, 1, 1),
            bias=True
        )

    def forward(self, x):
        """
        x: [B, C_in, T, H, W]
        """
        # Spatial features for every timestep
        f = self.spatial(x)                 # [B, C_feat, T, H, W]

        # Causal AR(8) along the time axis
        ar = self.temporal(f)               # [B, C_feat, T-7, H, W]

        # Project to target channels
        y = self.head(ar)                   # [B, C_out, T-7, H, W]

        if self.predict_sequence:
            return y                        # [B, C_out, T-7, H, W]
        else:
            # Return only the last step (one-step-ahead target aligned to the end)
            return y[:, :, -1:, :, :]       # [B, C_out, 1, H, W]


if __name__ == "__main__":
    # ---- quick sanity check with fake data ----
    B, C_in, T, H, W = 2, 3, 16, 32, 48  # e.g., 3 forcings, 16-day window
    C_out = 3                            # e.g., [O2, DIC, CT]

    model = SimpleSequentialCNN_AR8(
        in_channels=C_in,
        feat_channels=32,
        out_channels=C_out,
        num_spatial_blocks=2,
        predict_sequence=False,  # set True to get T-7 steps back
    )

    x = torch.randn(B, C_in, T, H, W)
    y = model(x)
    print("Output shape:", y.shape)  # expect [B, C_out, 1, H, W]
