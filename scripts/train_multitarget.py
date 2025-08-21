import argparse, json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import yaml

from loaders.ts_windows_singlecell import make_dataloaders

cfg = yaml.safe_load(open(args.config))
exp = cfg.get("experiment", cfg)

from models.seqcnn_multitask import SimpleSpatioTemporalCNN

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

    model = SimpleSpatioTemporalCNN(
        in_channels=exp['model']['in_channels'],
        hidden=exp['model']['hidden'],
        out_channels=len(exp['dataset']['variables']['targets']),
        kernel_size=exp['model']['kernel_size'],
        num_tblocks=exp['model']['num_tblocks'],
        dropout=exp['model']['dropout'],
    )

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
