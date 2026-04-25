import os
import glob
import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset

from VAE_graph import BRepTransformerVAE, chamfer_loss

CONFIG = {
    "data_dir": "./dataset_graph_train",
    "save_dir": "./checkpoints_vae_resnet",
    "feature_dim": 394,
    "hidden_dim": 512,
    "latent_dim": 128,
    "batch_size": 32,
    "epochs": 300,

    # [黄金配置]
    "base_lr": 2e-4,  # 稳中求进
    "weight_decay": 1e-5,
    "clip_grad": 0.5,  # 严格一点，防止 Epoch 10 惨案

    "kl_start": 100,
    "kl_max": 0.001,
    "kl_cycle": 200,
    "kl_ratio": 0.5,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}


class RAMGraphDataset(Dataset):
    def __init__(self, root_dir):
        super().__init__()
        self.data_list = []
        files = glob.glob(os.path.join(root_dir, "*.pt"))
        print(f"Loading {len(files)} files...")
        for f in tqdm(files):
            try:
                data = torch.load(f)
                if not torch.isnan(data.x).any() and not torch.isinf(data.x).any():
                    if data.x.abs().max() < 100.0:
                        self.data_list.append(data)
            except:
                pass

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]


def get_kl_weight(epoch):
    if epoch < CONFIG["kl_start"]: return 0.0
    epoch_rel = (epoch - CONFIG["kl_start"]) % CONFIG["kl_cycle"]
    ratio = epoch_rel / CONFIG["kl_cycle"]
    if ratio < CONFIG["kl_ratio"]:
        return CONFIG["kl_max"] * (ratio / CONFIG["kl_ratio"])
    else:
        return CONFIG["kl_max"]


def train():
    os.makedirs(CONFIG["save_dir"], exist_ok=True)
    device = CONFIG["device"]
    print(f"🚀 Start Absolute Stability Training...")

    # 开启这个可以帮你找到到底是哪一层出了 NaN，如果还炸，把这个 log 发给我
    # torch.autograd.set_detect_anomaly(True)

    dataset = RAMGraphDataset(CONFIG["data_dir"])
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=0)

    model = BRepTransformerVAE(CONFIG).to(device)

    # 不再手动修改 scale_factor，用默认的 8.0 只是为了推理还原

    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["base_lr"], weight_decay=CONFIG["weight_decay"])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    best_val_loss = float('inf')

    for epoch in range(CONFIG["epochs"]):
        model.train()
        kl_w = get_kl_weight(epoch)
        train_loss = 0
        valid_batches = 0

        pbar = tqdm(train_loader, desc=f"Ep {epoch + 1}")

        for data in pbar:
            data = data.to(device)
            optimizer.zero_grad()

            recon_x, mu, logvar, z, target_padded, padding_mask = model(data)
            loss = chamfer_loss(recon_x, target_padded, padding_mask, mu, logvar, kl_w)

            # 严格跳过
            if torch.isnan(loss):
                optimizer.zero_grad()
                continue

            loss.backward()

            # 极严裁剪
            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["clip_grad"])

            if torch.isnan(total_norm):
                optimizer.zero_grad()
                continue

            optimizer.step()

            train_loss += loss.item()
            valid_batches += 1
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'gn': f"{total_norm:.2f}"})

        if valid_batches > 0:
            scheduler.step()
            avg_train_loss = train_loss / valid_batches

            model.eval()
            val_loss = 0
            with torch.no_grad():
                for data in val_loader:
                    data = data.to(device)
                    recon_x, mu, logvar, z, target_padded, padding_mask = model(data)
                    v_loss = chamfer_loss(recon_x, target_padded, padding_mask, mu, logvar, kl_w)
                    if not torch.isnan(v_loss):
                        val_loss += v_loss.item()

            avg_val_loss = val_loss / len(val_loader)
            current_lr = optimizer.param_groups[0]['lr']

            print(f"Ep {epoch + 1} | Tr: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f} | LR: {current_lr:.1e}")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), os.path.join(CONFIG["save_dir"], "best_vae.pth"))
        else:
            print("❌ Epoch failed.")


if __name__ == "__main__":
    train()
