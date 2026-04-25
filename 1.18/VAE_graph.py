import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

MAX_FACES = 24

# ================= [新增] 残差块 (减震器) =================
class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim), # LayerNorm 是防炸的关键
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim)
        )
    def forward(self, x):
        # x + f(x): 即使 f(x) 炸了，x 还能保底
        return x + self.block(x)

class BRepTransformerVAE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        feature_dim = config['feature_dim']
        hidden_dim = config['hidden_dim'] 
        latent_dim = config['latent_dim'] 
        n_heads = 8 
        n_layers = 6 
        
        # Scaling
        self.register_buffer('scale_factor', torch.tensor(8.0)) 

        # ================= ENCODER =================
        self.enc_embedding = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=n_heads, dim_feedforward=hidden_dim*4, batch_first=True, norm_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        self.mu_head = nn.Linear(hidden_dim, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim, latent_dim)
        
        # ================= DECODER (ResNet 升级版) =================
        # 1. 先映射到 hidden_dim
        self.latent_proj = nn.Linear(latent_dim, hidden_dim)
        
        # 2. Deep Residual Expander (代替之前的普通 MLP)
        # 用 3 个残差块，深度够深，但绝对稳
        self.residual_decoder = nn.Sequential(
            ResidualBlock(hidden_dim),
            ResidualBlock(hidden_dim),
            ResidualBlock(hidden_dim)
        )
        
        # 3. 展开层: 512 -> 24 * 512
        self.expansion = nn.Linear(hidden_dim, MAX_FACES * hidden_dim)
        
        self.pos_embeddings = nn.Parameter(torch.randn(1, MAX_FACES, hidden_dim) * 0.02)
        
        # Refiner
        refine_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=n_heads, dim_feedforward=hidden_dim*4, batch_first=True, norm_first=True)
        self.face_refiner = nn.TransformerEncoder(refine_layer, num_layers=2)
        
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim), 
            nn.Linear(hidden_dim, feature_dim) 
        )

    def reparameterize(self, mu, logvar):
        # 安全 clamp
        logvar = torch.clamp(logvar, min=-5, max=5)
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def forward(self, data):
        x, batch = data.x, data.batch
        x_scaled = x / self.scale_factor
        
        batch_size = batch.max().item() + 1
        device = x.device
        
        x_padded = torch.zeros(batch_size, MAX_FACES, x.size(1), device=device)
        padding_mask = torch.ones(batch_size, MAX_FACES, device=device).bool() 
        
        for i in range(batch_size):
            mask_indices = (batch == i)
            nodes = x_scaled[mask_indices]
            num = nodes.size(0)
            if num > MAX_FACES: nodes = nodes[:MAX_FACES]; num = MAX_FACES
            x_padded[i, :num, :] = nodes
            padding_mask[i, :num] = False 

        # Encode
        h = self.enc_embedding(x_padded)
        h_enc = self.transformer_encoder(h, src_key_padding_mask=padding_mask)
        
        mask_float = (~padding_mask).float().unsqueeze(-1)
        h_pooled = (h_enc * mask_float).sum(dim=1) / (mask_float.sum(dim=1) + 1e-6)
        
        mu = self.mu_head(h_pooled)
        logvar = self.logvar_head(h_pooled)
        z = self.reparameterize(mu, logvar)
        
        # Decode (ResNet 流程)
        # A. Proj
        z_feat = self.latent_proj(z) 
        # B. Residual Process (这里最稳)
        z_deep = self.residual_decoder(z_feat)
        # C. Expand
        faces_coarse = self.expansion(z_deep).view(batch_size, MAX_FACES, -1)
        
        faces_input = faces_coarse + self.pos_embeddings
        faces_refined = self.face_refiner(faces_input, src_key_padding_mask=padding_mask)
        recon_x_scaled = self.output_head(faces_refined) 
        
        return recon_x_scaled, mu, logvar, z, x_padded, padding_mask

    def decode(self, z):
        with torch.no_grad():
            batch_size = z.size(0)
            z_feat = self.latent_proj(z) 
            z_deep = self.residual_decoder(z_feat)
            faces_coarse = self.expansion(z_deep).view(batch_size, MAX_FACES, -1)
            
            faces_input = faces_coarse + self.pos_embeddings
            faces_refined = self.face_refiner(faces_input)
            recon_scaled = self.output_head(faces_refined)
            recon_original = recon_scaled * self.scale_factor
        if batch_size == 1: return recon_original.squeeze(0)
        return recon_original

# ================= Loss (带 NaN 检查) =================
def chamfer_loss(recon_x, target_x, padding_mask, mu, logvar, kl_weight, scale_factor=8.0):
    
    # 物理尺寸还原
    recon_real = recon_x * scale_factor
    target_real = target_x * scale_factor
    
    # [最后一道防线] 检查是否有 NaN 进入 Loss
    if torch.isnan(recon_real).any():
        return torch.tensor(float('nan'), device=recon_real.device, requires_grad=True)

    # L1 Distance
    dist_matrix = torch.cdist(recon_real, target_real, p=1)
    
    target_mask = (~padding_mask).float().unsqueeze(1) 
    huge_val = 1e6 
    dist_matrix_masked = dist_matrix + (1.0 - target_mask) * huge_val
    
    min_dist_recon, _ = torch.min(dist_matrix_masked, dim=2) 
    term1 = torch.mean(min_dist_recon) 
    
    min_dist_target, _ = torch.min(dist_matrix, dim=1) 
    valid_count = target_mask.sum() + 1e-6
    term2 = torch.sum(min_dist_target * target_mask.squeeze(1)) / valid_count

    chamfer_dist = term1 + term2
    kld_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    return chamfer_dist + kld_loss * kl_weight
