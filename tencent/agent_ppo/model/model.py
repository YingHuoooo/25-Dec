#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import numpy as np
import torch
import torch.nn as nn
from agent_ppo.conf.conf import Config


def make_fc_layer(in_features, out_features, gain=1.0):
    fc = nn.Linear(in_features, out_features)
    nn.init.orthogonal_(fc.weight.data, gain=gain)
    nn.init.zeros_(fc.bias.data)
    return fc


class Model(nn.Module):
    def __init__(self, device=None):
        super().__init__()
        self.device = device
        self.cnn_flat = Config.CNN_FLAT_DIM  # 1764
        self.vec_dim = Config.VEC_DIM        # 55

        # CNN branch: 4ch 21x21 -> 256
        self.cnn = nn.Sequential(
            nn.Conv2d(Config.CNN_CHANNELS, 32, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            make_fc_layer(64 * 6 * 6, 256, gain=np.sqrt(2)),
            nn.ReLU(),
        )

        # MLP branch: vec_dim -> 128
        self.vec_mlp = nn.Sequential(
            make_fc_layer(self.vec_dim, 128, gain=np.sqrt(2)),
            nn.ReLU(),
        )

        # Fusion: 384 -> 128
        self.fusion = nn.Sequential(
            make_fc_layer(256 + 128, 256, gain=np.sqrt(2)),
            nn.ReLU(),
            make_fc_layer(256, 128, gain=np.sqrt(2)),
            nn.ReLU(),
        )

        # Actor / Critic heads
        self.actor_head = make_fc_layer(128, Config.ACTION_NUM, gain=0.01)
        self.critic_head = make_fc_layer(128, Config.VALUE_NUM, gain=1.0)

        # Init conv weights
        for m in self.cnn.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight.data, gain=np.sqrt(2))
                nn.init.zeros_(m.bias.data)

    def forward(self, obs, **kwargs):
        cnn_in = obs[:, :self.cnn_flat].reshape(-1, Config.CNN_CHANNELS, Config.CNN_MAP_SIZE, Config.CNN_MAP_SIZE)
        vec_in = obs[:, self.cnn_flat:]

        cnn_out = self.cnn(cnn_in)
        vec_out = self.vec_mlp(vec_in)
        fused = self.fusion(torch.cat([cnn_out, vec_out], dim=-1))

        logits = self.actor_head(fused)
        value = self.critic_head(fused)
        return logits, value

    def set_train_mode(self):
        self.train()

    def set_eval_mode(self):
        self.eval()
