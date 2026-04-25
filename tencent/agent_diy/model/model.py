#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import torch
import torch.nn as nn
import numpy as np
from agent_diy.conf.conf import Config

def make_fc_layer(in_features, out_features):
    fc = nn.Linear(in_features, out_features)
    nn.init.orthogonal_(fc.weight.data, gain=np.sqrt(2))
    nn.init.zeros_(fc.bias.data)
    return fc

class Model(nn.Module):
    def __init__(self, device=None):
        super().__init__()
        self.device = device
        
        # 极简骨干网络：丢掉残差，直接用两层128维的线性层，速度起飞！
        self.backbone = nn.Sequential(
            make_fc_layer(Config.DIM_OF_OBSERVATION, 128),
            nn.ReLU(),
            make_fc_layer(128, 128),
            nn.ReLU(),
        )
        
        # Actor 策略头
        self.actor_head = make_fc_layer(128, Config.ACTION_NUM)
        nn.init.orthogonal_(self.actor_head.weight.data, gain=0.01)

        # Critic 价值头
        self.critic_head = make_fc_layer(128, Config.VALUE_NUM)
        nn.init.orthogonal_(self.critic_head.weight.data, gain=1.0)

    def forward(self, obs, inference=False):
        hidden = self.backbone(obs)
        logits = self.actor_head(hidden)
        value = self.critic_head(hidden)
        return logits, value

    def set_train_mode(self): self.train()
    def set_eval_mode(self): self.eval()