#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import os
import time
import torch
from agent_diy.conf.conf import Config

class Algorithm:
    def __init__(self, model, optimizer, device=None, logger=None, monitor=None):
        self.device = device
        self.model = model
        self.optimizer = optimizer
        self.parameters = [p for pg in self.optimizer.param_groups for p in pg["params"]]
        self.logger = logger
        self.monitor = monitor
        self.last_report_time = 0

    def learn(self, list_sample_data):
        obs = torch.stack([torch.tensor(f.obs) for f in list_sample_data]).to(self.device)
        legal_action = torch.stack([torch.tensor(f.legal_action) for f in list_sample_data]).to(self.device)
        act = torch.stack([torch.tensor(f.act) for f in list_sample_data]).to(self.device).view(-1, 1)
        old_prob = torch.stack([torch.tensor(f.prob) for f in list_sample_data]).to(self.device)
        reward = torch.stack([torch.tensor(f.reward) for f in list_sample_data]).to(self.device)
        advantage = torch.stack([torch.tensor(f.advantage) for f in list_sample_data]).to(self.device)
        old_value = torch.stack([torch.tensor(f.value) for f in list_sample_data]).to(self.device)
        reward_sum = torch.stack([torch.tensor(f.reward_sum) for f in list_sample_data]).to(self.device)

        self.model.set_train_mode()
        self.optimizer.zero_grad()
        logits, value_pred = self.model(obs)

        # 掩码机制，过滤非法动作
        label_max, _ = torch.max(logits * legal_action, dim=1, keepdim=True)
        label = (logits - label_max) * legal_action + 1e5 * (legal_action - 1)
        prob_dist = torch.nn.functional.softmax(label, dim=1)

        # 策略损失 (PPO Clip)
        one_hot = torch.nn.functional.one_hot(act[:, 0].long(), Config.ACTION_NUM).float()
        new_prob = (one_hot * prob_dist).sum(1, keepdim=True)
        old_action_prob = (one_hot * old_prob).sum(1, keepdim=True).clamp(1e-9)
        ratio = new_prob / old_action_prob
        adv = advantage.view(-1, 1)
        
        policy_loss1 = -ratio * adv
        policy_loss2 = -ratio.clamp(1 - Config.CLIP_PARAM, 1 + Config.CLIP_PARAM) * adv
        policy_loss = torch.maximum(policy_loss1, policy_loss2).mean()

        # 价值损失 (Value Clip)
        value_clip = old_value + (value_pred - old_value).clamp(-Config.CLIP_PARAM, Config.CLIP_PARAM)
        value_loss = 0.5 * torch.maximum(torch.square(reward_sum - value_pred), torch.square(reward_sum - value_clip)).mean()

        # 熵正则化
        entropy_loss = (-prob_dist * torch.log(prob_dist.clamp(1e-9, 1.0))).sum(1).mean()

        total_loss = Config.VF_COEF * value_loss + policy_loss - Config.BETA_START * entropy_loss
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters, Config.GRAD_CLIP_RANGE)
        self.optimizer.step()

        # 监控上报 (完美缩进版)
        now = time.time()
        if now - self.last_report_time >= 60:
            res = {
                "total_loss": round(total_loss.item(), 4), 
                "value_loss": round(value_loss.item(), 4),
                "policy_loss": round(policy_loss.item(), 4), 
                "entropy_loss": round(entropy_loss.item(), 4),
                "reward": round(reward.mean().item(), 4),
            }
            if self.logger: self.logger.info(f"[DIY-Learn] total_loss: {res['total_loss']}")
            if self.monitor: self.monitor.put_data({os.getpid(): res})
            self.last_report_time = now