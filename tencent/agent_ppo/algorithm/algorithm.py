#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import os
import torch
import torch.nn.functional as F
import numpy as np

from agent_ppo.conf.conf import Config


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


class Algorithm:
    def __init__(self, model, optimizer, device, logger, monitor):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.logger = logger
        self.monitor = monitor

        self.train_step = 0
        self.clip_param = Config.CLIP_PARAM
        self.vf_coef = Config.VF_COEF
        self.var_beta = Config.BETA_START

        self.lr_start = Config.INIT_LEARNING_RATE_START
        self.lr_end = self.lr_start * 0.1

        self.sample_buffer = []
        self.min_batch_size = 256

    def learn(self, list_sample_data):
        self.sample_buffer.extend(list_sample_data)
        if len(self.sample_buffer) < self.min_batch_size:
            return None

        buffered = self.sample_buffer
        self.sample_buffer = []

        obs      = np.stack([_to_numpy(s.obs).flatten() for s in buffered]).astype(np.float32)
        act      = np.array([int(_to_numpy(s.act).flat[0]) for s in buffered], dtype=np.int64)
        old_prob = np.array([np.log(float(_to_numpy(s.prob).flat[int(_to_numpy(s.act).flat[0])]) + 1e-8) for s in buffered], dtype=np.float32)
        adv      = np.array([float(_to_numpy(s.advantage).flat[0]) for s in buffered], dtype=np.float32)
        ret      = np.array([float(_to_numpy(s.reward_sum).flat[0]) for s in buffered], dtype=np.float32)
        legal    = np.stack([_to_numpy(s.legal_action).flatten() for s in buffered]).astype(np.float32)
        old_val  = np.array([float(_to_numpy(s.value).flat[0]) for s in buffered], dtype=np.float32)

        adv_raw_mean = adv.mean()
        adv_raw_std = adv.std()
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        self.model.set_train_mode()

        result = None
        for _ in range(Config.PPO_EPOCHS):
            result = self._update(obs, act, old_prob, adv, ret, legal, old_val, adv_raw_mean, adv_raw_std)

        self.train_step += 1

        if self.monitor is not None and result is not None:
            try:
                self.monitor.put_data({os.getpid(): {
                    "total_loss":   result["total_loss"],
                    "policy_loss":  result["policy_loss"],
                    "value_loss":   result["value_loss"],
                    "entropy_loss": -self.var_beta * result["entropy"],
                }})
            except Exception:
                pass

        return result

    def _update(self, obs_np, act_np, old_prob_np, adv_np, ret_np, legal_np, old_val_np, adv_raw_mean, adv_raw_std):
        obs      = torch.tensor(obs_np, dtype=torch.float32).to(self.device)
        act      = torch.tensor(act_np, dtype=torch.long).to(self.device)
        old_prob = torch.tensor(old_prob_np, dtype=torch.float32).to(self.device)
        adv      = torch.tensor(adv_np, dtype=torch.float32).to(self.device)
        ret      = torch.tensor(ret_np, dtype=torch.float32).to(self.device)
        legal    = torch.tensor(legal_np, dtype=torch.float32).to(self.device)
        old_val  = torch.tensor(old_val_np, dtype=torch.float32).to(self.device)

        logits, value = self.model(obs)

        masked_logits = logits.clone()
        masked_logits[legal == 0] = -1e10
        dist = torch.distributions.Categorical(logits=masked_logits)
        new_prob = dist.log_prob(act)
        entropy = dist.entropy().mean()

        # PPO clipped surrogate
        ratio = torch.exp(new_prob - old_prob)
        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv
        policy_loss = -torch.mean(torch.min(surr1, surr2))

        # Value loss (clipped)
        value_pred = value.squeeze(-1)
        value_clipped = old_val + torch.clamp(value_pred - old_val, -self.clip_param, self.clip_param)
        vf_loss1 = (value_pred - ret) ** 2
        vf_loss2 = (value_clipped - ret) ** 2
        value_loss = 0.5 * torch.mean(torch.max(vf_loss1, vf_loss2))

        # Debug
        if self.train_step % 100 == 0:
            self.logger.info(f"[DEBUG] adv_raw: mean={adv_raw_mean:.4f} std={adv_raw_std:.4f}")
            self.logger.info(f"[DEBUG] ratio: mean={ratio.mean().item():.4f} std={ratio.std().item():.4f} min={ratio.min().item():.4f} max={ratio.max().item():.4f}")
            self.logger.info(f"[DEBUG] policy_loss={policy_loss.item():.4f} value_loss={value_loss.item():.4f}")

        # Entropy decay
        self.var_beta = max(Config.BETA_MIN, Config.BETA_START * (Config.BETA_DECAY ** self.train_step))
        entropy_loss = -self.var_beta * entropy

        # Learning rate decay
        decay = max(0.0, 1.0 - self.train_step / 50000.0)
        lr = self.lr_end + (self.lr_start - self.lr_end) * decay
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        # Combined update
        total_loss = policy_loss + entropy_loss + self.vf_coef * value_loss
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), Config.GRAD_CLIP_RANGE)
        self.optimizer.step()

        return {
            "total_loss":  total_loss.item(),
            "policy_loss": policy_loss.item(),
            "value_loss":  value_loss.item(),
            "entropy":     entropy.item(),
        }
