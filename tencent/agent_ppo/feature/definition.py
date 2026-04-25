#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import numpy as np
from common_python.utils.common_func import create_cls
from agent_ppo.conf.conf import Config


# ================================
# ObsData
# ================================
ObsData = create_cls(
    "ObsData",
    feature=None,
    legal_action=None
)


# ================================
# ActData
# ================================
ActData = create_cls(
    "ActData",
    action=None,
    d_action=None,
    prob=None,
    value=None
)


# ================================
# SampleData
# ================================
SampleData = create_cls(
    "SampleData",
    obs=Config.DIM_OF_OBSERVATION,   # 1819
    legal_action=Config.ACTION_NUM,  # 16
    act=1,
    reward=Config.VALUE_NUM,
    reward_sum=Config.VALUE_NUM,
    done=1,
    value=Config.VALUE_NUM,
    next_value=Config.VALUE_NUM,
    advantage=Config.VALUE_NUM,
    prob=Config.ACTION_NUM,          # 16
)


def _scalar(x):
    """将 tensor / ndarray / list / scalar 统一转为 Python float。"""
    import torch
    if isinstance(x, torch.Tensor):
        return float(x.detach().cpu().item() if x.numel() == 1 else x.detach().cpu().numpy().flat[0])
    return float(np.asarray(x).flat[0])


def _vec(x):
    """将 tensor / ndarray / list 统一转为 1-D numpy float32 array。"""
    import torch
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().flatten().astype(np.float32)
    return np.asarray(x, dtype=np.float32).flatten()


# ================================
# 样本处理（填 next_value + GAE）
# ================================
def sample_process(list_sample_data):
    """填充next_value并计算GAE"""
    if not list_sample_data:
        return list_sample_data

    # 填充中间步的next_value
    for i in range(len(list_sample_data) - 1):
        list_sample_data[i].next_value = list_sample_data[i + 1].value

    # 最后一步的next_value处理
    last_sample = list_sample_data[-1]
    if _scalar(last_sample.done) > 0.5:  # terminated（被抓）
        # 终止状态：next_value = 0
        last_sample.next_value = np.zeros(1, dtype=np.float32)
    else:  # truncated（超时）
        # 截断状态：bootstrap当前value
        last_sample.next_value = last_sample.value

    _calc_gae(list_sample_data)
    return list_sample_data


def _calc_gae(list_sample_data):
    """GAE advantage 计算，全程 Python float 避免 tensor 污染。"""
    gae   = 0.0
    gamma = Config.GAMMA
    lamda = Config.LAMDA

    for sample in reversed(list_sample_data):
        reward     = _scalar(sample.reward)
        value      = _scalar(sample.value)
        next_value = _scalar(sample.next_value)
        done       = _scalar(sample.done)

        delta = reward + gamma * next_value - value
        mask = 1.0 - done
        gae = mask * gae * gamma * lamda + delta

        sample.advantage  = np.array([gae],           dtype=np.float32)
        sample.reward_sum = np.array([gae + value],   dtype=np.float32)
