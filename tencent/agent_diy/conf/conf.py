#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

class Config:
    # 特征维度：148维鹰眼雷达
    DIM_OF_OBSERVATION = 148
    # 动作维度：16维（8方向移动 + 8方向闪现）
    ACTION_NUM = 16
    VALUE_NUM = 1

    # 高阶 PPO 训练超参数
    GAMMA = 0.99
    LAMDA = 0.95
    START_LR = 0.0003
    BETA_START = 0.015 # 熵正则化，鼓励探索
    CLIP_PARAM = 0.2
    VF_COEF = 1.0
    GRAD_CLIP_RANGE = 0.5