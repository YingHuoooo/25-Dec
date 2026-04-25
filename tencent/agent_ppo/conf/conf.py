#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

class Config:
    # =========================
    # 特征维度配置 — CNN+MLP混合架构
    # CNN: 4通道21×21地图 (flattened=1764)
    # MLP: 55维标量特征（含4维局势压力）
    # 总计 1819 维
    # =========================
    CNN_CHANNELS = 4
    CNN_MAP_SIZE = 21
    CNN_FLAT_DIM = CNN_CHANNELS * CNN_MAP_SIZE * CNN_MAP_SIZE  # 1764
    VEC_DIM = 55
    DIM_OF_OBSERVATION = CNN_FLAT_DIM + VEC_DIM  # 1819

    # =========================
    # LSTM 时序配置
    # =========================
    USE_LSTM = False               
    LSTM_HIDDEN_DIM = 128          
    LSTM_NUM_LAYERS = 1            
    SEQUENCE_LEN = 5               
    HISTORY_FEATURE_DIM = 16       
    DIM_WITH_HISTORY = DIM_OF_OBSERVATION + HISTORY_FEATURE_DIM # 500

    # =========================
    # 动作与价值空间
    # =========================
    ACTION_NUM = 16   # 8移动 + 8闪现
    VALUE_NUM = 1

    # =========================
    # PPO 超参数（调优平衡版）
    # =========================
    GAMMA  = 0.99
    LAMDA  = 0.95
    INIT_LEARNING_RATE_START = 0.0003
    BETA_START = 0.015
    BETA_MIN   = 0.005
    BETA_DECAY = 0.999
    CLIP_PARAM     = 0.2
    VF_COEF        = 0.5
    GRAD_CLIP_RANGE = 1.0
    PPO_EPOCHS = 4         # 减少榨取次数，防止过拟合到自杀策略