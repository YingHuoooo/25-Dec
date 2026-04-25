#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Monitor panel configuration builder for Gorge Chase.
峡谷追猎监控面板配置构建器。
"""

from kaiwudrl.common.monitor.monitor_config_builder import MonitorConfigBuilder


def build_monitor():
    monitor = MonitorConfigBuilder()

    config_dict = (
        monitor.title("峡谷追猎")
        .add_group(group_name="算法指标", group_name_en="algorithm")
        .add_panel(name="累积回报", name_en="reward", type="line")
        .add_metric(metrics_name="reward", expr="avg(reward{})")
        .end_panel()
        .add_panel(name="总损失", name_en="total_loss", type="line")
        .add_metric(metrics_name="total_loss", expr="avg(total_loss{})")
        .end_panel()
        .add_panel(name="价值损失", name_en="value_loss", type="line")
        .add_metric(metrics_name="value_loss", expr="avg(value_loss{})")
        .end_panel()
        .add_panel(name="策略损失", name_en="policy_loss", type="line")
        .add_metric(metrics_name="policy_loss", expr="avg(policy_loss{})")
        .end_panel()
        .add_panel(name="熵损失", name_en="entropy_loss", type="line")
        .add_metric(metrics_name="entropy_loss", expr="avg(entropy_loss{})")
        .end_panel()
        .end_group()
        .add_group(group_name="游戏指标", group_name_en="game")
        .add_panel(name="存活步数", name_en="episode_steps", type="line")
        .add_metric(metrics_name="episode_steps", expr="avg(episode_steps{})")
        .end_panel()
        .add_panel(name="存活率", name_en="survival_ratio", type="line")
        .add_metric(metrics_name="survival_ratio", expr="avg(survival_ratio{})")
        .end_panel()
        .add_panel(name="胜率", name_en="win", type="line")
        .add_metric(metrics_name="win", expr="avg(win{})")
        .end_panel()
        .add_panel(name="宝箱收集数", name_en="treasures_collected", type="line")
        .add_metric(metrics_name="treasures_collected", expr="avg(treasures_collected{})")
        .end_panel()
        .end_group()
        .build()
    )
    return config_dict
