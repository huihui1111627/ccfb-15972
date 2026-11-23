# -*- coding: utf-8 -*-
"""实验配置。时间单位均为毫秒(虚拟时间)。"""
from dataclasses import dataclass, asdict, replace
from typing import Optional


@dataclass
class Config:
    name: str = "run"
    seed: int = 0

    # ---- 三种模式 ----
    # sync              : 同步采样,所有 worker 用同一版本跑完一轮 barrier 后才更新
    # async_no_correct  : 异步采样,不做重要性修正
    # async_clip        : 异步采样,重要性权重截断
    mode: str = "async_clip"

    # ---- 规模 ----
    n_workers: int = 64
    total_env_steps: int = 120_000
    rollout_len: int = 32          # 每个 worker 每次连跑多少步再上报
    batch_size: int = 512          # learner 一次更新消耗多少条 transition

    # ---- 环境步耗时(均值由 worker 速度决定) ----
    base_step_time_ms: float = 20.0      # 速度=1 的 worker 单步期望耗时
    step_time_cv: float = 0.5            # 单步耗时变异系数(gamma 分布)
    # 异质性:
    #   mild: lognormal(sigma=0.3),无常驻慢节点
    #   high: lognormal(sigma=0.9) + 2 个速度仅 0.15 的 straggler
    heterogeneity: str = "mild"
    stall_prob: float = 0.004            # 每步开始时进入卡顿的概率
    stall_mean_ms: float = 400.0         # 卡顿持续时间(指数分布均值)

    # ---- learner ----
    learner_update_ms: float = 120.0     # 一次梯度更新计算耗时(调到 learner 接近瓶颈)

    # ---- 策略滞后与重要性采样 ----
    lag_cap: int = 8                     # 滞后版本数超过该值的 rollout 丢弃
    clip_eps: float = 0.2                # 重要性权重截断阈值(1±eps)

    # ---- 参数分发 ----
    # pull: worker 每完成一个 rollout 主动拉取(带 piggyback,无额外 RTT)
    # push: learner 每 push_interval_ms 广播一次
    param_mode: str = "push"
    push_interval_ms: float = 500.0
    bandwidth_mbps: float = 100.0        # 每条网络链路带宽(兆比特/秒)
    rtt_ms: float = 2.0                  # 消息传播时延(往返的一半)

    # ---- 训练超参 ----
    lr: float = 1.5e-2
    gamma: float = 0.99
    hidden: int = 32
    ent_coef: float = 0.02

    # ---- 监控 ----
    eval_every_updates: int = 4          # 每 N 次更新做一次贪心评估
    eval_episodes: int = 5

    def clone(self, **kw):
        return replace(self, **kw)

    def to_dict(self):
        return asdict(self)
