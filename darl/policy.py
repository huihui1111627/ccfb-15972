# -*- coding: utf-8 -*-
"""NumPy 小型 actor-critic 策略网络(无任何深度学习框架)。

结构:
  共享输入 -> 两层各自独立的单隐层 MLP
  actor : obs -> W1,b1 -> tanh -> Wa,ba -> softmax(随机策略)
  critic: obs -> W1,b1 -> tanh -> Wv,bv -> 标量状态价值
训练: REINFORCE with value baseline,重要性权重截断(PPO 式 clipped surrogate),
      手写反向传播 + 手写 Adam。
"""
from dataclasses import dataclass

import numpy as np


def init_params(obs_dim, n_actions, hidden=32, seed=0):
    rng = np.random.default_rng(seed)
    scale_a = np.sqrt(1.0 / obs_dim)
    scale_h = np.sqrt(1.0 / hidden)
    return {
        "W1": rng.normal(0, scale_a, size=(hidden, obs_dim)),
        "b1": np.zeros(hidden),
        "Wa": rng.normal(0, scale_h, size=(n_actions, hidden)),
        "ba": np.zeros(n_actions),
        "Wv": rng.normal(0, scale_h, size=(1, hidden)),
        "bv": np.zeros(1),
    }


def clone_params(params):
    return {k: v.copy() for k, v in params.items()}


def params_bytes(params):
    """策略参数在网络上传输的字节数(按 float64 计)。"""
    return int(sum(v.size for v in params.values()) * 8)


def _softmax(logits):
    z = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def forward(params, obs):
    """单条观测 -> (动作概率, 对数概率, 价值, 隐层激活)。"""
    z1 = params["W1"] @ obs + params["b1"]
    h = np.tanh(z1)
    logits = params["Wa"] @ h + params["ba"]
    probs = _softmax(logits)
    logp = np.log(probs + 1e-12)
    value = float((params["Wv"] @ h + params["bv"])[0])
    return probs, logp, value, h


def sample_action(params, obs, rng):
    probs, logp, value, _ = forward(params, obs)
    a = int(rng.choice(len(probs), p=probs))
    return a, float(logp[a]), float(probs[a]), value


def greedy_action(params, obs):
    probs, _, value, _ = forward(params, obs)
    return int(np.argmax(probs)), value


def init_adam_state(params):
    return {
        k: {"m": np.zeros_like(v), "v": np.zeros_like(v)}
        for k, v in params.items()
    }


def _adam_step(params, grads, state, lr, t, beta1=0.9, beta2=0.999, eps=1e-8,
               max_grad_norm=5.0):
    # 全局梯度裁剪,防止陈旧数据导致的梯度尖峰把参数打飞
    total_norm = np.sqrt(sum((g * g).sum() for g in grads.values()))
    if np.isfinite(total_norm) and total_norm > max_grad_norm:
        scale = max_grad_norm / (total_norm + 1e-12)
        grads = {k: g * scale for k, g in grads.items()}
    for k in params:
        g = grads[k]
        m = state[k]["m"]
        v = state[k]["v"]
        m *= beta1
        m += (1.0 - beta1) * g
        v *= beta2
        v += (1.0 - beta2) * g * g
        m_hat = m / (1.0 - beta1 ** t)
        v_hat = v / (1.0 - beta2 ** t)
        params[k] -= lr * m_hat / (np.sqrt(v_hat) + eps)


@dataclass
class UpdateStats:
    mean_ratio: float
    clip_frac: float
    approx_kl: float
    entropy: float
    value_loss: float
    mean_adv: float
    effective_weight: float


def update(params, adam_state, obs, actions, old_logp, advantages, returns,
           lr=1e-2, clip_eps=0.2, ent_coef=0.01, vf_coef=0.5, t=1,
           use_is_correction=True):
    """一次小批量梯度更新。

    obs        : (B, obs_dim)
    old_logp   : (B,) 采样时动作的对数概率(行为策略)
    advantages : (B,) 已标准化的优势
    returns    : (B,) 折扣 reward-to-go,作为 critic 目标
    use_is_correction=False 时权重恒为 1(模拟"不做修正的异步采样")。
    """
    B = obs.shape[0]

    # ---- 前向(向量化) ----
    Z1 = obs @ params["W1"].T + params["b1"]
    H = np.tanh(Z1)
    logits = H @ params["Wa"].T + params["ba"]
    probs = _softmax(logits)
    logp = np.log(probs + 1e-12)
    values = (H @ params["Wv"].T + params["bv"].T).reshape(-1)

    logp_taken = logp[np.arange(B), actions]
    ratio = np.exp(logp_taken - old_logp)

    if use_is_correction:
        w_raw = ratio
        w_eff = np.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
    else:
        w_raw = ratio
        w_eff = np.ones_like(ratio)

    adv = advantages
    # 标准化优势(批内)
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    q = w_eff * adv  # 每个样本的策略梯度系数

    # ---- actor 输出层梯度: dL/dlogits ----
    dlogits = probs.copy()
    dlogits[np.arange(B), actions] -= 1.0
    dlogits *= q.reshape(-1, 1) / B

    # ---- 熵正则梯度 ----
    E = (probs * logp).sum(axis=1, keepdims=True)      # = -H, (B,1)
    ent_grad = -ent_coef * probs * (E + logp) / B
    dlogits += ent_grad

    # ---- critic 梯度 ----
    dv = (2.0 * vf_coef * (values - returns) / B).reshape(-1, 1)

    # ---- 反向传播 ----
    dHid = dlogits @ params["Wa"] + dv @ params["Wv"]
    dZ1 = dHid * (1.0 - H ** 2)
    grads = {
        "W1": dZ1.T @ obs,
        "b1": dZ1.sum(axis=0),
        "Wa": dlogits.T @ H,
        "ba": dlogits.sum(axis=0),
        "Wv": dv.T @ H,
        "bv": dv.sum(axis=0).reshape(1),
    }
    _adam_step(params, grads, adam_state, lr, t)

    entropy = -(probs * logp).sum(axis=1).mean()
    approx_kl = float(((ratio - 1.0) - np.log(ratio + 1e-12)).mean())
    return UpdateStats(
        mean_ratio=float(w_raw.mean()),
        clip_frac=float((np.abs(w_raw - 1.0) > clip_eps).mean())
        if use_is_correction else 0.0,
        approx_kl=approx_kl,
        entropy=float(entropy),
        value_loss=float(((values - returns) ** 2).mean()),
        mean_adv=float(adv.mean()),
        effective_weight=float(w_eff.mean()),
    )
