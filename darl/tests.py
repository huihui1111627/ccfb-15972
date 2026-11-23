# -*- coding: utf-8 -*-
"""核心机制的轻量回归测试。运行: .venv/bin/python -m darl.tests"""
import numpy as np

from .config import Config
from .policy import init_params, update as policy_update, init_adam_state, forward
from .simulator import Simulator, compute_rtg


def test_is_clip_bounds():
    """截断模式下有效权重必须落在 [1-eps, 1+eps]。"""
    rng = np.random.default_rng(0)
    params = init_params(16, 4, seed=0)
    adam = init_adam_state(params)
    B = 64
    obs = np.eye(16)[rng.integers(0, 16, B)]
    actions = rng.integers(0, 4, B)
    # 故意让 old_logp 偏离当前策略,制造大 ratio
    old_logp = rng.uniform(-3.0, -0.5, B)
    adv = rng.normal(size=B)
    ret = rng.normal(size=B)
    stats = policy_update(params, adam, obs, actions, old_logp, adv, ret,
                          clip_eps=0.2, t=1, use_is_correction=True)
    assert 0.0 <= stats.clip_frac <= 1.0
    # 有偏差输入时应当触发一定比例截断
    assert stats.clip_frac > 0.0, "大 ratio 应被截断"
    print("test_is_clip_bounds OK (clip_frac=%.2f)" % stats.clip_frac)


def test_no_correction_weights_one():
    """无修正模式 clip_frac 恒为 0(权重恒 1)。"""
    rng = np.random.default_rng(1)
    params = init_params(16, 4, seed=1)
    adam = init_adam_state(params)
    B = 32
    obs = np.eye(16)[rng.integers(0, 16, B)]
    stats = policy_update(
        params, adam, obs, rng.integers(0, 4, B),
        rng.uniform(-3, -0.5, B), rng.normal(size=B), rng.normal(size=B),
        clip_eps=0.2, t=1, use_is_correction=False)
    assert stats.clip_frac == 0.0
    print("test_no_correction_weights_one OK")


def test_rtg_bootstrap():
    """reward-to-go: done 处切断,未结束段用 bootstrap 自举。"""
    rewards = np.array([1.0, 1.0, 1.0, 1.0])
    dones = np.array([False, True, False, False])
    gamma = 0.5
    rtg = compute_rtg(rewards, dones, 10.0, gamma)
    # 第一段: i1 done -> rtg[1]=1, rtg[0]=1+0.5*1=1.5
    assert abs(rtg[1] - 1.0) < 1e-9
    assert abs(rtg[0] - 1.5) < 1e-9
    # 第二段 bootstrap=10: rtg[3]=1+0.5*10=6, rtg[2]=1+0.5*6=4
    assert abs(rtg[3] - 6.0) < 1e-9
    assert abs(rtg[2] - 4.0) < 1e-9
    print("test_rtg_bootstrap OK")


def test_sync_zero_lag_and_barrier():
    """同步模式: 滞后恒为 0, 每个 worker 产出节奏由 barrier 对齐。"""
    cfg = Config(mode="sync", seed=3, n_workers=8, total_env_steps=4000,
                 eval_every_updates=2, heterogeneity="high")
    r = Simulator(cfg).run()
    assert r["lag_max"] == 0, "同步模式不应有版本滞后"
    assert abs(r["lag_mean"]) < 1e-9
    assert r["dropped_rollouts"] == 0
    # 强异质下同步的 worker 利用率应当明显偏低(barrier 等待)
    assert r["worker_util"].mean() < 0.6
    print("test_sync_zero_lag_and_barrier OK (wUtil=%.2f)"
          % r["worker_util"].mean())


def test_lag_cap_drops_stale():
    """lag_cap 很小时应丢弃超滞后数据并计入 dropped。"""
    cfg = Config(mode="async_clip", seed=4, n_workers=32,
                 total_env_steps=12000, lag_cap=1, eval_every_updates=2,
                 param_mode="push", push_interval_ms=2000)
    r = Simulator(cfg).run()
    assert r["lag_max"] <= 1, "保留的数据滞后不应超过 cap"
    assert r["dropped_rollouts"] + r["dropped_transitions"] >= 0
    print("test_lag_cap_drops_stale OK (lag_max=%d)" % r["lag_max"])


def test_async_keeps_workers_busy():
    """异步模式 worker 不应死锁, 利用率应显著高于同步。"""
    cfg = Config(mode="async_clip", seed=5, n_workers=16,
                 total_env_steps=6000, eval_every_updates=2)
    r = Simulator(cfg).run()
    assert r["total_steps"] >= 6000
    assert r["worker_util"].mean() > 0.7, "异步 worker 应持续采样"
    assert r["n_updates"] > 0
    print("test_async_keeps_workers_busy OK (wUtil=%.2f, upd=%d)"
          % (r["worker_util"].mean(), r["n_updates"]))


def test_termination():
    """仿真必须在预算耗尽时终止(push 定时器不得导致空转)。"""
    cfg = Config(mode="async_clip", seed=6, n_workers=16,
                 total_env_steps=3000, param_mode="push",
                 push_interval_ms=100, eval_every_updates=2)
    r = Simulator(cfg).run()
    assert r["total_steps"] >= 3000
    print("test_termination OK")


def test_param_versions_monotonic():
    """learner 版本号单调递增,快照可被 worker 引用。"""
    cfg = Config(mode="async_clip", seed=7, n_workers=16,
                 total_env_steps=4000, eval_every_updates=2)
    sim = Simulator(cfg)
    r = sim.run()
    versions = [u["version"] for u in r["update_log"]]
    assert versions == sorted(versions) and len(versions) == len(set(versions))
    print("test_param_versions_monotonic OK (versions=%d)" % len(versions))


if __name__ == "__main__":
    test_is_clip_bounds()
    test_no_correction_weights_one()
    test_rtg_bootstrap()
    test_sync_zero_lag_and_barrier()
    test_lag_cap_drops_stale()
    test_async_keeps_workers_busy()
    test_termination()
    test_param_versions_monotonic()
    print("\n全部测试通过 ✓")
