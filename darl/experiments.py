# -*- coding: utf-8 -*-
"""实验编排: 在相同总环境步数下跑全部对比实验并汇总。

实验分组
  A. 验收主对比(两种异质性 x 三种模式)
  B. 截断阈值扫描(0.1 / 0.2 / 0.3 / 不修正)
  C. 滞后上限扫描(lag_cap 2/4/8/16)
  D. 参数广播: pull vs push 不同间隔 + 低带宽压力
  E. 规模扫描(worker 数 16/32/64/128)定位瓶颈
所有 run 的总环境步数完全相同,保证"同总计算量"对比。
"""
import json
import time

import numpy as np

from .config import Config
from .simulator import Simulator

TOTAL_STEPS = 120_000
SEEDS = (1, 2)


def _run(cfg, tag, group, results):
    t0 = time.time()
    try:
        r = Simulator(cfg).run()
        r["tag"] = tag
        r["group"] = group
        results.append(r)
        er = r["eval_return"]
        print("  [%s] %-28s final=%+.3f last10=%+.3f worst10=%+.3f "
              "horizon=%7.0fms lUtil=%.2f wUtil=%.2f lag=%.2f drop=%d (%.1fs)"
              % (group, tag, er[-1], er[-10:].mean(), np.sort(er)[:10].mean(),
                 r["horizon_ms"], r["learner_util"].mean(),
                 r["worker_util"].mean(), r["lag_mean"],
                 r["dropped_rollouts"], time.time() - t0))
        return r
    except FloatingPointError as e:
        print("  [%s] %-28s DIVERGED: %s (%.1fs)" % (group, tag, e,
                                                      time.time() - t0))
        return None


def build_experiments():
    """返回 [(tag, group, kwargs), ...]。"""
    exps = []

    # A. 主对比
    for het in ("mild", "high"):
        for mode in ("sync", "async_no_correct", "async_clip"):
            exps.append(("%s/%s" % (het, mode), "A_main",
                         dict(mode=mode, heterogeneity=het)))

    # B. 截断阈值扫描(高异质,压力更大)
    for clip in (0.1, 0.2, 0.3):
        exps.append(("high/clip_%.1f" % clip, "B_clip",
                     dict(mode="async_clip", heterogeneity="high",
                          clip_eps=clip)))
    exps.append(("high/no_correct", "B_clip",
                 dict(mode="async_no_correct", heterogeneity="high")))

    # C. 滞后上限扫描
    for cap in (2, 4, 8, 16):
        exps.append(("mild/lagcap_%d" % cap, "C_lagcap",
                     dict(mode="async_clip", heterogeneity="mild",
                          lag_cap=cap)))

    # D. 参数广播
    exps.append(("pull", "D_broadcast",
                 dict(mode="async_clip", heterogeneity="mild",
                      param_mode="pull")))
    for interval in (250, 500, 1000, 2000):
        exps.append(("push_%dms" % interval, "D_broadcast",
                     dict(mode="async_clip", heterogeneity="mild",
                          param_mode="push", push_interval_ms=interval)))
    # 低带宽压力(10 Mbps,大策略参数时传输成本显现)
    exps.append(("push_500ms_10Mbps", "D_broadcast",
                 dict(mode="async_clip", heterogeneity="mild",
                      param_mode="push", push_interval_ms=500,
                      bandwidth_mbps=10.0)))

    # E. 规模扫描
    for n in (16, 32, 64, 128):
        exps.append(("workers_%d" % n, "E_scale",
                     dict(mode="async_clip", heterogeneity="mild",
                          n_workers=n)))
    return exps


def run_all(seeds=SEEDS, total_steps=TOTAL_STEPS):
    exps = build_experiments()
    results = []
    t_start = time.time()
    for seed in seeds:
        print("=== seed %d ===" % seed)
        for tag, group, kw in exps:
            cfg = Config(name=tag, seed=seed,
                         total_env_steps=total_steps, **kw)
            _run(cfg, tag, group, results)
    print("全部完成, 共 %d 个 run, 用时 %.1fs"
          % (len(results), time.time() - t_start))
    return results


def serialize(results):
    """把 ndarray 转 list,便于写 JSON。"""
    out = []
    for r in results:
        d = {}
        for k, v in r.items():
            if isinstance(v, np.ndarray):
                d[k] = v.tolist()
            elif isinstance(v, (np.floating, np.integer)):
                d[k] = v.item()
            elif k == "update_log":
                d[k] = v
            else:
                d[k] = v
        out.append(d)
    return out


if __name__ == "__main__":
    import os
    import pickle

    results = run_all()
    os.makedirs("results", exist_ok=True)
    with open("results/runs.pkl", "wb") as f:
        pickle.dump(results, f)
    with open("results/runs.json", "w") as f:
        json.dump(serialize(results), f)
    print("已写出 results/runs.pkl 和 results/runs.json")
