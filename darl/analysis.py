# -*- coding: utf-8 -*-
"""从原始 run 结果计算汇总指标与瓶颈分析。"""
import numpy as np


def _safe(x, default=0.0):
    x = np.asarray(x, dtype=float)
    return default if x.size == 0 else float(x.mean())


def summarize_run(r):
    """单个 run 的关键指标。"""
    er = np.asarray(r["eval_return"], dtype=float)
    ep = np.asarray(r["episode_return"], dtype=float)
    cfg = r["config"]
    horizon = r["horizon_ms"]

    # 稳定性: 评估回报的标准差、后期崩溃次数、最差 10 次均值。
    # 训练早期策略还随机,走不到目标得低分属正常,因此"崩溃"只统计
    # 训练过半之后跌破 -1.0 的评估点,避免把早期学习过程误判为发散。
    half = er.size // 2
    crashes = int(np.sum(er[half:] < -1.0)) if er.size else 0
    stability = {
        "eval_std": float(er.std()) if er.size else 0.0,
        "crashes": crashes,
        "worst10": float(np.sort(er)[:10].mean()) if er.size else 0.0,
        "last10": float(er[-10:].mean()) if er.size else 0.0,
        "final": float(er[-1]) if er.size else 0.0,
    }

    # 利用率
    learner_util = float(np.mean(r["learner_util"]))
    worker_util = float(np.mean(r["worker_util"]))

    # 吞吐: 每虚拟秒产生的环境步数 / 更新数
    throughput = {
        "steps_per_s": r["total_steps"] / max(horizon, 1e-9) * 1000.0,
        "updates_per_s": r["n_updates"] / max(horizon, 1e-9) * 1000.0,
    }

    # 数据丢弃率
    total_roll = max(r["n_rollouts"], 1)
    drop = {
        "rollout_drop_rate": r["dropped_rollouts"] / total_roll,
        "transitions_dropped": r["dropped_transitions"],
    }

    # 更新诊断的时间序列(KL / clip 比例 / 熵)
    log = r.get("update_log", [])
    diag = {
        "kl": [u["kl"] for u in log],
        "clip_frac": [u["clip_frac"] for u in log],
        "entropy": [u["entropy"] for u in log],
        "lag_mean_per_update": [u["lag_mean"] for u in log],
        "queue_len": [u["queue_len"] for u in log],
    }

    return {
        "tag": r["tag"],
        "group": r["group"],
        "mode": cfg["mode"],
        "heterogeneity": cfg["heterogeneity"],
        "seed": cfg["seed"],
        "horizon_s": horizon / 1000.0,
        "n_updates": r["n_updates"],
        "learner_util": learner_util,
        "worker_util": worker_util,
        "lag_mean": r["lag_mean"],
        "lag_p90": r["lag_p90"],
        "lag_max": r["lag_max"],
        "bytes_mb": r["bytes_pushed"] / 1e6,
        "broadcasts": r["broadcast_count"],
        **stability,
        **throughput,
        **drop,
        "diag": diag,
    }


def summarize_all(results):
    return [summarize_run(r) for r in results]


def bottleneck_report(r):
    """针对单个 run 给出瓶颈诊断文字。"""
    s = summarize_run(r)
    lines = []
    cfg = r["config"]
    lines.append("模式=%s, 异质性=%s, 虚拟耗时=%.1fs, 更新=%d 次"
                 % (cfg["mode"], cfg["heterogeneity"],
                    s["horizon_s"], s["n_updates"]))
    lines.append("  learner 利用率=%.1f%%, worker 平均利用率=%.1f%%"
                 % (s["learner_util"] * 100, s["worker_util"] * 100))
    if s["learner_util"] > 0.85:
        lines.append("  -> learner 是瓶颈: 梯度计算跟不上采样产出,经验在队列堆积")
    elif s["learner_util"] > 0.6:
        lines.append("  -> learner 高负载(>60%): 再加采样进程将很快饱和,滞后会恶化")
    elif s["worker_util"] < 0.5:
        lines.append("  -> 采样侧是瓶颈: worker 大量时间在等待(barrier/参数/卡顿)")
    else:
        lines.append("  -> 采样与学习基本均衡")

    # 慢 worker 影响
    speeds = np.asarray(r["worker_speeds"])
    rolls = np.asarray(r["worker_rollouts"], dtype=float)
    if speeds.size and rolls.sum() > 0:
        share = rolls / rolls.sum()
        slowest = int(np.argmin(speeds))
        fastest = int(np.argmax(speeds))
        lines.append("  最快 worker 产出占比=%.1f%%, 最慢=%.1f%% (速度比 %.1fx)"
                     % (share[fastest] * 100, share[slowest] * 100,
                        speeds[fastest] / max(speeds[slowest], 1e-9)))
    stall = np.asarray(r["worker_stall_ms"])
    if stall.sum() > 0:
        lines.append("  卡顿总时长=%.1fs, 占 worker 在线时间 %.1f%%"
                     % (stall.sum() / 1000,
                        stall.sum() / max(r["horizon_ms"], 1) * 100
                        / max(cfg["n_workers"], 1)))
    lines.append("  平均滞后=%.2f 版, p90=%.1f, 最大=%d; rollout 丢弃率=%.1f%%"
                 % (s["lag_mean"], s["lag_p90"], s["lag_max"],
                    s["rollout_drop_rate"] * 100))
    if s["crashes"] > 0:
        lines.append("  训练后期崩溃 %d 次(过半后评估回报跌破 -1.0), 稳定性差"
                     % s["crashes"])
    return "\n".join(lines)
