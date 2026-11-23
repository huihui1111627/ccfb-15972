# -*- coding: utf-8 -*-
"""统一入口。

用法:
  python -m darl.main smoke                 # 快速冒烟测试(小规模, 三种模式)
  python -m darl.main exp                   # 跑完整实验集(默认 2 个 seed)
  python -m darl.main exp --seeds 1 --steps 60000
  python -m darl.main dashboard             # 从 results/runs.pkl 生成 HTML
  python -m darl.main bottleneck --tag high/async_clip
"""
import argparse
import os
import pickle
import sys

import numpy as np

from .config import Config
from .simulator import Simulator
from .analysis import bottleneck_report, summarize_all
from .dashboard import build_html


def smoke():
    print("== 冒烟测试: 三种模式 x 两种异质性 (24000 步) ==")
    for het in ("mild", "high"):
        for mode in ("sync", "async_no_correct", "async_clip"):
            cfg = Config(mode=mode, heterogeneity=het, seed=1,
                         n_workers=32, total_env_steps=24000,
                         batch_size=512, eval_every_updates=3)
            r = Simulator(cfg).run()
            er = r["eval_return"]
            print("%-5s %-18s final=%+.3f horizon=%7.0fms lUtil=%.2f "
                  "wUtil=%.2f lag=%.2f drop=%d"
                  % (het, mode, er[-1], r["horizon_ms"],
                     r["learner_util"].mean(), r["worker_util"].mean(),
                     r["lag_mean"], r["dropped_rollouts"]))


def run_experiments(seeds, steps):
    from .experiments import run_all, serialize
    import json
    results = run_all(seeds=seeds, total_steps=steps)
    os.makedirs("results", exist_ok=True)
    with open("results/runs.pkl", "wb") as f:
        pickle.dump(results, f)
    with open("results/runs.json", "w") as f:
        json.dump(serialize(results), f)
    print("\n== 瓶颈诊断(每个 seed 的主对比 run) ==")
    seen = set()
    for r in results:
        if r["group"] == "A_main" and r["tag"] not in seen:
            seen.add(r["tag"])
            print(bottleneck_report(r))
            print("-" * 60)
    html = build_html(results)
    with open("results/dashboard.html", "w", encoding="utf-8") as f:
        f.write(html)
    print("已生成 results/dashboard.html (用浏览器打开)")


def make_dashboard():
    path = "results/runs.pkl"
    if not os.path.exists(path):
        print("未找到 %s, 请先运行: python -m darl.main exp" % path)
        sys.exit(1)
    with open(path, "rb") as f:
        results = pickle.load(f)
    html = build_html(results)
    with open("results/dashboard.html", "w", encoding="utf-8") as f:
        f.write(html)
    print("已生成 results/dashboard.html")


def bottleneck(tag):
    with open("results/runs.pkl", "rb") as f:
        results = pickle.load(f)
    for r in results:
        if r["tag"] == tag:
            print(bottleneck_report(r))
            return
    print("未找到 tag=%s, 可选: %s"
          % (tag, sorted(set(r["tag"] for r in results))))


def main():
    p = argparse.ArgumentParser(description="分布式 RL 采样/学习仿真")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("smoke")
    pe = sub.add_parser("exp")
    pe.add_argument("--seeds", type=int, nargs="+", default=[1, 2])
    pe.add_argument("--steps", type=int, default=120000)
    sub.add_parser("dashboard")
    pb = sub.add_parser("bottleneck")
    pb.add_argument("--tag", required=True)
    args = p.parse_args()

    if args.cmd == "smoke":
        smoke()
    elif args.cmd == "exp":
        run_experiments(tuple(args.seeds), args.steps)
    elif args.cmd == "dashboard":
        make_dashboard()
    elif args.cmd == "bottleneck":
        bottleneck(args.tag)


if __name__ == "__main__":
    main()
