# 分布式强化学习采样/学习系统仿真

用**虚拟时间事件驱动仿真**复现"几百个采样进程 + 一个学习进程"的分布式 RL 架构，
研究异步采样中**策略版本滞后**带来的偏差、重要性采样截断修正、参数广播带宽权衡与系统瓶颈。

环境、采样 worker、learner、网络全部为仿真，策略网络用纯 NumPy 手写（含反向传播与 Adam）。

## 为什么用虚拟时间仿真而不是真起进程

真起几百个进程在单机上会被 GIL、CPU 核数、调度抖动污染，且三种模式无法在
**严格相同的总计算量**下对比。本仿真用一个最小堆按虚拟时刻调度四类事件
（环境步完成、网络消息到达、learner 计算完成、广播定时器），从而：

- 确定性可复现（只依赖 seed）；
- 显式建模每步耗时差异、worker 速度异质、偶发卡顿、链路带宽/传播时延、learner 计算耗时；
- 三种模式跑**完全相同的总环境步数**，对比公平；
- 忙闲时间线、滞后分布等指标直接来自事件时间戳。

## 目录结构

| 文件 | 职责 |
|---|---|
| `darl/env.py` | 8×8 GridWorld 离散决策环境（打滑、撞墙、有界回报） |
| `darl/policy.py` | NumPy actor-critic 网络、softmax 策略、手写反向传播 + Adam、IS 截断 |
| `darl/config.py` | 全部仿真/训练/网络参数 |
| `darl/simulator.py` | 事件驱动仿真器：worker、learner、带宽链路、barrier、广播、指标统计 |
| `darl/experiments.py` | 五组实验的编排与序列化 |
| `darl/analysis.py` | 汇总指标与瓶颈诊断 |
| `darl/dashboard.py` | 生成自包含单文件 HTML 仪表盘（原生 canvas，离线可开） |
| `darl/main.py` | 命令行入口 |

## 三种对比模式

1. **`sync` 同步采样**：所有 worker 用同一版本各跑一个 rollout，learner 收齐一轮
   （barrier）后才更新并广播，worker 全程等待最慢的那个。
2. **`async_no_correct` 异步无修正**：worker 持续采样、定期收新参数，learner 按批
   更新；旧策略数据直接使用，重要性权重恒为 1。
3. **`async_clip` 异步截断修正**：每条 transition 记录采样时的动作对数概率与策略版本，
   用重要性权重 `ratio = π_new(a|s)/π_old(a|s)` 并截断到 `[1-ε, 1+ε]`；
   滞后版本数超过 `lag_cap` 的数据直接丢弃。

## 运行

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m darl.tests        # 8 项核心机制回归测试
.venv/bin/python -m darl.main smoke   # 快速冒烟（~5 秒）
.venv/bin/python -m darl.main exp     # 完整实验（2 seed × 24 配置，约 3.5 分钟）
.venv/bin/python -m darl.main dashboard   # 重新生成 HTML
.venv/bin/python -m darl.main bottleneck --tag high/async_clip
```

产物：`results/runs.pkl`、`results/runs.json`、`results/dashboard.html`、`results/exp_log.txt`。

## 仪表盘页面

用浏览器打开 `results/dashboard.html`，包含：

1. 三种模式训练回报曲线（虚拟时间/更新次数轴可切，mild/high 异质可切）；
2. 数据滞后版本数分布直方图（含 lag_cap 丢弃区）；
3. 截断阈值与滞后上限扫描；
4. 各 worker 与 learner 的忙闲时间线热力图（worker 按速度排序）；
5. 参数广播的带宽–滞后权衡；
6. worker 规模扫描下的利用率与瓶颈；
7. 可排序汇总表。

## 关键设计点

- **滞后记账**：worker 持有的参数带版本号，rollout 上报时附带；learner 入队时
  计算 `lag = current_version - sampled_version`，逐条记录并参与滞后分布统计。
- **IS 截断**：策略梯度系数用截断后的权重，未截断 ratio 仅用于统计 clip 比例与 KL；
  梯度做全局范数裁剪，防止陈旧数据的梯度尖峰打飞参数。
- **参数广播**：
  - `pull`：worker 上报 rollout 时 piggyback 拉最新参数，无额外 RTT、零空闲等待，
    但慢 worker 长期用旧策略，滞后最高；
  - `push`：learner 按固定间隔广播，间隔越小滞后越低但广播字节越多，且占共享上行带宽。
- **worker 语义**：异步模式发送不阻塞，立即用当前参数继续采样，新参数在步骤间隙
  热替换；同步模式才在 barrier 处停下等待。
- **速度异质**：worker 速度服从对数正态（`mild` σ=0.3 / `high` σ=0.9 加两个 straggler），
  归一化到相同均值，保证两种场景总算力一致；每步耗时 gamma 分布，另有指数分布的偶发卡顿。

## 主要结论（见仪表盘与 exp_log，均为 120k 总环境步、2 个 seed 实测）

- **同步 vs 异步吞吐**：强异质（high）下同步 worker 利用率仅 14–17%、虚拟耗时 ~530s；
  异步两种模式 worker 利用率 92–93%、耗时 ~43s——同步被最慢的 straggler（速度比可达 77×）
  的 barrier 拖慢约 12 倍，但同步始终零滞后、零崩溃。
- **截断修正的稳定性价值**：强异质 + seed2 这一最严苛工况下，无修正异步策略崩溃
  （评估回报 -1.43，训练过半后 28 个评估点跌破 -1，熵坍缩到接近 0 且无法恢复）；
  截断修正（ε=0.1/0.2/0.3）全部收敛到 +0.83 以上、后期 0 崩溃。温和异质下两种异步都能收敛，
  说明偏差风险随滞后增大才显现——这与"落后多了就发散"的工程经验一致。
- **截断阈值 ε**：越紧单步 KL 越小、越稳；ε=0.1 在所有工况都稳定，ε=0.3 容忍度高但波动更大。
- **滞后上限 lag_cap**：cap=2 时丢弃率显著上升（约 800+ rollout）、有效样本变少；
  cap≥4 在本规模下滞后本就不高，差异不大。
- **learner 瓶颈随规模显现**：worker 16→128，learner 利用率 17%→97%；128 时队列堆积、
  平均滞后升到 6.2 版、训练崩溃——量化呈现"学习进程算不过来、采样进程持续产出旧数据"。
- **参数广播权衡**：pull 零额外 RTT、吞吐最高但滞后最高（~2.4 版）；push 间隔越小滞后越低
  （250ms→1.06 版）但广播字节越多；带宽降到 10Mbps 时广播本身阻塞 learner（利用率掉到 5%）、
  滞后 5.75、训练崩溃——带宽是 push 方案不可忽略的约束。

> 校准说明：默认学习率 1.5e-2、熵正则系数 0.02。熵正则偏弱时截断权重会压缩有效梯度尺度、
> 导致探索过早坍缩到坏策略，因此把熵系数从 0.01 提到 0.02；"崩溃"只统计训练过半后的评估点，
> 避免把早期随机策略的正常低分误判为发散。
