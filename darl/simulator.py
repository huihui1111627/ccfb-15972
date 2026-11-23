# -*- coding: utf-8 -*-
"""虚拟时间事件驱动的分布式 RL 仿真器。

不真的 fork 进程,而是用一个最小堆按"虚拟时刻"调度四类事件:
  wstep  : 某个采样 worker 完成一个环境步
  recv   : 一条网络消息到达(learner 收 rollout / worker 收参数 / 广播定时器)
  update : learner 完成一次梯度计算
  stop   : 总环境步数耗尽,结束

这样可以:
  - 确定性复现(只依赖 seed);
  - 在完全相同的总环境步数(总计算量)下对比三种模式;
  - 显式建模链路带宽/传播时延、worker 速度异质与卡顿、learner 计算耗时;
  - 忙闲时间线、滞后分布等指标直接来自事件时间戳。
"""
import heapq
import warnings
from collections import defaultdict

import numpy as np

# macOS Accelerate/BLAS 在 matmul 上会误报 divide/overflow 等浮点警告,
# 但结果有限且正确(纯随机矩阵相乘也会触发)。过滤这组误报,
# 真正的数值问题仍由结果有限性校验捕获。
warnings.filterwarnings(
    "ignore",
    message=".*encountered in matmul",
    category=RuntimeWarning,
)

from .env import GridWorld
from .policy import (
    clone_params,
    init_params,
    init_adam_state,
    params_bytes,
    sample_action,
    greedy_action,
    forward as fwd,
    update as policy_update,
)

BUCKET_MS = 100.0          # 忙闲时间线的时间桶宽度


# --------------------------------------------------------------------------
# 工具函数
# --------------------------------------------------------------------------
def compute_rtg(rewards, dones, last_bootstrap, gamma):
    """按 done 切段计算折扣 reward-to-go,末段用 critic 自举。"""
    n = len(rewards)
    rtg = np.zeros(n, dtype=np.float64)
    ends = [i for i, d in enumerate(dones) if d]
    bounds = ends + [n - 1]
    start = 0
    for end in bounds:
        running = 0.0 if dones[end] else last_bootstrap
        for i in range(end, start - 1, -1):
            running = rewards[i] + gamma * running
            rtg[i] = running
        start = end + 1
    return rtg


class Link:
    """带宽受限的串行链路(传输完成时刻 = max(当前时刻, 链路空闲时刻) + 时延)。"""

    def __init__(self, bandwidth_mbps):
        self.bytes_per_ms = bandwidth_mbps * 1_000_000.0 / 8.0 / 1000.0
        self.free_at = 0.0

    def schedule(self, now, nbytes, rtt_ms):
        tx = nbytes / self.bytes_per_ms
        arrive = max(now, self.free_at) + tx + rtt_ms
        self.free_at = arrive - rtt_ms
        return arrive


class Worker:
    __slots__ = ("wid", "env", "rng", "speed", "params", "version",
                 "buf", "steps_in_rollout", "last_bucket")

    def __init__(self, wid, env, rng, speed, params):
        self.wid = wid
        self.env = env
        self.rng = rng
        self.speed = speed
        self.params = params
        self.version = 0
        self.buf = None
        self.steps_in_rollout = 0
        self.last_bucket = -1


# --------------------------------------------------------------------------
# 仿真器
# --------------------------------------------------------------------------
class Simulator:

    def __init__(self, cfg):
        self.cfg = cfg
        c = cfg
        self.rng = np.random.default_rng(c.seed)
        self.eval_rng = np.random.default_rng(c.seed + 999_983)

        # 策略: learner 持有主副本
        self.params = init_params(64, 4, hidden=c.hidden, seed=c.seed)
        self.param_size = params_bytes(self.params)
        self.adam = init_adam_state(self.params)
        self.adam_t = 0
        self.version = 0
        self.snapshots = {0: clone_params(self.params)}

        # 事件堆
        self.heap = []
        self._seq = 0
        self.now = 0.0

        # 网络: learner 上下行 + 每个 worker 一条接入链路
        self.learner_up = Link(c.bandwidth_mbps)
        self.learner_down = Link(c.bandwidth_mbps)
        self.worker_links = [Link(c.bandwidth_mbps) for _ in range(c.n_workers)]

        # worker
        self.workers = []
        self._make_workers()

        # learner 经验队列(异步模式)
        self.exp_queue = []          # 每条: (obs, act, old_logp, adv_target, rtg, version)
        self.learner_busy_until = None
        self.learner_last_bucket = -1

        # sync 模式 barrier 簿记
        self.round = 0
        self.round_received = 0
        self.round_rollouts = []

        # push 模式簿记
        self.push_pending_version = None
        self.push_timer_armed = False

        # 指标
        self.total_steps = 0
        self.rollout_count = 0
        self.dropped_rollouts = 0
        self.dropped_transitions = 0
        self.dropped_lags = []
        self.lag_samples = []
        self.episode_returns = []          # (time, return)
        self.eval_curve = []               # (time, update_no, mean_return)
        self.update_log = []               # 每次更新的诊断字典
        self.worker_busy = [defaultdict(float) for _ in range(c.n_workers)]
        self.worker_rollouts = [0] * c.n_workers
        self.learner_busy_buckets = defaultdict(float)
        self.worker_stall_time = [0.0] * c.n_workers
        self.bytes_pushed = 0
        self.broadcast_count = 0

    # ------------------------------------------------------------------
    # 初始化
    # ------------------------------------------------------------------
    def _make_workers(self):
        c = self.cfg
        if c.heterogeneity == "mild":
            speeds = np.exp(self.rng.normal(0.0, 0.3, size=c.n_workers))
        elif c.heterogeneity == "high":
            speeds = np.exp(self.rng.normal(0.0, 0.9, size=c.n_workers))
            # 两个常驻 straggler
            speeds[0] = 0.15
            speeds[1] = 0.2
        else:
            raise ValueError(c.heterogeneity)
        speeds = speeds / speeds.mean()       # 归一化: 总算力两种场景一致,对比才公平
        for i in range(c.n_workers):
            env = GridWorld(seed=c.seed * 10_003 + i)
            rng = np.random.default_rng(c.seed * 7919 + i + 1)
            w = Worker(i, env, rng, float(speeds[i]), clone_params(self.params))
            self.workers.append(w)

    def schedule(self, time, kind, *payload):
        heapq.heappush(self.heap, (time, self._seq, kind, payload))
        self._seq += 1

    # ------------------------------------------------------------------
    # 忙闲记录(时间桶)
    # ------------------------------------------------------------------
    def _mark(self, bucket_map, t0, t1, busy):
        if t1 <= t0:
            return
        b0 = int(t0 // BUCKET_MS)
        b1 = int((t1 - 1e-9) // BUCKET_MS)
        for b in range(b0, b1 + 1):
            lo = max(t0, b * BUCKET_MS)
            hi = min(t1, (b + 1) * BUCKET_MS)
            if hi > lo:
                bucket_map[b] += (hi - lo) if busy else 0.0
        # 空闲 = 桶宽 - 忙(后处理时计算)

    def _mark_worker(self, w, t0, t1, busy):
        self._mark(self.worker_busy[w.wid], t0, t1, busy)

    def _mark_learner(self, t0, t1, busy):
        self._mark(self.learner_busy_buckets, t0, t1, busy)

    # ------------------------------------------------------------------
    # 网络
    # ------------------------------------------------------------------
    def _worker_to_learner(self, wid, nbytes, kind, *payload):
        """worker 上行: 占 worker 接入链路 + learner 下行链路。"""
        link = self.worker_links[wid]
        depart = max(self.now, link.free_at)
        arrive_link = depart + nbytes / link.bytes_per_ms
        link.free_at = arrive_link
        # learner 下行汇聚
        arrive = self.learner_down.schedule(arrive_link, nbytes, 0.0)
        self.schedule(arrive + self.cfg.rtt_ms, "recv", "learner",
                      (kind, wid) + payload)

    def _learner_to_worker(self, wid, nbytes, kind, *payload):
        """learner 下行: 占 learner 共享上行 + worker 接入链路。"""
        depart_up = max(self.now, self.learner_up.free_at)
        up_done = depart_up + nbytes / self.learner_up.bytes_per_ms
        self.learner_up.free_at = up_done
        link = self.worker_links[wid]
        arrive = link.schedule(up_done, nbytes, self.cfg.rtt_ms)
        self.schedule(arrive, "recv", ("worker", wid), (kind,) + payload)

    # ------------------------------------------------------------------
    # worker 生命周期
    # ------------------------------------------------------------------
    def start_rollout(self, wid):
        w = self.workers[wid]
        w.buf = {"obs": [], "act": [], "logp": [], "rew": [],
                 "val": [], "done": []}
        w.steps_in_rollout = 0
        obs = w.env.reset()
        self._schedule_step(w, obs)

    def _schedule_step(self, w, obs):
        c = self.cfg
        t_start = self.now
        stall = 0.0
        if w.rng.random() < c.stall_prob:
            stall = float(w.rng.exponential(c.stall_mean_ms))
            self.worker_stall_time[w.wid] += stall
            self._mark_worker(w, t_start, t_start + stall, busy=False)
            t_start += stall
        mean_dt = c.base_step_time_ms / w.speed
        shape = 1.0 / (c.step_time_cv ** 2)
        dt = float(w.rng.gamma(shape, mean_dt / shape))
        self._mark_worker(w, t_start, t_start + dt, busy=True)
        self.schedule(t_start + dt, "wstep", w.wid, obs)

    def on_wstep(self, wid, obs):
        w = self.workers[wid]
        c = self.cfg
        action, logp, _, value = sample_action(w.params, obs, w.rng)
        next_obs, reward, done = w.env.step(action)
        buf = w.buf
        buf["obs"].append(obs)
        buf["act"].append(action)
        buf["logp"].append(logp)
        buf["rew"].append(reward)
        buf["val"].append(value)
        buf["done"].append(done)
        w.steps_in_rollout += 1
        self.total_steps += 1

        if done:
            self.episode_returns.append((self.now, self._episode_return(buf)))

        if w.steps_in_rollout >= c.rollout_len or done or \
                self.total_steps >= c.total_env_steps:
            self._finish_rollout(w)
            return
        self._schedule_step(w, next_obs)

    @staticmethod
    def _episode_return(buf):
        # 本次 done 对应的整条 episode 回报: 简化为当前 rollout 内
        # 最后一个 done 段的奖励和(episode 通常远短于 rollout)
        rews, dones = buf["rew"], buf["done"]
        end = len(rews) - 1
        start = 0
        for i in range(end - 1, -1, -1):
            if dones[i]:
                start = i + 1
                break
        return float(sum(rews[start:end + 1]))

    def _finish_rollout(self, w):
        c = self.cfg
        buf = w.buf
        # 未自然结束的末段用最后一个状态的 critic 值自举
        last_bootstrap = 0.0 if buf["done"][-1] else \
            fwd(w.params, buf["obs"][-1])[2]
        rtg = compute_rtg(np.array(buf["rew"]), np.array(buf["done"]),
                          last_bootstrap, c.gamma)
        payload = (w.version, w.steps_in_rollout,
                   np.array(buf["obs"]), np.array(buf["act"]),
                   np.array(buf["logp"]), np.array(buf["val"]),
                   rtg, np.array(buf["done"]))
        nbytes = int(buf["obs"][0].shape[0] * w.steps_in_rollout * 8 * 4)
        self.worker_rollouts[w.wid] += 1
        self.rollout_count += 1
        self._worker_to_learner(w.wid, nbytes, "rollout", *payload)
        w.buf = None
        if c.mode == "sync":
            # 同步模式: 停下等 learner 收齐一轮后广播新参数,再由
            # on_worker_recv 唤醒(这段等待时间计入空闲)
            return
        # 异步模式: 发送不阻塞,worker 立刻用当前持有的参数继续采样;
        # 新参数到达时在步骤间隙热替换(on_worker_recv)。
        self.start_rollout(w.wid)

    # ------------------------------------------------------------------
    # learner 收消息
    # ------------------------------------------------------------------
    def on_learner_recv(self, msg):
        kind, wid = msg[0], msg[1]
        if kind != "rollout":
            return
        if self.total_steps > self.cfg.total_env_steps:
            return
        (version, n, obs, act, logp, val, rtg, dones) = msg[2:]
        if self.cfg.mode == "sync":
            self._on_rollout_sync(wid, version, n, obs, act, logp, val, rtg)
            return

        lag = self.version - version
        if lag > self.cfg.lag_cap:
            self.dropped_rollouts += 1
            self.dropped_transitions += n
            self.dropped_lags.append(lag)
        else:
            adv = rtg - val
            for i in range(n):
                self.exp_queue.append(
                    (obs[i], act[i], logp[i], adv[i], rtg[i], version))
        # pull 模式:piggyback 带回最新参数,不额外花 RTT
        if self.cfg.param_mode == "pull":
            self._learner_to_worker(
                wid, self.param_size, "params",
                self.version, self.snapshots[self.version])
            self.bytes_pushed += self.param_size
            self.broadcast_count += 1
        self._maybe_dispatch_update()

    # ------------------------------------------------------------------
    # 异步 learner 批处理
    # ------------------------------------------------------------------
    def _maybe_dispatch_update(self):
        c = self.cfg
        if self.learner_busy_until is not None and \
                self.now < self.learner_busy_until:
            return
        # 丢弃队列中滞后超限的残留数据
        kept = []
        for tr in self.exp_queue:
            if self.version - tr[5] <= c.lag_cap:
                kept.append(tr)
            else:
                self.dropped_transitions += 1
                self.dropped_lags.append(self.version - tr[5])
        self.exp_queue = kept
        if len(self.exp_queue) < c.batch_size:
            return
        batch = self.exp_queue[:c.batch_size]
        self.exp_queue = self.exp_queue[c.batch_size:]
        self._run_update(batch)

    def _run_update(self, batch):
        c = self.cfg
        obs = np.stack([b[0] for b in batch])
        act = np.array([b[1] for b in batch], dtype=np.int64)
        old_logp = np.array([b[2] for b in batch])
        adv = np.array([b[3] for b in batch])
        rtg = np.array([b[4] for b in batch])
        versions = np.array([b[5] for b in batch])
        lags = self.version - versions
        self.lag_samples.extend(lags.tolist())

        t0 = self.now
        self.adam_t += 1
        stats = policy_update(
            self.params, self.adam, obs, act, old_logp, adv, rtg,
            lr=c.lr, clip_eps=c.clip_eps, ent_coef=c.ent_coef,
            t=self.adam_t, use_is_correction=(c.mode == "async_clip"))
        t1 = t0 + c.learner_update_ms
        self.learner_busy_until = t1
        self._mark_learner(t0, t1, busy=True)

        old_version = self.version
        self.version += 1
        self.snapshots[self.version] = clone_params(self.params)
        # 只保留近期快照,控制内存
        if len(self.snapshots) > c.lag_cap + 4:
            oldest = min(self.snapshots)
            del self.snapshots[oldest]

        self.update_log.append({
            "t": t1, "version": self.version,
            "lag_mean": float(lags.mean()), "lag_max": int(lags.max()),
            "queue_len": len(self.exp_queue),
            "ratio": stats.mean_ratio, "clip_frac": stats.clip_frac,
            "kl": stats.approx_kl, "entropy": stats.entropy,
            "vloss": stats.value_loss,
        })

        # 参数分发
        if c.mode == "sync":
            pass  # 由 barrier 统一广播
        elif c.param_mode == "pull":
            pass  # worker 下次上报时 piggyback
        else:  # push
            self._on_push_after_update(old_version)

        # 定时评估(零耗时,只占用 learner 之外的统计资源)
        if len(self.update_log) % c.eval_every_updates == 0:
            ret = self._evaluate()
            self.eval_curve.append((t1, len(self.update_log), ret))

        self.schedule(t1, "update",)

    def on_update_done(self):
        self.learner_busy_until = None
        self._maybe_dispatch_update()

    # ------------------------------------------------------------------
    # push 广播
    # ------------------------------------------------------------------
    def _on_push_after_update(self, old_version):
        c = self.cfg
        self.push_pending_version = self.version
        if not self.push_timer_armed:
            self.push_timer_armed = True
            self.schedule(self.now + c.push_interval_ms, "recv",
                          "push_timer", None)

    def on_push_timer(self):
        c = self.cfg
        self.push_timer_armed = False
        # 预算耗尽后不再重挂,否则定时器事件永不枯竭,仿真无法终止
        if self.total_steps >= c.total_env_steps:
            return
        if self.push_pending_version is not None:
            ver = self.push_pending_version
            self._broadcast(ver)
            self.push_pending_version = None
        # 重新挂定时器;若期间又有新版本,定时器到点即推
        self.push_timer_armed = True
        self.schedule(self.now + c.push_interval_ms, "recv",
                      "push_timer", None)

    def _broadcast(self, version):
        params = self.snapshots[version]
        for w in self.workers:
            self._learner_to_worker(w.wid, self.param_size, "params",
                                    version, params)
        self.bytes_pushed += self.param_size * self.cfg.n_workers
        self.broadcast_count += 1

    # ------------------------------------------------------------------
    # sync 模式 barrier
    # ------------------------------------------------------------------
    def _on_rollout_sync(self, wid, version, n, obs, act, logp, val, rtg):
        c = self.cfg
        if version != self.round:
            # 迟到的上一轮数据(理论上不会发生,防御性丢弃)
            self.dropped_rollouts += 1
            return
        self.round_received += 1
        adv = rtg - val
        self.round_rollouts.append(
            (obs, act, logp, adv, rtg, np.full(n, version, dtype=np.int64)))
        if self.round_received < c.n_workers:
            return

        # barrier 到齐: 合并整轮数据,按 batch 串行更新。
        # 滞后样本由 _run_update 统一记录(sync 下版本一致, lag 全为 0)。
        batch = []
        for (o, a, lp, ad, r, ver) in self.round_rollouts:
            for i in range(len(a)):
                batch.append((o[i], a[i], lp[i], ad[i], r[i], int(ver[i])))
        self.round_rollouts = []
        self.round_received = 0
        # adam_t 由 _run_update 内部统一自增,这里不要重复计数
        # 一轮数据可能够多个 batch,全部更新完再广播(串行计算)
        idx = 0
        update_no = 0
        while idx + c.batch_size <= len(batch):
            mini = batch[idx:idx + c.batch_size]
            idx += c.batch_size
            self._run_update(mini)
            update_no += 1
        if update_no == 0:
            # batch 比一轮数据还大时,至少更新一次
            self._run_update(batch)
        # 广播新参数,所有 worker 收到后才能开下一轮(天然 barrier)
        self.round = self.version
        self._broadcast(self.version)

    # ------------------------------------------------------------------
    # worker 收消息
    # ------------------------------------------------------------------
    def on_worker_recv(self, wid, msg):
        kind = msg[0]
        if kind != "params":
            return
        _, version, params = msg
        w = self.workers[wid]
        w.params = params
        w.version = version
        if w.buf is None:
            # worker 正空闲等待参数:收到后立刻开跑
            self.start_rollout(wid)
        # 否则正在 rollout 中:热替换参数,当前 rollout 用新版本继续
        # (真实系统里参数也是在步骤间隙原子生效,不会打断轨迹)

    # ------------------------------------------------------------------
    # 评估
    # ------------------------------------------------------------------
    def _evaluate(self):
        c = self.cfg
        env = GridWorld(seed=int(self.eval_rng.integers(1 << 30)))
        returns = []
        for _ in range(c.eval_episodes):
            obs = env.reset()
            ep_ret, done = 0.0, False
            guard = 0
            while not done and guard < 200:
                a, _ = greedy_action(self.params, obs)
                obs, r, done = env.step(a)
                ep_ret += r
                guard += 1
            returns.append(ep_ret)
        return float(np.mean(returns))

    # ------------------------------------------------------------------
    # 主循环
    # ------------------------------------------------------------------
    def run(self):
        c = self.cfg
        for w in self.workers:
            self.start_rollout(w.wid)

        while self.heap:
            t, _, kind, payload = heapq.heappop(self.heap)
            self.now = t
            # 预算已耗尽: 丢弃后续所有事件(含在途的参数/定时器),直接收尾
            if self.total_steps >= c.total_env_steps:
                break
            if kind == "wstep":
                wid, obs = payload
                self.on_wstep(wid, obs)
            elif kind == "recv":
                target, msg = payload
                if target == "learner":
                    self.on_learner_recv(msg)
                elif target == "push_timer":
                    self.on_push_timer()
                else:
                    role, wid = target
                    self.on_worker_recv(wid, msg)
            elif kind == "update":
                self.on_update_done()

            if self.total_steps >= c.total_env_steps:
                break

        # 真正的数值健康校验(区别于 BLAS 误报)
        bad = [k for k, v in self.params.items()
               if not np.isfinite(v).all()]
        if bad:
            raise FloatingPointError(
                "learner 参数出现非有限值: %s —— 训练发散" % ",".join(bad))
        return self.build_results()

    # ------------------------------------------------------------------
    # 结果汇总
    # ------------------------------------------------------------------
    def build_results(self):
        c = self.cfg
        horizon = self.now

        # worker 忙闲(时间桶 -> 忙占比)
        n_buckets = max(1, int(horizon // BUCKET_MS) + 1)
        worker_util = np.zeros((c.n_workers, n_buckets))
        for i, bm in enumerate(self.worker_busy):
            for b, busy_ms in bm.items():
                if b < n_buckets:
                    worker_util[i, b] = min(1.0, busy_ms / BUCKET_MS)
        learner_util = np.zeros(n_buckets)
        for b, busy_ms in self.learner_busy_buckets.items():
            if b < n_buckets:
                learner_util[b] = min(1.0, busy_ms / BUCKET_MS)

        lags = np.array(self.lag_samples if self.lag_samples else [0])
        dropped = np.array(self.dropped_lags if self.dropped_lags else [])

        # 平滑训练回报
        ep = np.array([r for _, r in self.episode_returns])
        ep_t = np.array([t for t, _ in self.episode_returns])

        return {
            "config": c.to_dict(),
            "horizon_ms": horizon,
            "total_steps": self.total_steps,
            "n_updates": len(self.update_log),
            "n_rollouts": self.rollout_count,
            "dropped_rollouts": self.dropped_rollouts,
            "dropped_transitions": self.dropped_transitions,
            "broadcast_count": self.broadcast_count,
            "bytes_pushed": self.bytes_pushed,
            "worker_speeds": [w.speed for w in self.workers],
            "worker_rollouts": self.worker_rollouts,
            "worker_stall_ms": self.worker_stall_time,
            "worker_util": worker_util,
            "learner_util": learner_util,
            "bucket_ms": BUCKET_MS,
            "lag_mean": float(lags.mean()),
            "lag_p50": float(np.percentile(lags, 50)),
            "lag_p90": float(np.percentile(lags, 90)),
            "lag_max": int(lags.max()),
            "lag_hist": np.bincount(lags.astype(int),
                                    minlength=c.lag_cap + 2).tolist(),
            "dropped_lag_mean": float(dropped.mean()) if dropped.size else 0.0,
            "episode_t": ep_t,
            "episode_return": ep,
            "eval_t": np.array([x[0] for x in self.eval_curve]),
            "eval_update": np.array([x[1] for x in self.eval_curve]),
            "eval_return": np.array([x[2] for x in self.eval_curve]),
            "update_log": self.update_log,
            "final_return": float(self.eval_curve[-1][2])
            if self.eval_curve else float(ep[-10:].mean() if ep.size else 0.0),
        }
