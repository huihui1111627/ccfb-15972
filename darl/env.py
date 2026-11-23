# -*- coding: utf-8 -*-
"""离散决策环境: 5x5 GridWorld(虚拟时间仿真用)。

智能体从左上角出发,目标是右下角。每步:
  - 以 1-slip_prob 的概率执行所选动作,否则随机滑向另一个方向;
  - 撞墙留在原地;
  - 到达目标立即结束回合。
观测为 one-hot 位置向量(25 维),动作空间 4 个离散动作。
奖励设计(未折扣回报有界,便于对比训练曲线):
  每步 -0.01,到达目标 +1.0。
最优策略 8 步到达,最优回合回报约 0.92。
"""
import numpy as np

UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3
_ACTION_DELTA = {
    UP: (-1, 0),
    RIGHT: (0, 1),
    DOWN: (1, 0),
    LEFT: (0, -1),
}


class GridWorld:
    def __init__(self, size=8, slip_prob=0.15, seed=0):
        self.size = size
        self.slip_prob = slip_prob
        self.rng = np.random.default_rng(seed)
        self.n_actions = 4
        self.obs_dim = size * size
        self.start = (0, 0)
        self.goal = (size - 1, size - 1)
        self.pos = self.start
        self.steps = 0

    def reset(self):
        self.pos = self.start
        self.steps = 0
        return self._obs()

    def _obs(self):
        x = np.zeros(self.obs_dim, dtype=np.float64)
        r, c = self.pos
        x[r * self.size + c] = 1.0
        return x

    def step(self, action):
        self.steps += 1
        if self.rng.random() < self.slip_prob:
            action = int(self.rng.integers(self.n_actions))
        dr, dc = _ACTION_DELTA[action]
        nr, nc = self.pos[0] + dr, self.pos[1] + dc
        if 0 <= nr < self.size and 0 <= nc < self.size:
            self.pos = (nr, nc)
        done = self.pos == self.goal
        reward = 1.0 if done else -0.01
        # 兜底:防止极端情况下回合无限长
        if self.steps >= self.size * self.size * 4:
            done = True
        return self._obs(), reward, done
