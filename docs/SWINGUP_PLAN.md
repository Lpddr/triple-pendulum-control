# Swing-up 训练设计与烟雾测试记录

目标：满足 HR 任务 —— ① 自然下垂/大角度起始 → ② 自主摆起 → ③ 保持直立 ≥ 10 s → ④ 同一策略。
10 s = 200 个连续控制步（`env.dt = 0.05`）。

---

## 1. 为什么必须改 env（物理上绕不开）

原任务定义下 `termination_height = 1.5` 会让任何下垂起始（`tip_y = -1.8`）**第 1 步就判负**，
episode 没有机会摆起。而已验证的作者模型是 balance-only（见 `AUTHOR_MODEL_CAPABILITY.md`），
fine-tune 路线已实测 5 次失败，原因是结构性的：

1. 下垂状态域里作者策略动作乱打（32% 饱和），没有蓄能先验可继承；
2. 大角度 reset 的梯度必然破坏直立附近的精准行为（错误 5 的"越训越差"）；
3. 原 reward 没有能量塑形项，从下垂起步时梯度 ≈ 0。

## 2. 设计（全部放在 `swingup=True` 开关后面）

**默认关闭，`swingup=False` 时与上游行为逐字节一致** —— 25 个单元测试保持通过，
作者模型照常可评估。新增构造参数：

```python
swingup: bool = False            # 总开关
reset_angle_limit: float | None  # 课程角度上限；None 时用上游 reset
energy_weight: float = 1.0       # 能量塑形权重
dist_weight: float = 0.2         # swingup 模式下 distance_penalty 权重
```

### 2.1 条件终止（`env.step`）

```python
if y > 1.7: self._ever_upright = True
terminated = self._ever_upright and (y <= termination_height)
```

效果：下垂起始不再首步判负，但"上去过又摔下来"仍然判负 —— 保留了跌倒信号。
实测随机策略 200 局中 62 局提前判负（都是上去过又掉的），而不是旧逻辑下的"~60% 首步必死"。

### 2.2 能量塑形（`env._get_rew`）

摆起的经典做法：先付"把能量推到直立对应能量"的奖励，再靠距离项把位形收上来。

```python
E        = KE + PE - 0.5 * m_cart * vx²    # 甩掉小车动能，防止靠抖车刷能量
E_target = 直立静止时的 E（init_qpos 处实测 100.944 J）
energy_penalty = ((E - E_target) / E_target)²
```

- `E_target` 在 `__init__` 里用 `set_state(init_qpos, 0)` 实测，XML 改了会自动跟上；
- 用 MuJoCo 的 `mjENBL_ENERGY` 标志读 `data.energy`，不手算质量矩阵；
- 甩掉小车动能是关键：小车限位 ±1 m、速度可达 7 m/s，其动能 (~260 J) 与目标能量同量级。

### 2.3 Reward 重排（swingup 模式）

```
r = alive_bonus - 0.2 * dist_penalty - vel_penalty - energy_penalty
alive_bonus = 10 if tip_y > 1.5 else 0
```

- **alive bonus 门控**：`tip_y > 1.5` 才发。否则下垂局白拿 10/步 × 1000 步 = 10000，
  策略躺平就能拿高分。
- **dist_penalty 权重降到 0.2**：原权重下垂时是 12.96，会淹没能量项（最大 4.15），
  策略会继续掉进"贪图 tip 高度"的局部最优，而不是先蓄能。
- 直立附近：energy_penalty → 0、dist → 0、alive = 10，与原任务 reward 一致。

### 2.4 课程 reset（`reset_model`）

```
50%  自然下垂链（θ₁ = π ± 0.15，其余 ±0.15）
50%  每个铰链 uniform(±reset_angle_limit)
qvel ~ N(0, 0.2)，cart U(-0.1, 0.1)
```

晋级：`--angle-limit 0.6 → 1.2 → 3.14`，晋级判据用
`tools/eval_policy.py` 的 "episodes passing 10 s ≥ 80%"。

## 3. 训练脚本改动（`sac_train.py`）

- `make_env` 传入 `swingup / reset_angle_limit`，**删掉了硬编码的 `termination_height=-999`**；
- 新增 argparse：`--total-timesteps / --angle-limit / --n-envs / --seed / --save-dir / --no-swingup`；
- **checkpoint 改存到 `runs/<save-dir>/sac_swingup_*.pt`**。
  原脚本把 `sac_triple_final.pt` 写到仓库根目录，这就是上次作者模型被覆盖的原因。
  现在不可能再覆盖。

## 4. 烟雾测试结果（20 万步，`--angle-limit 1.2`）

同一分布、关闭提前终止、50 局：

| 指标 | 随机策略基线 | 作者模型(hanging) | **烟雾 20 万步** |
|------|------------|-----------------|----------------|
| mean best tip_y | 1.632 | 1.476 | **1.718** |
| 动作饱和率(>0.95) | 32% | 32% | **6%** |
| 连续保持 >1.7 | 0.02 s | 0.00 s | 0.11 s |
| 通过 10 s | 0/30 | 0/50 | 0/50 |

**判读：设计成立，但 20 万步只够走 10% 的路。**

- 饱和率 32% → 6% 是决定性信号：随机/作者策略在下垂域都是乱打，
  训练后的策略已经学会"不瞎打满"，说明能量塑形给了可学的梯度；
- best tip_y 1.632 → 1.718，47/50 局能碰到 1.5 以上；
- 连续保持还没起来（0.11 s）—— 稳定阶段需要更多样本，这正是要跑完整训练的原因。

## 5. 正在进行

```
WANDB_MODE=offline python algorithms/sac/sac_train.py \
    --total-timesteps 2000000 --angle-limit 3.14 --save-dir runs/sac_swingup_pi
```

- 200 万步 ≈ 75 分钟（实测 ~440 SPS）；
- 每 25 万步存 `runs/sac_swingup_pi/sac_swingup_latest.pt`；
- 结束后用 `tools/eval_policy.py --checkpoint ... --start swingup --swingup` 逐档评估，
  拿真实的学习曲线（而不是 wandb return，后者混着 alive bonus 噪声）。

### 晋级/止损判据

| 里程碑 | 判据 | 不达标时的动作 |
|--------|------|--------------|
| 50 万步 | best tip_y 均值 > 1.75，饱和率 < 10% | 检查 reward 权重 |
| 100 万步 | 连续保持 > 2 s | 加大 `energy_weight` 或做两级课程 |
| 200 万步 | ≥ 30% 局通过 10 s | 继续加到 500 万步 |
| 500 万步 | ≥ 80% 局通过 10 s | 转 B 路线（改 XML 降物理难度） |

## 6. 评估命令备忘

```bash
# 训练分布（摆起任务口径）
python tools/eval_policy.py --checkpoint runs/sac_swingup_pi/sac_swingup_final.pt \
    --start swingup --swingup --episodes 100
# 平衡能力是否保住（近直立）
python tools/eval_policy.py --checkpoint runs/sac_swingup_pi/sac_swingup_final.pt \
    --start author --episodes 100 --terminate
```
