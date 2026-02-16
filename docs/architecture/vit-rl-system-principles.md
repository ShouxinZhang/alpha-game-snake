# ViT + RL（AlphaZero 风格）Snake 系统原理（工程可推导版）

## 1. 系统目标与训练闭环总览

### 1.1 业务目标

本系统的目标不是“离线做一次实验”，而是构建一个可持续迭代的训练闭环：

- 高吞吐并行采样（Rust 批量环境）
- 稳定策略改进（ViT 策略价值网络 + 自博弈）
- 可观测可调参（JSONL 指标 + UI 面板）

核心链路：

```text
Env -> Planner -> Transition -> Replay -> SGD -> Updated Model
```

### 1.2 单次迭代 I/O

- 输入：
1. 当前模型参数 `theta_k`
2. 并行环境当前状态批次 `s_t`
3. 训练配置（MCTS、优化器、runtime）
- 输出：
1. 新参数 `theta_(k+1)`
2. 指标记录（reward/score/steps/loss/policy 分布等）
3. 周期性 checkpoint

代码入口：`python/snake_rl/trainers/train_loop.py`

## 2. MDP 形式化定义（面向实现）

我们把 Snake 建模为离散时间 MDP：

- 状态空间：`S`，每个状态由 `7 x H x W` 张量表示
- 动作空间：`A = {Up, Down, Left, Right}`，`|A| = 4`
- 转移：`P(s_(t+1) | s_t, a_t)`，由环境规则决定
- 奖励：`r_t = R(s_t, a_t, s_(t+1))`
- 折扣：`gamma in (0, 1]`（当前默认 `0.99`）

目标是最大化折扣回报：

```text
G_t = sum_(k=0..inf) gamma^k * r_(t+k)
```

episode 终止条件（任一满足）：

1. 撞墙
2. 撞到自身（考虑尾巴移动后的合法性）
3. 达到 `max_steps`

规则实现：`crates/snake-core/src/env.rs`

## 3. 状态编码与动作合法性

### 3.1 观测张量

每个环境输出：

```text
x_t in R^(7 x H x W)
```

7 个通道定义（与 `OBS_CHANNELS=7` 一致）：

1. `channel 0`：蛇头位置 one-hot
2. `channel 1`：蛇身位置（实现里包含蛇头位置，头部通道单独覆盖）
3. `channel 2`：食物位置 one-hot
4. `channel 3`：当前方向为 Up 时的全 1 平面
5. `channel 4`：当前方向为 Down 时的全 1 平面
6. `channel 5`：当前方向为 Left 时的全 1 平面
7. `channel 6`：当前方向为 Right 时的全 1 平面

代码：

- Rust 生成：`crates/snake-core/src/env.rs` `observation`
- Python fallback：`python/snake_rl/runtime/rust_env.py` `_obs`

### 3.2 合法动作掩码

`legal_actions_mask in {0,1}^4`，仅禁止“与当前方向相反”的动作：

```text
mask[opposite(dir)] = 0
```

这个掩码用于 planner 的 logits 屏蔽，避免策略学习到非法反向动作。

代码：

- `crates/snake-core/src/env.rs` `legal_actions_mask`
- `python/snake_rl/trainers/self_play.py` 中作为 `planner.plan` 输入

## 4. 环境动力学与奖励函数

每一步基础奖励初始化为 `-0.01`，然后叠加事件项。

| 事件 | 奖励增量 | 说明 |
|---|---:|---|
| 生存一步 | `-0.01` | 轻微步长惩罚，鼓励更短路径找食物 |
| 请求反向动作 | `-0.1` | 动作被替换为当前方向，防止抖动策略 |
| 吃到食物 | `+1.0` | 分数 +1，蛇增长并重生食物 |
| 撞墙/撞自身 | `-1.0` | 终止 episode |
| 达到 `max_steps` | `-0.2` | 终止 episode，抑制无效拖延 |

实现上单步总奖励示例：

- 吃到食物：`-0.01 + 1.0 = 0.99`
- 反向且随后撞墙：`-0.01 -0.1 -1.0 = -1.11`

### 并行语义：`reset_done`

批量环境中，`done=True` 的子环境不会阻塞其他环境：

1. `step` 后收集全批次 `done`
2. 调用 `reset_done` 只重置已结束环境
3. 其余环境持续 rollout

这能稳定并行吞吐，减少“全批等待最慢 episode”的问题。

代码：`crates/snake-core/src/batch.rs` `reset_done`

## 5. ViT 策略价值网络

代码：`python/snake_rl/models/vit_policy_value.py`

### 5.1 形状流转

设输入 `obs` 形状为 `[B, C, H, W]`，其中 `C=7`。

1. Patch Embedding（`Conv2d`，kernel=stride=`patch_size`）：
```text
[B, C, H, W] -> [B, D, H/P, W/P]
```
2. 展平为 token 序列：
```text
[B, D, H/P, W/P] -> [B, N, D], N=(H/P)*(W/P)
```
3. 加可学习位置编码 `pos_embed`
4. Transformer Encoder（`depth` 层）
5. token 均值池化 + `LayerNorm` 得到全局表示 `h in R^D`

### 5.2 双头输出

- 策略头：`pi_logits = W_p2 * GELU(W_p1 * h)`，输出维度 4
- 价值头：`v = tanh(W_v2 * GELU(W_v1 * h))`，输出范围 `[-1,1]`

`forward` 返回：

```text
policy_logits: [B, 4]
value: [B, 1]
```

## 6. Planner（当前实现）如何产出策略与动作

代码：`python/snake_rl/mcts/alpha_zero_mcts.py` `AlphaZeroPlanner.plan`

### 6.1 掩码 softmax + 温度

1. 非法动作 logits 置为极小值：
```text
masked_logits[a] = -1e9, if legal_mask[a] <= 0
```
2. 温度化分布：
```text
p = softmax(masked_logits / max(T, 1e-3))
```

温度 `T` 越低，分布越尖锐，动作更接近贪心。

### 6.2 训练态探索噪声

训练时引入 Dirichlet 噪声：

```text
n ~ Dirichlet(alpha * 1_4)
p <- 0.75 * p + 0.25 * n
p <- normalize(p * legal_mask)
```

作用：增加探索，降低早期策略塌缩风险。

### 6.3 多轮 sharpen（近似搜索）

当前实现不构建完整树结构，而做迭代分布锐化：

```text
for i in 1..max(simulations//16, 1):
    p <- normalize((p + 1e-8)^(1 + c_puct*0.05) * legal_mask)
```

直觉：当 `simulations` 或 `c_puct` 更大时，分布会逐步更集中到高概率动作，模拟“多次选择-强化”的效果。

### 6.4 动作采样

最终按分布采样：

```text
a_t ~ Categorical(p)
```

代码对应：`torch.multinomial(policy, num_samples=1)`

## 7. Self-Play 样本构造

代码：`python/snake_rl/trainers/self_play.py`

每条样本定义：

```text
Transition = (obs, target_policy, target_value)
```

- `obs`: 当前状态张量
- `target_policy`: planner 输出策略分布 `p_t`
- `target_value`: TD(0) bootstrap 目标

价值标签：

```text
y_t = r_t + gamma * V_theta(s_(t+1)) * (1 - done_t)
```

其中 `V_theta(s_(t+1))` 来自同一模型在 next state 的估计。

### 为什么不用完整蒙特卡洛回报

当前系统选择 TD(0) 的工程原因：

1. 并行环境短时自重置，完整轨迹回传管理更复杂
2. TD(0) 方差更低，适合高吞吐在线训练
3. 与当前“每步都可落地 replay”的数据管线更匹配

## 8. Replay Buffer 与采样机制

代码：`python/snake_rl/runtime/replay_buffer.py`

- 容器：`deque(maxlen=capacity)`，超出容量自动覆盖最旧样本
- 写入：`push_batch(transitions)` 批量追加
- 采样：`torch.randint` 随机下标（有放回），构建 mini-batch

作用：打散时间相关性，提升 SGD 样本多样性，减缓相邻状态导致的梯度偏置。

## 9. 训练目标函数与优化步骤

代码：`python/snake_rl/trainers/train_loop.py`

### 9.1 联合目标

对 mini-batch，模型输出 `policy_logits, pred_value`，定义：

策略损失（交叉熵形式）：

```text
L_policy = - E[ sum_a target_policy(a|s) * log softmax(policy_logits)_a ]
```

价值损失（MSE）：

```text
L_value = E[ (pred_value - target_value)^2 ]
```

总损失：

```text
L = L_policy + value_coef * L_value
```

其中 `value_coef` 来自配置 `optim.value_coef`。

### 9.2 优化器与稳定化

1. 优化器：`AdamW(lr, weight_decay)`
2. 混合精度：`torch.autocast` + `torch.amp.GradScaler`
3. 反向传播：`scaler.scale(loss).backward()`
4. 梯度裁剪：`clip_grad_norm_(..., grad_clip)`
5. `scaler.step(optimizer)` + `scaler.update()`

## 10. 端到端一次迭代的时序分解

以下流程对应 `train_loop.train` 的每个 `iter`：

1. `model.eval()`，切换到采样模式。  
2. `collect_self_play_batch` 读取 `env.get_obs_tensor()`。  
3. planner 基于当前状态输出 `policy` 和 `actions`。  
4. 环境执行 `env.step(actions)`，返回 `next_obs/reward/done/score/legal_mask`。  
5. 模型估计 `V(next_obs)`，构建 TD 目标 `y_t`。  
6. 生成并缓存 `Transition(obs, policy, y_t)`。  
7. 对已结束子环境调用 `env.reset_done()`。  
8. 本轮采样结束后，`buffer.push_batch(transitions)`。  
9. 切换 `model.train()`，执行 `updates_per_iter` 次参数更新。  
10. 每次更新：`sample -> forward -> loss -> backward -> clip -> step`。  
11. 汇总 rolling 指标并写入 `artifacts/metrics/latest.jsonl`。  
12. 若到间隔，保存 checkpoint 到 `checkpoints/`。  

源码入口映射：

- 迭代主循环：`python/snake_rl/trainers/train_loop.py` `train`
- 采样构造：`python/snake_rl/trainers/self_play.py` `collect_self_play_batch`
- planner：`python/snake_rl/mcts/alpha_zero_mcts.py` `plan`

## 11. 关键超参数与调参杠杆

| 参数 | 主要作用 | 调大典型影响 | 调小典型影响 |
|---|---|---|---|
| `simulations` | 控制 planner 锐化轮数 | 策略更集中，探索下降，CPU 成本上升 | 分布更平滑，探索多，决策稳定性下降 |
| `c_puct` | 控制 sharpen 指数强度 | 更偏向高概率动作 | 更保守，分布更均匀 |
| `dirichlet_alpha` | 探索噪声形状 | 噪声更均匀，扰动更强 | 噪声更尖锐，随机性集中 |
| `temperature` | softmax 温度 | 高温更随机 | 低温更贪心 |
| `gamma` | 未来回报权重 | 更看重长期收益 | 更看重即时奖励 |
| `batch_size` | 单次更新样本数 | 梯度更平滑，显存/延迟上升 | 更新更快但噪声增大 |
| `updates_per_iter` | 每轮采样后的学习强度 | 学习更充分但过拟合 replay 风险增大 | 学习不足，收敛变慢 |

工程排障（现象 -> 参数 -> 调整方向）：

| 现象 | 可能参数 | 建议方向 |
|---|---|---|
| `policy_*` 长期接近单一动作 | `temperature` 过低，`simulations/c_puct` 过高 | 提高 `temperature` 或降低锐化强度 |
| reward 不涨且 steps 偏高 | `gamma` 偏低、探索不足 | 适度提高 `gamma`，增强探索（温度/噪声） |
| loss 大幅抖动 | `batch_size` 偏小、lr 偏高 | 增大 `batch_size` 或降低 `lr` |
| buffer 很大但性能无提升 | `updates_per_iter` 偏低 | 适当提高更新步数 |

## 12. 指标解读与训练健康度判断

指标来源：`train_loop.py` 写入 `artifacts/metrics/latest.jsonl`

核心字段：

- `rolling_avg_reward`：窗口平均 reward，主目标趋势指标
- `avg_score`：窗口平均得分，反映吃食效率
- `avg_steps`：窗口平均存活步数，反映生存与拖延平衡
- `loss`：窗口平均训练损失
- `episodes_finished`：本迭代完成的 episode 数
- `policy_up/down/left/right`：策略分布均值

常见异常与定位：

1. 策略塌缩：`policy_right` 等单一维度长期接近 1  
优先检查 `temperature/simulations/c_puct`。  
2. `loss` 抖动大且 reward 不升  
优先检查 `lr`、`batch_size`、`grad_clip`。  
3. `avg_steps` 高但 `avg_score` 低  
说明“存活但不吃食”，优先检查探索强度与奖励设计。  

## 13. 附录：核心数据结构与接口速查

### 13.1 `RustBatchEnv.step`

位置：`python/snake_rl/runtime/rust_env.py`

输入：

- `actions: np.ndarray[num_envs]`，每项为 `0..3`

输出：

1. `obs: np.ndarray[num_envs, 7, H, W]`
2. `rewards: np.ndarray[num_envs]`
3. `dones: np.ndarray[num_envs]`
4. `info["score"]: np.ndarray[num_envs]`
5. `info["legal_actions_mask"]: np.ndarray[num_envs, 4]`

### 13.2 `ReplayBuffer.sample`

位置：`python/snake_rl/runtime/replay_buffer.py`

输入：

- `batch_size: int`
- `device: torch.device`

输出：

1. `obs: torch.Tensor[B, 7, H, W]`
2. `policy: torch.Tensor[B, 4]`
3. `value: torch.Tensor[B, 1]`

### 13.3 `collect_self_play_batch`

位置：`python/snake_rl/trainers/self_play.py`

输入：环境、planner、模型、device、`steps_per_iter`、`gamma`。  
输出：

1. `transitions: list[Transition]`
2. `metrics: SelfPlayMetrics`

### 13.4 `ViTPolicyValueNet.forward`

位置：`python/snake_rl/models/vit_policy_value.py`

输入：

- `obs: torch.Tensor[B, 7, H, W]`

输出：

1. `policy_logits: torch.Tensor[B, 4]`
2. `value: torch.Tensor[B, 1]`
