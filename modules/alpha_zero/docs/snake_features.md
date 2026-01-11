# Snake: 状态特征与动作空间（AlphaZero/MCTS 视角）

目标：给 `snake_gui` 的规则引擎（`Game`）配一套 **可被网络学习**、同时也 **便于 MCTS 枚举/回滚** 的 state/action 定义。

## 动作空间（推荐）

### 方案 A：相对动作（3 个）

- `TurnLeft`
- `Straight`
- `TurnRight`

为什么推荐：
- 分支因子从 4 降到 3，MCTS 更省算力。
- 天然避免“直接掉头”这种非法输入（`Direction::is_opposite`）。
- 与人类操作一致（相对方向更稳定），训练更容易收敛。

落地方式：
- 环境里保留当前朝向 `Direction`，把相对动作映射成下一步绝对 `Direction`。

### 方案 B：绝对动作（4 个）

- `Up/Down/Left/Right`

适用场景：
- 想最大程度复用现有输入/方向代码。

注意：
- 需要在 `legal_actions` 中屏蔽掉头方向；可选再屏蔽“立即撞墙/撞身”的动作（但一般留给环境推进去判负也可以）。

## 状态特征（网络输入）

Snake 是完全可观测网格世界，最稳妥的是“多通道棋盘张量”。

### 基础棋盘通道（强推荐）

对一个 $W \times H$ 的棋盘，构造 $C \times H \times W$：

- `head`：蛇头位置为 1
- `body`：蛇身（不含头）为 1
- `food`：食物位置为 1
- `walls`（可选）：边界为 1（也可以不加，因为越界会被规则判负，但加了更好学）

### 方向与回合信息（必须补充）

仅有棋盘通道，网络无法知道“下一步会往哪走”，需要加入当前朝向：

- `dir_onehot`：4 个通道（Up/Down/Left/Right），整张图广播为 1

（也可以用 4 维向量拼到 MLP，但 CNN 版本更常见）

### 训练更友好的增强特征（可选）

这些不是必须，但通常会显著提升学习效率：

- `tail` 通道：尾巴位置为 1（对“尾巴会移动”这种动态很有帮助）
- `distance-to-food` 类特征：例如 food 的相对方向（上/下/左/右）one-hot，或者 $(dx, dy)$ 标准化向量
- `danger` 三通道：对相对动作 Straight/Left/Right，标记“下一步是否立即死亡”（撞墙/撞身）

## 价值定义（Value, z）

对于单人 Snake（不对战）
- 胜负不是天然的二人零和；建议把终局价值定义为：
  - 撞死：-1
  - 达成 Victory（填满棋盘）：+1
  - 中途状态：由网络预测（训练信号来自终局回传）

如果要做 AlphaZero 风格的“对战”版本（双蛇/对抗）
- 价值可以回到经典零和：赢 +1 / 输 -1 / 平 0。

## 与现有 `snake_gui` 的对齐点

- 当前实现的核心状态：`snake: VecDeque<(i32,i32)>`, `direction`, `food`, `Grid(width,height)`
- 动作合法性：`queue_direction` 禁止掉头；`step` 内部判定 HitWall/HitSelf。

下一步实现环境时，建议做一层“可 clone 的 GameState”，把 `VecDeque` 和缓存都封装起来，避免 UI 依赖。
