# Repository Structure

## 目录结构

<!-- REPO-TREE-START -->
```
REPO/
├── .agents/                                 # AI Agent 配置与技能根目录
│   └── skills/                              # Agent 技能集合目录
│       ├── build-check/                     # 构建质量门禁技能
│       ├── dependency-review-system/        # 依赖驱动的 AI Review 调度技能
│       ├── dev-logs/                        # 开发日志记录技能
│       ├── domain-data-update/              # 领域数据维护技能
│       ├── git-management/                  # Git 管理技能
│       ├── local-dev-workflow/              # 本地开发全链路 SOP 技能
│       ├── modularization-governance/       # 模块化量化治理技能
│       └── repo-structure-sync/             # 仓库结构文档同步技能
├── configs/                                 # 训练与运行配置文件目录
│   └── train/                               # RL 训练超参数配置目录
│       └── vit_mae_icm_ppo.yaml             # ViT-MAE+ICM+PPO 训练超参数配置
├── crates/                                  # Rust crate 工作区成员目录
│   ├── snake-core/                          # 贪吃蛇 RL 环境核心引擎 crate
│   │   ├── src/                             # snake-core 源代码目录
│   │   ├── tests/                           # snake-core 集成测试目录
│   │   └── Cargo.toml                       # snake-core 包配置与 PyO3 特性声明
│   └── snake-ui/                            # egui 可视化监控面板 crate
│       ├── src/                             # snake-ui 源代码目录
│       └── Cargo.toml                       # snake-ui 包配置与 egui 依赖声明
├── docs/                                    # 项目文档根目录
│   ├── architecture/                        # 架构文档与仓库元数据
│   │   ├── repo-metadata.json               # 仓库路径元数据 JSON 快照
│   │   └── repository-structure.md          # 自动生成的仓库目录树文档
│   └── dev_logs/                            # 开发日志按日期归档
├── python/                                  # Python 训练代码根目录
│   └── snake_rl/                            # Snake RL 训练包
│       ├── models/                          # RL 模型定义模块
│       ├── runtime/                         # 运行时环境适配层
│       ├── trainers/                        # 训练流程与损失计算模块
│       └── __init__.py                      # snake_rl 包初始化
├── scripts/                                 # 工程脚本与工具集
│   ├── repo-metadata/                       # 仓库元数据管理工具集(SQLite)
│   │   ├── lib/                             # 核心库: SQLite 操作与工具函数
│   │   ├── scripts/                         # 元数据 CLI 子命令脚本
│   │   ├── sql/                             # PostgreSQL DDL 脚本 (已废弃)
│   │   ├── mcp-server.mjs                   # 元数据 MCP Server(stdio)
│   │   ├── package.json                     # repo-metadata Node.js 依赖
│   │   └── README.md                        # repo-metadata 使用说明
│   ├── review/                              # 依赖关系代码评审流水线
│   │   ├── config/                          # 评审策略配置目录
│   │   ├── input/                           # LLM 评审报告输入目录
│   │   ├── scripts/                         # 评审流水线子步骤脚本
│   │   ├── templates/                       # 评审报告模板目录
│   │   ├── mcp-server.mjs                   # Review MCP Server(stdio)
│   │   ├── README.md                        # Review 流水线使用说明
│   │   └── run.sh                           # 一键执行评审流水线入口
│   ├── tools/                               # Python 质量检查工具集
│   │   └── check_errors/                    # Python 代码错误检查模块
│   └── check_errors.sh                      # 通用质量门禁检查脚本
├── .gitignore                               # Git 版本控制忽略规则
├── AGENTS.md                                # AI 编码代理工作流与规范指令
├── Cargo.lock                               # Rust 依赖精确版本锁定文件
├── Cargo.toml                               # Rust workspace 根配置与依赖声明
├── restart.sh                               # 一键重启训练与 GUI 脚本
└── test.txt                                 # 临时测试文件
```
<!-- REPO-TREE-END -->
