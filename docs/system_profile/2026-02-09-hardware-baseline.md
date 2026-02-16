# Hardware Baseline (2026-02-09)

- Probe Time: 2026-02-09 11:44:45
- CPU: AMD Ryzen 9 9955HX3D 16-Core Processor
- CPU Cores/Threads: 16/32
- CPU Max MHz: 5460.0
- RAM Total/Available: 60.51GB / 19.21GB
- GPU: NVIDIA GeForce RTX 5090 Laptop GPU
- GPU VRAM Total/Free: 23.89GB / 23.41GB
- PyTorch: 2.9.1+cu128
- CUDA Available: True (1 device)

## Recommended Runtime Cap

- env_concurrency: 16
- rust_num_threads: 24
- mcts_workers: 20
- train_batch_size: 1024
- replay_capacity: 1000000
- ui_fps: 20
- amp: True
- torch_compile: True
