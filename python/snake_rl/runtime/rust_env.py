from __future__ import annotations

from dataclasses import dataclass
import importlib
from pathlib import Path

import numpy as np


def _load_local_extension() -> object | None:
    repo_root = Path(__file__).resolve().parents[3]
    candidates = [
        repo_root / "target" / "debug" / "libsnake_core.so",
        repo_root / "target" / "release" / "libsnake_core.so",
        repo_root / "target" / "debug" / "snake_core.dll",
        repo_root / "target" / "release" / "snake_core.dll",
        repo_root / "target" / "debug" / "libsnake_core.dylib",
        repo_root / "target" / "release" / "libsnake_core.dylib",
    ]

    for lib_path in candidates:
        if not lib_path.exists():
            continue
        spec = importlib.util.spec_from_file_location("snake_core_py", lib_path)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        try:  # pragma: no cover - runtime dependent
            spec.loader.exec_module(module)
            return module
        except Exception:
            continue
    return None


def _load_snake_core_module() -> object | None:
    for module_name in ("snake_core_py", "snake_core"):
        try:  # pragma: no cover - runtime dependent
            return importlib.import_module(module_name)
        except ImportError:
            continue
    return _load_local_extension()


snake_core_py = _load_snake_core_module()


@dataclass
class EnvConfig:
    board_w: int
    board_h: int
    max_steps: int
    seed: int
    num_envs: int
    num_threads: int
    obs_channels: int


class RustBatchEnv:
    def __init__(self, config: dict) -> None:
        self.config = EnvConfig(
            board_w=int(config.get("board_w", 12)),
            board_h=int(config.get("board_h", 12)),
            max_steps=int(config.get("max_steps", 512)),
            seed=int(config.get("seed", 7)),
            num_envs=int(config.get("num_envs", 16)),
            num_threads=int(config.get("num_threads", 24)),
            obs_channels=int(config.get("obs_channels", 7)),
        )
        self._obs_shape = (
            self.config.num_envs,
            self.config.obs_channels,
            self.config.board_h,
            self.config.board_w,
        )

        if snake_core_py is not None:
            self._backend = snake_core_py.RustBatchEnv(
                self.config.board_w,
                self.config.board_h,
                self.config.max_steps,
                self.config.seed,
                self.config.num_envs,
                self.config.num_threads,
            )
            self._mode = "rust"
            if hasattr(self._backend, "obs_channels"):
                self.config.obs_channels = int(self._backend.obs_channels())
        else:
            self._backend = _PythonBatchEnv(self.config)
            self._mode = "python"

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def obs_shape(self) -> tuple[int, int, int, int]:
        return self._obs_shape

    def reset(self) -> np.ndarray:
        data, shape = self._backend.reset()
        self._obs_shape = tuple(shape)
        return np.asarray(data, dtype=np.float32).reshape(shape)

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        action_list = actions.astype(np.int64).tolist()
        if self._mode == "rust":
            obs, shape, rewards, dones, scores, legal_flat = self._backend.step(action_list)
            legal = np.asarray(legal_flat, dtype=np.uint8).reshape((self.config.num_envs, 4))
        else:
            obs, shape, rewards, dones, scores, legal = self._backend.step(action_list)

        self._obs_shape = tuple(shape)
        obs_np = np.asarray(obs, dtype=np.float32).reshape(shape)
        return (
            obs_np,
            np.asarray(rewards, dtype=np.float32),
            np.asarray(dones, dtype=np.bool_),
            {
                "score": np.asarray(scores, dtype=np.int32),
                "legal_actions_mask": np.asarray(legal, dtype=np.uint8),
            },
        )

    def reset_done(self) -> np.ndarray:
        return np.asarray(self._backend.reset_done(), dtype=np.int64)

    def get_obs_tensor(self) -> np.ndarray:
        data, shape = self._backend.get_obs_tensor()
        self._obs_shape = tuple(shape)
        return np.asarray(data, dtype=np.float32).reshape(shape)

    def seed(self, seed: int) -> None:
        self._backend.seed(int(seed))


class _PythonBatchEnv:
    def __init__(self, config: EnvConfig) -> None:
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.snakes: list[list[tuple[int, int]]] = []
        self.dirs: list[int] = []
        self.foods: list[tuple[int, int]] = []
        self.done: np.ndarray = np.zeros((config.num_envs,), dtype=np.bool_)
        self.scores: np.ndarray = np.zeros((config.num_envs,), dtype=np.int32)
        self.steps: np.ndarray = np.zeros((config.num_envs,), dtype=np.int32)
        self._reset_all()

    def reset(self) -> tuple[np.ndarray, tuple[int, int, int, int]]:
        self._reset_all()
        obs = self._obs()
        return obs.flatten(), obs.shape

    def step(
        self, actions: list[int]
    ) -> tuple[np.ndarray, tuple[int, int, int, int], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        rewards = np.full((self.config.num_envs,), -0.01, dtype=np.float32)
        legal = self._legal_mask()

        for i, action in enumerate(actions):
            if self.done[i]:
                rewards[i] = 0.0
                continue

            direction = int(action)
            if legal[i, direction] == 0:
                direction = self.dirs[i]
                rewards[i] -= 0.1
            self.dirs[i] = direction

            head_x, head_y = self.snakes[i][-1]
            delta = [(0, -1), (0, 1), (-1, 0), (1, 0)][direction]
            nx = head_x + delta[0]
            ny = head_y + delta[1]

            if nx < 0 or ny < 0 or nx >= self.config.board_w or ny >= self.config.board_h:
                self.done[i] = True
                rewards[i] -= 1.0
                continue

            next_cell = (nx, ny)
            eat = next_cell == self.foods[i]
            body = self.snakes[i][1:] if not eat else self.snakes[i]
            if next_cell in body:
                self.done[i] = True
                rewards[i] -= 1.0
                continue

            self.snakes[i].append(next_cell)
            if eat:
                self.scores[i] += 1
                rewards[i] += 1.0
                self.foods[i] = self._spawn_food(i)
            else:
                self.snakes[i].pop(0)

            self.steps[i] += 1
            if self.steps[i] >= self.config.max_steps:
                self.done[i] = True
                rewards[i] -= 0.2

        obs = self._obs()
        return obs.flatten(), obs.shape, rewards, self.done.copy(), self.scores.copy(), self._legal_mask()

    def reset_done(self) -> list[int]:
        ids: list[int] = []
        for i in range(self.config.num_envs):
            if self.done[i]:
                self._reset_one(i)
                ids.append(i)
        return ids

    def get_obs_tensor(self) -> tuple[np.ndarray, tuple[int, int, int, int]]:
        obs = self._obs()
        return obs.flatten(), obs.shape

    def seed(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)
        self._reset_all()

    def _reset_all(self) -> None:
        self.snakes.clear()
        self.dirs.clear()
        self.foods.clear()
        self.done[:] = False
        self.scores[:] = 0
        self.steps[:] = 0

        for i in range(self.config.num_envs):
            self._reset_one(i)

    def _reset_one(self, i: int) -> None:
        cx = self.config.board_w // 2
        cy = self.config.board_h // 2
        snake = [(cx - 1, cy), (cx, cy), (cx + 1, cy)]
        if i < len(self.snakes):
            self.snakes[i] = snake
            self.dirs[i] = 3
            self.done[i] = False
            self.scores[i] = 0
            self.steps[i] = 0
            self.foods[i] = self._spawn_food(i)
        else:
            self.snakes.append(snake)
            self.dirs.append(3)
            self.foods.append(self._spawn_food(i))

    def _spawn_food(self, i: int) -> tuple[int, int]:
        occupied = set(self.snakes[i])
        while True:
            x = int(self.rng.integers(0, self.config.board_w))
            y = int(self.rng.integers(0, self.config.board_h))
            if (x, y) not in occupied:
                return (x, y)

    def _legal_mask(self) -> np.ndarray:
        mask = np.ones((self.config.num_envs, 4), dtype=np.uint8)
        opposite = [1, 0, 3, 2]
        for i, direction in enumerate(self.dirs):
            mask[i, opposite[direction]] = 0
        return mask

    def _obs(self) -> np.ndarray:
        obs = np.zeros(
            (
                self.config.num_envs,
                self.config.obs_channels,
                self.config.board_h,
                self.config.board_w,
            ),
            dtype=np.float32,
        )
        for i in range(self.config.num_envs):
            for x, y in self.snakes[i][:-1]:
                obs[i, 1, y, x] = 1.0
            head_x, head_y = self.snakes[i][-1]
            obs[i, 0, head_y, head_x] = 1.0
            food_x, food_y = self.foods[i]
            obs[i, 2, food_y, food_x] = 1.0
            obs[i, 3 + self.dirs[i], :, :] = 1.0
        return obs
