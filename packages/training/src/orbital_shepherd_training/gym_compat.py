from __future__ import annotations

from collections.abc import Mapping
from typing import Any

try:  # pragma: no cover - exercised when gymnasium is installed.
    import gymnasium as gym
    from gymnasium import spaces

    Env = gym.Env
    Wrapper = gym.Wrapper
    ObservationWrapper = gym.ObservationWrapper
    GYMNASIUM_AVAILABLE = True
except ModuleNotFoundError:  # pragma: no cover - covered by local tests.
    GYMNASIUM_AVAILABLE = False

    class Space:
        def __init__(self, *, shape: tuple[int, ...], dtype: str) -> None:
            self.shape = shape
            self.dtype = dtype

        def contains(self, value: Any) -> bool:
            del value
            return True

    class Box(Space):
        def __init__(
            self,
            low: float | int,
            high: float | int,
            *,
            shape: tuple[int, ...],
            dtype: str,
        ) -> None:
            super().__init__(shape=shape, dtype=dtype)
            self.low = low
            self.high = high

        def contains(self, value: Any) -> bool:
            return _shape_of(value) == self.shape

    class Discrete(Space):
        def __init__(self, n: int) -> None:
            if n <= 0:
                raise ValueError("Discrete space requires n >= 1")
            super().__init__(shape=(), dtype="int64")
            self.n = n

        def contains(self, value: Any) -> bool:
            return isinstance(value, int) and 0 <= value < self.n

    class Dict(Space):
        def __init__(self, spaces_by_key: Mapping[str, Space]) -> None:
            self.spaces = dict(spaces_by_key)
            super().__init__(shape=(), dtype="dict")

        def __getitem__(self, key: str) -> Space:
            return self.spaces[key]

        def contains(self, value: Any) -> bool:
            if not isinstance(value, Mapping):
                return False
            return all(
                key in value and space.contains(value[key])
                for key, space in self.spaces.items()
            )

    class _SpacesModule:
        Box = Box
        Dict = Dict
        Discrete = Discrete

    spaces = _SpacesModule()

    class Env:
        metadata: dict[str, Any] = {}
        action_space: Space
        observation_space: Space

        def reset(
            self,
            *,
            seed: int | None = None,
            options: dict[str, Any] | None = None,
        ) -> tuple[Any, dict[str, Any]]:
            del seed, options
            raise NotImplementedError

        def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
            del action
            raise NotImplementedError

        def close(self) -> None:
            return None

        @property
        def unwrapped(self) -> Env:
            return self

    class Wrapper(Env):
        def __init__(self, env: Env) -> None:
            self.env = env
            self.action_space = env.action_space
            self.observation_space = env.observation_space
            self.metadata = getattr(env, "metadata", {})

        def reset(
            self,
            *,
            seed: int | None = None,
            options: dict[str, Any] | None = None,
        ) -> tuple[Any, dict[str, Any]]:
            return self.env.reset(seed=seed, options=options)

        def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
            return self.env.step(action)

        def close(self) -> None:
            self.env.close()

        @property
        def unwrapped(self) -> Env:
            return self.env.unwrapped

    class ObservationWrapper(Wrapper):
        def observation(self, observation: Any) -> Any:
            return observation

        def reset(
            self,
            *,
            seed: int | None = None,
            options: dict[str, Any] | None = None,
        ) -> tuple[Any, dict[str, Any]]:
            observation, info = self.env.reset(seed=seed, options=options)
            return self.observation(observation), info

        def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
            observation, reward, terminated, truncated, info = self.env.step(action)
            return self.observation(observation), reward, terminated, truncated, info


def _shape_of(value: Any) -> tuple[int, ...]:
    if hasattr(value, "shape"):
        shape = value.shape
        if isinstance(shape, tuple):
            return shape
    if isinstance(value, (list, tuple)):
        if not value:
            return (0,)
        return (len(value), *_shape_of(value[0]))
    return ()
