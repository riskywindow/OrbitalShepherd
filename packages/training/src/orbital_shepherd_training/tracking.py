from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from orbital_shepherd_training.models import WandbConfig


@dataclass(slots=True)
class NoopWandbRun:
    config: dict[str, Any] = field(default_factory=dict)

    def log(self, _: dict[str, Any], *, step: int | None = None) -> None:
        return None

    def finish(self) -> None:
        return None


def maybe_init_wandb(
    config: WandbConfig,
    *,
    run_name: str,
    run_config: dict[str, Any] | None = None,
) -> Any:
    if not config.enabled or config.mode == "disabled":
        return NoopWandbRun(config=run_config or {})
    try:
        import wandb  # type: ignore[import-not-found]
    except ModuleNotFoundError:
        return NoopWandbRun(config=run_config or {})
    return wandb.init(
        project=config.project,
        entity=config.entity,
        group=config.group,
        tags=config.tags,
        mode=config.mode,
        name=run_name,
        config=run_config or {},
    )
