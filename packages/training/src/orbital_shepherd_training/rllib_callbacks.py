from __future__ import annotations

from typing import Any

from ray.rllib.callbacks.callbacks import RLlibCallback


class OrbitalTrainingMetricsCallback(RLlibCallback):
    def on_episode_end(
        self,
        *,
        episode: Any,
        metrics_logger: Any | None = None,
        env: Any | None = None,
        **kwargs,
    ) -> None:
        del episode, kwargs
        if metrics_logger is None or env is None:
            return
        env_ref = getattr(env, "unwrapped", env)
        metric_values = getattr(env_ref, "episode_metric_values", None)
        if metric_values is None:
            return
        for metric_name, metric_value in metric_values().items():
            metrics_logger.log_value(
                key=("orbital", metric_name),
                value=float(metric_value),
                reduce="mean",
            )
