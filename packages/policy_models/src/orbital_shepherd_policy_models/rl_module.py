from __future__ import annotations

from typing import Any

import gymnasium as gym
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.apis.value_function_api import ValueFunctionAPI
from ray.rllib.core.rl_module.torch import TorchRLModule

from orbital_shepherd_policy_models.phase2_policy import (
    OrbitalPhase2PolicyModel,
    Phase2PolicyModelConfig,
)


class OrbitalMaskedActionTorchRLModule(TorchRLModule, ValueFunctionAPI):
    def setup(self) -> None:
        if not isinstance(self.observation_space, gym.spaces.Dict):
            raise ValueError("OrbitalMaskedActionTorchRLModule requires a Dict observation space")
        if not isinstance(self.action_space, gym.spaces.Discrete):
            raise ValueError("OrbitalMaskedActionTorchRLModule requires a Discrete action space")
        model_config = dict(self.model_config or {})
        self.policy_model = OrbitalPhase2PolicyModel(
            Phase2PolicyModelConfig(
                global_feature_dim=int(self.observation_space["global_features"].shape[0]),
                candidate_feature_dim=int(
                    self.observation_space["candidate_features"].shape[-1]
                ),
                action_dim=int(self.action_space.n),
                top_k=int(self.observation_space["candidate_features"].shape[0]),
                hidden_dim=int(model_config.get("hidden_dim", 256)),
                encoder_layers=int(model_config.get("encoder_layers", 2)),
                attention_heads=int(model_config.get("attention_heads", 4)),
                dropout=float(model_config.get("dropout", 0.1)),
            )
        )

    def _forward(self, batch: dict[str, Any], **kwargs) -> dict[str, Any]:
        del kwargs
        return self._forward_common(batch=batch, include_embeddings=False)

    def _forward_inference(self, batch: dict[str, Any], **kwargs) -> dict[str, Any]:
        del kwargs
        return self._forward_common(batch=batch, include_embeddings=False)

    def _forward_exploration(self, batch: dict[str, Any], **kwargs) -> dict[str, Any]:
        del kwargs
        return self._forward_common(batch=batch, include_embeddings=False)

    def _forward_train(self, batch: dict[str, Any], **kwargs) -> dict[str, Any]:
        del kwargs
        return self._forward_common(batch=batch, include_embeddings=True)

    def compute_values(
        self,
        batch: dict[str, Any],
        embeddings: Any = None,
    ) -> Any:
        if embeddings is not None:
            return self.policy_model.value_from_features(embeddings)
        outputs = self.policy_model.forward_observation(batch[Columns.OBS])
        return outputs.values

    def _forward_common(
        self,
        *,
        batch: dict[str, Any],
        include_embeddings: bool,
    ) -> dict[str, Any]:
        outputs = self.policy_model.forward_observation(batch[Columns.OBS])
        result: dict[str, Any] = {
            Columns.ACTION_DIST_INPUTS: outputs.masked_logits,
            Columns.VF_PREDS: outputs.values,
        }
        if include_embeddings:
            result[Columns.EMBEDDINGS] = outputs.policy_context
        return result
