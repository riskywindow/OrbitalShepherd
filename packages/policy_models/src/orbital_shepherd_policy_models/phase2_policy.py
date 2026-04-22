from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import Tensor, nn

INVALID_ACTION_LOGIT = -1.0e9


@dataclass(frozen=True, slots=True)
class Phase2PolicyModelConfig:
    global_feature_dim: int
    candidate_feature_dim: int
    action_dim: int
    top_k: int
    hidden_dim: int = 256
    encoder_layers: int = 2
    attention_heads: int = 4
    dropout: float = 0.1

    def __post_init__(self) -> None:
        if self.global_feature_dim <= 0:
            raise ValueError("global_feature_dim must be >= 1")
        if self.candidate_feature_dim <= 0:
            raise ValueError("candidate_feature_dim must be >= 1")
        if self.top_k <= 0:
            raise ValueError("top_k must be >= 1")
        if self.action_dim != self.top_k + 1:
            raise ValueError("action_dim must match top_k + 1 for candidate slots plus noop")
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be >= 1")
        if self.encoder_layers <= 0:
            raise ValueError("encoder_layers must be >= 1")
        if self.attention_heads <= 0:
            raise ValueError("attention_heads must be >= 1")
        if self.hidden_dim % self.attention_heads != 0:
            raise ValueError("hidden_dim must be divisible by attention_heads")


@dataclass(frozen=True, slots=True)
class Phase2PolicyOutputs:
    logits: Tensor
    masked_logits: Tensor
    values: Tensor
    policy_context: Tensor
    candidate_embeddings: Tensor


def apply_action_mask(logits: Tensor, action_mask: Tensor) -> Tensor:
    if logits.ndim != 2:
        raise ValueError(f"logits must be rank-2, got shape={tuple(logits.shape)}")
    if action_mask.shape != logits.shape:
        raise ValueError(
            "action_mask must have the same shape as logits, "
            f"got logits={tuple(logits.shape)} action_mask={tuple(action_mask.shape)}"
        )
    mask = action_mask.to(device=logits.device, dtype=torch.bool)
    return logits.masked_fill(~mask, INVALID_ACTION_LOGIT)


class OrbitalPhase2PolicyModel(nn.Module):
    def __init__(self, config: Phase2PolicyModelConfig) -> None:
        super().__init__()
        self.config = config
        self.global_encoder = _FeatureEncoder(
            input_dim=config.global_feature_dim,
            hidden_dim=config.hidden_dim,
            depth=max(1, config.encoder_layers),
            dropout=config.dropout,
        )
        self.candidate_encoder = _FeatureEncoder(
            input_dim=config.candidate_feature_dim,
            hidden_dim=config.hidden_dim,
            depth=max(1, config.encoder_layers),
            dropout=config.dropout,
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.attention_heads,
            dim_feedforward=config.hidden_dim * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.set_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=max(1, config.encoder_layers),
        )
        self.policy_context_head = nn.Sequential(
            nn.LayerNorm(config.hidden_dim * 2),
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.noop_head = nn.Linear(config.hidden_dim, 1)
        self.candidate_head = nn.Sequential(
            nn.LayerNorm(config.hidden_dim * 2),
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, 1),
        )
        self.value_head = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, 1),
        )

    def forward(
        self,
        *,
        global_features: Tensor,
        candidate_features: Tensor,
        action_mask: Tensor,
    ) -> Phase2PolicyOutputs:
        global_features = _ensure_batch_rank(
            global_features,
            expected_last_dim=self.config.global_feature_dim,
        )
        candidate_features = _ensure_batch_rank(
            candidate_features,
            expected_last_dim=self.config.candidate_feature_dim,
            expected_rank=3,
        )
        action_mask = _ensure_batch_rank(
            action_mask,
            expected_last_dim=self.config.action_dim,
        )
        candidate_valid_mask = action_mask[:, 1:] > 0

        global_tokens = self.global_encoder(global_features).unsqueeze(1)
        candidate_tokens = self.candidate_encoder(candidate_features)
        transformer_tokens = torch.cat([global_tokens, candidate_tokens], dim=1)
        padding_mask = torch.cat(
            [
                torch.zeros(
                    (candidate_valid_mask.shape[0], 1),
                    device=candidate_valid_mask.device,
                    dtype=torch.bool,
                ),
                ~candidate_valid_mask,
            ],
            dim=1,
        )
        contextual_tokens = self.set_encoder(
            transformer_tokens,
            src_key_padding_mask=padding_mask,
        )
        global_context = contextual_tokens[:, 0]
        candidate_context = contextual_tokens[:, 1:]
        pooled_candidates = _masked_mean(candidate_context, candidate_valid_mask)
        policy_context = self.policy_context_head(
            torch.cat([global_context, pooled_candidates], dim=-1)
        )
        noop_logits = self.noop_head(policy_context)
        expanded_context = policy_context.unsqueeze(1).expand_as(candidate_context)
        candidate_logits = self.candidate_head(
            torch.cat([candidate_context, expanded_context], dim=-1)
        ).squeeze(-1)
        logits = torch.cat([noop_logits, candidate_logits], dim=-1)
        masked_logits = apply_action_mask(logits, action_mask)
        values = self.value_head(policy_context).squeeze(-1)
        return Phase2PolicyOutputs(
            logits=logits,
            masked_logits=masked_logits,
            values=values,
            policy_context=policy_context,
            candidate_embeddings=candidate_context,
        )

    def forward_observation(self, observation: Mapping[str, Tensor]) -> Phase2PolicyOutputs:
        return self(
            global_features=observation["global_features"],
            candidate_features=observation["candidate_features"],
            action_mask=observation["action_mask"],
        )

    def value_from_features(self, policy_context: Tensor) -> Tensor:
        policy_context = _ensure_batch_rank(
            policy_context,
            expected_last_dim=self.config.hidden_dim,
        )
        return self.value_head(policy_context).squeeze(-1)


class _FeatureEncoder(nn.Module):
    def __init__(self, *, input_dim: int, hidden_dim: int, depth: int, dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current_dim = input_dim
        for _ in range(depth):
            layers.extend(
                [
                    nn.Linear(current_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            current_dim = hidden_dim
        self.network = nn.Sequential(*layers)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.network(inputs)


def _ensure_batch_rank(
    tensor: Tensor,
    *,
    expected_last_dim: int,
    expected_rank: int = 2,
) -> Tensor:
    if tensor.ndim == expected_rank - 1:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != expected_rank:
        raise ValueError(f"expected rank {expected_rank}, got shape={tuple(tensor.shape)}")
    if tensor.shape[-1] != expected_last_dim:
        raise ValueError(
            f"expected final dimension {expected_last_dim}, got shape={tuple(tensor.shape)}"
        )
    return tensor.float()


def _masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    weights = mask.to(device=values.device, dtype=values.dtype).unsqueeze(-1)
    total = torch.sum(values * weights, dim=1)
    count = torch.clamp(torch.sum(weights, dim=1), min=1.0)
    return total / count
