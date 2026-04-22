from __future__ import annotations

import torch

from orbital_shepherd_policy_models import (
    INVALID_ACTION_LOGIT,
    OrbitalPhase2PolicyModel,
    Phase2PolicyModelConfig,
    apply_action_mask,
)
from orbital_shepherd_training import CANDIDATE_FEATURE_SPECS, GLOBAL_FEATURE_SPECS


def test_phase2_policy_model_forward_shapes() -> None:
    model = OrbitalPhase2PolicyModel(
        Phase2PolicyModelConfig(
            global_feature_dim=len(GLOBAL_FEATURE_SPECS),
            candidate_feature_dim=len(CANDIDATE_FEATURE_SPECS),
            action_dim=5,
            top_k=4,
            hidden_dim=32,
            encoder_layers=2,
            attention_heads=4,
            dropout=0.0,
        )
    )
    outputs = model(
        global_features=torch.randn(2, len(GLOBAL_FEATURE_SPECS)),
        candidate_features=torch.randn(2, 4, len(CANDIDATE_FEATURE_SPECS)),
        action_mask=torch.tensor(
            [
                [1, 1, 1, 0, 0],
                [1, 0, 1, 1, 0],
            ],
            dtype=torch.int8,
        ),
    )

    assert outputs.logits.shape == (2, 5)
    assert outputs.masked_logits.shape == (2, 5)
    assert outputs.values.shape == (2,)
    assert outputs.policy_context.shape == (2, 32)
    assert outputs.candidate_embeddings.shape == (2, 4, 32)


def test_apply_action_mask_sets_invalid_logits_to_large_negative_values() -> None:
    logits = torch.tensor([[0.4, -0.2, 0.1, 0.7]])
    masked_logits = apply_action_mask(
        logits,
        torch.tensor([[1, 0, 1, 0]], dtype=torch.int8),
    )

    assert masked_logits[0, 0].item() == logits[0, 0].item()
    assert masked_logits[0, 2].item() == logits[0, 2].item()
    assert masked_logits[0, 1].item() == INVALID_ACTION_LOGIT
    assert masked_logits[0, 3].item() == INVALID_ACTION_LOGIT
