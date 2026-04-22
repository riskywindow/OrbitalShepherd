"""Reusable PyTorch policy models for Orbital Shepherd training."""

from orbital_shepherd_policy_models.phase2_policy import (
    INVALID_ACTION_LOGIT,
    OrbitalPhase2PolicyModel,
    Phase2PolicyModelConfig,
    Phase2PolicyOutputs,
    apply_action_mask,
)
from orbital_shepherd_policy_models.projection import (
    CANDIDATE_FEATURE_SPECS,
    GLOBAL_FEATURE_SPECS,
    CandidateProjection,
    FeatureDescriptor,
    ProjectedActionSlot,
    TrainingObservationProjector,
    flatten_training_observation,
)
from orbital_shepherd_policy_models.registry import (
    LoadedTrainedPolicy,
    PolicyArchitectureSpec,
    PolicyCheckpointManifestRecord,
    PolicyModelRegistry,
    RegisteredTrainedPolicy,
    default_policy_model_registry,
)
from orbital_shepherd_policy_models.rl_module import OrbitalMaskedActionTorchRLModule

__all__ = [
    "CANDIDATE_FEATURE_SPECS",
    "CandidateProjection",
    "FeatureDescriptor",
    "GLOBAL_FEATURE_SPECS",
    "INVALID_ACTION_LOGIT",
    "LoadedTrainedPolicy",
    "OrbitalMaskedActionTorchRLModule",
    "OrbitalPhase2PolicyModel",
    "PolicyArchitectureSpec",
    "PolicyCheckpointManifestRecord",
    "PolicyModelRegistry",
    "ProjectedActionSlot",
    "Phase2PolicyModelConfig",
    "Phase2PolicyOutputs",
    "RegisteredTrainedPolicy",
    "TrainingObservationProjector",
    "apply_action_mask",
    "default_policy_model_registry",
    "flatten_training_observation",
]
