from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

NamespacedId = Annotated[str, StringConstraints(pattern=r"^[a-z0-9][a-z0-9._:-]{2,127}$")]
NonEmptyString = Annotated[str, StringConstraints(min_length=1)]
ArtifactFingerprint = Annotated[str, StringConstraints(pattern=r"^sha256:[a-f0-9]{64}$")]
ScenarioFamily = Literal[
    "sparse_frontier",
    "burst_outbreak",
    "cloud_trap",
    "downlink_crunch",
    "station_outage",
    "constellation_degradation",
]
SplitName = Literal["train", "val", "test", "ood"]
AlgorithmName = Literal["behavior_cloning", "ppo"]
BenchmarkMetricPolicy = Literal["audit_only"]
DatasetArtifactFormat = Literal["jsonl", "markdown", "npz", "parquet"]
ClassWeightingStrategy = Literal["none", "balanced", "inverse_frequency"]
ComputeDevice = Literal["auto", "cpu", "cuda"]
CheckpointSelection = Literal["latest"]
DatasetArtifactRole = Literal[
    "canonical_episodes",
    "canonical_steps",
    "dataset_card",
    "rllib_transitions",
    "training_arrays",
]
RewardComponentName = Literal[
    "observation_value",
    "downlink_value",
    "cloud_penalty",
    "latency_penalty",
    "buffer_pressure_penalty",
    "missed_incident_penalty",
]
EvaluationMetricName = Literal[
    "time_to_first_useful_observation_seconds",
    "useful_observation_value_captured",
    "cloud_waste_rate",
    "downlink_latency_seconds",
    "missed_urgent_incident_rate",
    "opportunity_utilization_efficiency",
    "mission_utility",
]
ReportFormat = Literal["json", "markdown", "csv"]
WandbMode = Literal["disabled", "offline", "online"]


class TrainingModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware")
    if value.utcoffset() != timedelta(0):
        raise ValueError("timestamp must be UTC")
    return value.astimezone(UTC)


class ArtifactLayoutConfig(TrainingModel):
    dataset_root: NonEmptyString
    checkpoint_root: NonEmptyString
    report_root: NonEmptyString
    manifest_root: NonEmptyString
    scenario_pack_root: NonEmptyString


class WandbConfig(TrainingModel):
    enabled: bool = False
    mode: WandbMode = "disabled"
    project: NonEmptyString
    entity: str | None = None
    group: str | None = None
    tags: list[NonEmptyString] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_mode(self) -> WandbConfig:
        if not self.enabled and self.mode != "disabled":
            raise ValueError("mode must be 'disabled' when W&B is disabled")
        if self.enabled and self.mode == "disabled":
            raise ValueError("enabled W&B config must use offline or online mode")
        return self


class RewardTermConfig(TrainingModel):
    term_id: NonEmptyString
    source_component: RewardComponentName
    weight: float
    description: NonEmptyString
    clamp_min: float | None = None
    clamp_max: float | None = None

    @model_validator(mode="after")
    def _validate_clamp(self) -> RewardTermConfig:
        if self.clamp_min is not None and self.clamp_max is not None:
            if self.clamp_max < self.clamp_min:
                raise ValueError("clamp_max must be greater than or equal to clamp_min")
        return self


class RewardShapingConfig(TrainingModel):
    config_type: Literal["reward_shaping"] = "reward_shaping"
    reward_id: NamespacedId
    benchmark_id: NonEmptyString
    benchmark_metric_policy: BenchmarkMetricPolicy = "audit_only"
    decision_interval_seconds: int = Field(ge=1)
    normalize_by_decision_interval: bool = True
    emit_component_trace: bool = True
    terms: list[RewardTermConfig] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_unique_terms(self) -> RewardShapingConfig:
        term_ids = [term.term_id for term in self.terms]
        if len(term_ids) != len(set(term_ids)):
            raise ValueError("reward term ids must be unique")
        return self


class CurriculumStageConfig(TrainingModel):
    stage_id: NamespacedId
    display_name: NonEmptyString
    description: NonEmptyString
    policy_splits: list[SplitName] = Field(min_length=1)
    evaluation_splits: list[SplitName] = Field(min_length=1)
    policy_family_allowlist: list[ScenarioFamily] = Field(min_length=1)
    evaluation_family_allowlist: list[ScenarioFamily] = Field(min_length=1)
    difficulty_tiers: list[int] = Field(min_length=1)
    algorithms: list[AlgorithmName] = Field(min_length=1)
    max_stage_episodes: int = Field(ge=1)


class CurriculumConfig(TrainingModel):
    config_type: Literal["curriculum"] = "curriculum"
    curriculum_id: NamespacedId
    benchmark_id: NonEmptyString
    split_registry_path: NonEmptyString
    stages: list[CurriculumStageConfig] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_unique_stages(self) -> CurriculumConfig:
        stage_ids = [stage.stage_id for stage in self.stages]
        if len(stage_ids) != len(set(stage_ids)):
            raise ValueError("curriculum stage ids must be unique")
        return self


class ModelArchitectureConfig(TrainingModel):
    config_type: Literal["model_architecture"] = "model_architecture"
    model_id: NamespacedId
    observation_encoder_type: Literal["tabular_transformer", "mlp"]
    action_encoder_type: Literal["masked_action_mlp"]
    hidden_dim: int = Field(ge=32)
    encoder_layers: int = Field(ge=1)
    attention_heads: int = Field(ge=1)
    dropout: float = Field(ge=0, le=1)
    recurrent_memory_steps: int = Field(ge=1)
    shared_policy_value_backbone: bool = True


class EarlyStoppingConfig(TrainingModel):
    enabled: bool = False
    metric: Literal["val_loss", "val_accuracy"] = "val_loss"
    mode: Literal["min", "max"] = "min"
    patience: int = Field(default=3, ge=1)
    min_delta: float = Field(default=0.0, ge=0)

    @model_validator(mode="after")
    def _validate_metric_mode(self) -> EarlyStoppingConfig:
        expected_mode = "min" if self.metric == "val_loss" else "max"
        if self.mode != expected_mode:
            raise ValueError(
                f"early_stopping.mode must be '{expected_mode}' when metric is {self.metric}"
            )
        return self


class PolicyInitializationConfig(TrainingModel):
    mode: Literal["scratch", "checkpoint"] = "scratch"
    checkpoint_manifest_path: NonEmptyString | None = None
    source_run_id: NamespacedId | None = None
    selection: CheckpointSelection = "latest"

    @model_validator(mode="after")
    def _validate_reference(self) -> PolicyInitializationConfig:
        provided = int(self.checkpoint_manifest_path is not None) + int(
            self.source_run_id is not None
        )
        if self.mode == "scratch":
            if provided:
                raise ValueError("scratch initialization must not provide a checkpoint source")
            return self
        if provided != 1:
            raise ValueError(
                "checkpoint initialization requires exactly one of "
                "checkpoint_manifest_path or source_run_id"
            )
        return self


class BehaviorCloningConfig(TrainingModel):
    config_type: Literal["behavior_cloning"] = "behavior_cloning"
    run_id: NamespacedId
    benchmark_id: NonEmptyString
    model_id: NamespacedId
    reward_id: NamespacedId
    curriculum_id: NamespacedId
    artifact_layout: ArtifactLayoutConfig
    wandb: WandbConfig
    seed: int
    batch_size: int = Field(ge=1)
    epochs: int = Field(ge=1)
    learning_rate: float = Field(gt=0)
    weight_decay: float = Field(ge=0)
    label_smoothing: float = Field(ge=0, le=1)
    training_pack_path: NonEmptyString = (
        "data/training/manifests/phase2-training-pack-manifest.json"
    )
    split_registry_path: NonEmptyString = "training/configs/curriculum/phase2_splits.yaml"
    source_planner_ids: list[NamespacedId] = Field(default_factory=list)
    train_split: SplitName
    validation_split: SplitName
    top_k: int = Field(default=64, ge=1)
    dataset_build_id: NamespacedId | None = None
    limit_bundles_per_split: int | None = Field(default=None, ge=1)
    max_train_transitions: int | None = Field(default=None, ge=1)
    max_validation_transitions: int | None = Field(default=None, ge=1)
    class_weighting: ClassWeightingStrategy = "none"
    class_weight_clip: float | None = Field(default=None, gt=0)
    checkpoint_frequency: int = Field(default=1, ge=1)
    checkpoint_at_end: bool = True
    require_parquet: bool = False
    device: ComputeDevice = "auto"
    early_stopping: EarlyStoppingConfig = Field(default_factory=EarlyStoppingConfig)


class PpoTrainingConfig(TrainingModel):
    config_type: Literal["ppo_training"] = "ppo_training"
    run_id: NamespacedId
    benchmark_id: NonEmptyString
    model_id: NamespacedId
    reward_id: NamespacedId
    curriculum_id: NamespacedId
    artifact_layout: ArtifactLayoutConfig
    wandb: WandbConfig
    seed: int
    total_timesteps: int = Field(ge=1)
    rollout_steps: int = Field(ge=1)
    minibatch_size: int = Field(ge=1)
    update_epochs: int = Field(ge=1)
    gamma: float = Field(gt=0, le=1)
    gae_lambda: float = Field(gt=0, le=1)
    clip_range: float = Field(gt=0, lt=1)
    entropy_coef: float = Field(ge=0)
    value_loss_coef: float = Field(ge=0)
    learning_rate: float = Field(gt=0)
    max_grad_norm: float = Field(gt=0)
    target_kl: float = Field(gt=0)
    framework: Literal["torch"] = "torch"
    trainer_backend: Literal["rllib"] = "rllib"
    training_pack_path: NonEmptyString = (
        "data/training/manifests/phase2-training-pack-manifest.json"
    )
    split_registry_path: NonEmptyString = "training/configs/curriculum/phase2_splits.yaml"
    train_split: SplitName = "train"
    evaluation_split: SplitName = "val"
    top_k: int = Field(default=64, ge=1)
    scenario_limit: int | None = Field(default=None, ge=1)
    num_env_runners: int = Field(default=0, ge=0)
    num_envs_per_env_runner: int = Field(default=1, ge=1)
    num_cpus_per_env_runner: float = Field(default=1.0, gt=0)
    num_gpus: float = Field(default=0.0, ge=0)
    evaluation_interval: int = Field(default=1, ge=1)
    evaluation_duration: int = Field(default=2, ge=1)
    checkpoint_frequency: int = Field(default=1, ge=1)
    checkpoint_at_end: bool = True
    enable_rl_module_api: bool = True
    initialization: PolicyInitializationConfig = Field(default_factory=PolicyInitializationConfig)
    local_mode: bool = False
    log_level: Literal["DEBUG", "INFO", "WARN", "ERROR"] = "INFO"


class EvaluationConfig(TrainingModel):
    config_type: Literal["evaluation"] = "evaluation"
    evaluation_id: NamespacedId
    benchmark_id: NonEmptyString
    split_order: list[SplitName] = Field(min_length=1)
    planner_agnostic_metrics: list[EvaluationMetricName] = Field(min_length=1)
    compare_against_baselines: list[NonEmptyString] = Field(min_length=1)
    report_formats: list[ReportFormat] = Field(min_length=1)
    artifact_layout: ArtifactLayoutConfig
    wandb: WandbConfig
    bootstrap_replicates: int = Field(default=1000, ge=1)
    confidence_level: float = Field(default=0.95, gt=0, lt=1)
    notable_episode_count: int = Field(default=5, ge=1)
    primary_ranking_metric: EvaluationMetricName = "mission_utility"
    emit_reward_audit: bool = True


class ScenarioSplitEntry(TrainingModel):
    recipe_id: NamespacedId
    manifest_id: NamespacedId
    bundle_id: NamespacedId
    scenario_family: ScenarioFamily
    split: SplitName
    simulation_seed: int
    difficulty_tier: int = Field(ge=1)
    curriculum_stage: NamespacedId
    ood_axes: list[NonEmptyString] = Field(default_factory=list)


class ScenarioSplitRegistry(TrainingModel):
    config_type: Literal["scenario_split_registry"] = "scenario_split_registry"
    registry_id: NamespacedId
    benchmark_id: NonEmptyString
    seed_namespace: NonEmptyString
    entries: list[ScenarioSplitEntry] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_unique_entries(self) -> ScenarioSplitRegistry:
        bundle_ids = [entry.bundle_id for entry in self.entries]
        if len(bundle_ids) != len(set(bundle_ids)):
            raise ValueError("bundle ids must be unique within the split registry")
        return self


class TrainingPackEntry(TrainingModel):
    recipe_id: NamespacedId
    manifest_id: NamespacedId
    bundle_id: NamespacedId
    scenario_family: ScenarioFamily
    split: SplitName
    scenario_path: NonEmptyString
    difficulty_tier: int = Field(ge=1)


class TrainingPackManifest(TrainingModel):
    config_type: Literal["training_pack_manifest"] = "training_pack_manifest"
    training_pack_id: NamespacedId
    benchmark_id: NonEmptyString
    split_registry_id: NamespacedId
    generated_at_utc: datetime
    output_dir: NonEmptyString
    bundle_count: int = Field(ge=1)
    entries: list[TrainingPackEntry] = Field(min_length=1)
    artifact_fingerprint: ArtifactFingerprint

    _validate_generated_at = field_validator("generated_at_utc")(_ensure_utc)


class OfflineDatasetArtifact(TrainingModel):
    artifact_role: DatasetArtifactRole
    format: DatasetArtifactFormat
    path: NonEmptyString
    record_count: int | None = Field(default=None, ge=0)
    artifact_fingerprint: ArtifactFingerprint


class OfflineDatasetPlannerSource(TrainingModel):
    planner_id: NamespacedId
    planner_version: NonEmptyString
    description: NonEmptyString
    episode_count: int = Field(ge=0)
    record_count: int = Field(ge=0)


class OfflineDatasetFeatureDescriptor(TrainingModel):
    name: NonEmptyString
    lower_bound: float | None = None
    upper_bound: float | None = None
    description: NonEmptyString


class OfflineDatasetFeatureSchema(TrainingModel):
    top_k: int = Field(ge=1)
    global_features: list[OfflineDatasetFeatureDescriptor] = Field(min_length=1)
    candidate_features: list[OfflineDatasetFeatureDescriptor] = Field(min_length=1)
    action_mask_shape: list[int] = Field(min_length=1)
    action_mask_dtype: NonEmptyString
    slot_mapping_fields: list[NonEmptyString] = Field(min_length=1)


class OfflineDatasetActionSemantics(TrainingModel):
    noop_slot_index: int = Field(ge=0)
    projected_slots_start_index: int = Field(ge=0)
    projected_slots_end_index: int = Field(ge=0)
    canonical_action_mask_field: NonEmptyString
    projected_action_mask_field: NonEmptyString
    selected_slot_field: NonEmptyString
    selected_runtime_action_index_field: NonEmptyString
    selected_action_id_field: NonEmptyString


class OfflineDatasetManifest(TrainingModel):
    config_type: Literal["offline_dataset_manifest"] = "offline_dataset_manifest"
    dataset_id: NamespacedId
    benchmark_id: NonEmptyString
    source_training_pack_id: NamespacedId
    split_registry_id: NamespacedId
    split: SplitName
    algorithm: AlgorithmName
    created_at_utc: datetime
    dataset_schema_version: NonEmptyString
    dataset_path: NonEmptyString
    record_count: int = Field(ge=0)
    episode_count: int = Field(ge=0)
    top_k: int = Field(ge=1)
    max_legal_action_count: int = Field(ge=0)
    max_projected_candidate_count: int = Field(ge=0)
    source_bundle_ids: list[NamespacedId]
    source_manifest_ids: list[NamespacedId] = Field(default_factory=list)
    source_planners: list[OfflineDatasetPlannerSource] = Field(min_length=1)
    scenario_families: list[NonEmptyString] = Field(default_factory=list)
    reward_id: NamespacedId
    feature_schema: OfflineDatasetFeatureSchema
    action_semantics: OfflineDatasetActionSemantics
    artifacts: list[OfflineDatasetArtifact] = Field(min_length=1)
    artifact_fingerprint: ArtifactFingerprint

    _validate_created_at = field_validator("created_at_utc")(_ensure_utc)


class OfflineDatasetBuildManifest(TrainingModel):
    config_type: Literal["offline_dataset_build_manifest"] = "offline_dataset_build_manifest"
    build_id: NamespacedId
    benchmark_id: NonEmptyString
    source_training_pack_id: NamespacedId
    split_registry_id: NamespacedId
    created_at_utc: datetime
    output_dir: NonEmptyString
    planner_ids: list[NamespacedId] = Field(min_length=1)
    splits: list[SplitName] = Field(min_length=1)
    top_k: int = Field(ge=1)
    dataset_manifests: list[NonEmptyString] = Field(min_length=1)
    artifact_fingerprint: ArtifactFingerprint

    _validate_created_at = field_validator("created_at_utc")(_ensure_utc)


class PolicyCheckpointManifest(TrainingModel):
    config_type: Literal["policy_checkpoint_manifest"] = "policy_checkpoint_manifest"
    checkpoint_id: NamespacedId
    algorithm: AlgorithmName
    created_at_utc: datetime
    benchmark_id: NonEmptyString
    run_id: NamespacedId
    model_id: NamespacedId
    reward_id: NamespacedId
    trainer_backend: NonEmptyString
    framework: NonEmptyString
    source_dataset_ids: list[NamespacedId] = Field(default_factory=list)
    source_training_pack_id: NamespacedId | None = None
    source_bundle_ids: list[NamespacedId] = Field(default_factory=list)
    checkpoint_path: NonEmptyString
    global_step: int = Field(ge=0)
    metrics: dict[str, float] = Field(default_factory=dict)
    metadata: dict[str, str | int | float | bool | list[str] | None] = Field(default_factory=dict)
    artifact_fingerprint: ArtifactFingerprint

    _validate_created_at = field_validator("created_at_utc")(_ensure_utc)


class EvaluationReportManifest(TrainingModel):
    config_type: Literal["evaluation_report_manifest"] = "evaluation_report_manifest"
    report_id: NamespacedId
    benchmark_id: NonEmptyString
    evaluated_artifact_id: NamespacedId
    evaluation_config_id: NamespacedId
    split: SplitName
    generated_at_utc: datetime
    summary_path: NonEmptyString
    benchmark_metrics: dict[str, float] = Field(default_factory=dict)
    reward_audit_summary: dict[str, float] = Field(default_factory=dict)
    artifact_fingerprint: ArtifactFingerprint

    _validate_generated_at = field_validator("generated_at_utc")(_ensure_utc)


class EvaluationRunPlanner(TrainingModel):
    planner_key: NonEmptyString
    planner_kind: Literal["builtin", "policy_checkpoint"]
    display_name: NonEmptyString
    evaluated_artifact_id: NamespacedId
    description: NonEmptyString
    source: NonEmptyString
    checkpoint_manifest_path: NonEmptyString | None = None


class EvaluationRunManifest(TrainingModel):
    config_type: Literal["evaluation_run_manifest"] = "evaluation_run_manifest"
    run_id: NamespacedId
    benchmark_id: NonEmptyString
    evaluation_config_id: NamespacedId
    training_pack_id: NamespacedId
    split_registry_id: NamespacedId
    generated_at_utc: datetime
    report_dir: NonEmptyString
    summary_path: NonEmptyString
    markdown_path: NonEmptyString | None = None
    episode_metrics_path: NonEmptyString | None = None
    pairwise_comparisons_path: NonEmptyString | None = None
    notable_episodes_path: NonEmptyString | None = None
    selected_splits: list[SplitName] = Field(min_length=1)
    selected_bundle_ids: list[NamespacedId] = Field(min_length=1)
    planners: list[EvaluationRunPlanner] = Field(min_length=1)
    report_manifests: list[NonEmptyString] = Field(default_factory=list)
    artifact_fingerprint: ArtifactFingerprint

    _validate_generated_at = field_validator("generated_at_utc")(_ensure_utc)


Phase2ConfigDocument = (
    RewardShapingConfig
    | CurriculumConfig
    | ModelArchitectureConfig
    | BehaviorCloningConfig
    | PpoTrainingConfig
    | EvaluationConfig
    | ScenarioSplitRegistry
)
