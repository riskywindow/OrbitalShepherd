from __future__ import annotations

import argparse
import shutil
from collections.abc import Sequence
from pathlib import Path

from orbital_shepherd_training.bc_training import train_behavior_cloning
from orbital_shepherd_training.evaluation import run_phase2_evaluation
from orbital_shepherd_training.offline_dataset import (
    DEFAULT_EXPERT_PLANNER_IDS,
    build_offline_datasets,
    inspect_offline_dataset,
)
from orbital_shepherd_training.paths import (
    phase2_dataset_root,
    phase2_scenario_pack_dir,
    phase2_split_registry_path,
    phase2_training_pack_manifest_path,
    training_config_root,
    training_manifest_root,
)
from orbital_shepherd_training.registry import (
    build_phase2_split_registry,
    build_phase2_training_pack,
    validate_phase2_configs,
    validate_phase2_split_registry,
    validate_phase2_training_pack,
)
from orbital_shepherd_training.rllib_training import train_ppo_with_rllib

DEFAULT_EVALUATION_PLANNER_IDS: tuple[str, ...] = (
    "random_valid_action",
    "urgency_greedy",
    "value_density_greedy",
    "ortools_receding_horizon",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="orbital-shepherd-training")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_configs = subparsers.add_parser(
        "validate-configs",
        help="Validate the committed Phase 2 config files and cross-config references.",
    )
    validate_configs.add_argument(
        "--config-root",
        type=Path,
        default=training_config_root(),
        help="Training config directory root.",
    )
    validate_configs.set_defaults(handler=_handle_validate_configs)

    build_splits = subparsers.add_parser(
        "build-split-registry",
        help="Build the deterministic Phase 2 train/val/test/OOD split registry.",
    )
    build_splits.add_argument(
        "--output",
        type=Path,
        default=phase2_split_registry_path(),
        help="Destination path for the split registry.",
    )
    build_splits.set_defaults(handler=_handle_build_splits)

    validate_splits = subparsers.add_parser(
        "validate-split-registry",
        help="Validate a Phase 2 split registry document.",
    )
    validate_splits.add_argument(
        "--input",
        type=Path,
        default=phase2_split_registry_path(),
        help="Path to the split registry document.",
    )
    validate_splits.set_defaults(handler=_handle_validate_splits)

    build_pack = subparsers.add_parser(
        "build-training-pack",
        help="Build the deterministic Phase 2 training scenario pack and manifest.",
    )
    build_pack.add_argument(
        "--output-dir",
        type=Path,
        default=phase2_scenario_pack_dir(),
        help="Directory for compiled Phase 2 scenario bundles.",
    )
    build_pack.add_argument(
        "--manifest-output",
        type=Path,
        default=phase2_training_pack_manifest_path(),
        help="Destination path for the training pack manifest.",
    )
    build_pack.add_argument(
        "--split-output",
        type=Path,
        default=phase2_split_registry_path(),
        help="Destination path for the split registry.",
    )
    build_pack.set_defaults(handler=_handle_build_pack)

    validate_pack = subparsers.add_parser(
        "validate-training-pack",
        help="Validate the deterministic Phase 2 scenario pack and manifest.",
    )
    validate_pack.add_argument(
        "--input-dir",
        type=Path,
        default=phase2_scenario_pack_dir(),
        help="Directory containing compiled Phase 2 scenario bundles.",
    )
    validate_pack.add_argument(
        "--manifest",
        type=Path,
        default=phase2_training_pack_manifest_path(),
        help="Path to the Phase 2 training pack manifest.",
    )
    validate_pack.set_defaults(handler=_handle_validate_pack)

    build_dataset = subparsers.add_parser(
        "build-offline-dataset",
        help="Build deterministic offline expert datasets from actual planner rollouts.",
    )
    build_dataset.add_argument(
        "--training-pack-manifest",
        type=Path,
        default=phase2_training_pack_manifest_path(),
        help="Path to the training pack manifest.",
    )
    build_dataset.add_argument(
        "--split-registry",
        type=Path,
        default=phase2_split_registry_path(),
        help="Path to the split registry document.",
    )
    build_dataset.add_argument(
        "--output-root",
        type=Path,
        default=phase2_dataset_root(),
        help="Directory that will contain the dataset build outputs.",
    )
    build_dataset.add_argument(
        "--manifest-root",
        type=Path,
        default=training_manifest_root(),
        help="Directory that will contain the top-level build manifest.",
    )
    build_dataset.add_argument(
        "--planner",
        action="append",
        dest="planners",
        default=None,
        help="Planner id to include. Repeat to mix multiple expert planners.",
    )
    build_dataset.add_argument(
        "--split",
        action="append",
        dest="splits",
        default=None,
        help="Split to build. Repeat to build multiple splits.",
    )
    build_dataset.add_argument(
        "--top-k",
        type=int,
        default=64,
        help="Maximum number of projected non-noop action slots.",
    )
    build_dataset.add_argument(
        "--limit-bundles-per-split",
        type=int,
        default=None,
        help="Optional cap on bundles per split for smoke builds.",
    )
    build_dataset.add_argument(
        "--build-id",
        default=None,
        help="Optional stable id for the dataset build.",
    )
    build_dataset.add_argument(
        "--require-parquet",
        action="store_true",
        help="Fail the build if parquet output cannot be written.",
    )
    build_dataset.set_defaults(handler=_handle_build_offline_dataset)

    inspect_dataset = subparsers.add_parser(
        "inspect-offline-dataset",
        help="Print a dataset or build-manifest summary without using notebooks.",
    )
    inspect_dataset.add_argument(
        "source",
        type=Path,
        help="Path to an offline dataset manifest, build manifest, or dataset directory.",
    )
    inspect_dataset.add_argument(
        "--preview-steps",
        type=int,
        default=3,
        help="Number of step rows to preview when inspecting a split dataset manifest.",
    )
    inspect_dataset.set_defaults(handler=_handle_inspect_offline_dataset)

    train_bc = subparsers.add_parser(
        "train-bc",
        help="Run Phase 2 behavior cloning pretraining on offline expert traces.",
    )
    train_bc.add_argument(
        "--config",
        type=Path,
        default=training_config_root() / "bc" / "phase2_bc.yaml",
        help="Path to the behavior cloning config document.",
    )
    train_bc.add_argument(
        "--model-config",
        type=Path,
        default=training_config_root() / "model" / "phase2_policy.yaml",
        help="Path to the shared policy-model architecture config document.",
    )
    train_bc.add_argument(
        "--best-checkpoint-alias",
        type=Path,
        default=None,
        help="Optional stable path to copy the best BC checkpoint manifest to.",
    )
    train_bc.set_defaults(handler=_handle_train_bc)

    train_ppo = subparsers.add_parser(
        "train-ppo",
        help="Run Phase 2 online PPO training with RLlib and PyTorch.",
    )
    train_ppo.add_argument(
        "--config",
        type=Path,
        default=training_config_root() / "ppo" / "phase2_ppo.yaml",
        help="Path to the PPO training config document.",
    )
    train_ppo.add_argument(
        "--model-config",
        type=Path,
        default=training_config_root() / "model" / "phase2_policy.yaml",
        help="Path to the shared policy-model architecture config document.",
    )
    train_ppo.add_argument(
        "--latest-checkpoint-alias",
        type=Path,
        default=None,
        help="Optional stable path to copy the latest PPO checkpoint manifest to.",
    )
    train_ppo.set_defaults(handler=_handle_train_ppo)

    evaluate = subparsers.add_parser(
        "evaluate",
        help="Evaluate baseline planners and trained policies on Phase 2 held-out splits.",
    )
    evaluate.add_argument(
        "--config",
        type=Path,
        default=training_config_root() / "evaluation" / "phase2_eval.yaml",
        help="Path to the evaluation config document.",
    )
    evaluate.add_argument(
        "--training-pack-manifest",
        type=Path,
        default=phase2_training_pack_manifest_path(),
        help="Path to the Phase 2 training pack manifest.",
    )
    evaluate.add_argument(
        "--split-registry",
        type=Path,
        default=phase2_split_registry_path(),
        help="Path to the Phase 2 split registry.",
    )
    evaluate.add_argument(
        "--planner",
        action="append",
        dest="planners",
        default=None,
        help="Built-in planner id to include. Repeat to compare multiple baselines.",
    )
    evaluate.add_argument(
        "--checkpoint-manifest",
        action="append",
        dest="checkpoint_manifests",
        default=[],
        help="Policy checkpoint manifest to evaluate. Repeat to compare multiple trained policies.",
    )
    evaluate.add_argument(
        "--policy-label",
        action="append",
        dest="policy_labels",
        default=None,
        help=(
            "Optional display label for a checkpoint manifest. "
            "Repeat in the same order as --checkpoint-manifest."
        ),
    )
    evaluate.add_argument(
        "--split",
        action="append",
        dest="splits",
        default=None,
        help="Split to evaluate. Repeat to include multiple splits.",
    )
    evaluate.add_argument(
        "--bundle-id",
        action="append",
        dest="bundle_ids",
        default=[],
        help="Exact bundle id filter. Repeat to keep a small reproducible subset.",
    )
    evaluate.add_argument(
        "--limit-bundles-per-split",
        type=int,
        default=None,
        help="Optional cap on bundles per split for smoke evaluation runs.",
    )
    evaluate.add_argument(
        "--run-id",
        default=None,
        help="Optional stable evaluation run id.",
    )
    evaluate.set_defaults(handler=_handle_evaluate)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.handler(args))


def _handle_validate_configs(args: argparse.Namespace) -> int:
    validated = validate_phase2_configs(args.config_root)
    for name, document in validated.items():
        print(f"{name}\t{type(document).__name__}")
    return 0


def _handle_build_splits(args: argparse.Namespace) -> int:
    registry = build_phase2_split_registry(args.output)
    print(args.output)
    print(f"entries={len(registry.entries)}")
    return 0


def _handle_validate_splits(args: argparse.Namespace) -> int:
    from orbital_shepherd_training.config_io import load_yaml_document

    registry = validate_phase2_split_registry(load_yaml_document(args.input))
    print(args.input)
    print(f"entries={len(registry.entries)}")
    return 0


def _handle_build_pack(args: argparse.Namespace) -> int:
    manifest = build_phase2_training_pack(
        output_dir=args.output_dir,
        manifest_output=args.manifest_output,
        split_registry_output=args.split_output,
    )
    print(args.output_dir)
    print(args.manifest_output)
    print(f"bundles={manifest.bundle_count}")
    return 0


def _handle_validate_pack(args: argparse.Namespace) -> int:
    manifest = validate_phase2_training_pack(
        input_dir=args.input_dir,
        manifest_path=args.manifest,
    )
    print(args.input_dir)
    print(args.manifest)
    print(f"bundles={manifest.bundle_count}")
    return 0


def _handle_build_offline_dataset(args: argparse.Namespace) -> int:
    manifest = build_offline_datasets(
        training_pack=args.training_pack_manifest,
        split_registry=args.split_registry,
        output_root=args.output_root,
        manifest_root=args.manifest_root,
        planner_ids=tuple(args.planners or DEFAULT_EXPERT_PLANNER_IDS),
        splits=tuple(args.splits or ("train", "val", "test")),
        top_k=args.top_k,
        limit_bundles_per_split=args.limit_bundles_per_split,
        build_id=args.build_id,
        require_parquet=bool(args.require_parquet),
    )
    build_manifest_path = args.manifest_root / f"{str(manifest.build_id).replace(':', '--')}.json"
    print(build_manifest_path)
    print(manifest.output_dir)
    for dataset_manifest_path in manifest.dataset_manifests:
        print(dataset_manifest_path)
    return 0


def _handle_inspect_offline_dataset(args: argparse.Namespace) -> int:
    print(inspect_offline_dataset(args.source, preview_steps=args.preview_steps), end="")
    return 0


def _handle_train_bc(args: argparse.Namespace) -> int:
    summary = train_behavior_cloning(
        bc_config=args.config,
        model_config=args.model_config,
    )
    if args.best_checkpoint_alias is not None and summary.best_checkpoint_manifest_path is not None:
        args.best_checkpoint_alias.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(summary.best_checkpoint_manifest_path, args.best_checkpoint_alias)
    print(summary.run_dir)
    print(summary.metrics_path)
    print(summary.summary_report_path)
    print(summary.run_manifest_path)
    print(summary.dataset_build_manifest_path)
    print(summary.train_dataset_manifest_path)
    print(summary.validation_dataset_manifest_path)
    for manifest_path in summary.checkpoint_manifest_paths:
        print(manifest_path)
    return 0


def _handle_train_ppo(args: argparse.Namespace) -> int:
    summary = train_ppo_with_rllib(
        ppo_config=args.config,
        model_config=args.model_config,
    )
    if args.latest_checkpoint_alias is not None and summary.checkpoint_manifest_paths:
        args.latest_checkpoint_alias.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(summary.checkpoint_manifest_paths[-1], args.latest_checkpoint_alias)
    print(summary.run_dir)
    print(summary.metrics_path)
    print(summary.run_manifest_path)
    for manifest_path in summary.checkpoint_manifest_paths:
        print(manifest_path)
    return 0


def _handle_evaluate(args: argparse.Namespace) -> int:
    summary = run_phase2_evaluation(
        evaluation_config=args.config,
        training_pack=args.training_pack_manifest,
        split_registry=args.split_registry,
        planner_ids=tuple(args.planners or DEFAULT_EVALUATION_PLANNER_IDS),
        checkpoint_manifests=tuple(args.checkpoint_manifests),
        policy_labels=tuple(args.policy_labels or ()),
        splits=tuple(args.splits or ()),
        limit_bundles_per_split=args.limit_bundles_per_split,
        bundle_ids=tuple(args.bundle_ids),
        run_id=args.run_id,
    )
    print(summary.report_dir)
    print(summary.summary_path)
    if summary.markdown_path is not None:
        print(summary.markdown_path)
    print(summary.run_manifest_path)
    for manifest_path in summary.report_manifest_paths:
        print(manifest_path)
    return 0
