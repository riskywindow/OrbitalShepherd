
# Orbital Shepherd Phase 2 — Detailed Plan and Codex Prompt Pack

## What Phase 2 should achieve

Phase 2 is where Orbital Shepherd stops being only a replayable mission-control platform and becomes a **serious orbital RL lab**.

By the end of Phase 2, the repo should support:

- a training-ready orbital environment with a **fixed-size, action-masked policy interface**
- stronger non-learning baselines, including a **value-density greedy** planner and a **receding-horizon OR-Tools planner**
- deterministic **train / validation / test / OOD** scenario packs
- a canonical **offline trajectory dataset** generated from baseline rollouts
- an **RLlib + PyTorch** training stack for online RL
- **behavior cloning warm-start** from expert traces, then **PPO fine-tuning**
- reproducible experiments with **W&B** and **DVC** hooks
- evaluation reports with per-family breakdowns, confidence intervals, and failure slices
- trained-policy checkpoints that can be loaded through the existing API and replayed in the mission-control UI

This phase is still **single-agent orbital control**. Do **not** start the tactical street map or hierarchical globe-to-street controller yet.

## What should be true coming out of Phase 1

Phase 2 assumes Phase 1 already gave you:

- canonical contracts and schemas
- deterministic `ScenarioBundle`s
- an `OrbitalEnv`-style runtime with replay events
- scenario packs and a benchmark runner
- baseline planners for `random_valid_action` and `urgency_greedy`
- a FastAPI layer
- a Cesium mission-control UI that can run and replay episodes

If the repo deviates from that, Phase 2 work should normalize the repo gently rather than rewrite it.

## The important design shift in Phase 2

Phase 1 optimized for **systems credibility**.

Phase 2 must optimize for **training credibility**.

That means the critical engineering problems are now:

1. **How to expose a structured, variable candidate set to a neural policy without destroying determinism**
2. **How to keep benchmark metrics separate from training reward**
3. **How to generate stronger baselines and expert traces**
4. **How to run training reproducibly enough that the results feel defendable**
5. **How to replay trained-policy decisions through the same API/UI contract as non-learning planners**

## Recommended Phase 2 architecture

### Stack

- **Training framework:** Ray RLlib + PyTorch
- **Environment API:** Gymnasium-compatible single-agent env wrappers
- **Experiment tracking:** W&B
- **Reproducibility / pipelines:** DVC
- **Optimization baseline:** OR-Tools
- **Data formats:** canonical local artifacts + offline trajectory datasets + replay NDJSON
- **Serving / inference:** existing FastAPI planner surface extended with trained policy adapters
- **Frontend:** existing Cesium mission-control UI extended to compare trained policies vs baselines

### Why this stack is the right fit

Use **RLlib** because it is built for scalable RL workloads and supports Gymnasium environments, offline data workflows, and customizable RLModules for arbitrary PyTorch architectures. Use **W&B** because it logs metrics, hyperparameters, system metrics, and artifacts. Use **DVC** because it versions pipelines, parameters, metrics, and model artifacts in a way that feels like real ML systems work. Use **OR-Tools** because it gives you a real receding-horizon optimization baseline rather than toy heuristics. Gymnasium wrappers remain the cleanest place to expose a training-friendly observation/action boundary over the deterministic runtime. 

## Core design decisions for Phase 2

### 1. Keep the control problem single-agent and orbital-only
Do not introduce street-level dispatch or multi-agent orchestration yet. The prestige move here is **strong orbital control with strong baselines**, not premature hierarchy.

### 2. Separate benchmark metrics from training reward
The leaderboard contract must stay planner-agnostic. Training reward can be shaped, clipped, normalized, or decomposed more aggressively, but benchmark comparison should still be based on the same Phase 0/1 metrics.

### 3. Convert the dynamic candidate universe into a fixed policy interface
The training wrapper should present:

- `global_features`
- `candidate_features` with shape `[K, F]`
- `action_mask` with shape `[K + 1]` including `noop`
- `candidate_slot_to_action_id` mapping in `info`

Use a deterministic top-K projection so the policy sees a stable-sized interface. Keep `K` configurable. A good default is `K=64`.

### 4. Use explicit action masking
Illegal actions should be masked before sampling or argmax, not merely penalized after the fact.

### 5. Build an expert-data ladder
Generate traces from:
- urgency-greedy
- value-density greedy
- OR-Tools receding-horizon

Use those traces to:
- inspect policy targets
- build offline datasets
- warm-start policy learning with behavior cloning

### 6. Use a structured policy, not just a flat MLP
The flagship model for Phase 2 should be a **candidate-attention policy**:
- shared encoder over candidates
- pooled context from global features
- masked logits over candidate slots
- value head over pooled state

A flat MLP baseline is still worth keeping for ablations.

### 7. Make every trained checkpoint replayable in the same stack
A trained model should be loadable through the planner API and should emit replay/inference traces the UI can inspect.

## Definition of done for Phase 2

Phase 2 is complete when all of the following are true:

1. There is a **training-ready env interface** with stable tensor shapes and deterministic candidate projection.
2. There are stronger baselines:
   - `value_density_greedy`
   - `ortools_receding_horizon`
3. There are **train / val / test / OOD** scenario registries and reproducible dataset builds.
4. There is a canonical **offline trajectory dataset** builder.
5. There is a **candidate-attention RL policy** trainable with RLlib.
6. There is a **BC pretraining** path and a **PPO fine-tuning** path.
7. There is an evaluation suite that produces:
   - per-family metrics
   - paired comparisons vs baselines
   - confidence intervals
   - notable failure and success episodes
8. Trained checkpoints can be:
   - registered
   - evaluated
   - loaded via API
   - replayed in the Cesium UI
9. There is a clear Phase 2 completion report, even if the RL policy still loses to OR-Tools on some families.

## Recommended repo additions

```text
orbital-shepherd/
  apps/
    api/
    web/
  packages/
    contracts/
    core/
    env_runtime/
    benchmark/
    policy_interface/
    policy_models/
    expert_data/
    training_eval/
  training/
    configs/
      reward/
      curriculum/
      model/
      ppo/
      bc/
      eval/
    datasets/
    checkpoints/
    reports/
  docs/
    phase2/
  scripts/
```

## Suggested scenario-pack strategy for Phase 2

Phase 1 probably has a small deterministic pack. Phase 2 needs a proper split registry.

Recommended minimum:

- **Train:** 48 deterministic scenarios
- **Validation:** 12 deterministic scenarios
- **Test:** 18 deterministic scenarios
- **OOD / stress:** 12 deterministic scenarios

Recommended family mix:

- Sparse Frontier
- Burst Outbreak
- Cloud Trap
- Downlink Crunch
- Station Outage
- Constellation Degradation

Use curriculum stages so the easiest families come first.

## Run order for the Codex sessions

1. Training contract, reward/curriculum packs, and experiment scaffolding
2. Training-ready env wrappers and fixed action interface
3. Strong baselines: value-density greedy and OR-Tools receding-horizon
4. Expert trace extraction and offline dataset compiler
5. Candidate-attention policy model and RLlib training scaffold
6. Behavior cloning pretraining and PPO fine-tuning
7. Evaluation harness, robustness suite, and statistical reporting
8. Model registry, trained-policy inference adapter, and API integration
9. UI integration for trained-policy replay and compare mode
10. End-to-end hardening, DVC pipeline, and Phase 2 completion report

---

# Copy-paste Codex prompts

Each prompt below is written to be executable in a fresh Codex session. They assume the repository already contains prior merged work, but they do **not** assume chat history. Run them in order.

---

## Prompt 1 — Training contract, reward/curriculum packs, and experiment scaffolding

```text
You are working in the Orbital Shepherd repository. Do not assume prior chat history.

Start by inspecting the repo and any Phase 0 / Phase 1 docs, especially anything like:
- docs/phase0/
- docs/phase1/
- phase1 completion reports
- existing scenario packs, env runtime, benchmark runner, and API/UI code

Treat the current repo state as the source of truth. Improve and normalize what exists rather than rewriting it.

Your task is to create the Phase 2 training foundation.

Objectives:
1. Write `docs/phase2/ADR-0001-phase2-training-contract.md` that defines:
   - the Phase 2 scope
   - what remains out of scope
   - the relationship between benchmark metrics and training reward
   - the Phase 2 scenario split contract (train / val / test / OOD)
   - checkpoint, dataset, and experiment artifact expectations
2. Add Phase 2 config models and config files for:
   - reward shaping
   - curriculum stages
   - model architecture
   - PPO training
   - behavior cloning
   - evaluation
3. Extend the scenario generation/registry layer to support the full Phase 2 family set:
   - Sparse Frontier
   - Burst Outbreak
   - Cloud Trap
   - Downlink Crunch
   - Station Outage
   - Constellation Degradation
4. Create a deterministic scenario split registry, for example:
   - `training/configs/curriculum/phase2_splits.yaml`
   - `training/configs/curriculum/phase2_curriculum.yaml`
5. Add deterministic builders/CLIs/scripts to generate the Phase 2 packs on demand instead of committing giant generated blobs.
6. Add experiment-management scaffolding:
   - W&B integration that is optional and can run disabled locally
   - DVC pipeline scaffolding that works locally without a remote
   - stable directories for datasets, checkpoints, and reports
7. Define typed manifests for:
   - `TrainingPackManifest`
   - `OfflineDatasetManifest`
   - `PolicyCheckpointManifest`
   - `EvaluationReportManifest`
8. Document all of this in `docs/phase2/training-foundation.md`.

Important design guidance:
- Keep Phase 2 orbital-only and single-agent.
- Do not add the tactical street layer.
- Keep benchmark metrics planner-agnostic.
- Training reward may be separate but must remain auditable and decomposed.
- Make the scenario builders deterministic and seed-driven.
- Prefer small manifests plus build scripts over storing huge artifacts in git.

Deliverables:
- Phase 2 ADR
- config layer and config files
- scenario family extensions and split registry
- DVC scaffolding
- optional W&B integration hooks
- typed manifests
- documentation

Acceptance criteria:
- Phase 2 config files validate
- a developer can build the split registry and scenario manifests locally
- docs clearly distinguish train reward from benchmark metrics
- the repo now has a clean foundation for BC + PPO training

At the end:
- run validation/tests for the new configs/manifests
- summarize the scenario split contract and curriculum stages
- list any assumptions you had to make
```

---

## Prompt 2 — Training-ready env wrappers and fixed action interface

```text
You are working in the Orbital Shepherd repository. Do not assume prior chat history.

Inspect the current repo state, especially the deterministic orbital env/runtime, opportunity builder, and replay/event machinery from Phase 1.

Your task is to build the training-facing environment interface for Phase 2.

Objectives:
1. Create a training wrapper over the existing orbital env, for example:
   - `OrbitalTrainingEnv`
   - or a set of Gymnasium wrappers that expose a stable training interface
2. Convert the variable-size candidate universe into a fixed-size, action-masked interface:
   - default configurable top-K candidate projection, with a sensible default like `K=64`
   - slot 0 should be `noop`
   - remaining slots map deterministically to candidate actions
3. Expose a structured observation dict that includes at minimum:
   - `global_features`
   - `candidate_features` with shape `[K, F]`
   - `action_mask` with shape `[K + 1]`
   - optional `candidate_types`, `candidate_ids`, or debug metadata in `info`
4. Add deterministic candidate-ranking/projection logic.
   - projection must be deterministic for identical bundle + seed + tick
   - ties must be broken stably
5. Add feature extraction for candidates and global state.
   - keep it explicit and auditable
   - no hidden privileged future information
6. Add wrappers/helpers for:
   - normalization metadata
   - optional flattening for ablations
   - replay/debug inspection of slot-to-action mapping
7. Ensure training wrappers remain compatible with Gymnasium and the underlying deterministic runtime.
8. Add tests for:
   - observation shapes
   - action mask legality
   - deterministic slot mapping
   - equivalence between chosen slot and underlying runtime action
9. Document the interface in `docs/phase2/training-env.md`.

Important design guidance:
- Do not change the Phase 1 runtime semantics just to simplify training.
- Build wrappers/adapters around the runtime.
- The action mask should prevent invalid actions, not just punish them.
- Prefer explicit features over magical latent-state bundling.
- Keep the training interface suitable for RLlib and future offline learners.

Deliverables:
- training env wrapper(s)
- fixed action projection logic
- structured observation extractor
- deterministic action masking
- tests and docs

Acceptance criteria:
- the wrapped env can reset/step like a Gymnasium env
- the observation dict is stable and well-shaped
- action slots can be mapped back to canonical action IDs
- repeated runs with the same seed produce identical slot mappings and masks

At the end:
- run the relevant tests
- summarize the observation schema, action schema, and top-K policy interface
```

---

## Prompt 3 — Strong baselines: value-density greedy and OR-Tools receding-horizon

```text
You are working in the Orbital Shepherd repository. Do not assume prior chat history.

Inspect the current repo state, especially the planner interface, benchmark runner, and Phase 1 baselines.

Your task is to implement the serious Phase 2 non-learning baselines.

Required new planners:
- `value_density_greedy`
- `ortools_receding_horizon`

Objectives:
1. Implement `value_density_greedy` as a deterministic heuristic that ranks legal actions using an explicit score that combines factors such as:
   - expected mission value
   - freshness / urgency
   - predicted quality
   - cloud risk
   - retarget cost
   - downlink consequences
2. Document the scoring formula clearly and keep it configurable.
3. Implement `ortools_receding_horizon` using OR-Tools.
   - use a finite lookahead horizon over candidate actions
   - keep the model small enough for local execution
   - use deterministic solver settings where practical
4. Ensure both planners act through the same canonical action contract used by the runtime and future trained policies.
5. Add planner traces so replay and evaluation can inspect:
   - chosen action
   - considered candidates
   - planner score components
   - solver objective / status for OR-Tools
6. Add benchmark tests and controlled toy cases where the expected decision is obvious.
7. Update benchmark docs and planner registries.

Constraints:
- Do not give one planner privileged information that another planner lacks.
- Keep the OR-Tools horizon approximation defensible and documented.
- Favor a clear deterministic approximation over an overcomplicated solver that nobody can debug.
- Preserve replayability and inspection quality.

Deliverables:
- value-density greedy planner
- OR-Tools receding-horizon planner
- planner traces and docs
- tests and benchmark integration
- `docs/phase2/strong-baselines.md`

Acceptance criteria:
- both planners can run across the scenario packs
- planner outputs are deterministic under fixed seeds/configs
- replay artifacts include enough trace data to inspect why the planner acted
- the benchmark runner can compare all planners side by side

At the end:
- run benchmark subsets and planner tests
- summarize the heuristic formula and the OR-Tools lookahead approximation
```

---

## Prompt 4 — Expert trace extraction and offline dataset compiler

```text
You are working in the Orbital Shepherd repository. Do not assume prior chat history.

Inspect the current repo state, especially the replay/event layer, benchmark runner, training env wrapper, and planners.

Your task is to create the offline trajectory data layer for Phase 2.

Objectives:
1. Define a canonical offline dataset schema for single-agent orbital control, capturing at minimum:
   - episode metadata
   - observation tensors/features
   - action mask
   - chosen action
   - reward and reward decomposition
   - termination flags
   - next-step linkage or enough data to reconstruct transitions
   - source planner identity
2. Implement a dataset compiler that can generate expert traces from:
   - urgency-greedy
   - value-density greedy
   - OR-Tools receding-horizon
3. Produce datasets for train / val / test splits without data leakage.
4. Persist datasets in a practical format for both analysis and model training.
   - prefer an ergonomic format such as parquet/arrow for canonical storage
   - add an exporter or adapter if needed for RLlib/offline workflows
5. Create typed manifests and fingerprints for each dataset build.
6. Add a CLI/script such as:
   - `build-offline-dataset`
   - `inspect-offline-dataset`
7. Add lightweight dataset cards / docs describing:
   - source planners
   - split definitions
   - feature schema
   - action semantics
   - known limitations
8. Add tests for:
   - schema validity
   - split integrity
   - deterministic dataset builds
   - feature/action alignment

Constraints:
- Do not invent future information during export.
- Keep the dataset auditable and easy to inspect.
- Preserve the mapping from training slots back to canonical actions where needed.
- Build from actual rollouts, not from hypothetical planner labels.

Deliverables:
- offline dataset schema and manifests
- dataset builder and inspection CLI
- deterministic expert datasets
- docs / dataset cards
- tests
- `docs/phase2/offline-datasets.md`

Acceptance criteria:
- datasets can be regenerated deterministically
- train / val / test splits are clean
- a developer can inspect a dataset without opening ad hoc notebooks
- datasets are immediately usable by later BC / offline training code

At the end:
- run the dataset build on a small split
- summarize dataset outputs, formats, and manifests
```

---

## Prompt 5 — Candidate-attention policy model and RLlib training scaffold

```text
You are working in the Orbital Shepherd repository. Do not assume prior chat history.

Inspect the current repo state, especially the training env wrapper, config layer, and offline dataset schemas.

Your task is to build the online RL training scaffold for Phase 2 using RLlib + PyTorch.

Objectives:
1. Add RLlib as the primary online RL training framework for Phase 2.
2. Register the training env cleanly with RLlib.
3. Implement a reusable policy-model package, for example in `packages/policy_models/`, that includes:
   - a shared candidate encoder
   - a global-state encoder
   - a candidate-attention or set-style aggregation mechanism
   - masked action logits over candidate slots + noop
   - a value head
4. Keep the model reusable for both BC and PPO.
5. Implement the RLlib integration using the current recommended API surface in the version you install.
   - prefer current RLModule-based integration if practical
   - document any version-specific compromises clearly
6. Add training scripts/configs for a smoke-test PPO run on a small scenario subset.
7. Ensure checkpoints are written with manifests and enough metadata to reproduce the run.
8. Add training metrics logging hooks and local-friendly defaults.
9. Add tests or smoke validations for:
   - model forward pass
   - action masking behavior
   - env + model compatibility
   - a short end-to-end PPO smoke run
10. Document the stack in `docs/phase2/rllib-training.md`.

Constraints:
- Keep the implementation PyTorch-first.
- Do not hard-wire giant distributed cluster assumptions; local execution should work.
- Keep action masking explicit in the policy logic.
- Preserve deterministic configs and seeds wherever feasible.
- Avoid building a giant custom trainer if RLlib already solves the core loop.

Deliverables:
- policy model package
- RLlib env registration and training scaffold
- PPO smoke config and scripts
- checkpoint manifests
- docs and tests

Acceptance criteria:
- a short PPO training run completes locally
- masked logits behave correctly
- checkpoints are saved with metadata
- the policy/model code is shared cleanly enough for later BC warm-starting

At the end:
- run a smoke training job
- summarize the model architecture and RLlib integration choices
```

---

## Prompt 6 — Behavior cloning pretraining and PPO fine-tuning

```text
You are working in the Orbital Shepherd repository. Do not assume prior chat history.

Inspect the current repo state, especially:
- offline expert datasets
- policy model package
- RLlib training scaffold
- config/manifests for checkpoints and reports

Your task is to implement the two-stage learning path for Phase 2:
1. behavior cloning pretraining from expert traces
2. PPO fine-tuning in the online env

Objectives:
1. Implement a BC training pipeline that trains the same core policy architecture on the offline expert dataset.
2. Make BC configurable by:
   - source planners included
   - split selection
   - loss weighting or class weighting if needed
   - number of epochs / early stopping
3. Export BC checkpoints in a format that can initialize the PPO policy.
4. Implement a PPO fine-tuning path that can:
   - start from scratch
   - start from BC-pretrained weights
5. Add scripts/configs to compare:
   - PPO from scratch
   - BC only
   - BC -> PPO
6. Log training metrics, checkpoint metadata, and report artifacts.
7. Add tests or smoke validations showing:
   - BC can overfit a tiny slice
   - PPO can load BC weights
   - checkpoint manifests remain consistent
8. Document the workflow in `docs/phase2/bc-and-finetune.md`.

Constraints:
- Keep the shared model weights truly shared, not duplicated in subtly incompatible forms.
- Prefer explicit checkpoint translation/adapters over fragile hacks.
- Do not assume W&B credentials are present; local fallback must work.
- Keep the code reproducible and config-driven.

Deliverables:
- BC trainer
- BC checkpoint exporter/importer
- PPO fine-tuning flow
- comparison scripts/configs
- docs and smoke tests

Acceptance criteria:
- BC training runs end to end on the offline dataset
- PPO can initialize from BC weights
- there is a documented command path for scratch vs BC-warm-start runs
- artifacts and manifests are consistent enough for later evaluation

At the end:
- run the smallest practical BC and BC->PPO smoke jobs
- summarize the checkpoint flow and any assumptions
```

---

## Prompt 7 — Evaluation harness, robustness suite, and statistical reporting

```text
You are working in the Orbital Shepherd repository. Do not assume prior chat history.

Inspect the current repo state, especially:
- benchmark runner
- scenario split registry
- planners and trained-policy checkpoints
- replay and metrics artifacts

Your task is to implement the Phase 2 evaluation system.

Objectives:
1. Build an evaluation runner that can compare:
   - random_valid_action
   - urgency_greedy
   - value_density_greedy
   - ortools_receding_horizon
   - trained RL policies
2. Support evaluation on:
   - validation splits
   - held-out test splits
   - OOD / stress packs
3. Produce reports with:
   - core benchmark metrics
   - per-family breakdowns
   - paired planner comparisons
   - confidence intervals (e.g. bootstrap CIs)
   - win/loss/tie summaries by episode
4. Add failure and success slicing:
   - biggest RL wins
   - biggest RL losses
   - cloud-heavy episodes
   - downlink-heavy episodes
   - outage/degradation episodes
5. Emit machine-readable and human-readable reports:
   - JSON
   - Markdown
   - CSV where useful
6. Integrate optional artifact logging for W&B and metrics/plots for DVC.
7. Add tests for:
   - report generation on toy inputs
   - split integrity
   - paired comparison correctness
8. Document the evaluation story in `docs/phase2/evaluation.md`.

Constraints:
- Keep the evaluation code planner-agnostic.
- Do not rank planners using training reward; use benchmark metrics.
- Make it easy to trace any reported metric back to replay artifacts.
- Prefer explicit statistical summaries over vague charts.

Deliverables:
- evaluation runner
- report generators
- confidence-interval logic
- notable-episode selector
- docs and tests

Acceptance criteria:
- a trained policy and all baselines can be compared on held-out splits
- reports are reproducible and saved with manifests
- notable success/failure episodes are surfaced automatically
- another engineer can inspect results without opening notebooks

At the end:
- run the evaluation on a small but nontrivial subset
- summarize the generated report artifacts and where they are stored
```

---

## Prompt 8 — Model registry, trained-policy inference adapter, and API integration

```text
You are working in the Orbital Shepherd repository. Do not assume prior chat history.

Inspect the current repo state, especially:
- checkpoint manifests
- planner API
- runtime step interface
- replay / metrics endpoints
- trained policy code

Your task is to make trained policies first-class planners inside Orbital Shepherd.

Objectives:
1. Implement a model registry / checkpoint loader layer that can:
   - register trained policies
   - list available checkpoints
   - read checkpoint manifests and metadata
   - load models for local CPU inference
2. Implement a `trained_policy` planner adapter that acts through the same action contract as other planners.
3. Extend the API with non-breaking additions to support trained policies, for example:
   - list models
   - get model metadata
   - run a model over an episode
   - optionally request inference traces
4. Add inference trace support so replay can show:
   - top action probabilities or logits
   - selected slot
   - action entropy
   - value estimate if available
   - mask pressure / number of legal actions
5. Ensure the API, benchmark runner, and replay pipeline can all run trained policies consistently.
6. Add tests for:
   - model registration
   - checkpoint loading
   - policy inference through the planner adapter
   - API execution of a trained checkpoint
7. Document the model-serving layer in `docs/phase2/model-registry-and-api.md`.

Constraints:
- Keep inference simple and local-friendly first.
- Do not build distributed serving infrastructure yet.
- Do not break existing baseline routes or contracts.
- Preserve reproducibility: every run should record model ID, checkpoint fingerprint, and config.

Deliverables:
- model registry
- trained-policy planner adapter
- API extensions
- inference traces
- docs and tests

Acceptance criteria:
- trained policies can be selected and run through the same system as baselines
- replay artifacts record which model/checkpoint acted
- the API can expose both metrics and inference traces for trained runs
- local CPU inference is stable for demo/eval purposes

At the end:
- run API tests and one trained-policy demo path
- summarize the added endpoints and registry structure
```

---

## Prompt 9 — UI integration for trained-policy replay and compare mode

```text
You are working in the Orbital Shepherd repository. Do not assume prior chat history.

Inspect the current repo state, especially:
- the Cesium mission-control UI
- replay/event panels
- metrics panel
- any compare-mode or planner-selection code
- new trained-policy API endpoints

Your task is to extend the UI so trained policies feel like first-class citizens.

Objectives:
1. Add trained-policy selection to the existing planner selector / episode runner UX.
2. Add UI support for replaying trained-policy runs and comparing them with baselines.
3. Surface inference-specific telemetry in a technically legible way:
   - selected action slot and canonical action ID
   - top candidate scores/probabilities
   - action entropy
   - value estimate if available
   - mask pressure / legal-action count
4. Add a compare mode that can load at least:
   - one trained-policy run
   - one baseline run
   - side-by-side metrics and event navigation
5. Preserve the mission-control feel:
   - dark theme
   - globe remains the hero surface
   - panels prioritize clarity over flashy graphics
6. Add an evaluation/report viewer if practical, or at least a clean way to browse generated reports and notable episodes.
7. Add at least one smoke test if feasible, otherwise provide a strong manual verification script.
8. Document the UI changes in `docs/phase2/ui-trained-policy.md`.

Constraints:
- Do not add the tactical street map yet.
- Avoid overbuilding a complex design system.
- Prefer technical legibility and replay/debug power over aesthetics.
- Make sure the UI works even when only local file/API artifacts are available.

Deliverables:
- trained-policy UI integration
- compare mode
- inference telemetry panels
- docs and test/manual verification notes

Acceptance criteria:
- a user can run or load a trained-policy episode in the UI
- a user can compare trained-policy metrics against a baseline
- inference traces are visible and understandable
- the UI still feels like one coherent mission-control application

At the end:
- build the UI
- summarize the new flows and any rough edges
```

---

## Prompt 10 — End-to-end hardening, DVC pipeline, and Phase 2 completion report

```text
You are working in the Orbital Shepherd repository. Do not assume prior chat history.

Inspect the full repo state. Your task is to harden the completed Phase 2 stack into a reproducible, demoable RL system.

Objectives:
1. Verify the end-to-end Phase 2 path:
   - build/generate scenario packs
   - build offline expert datasets
   - run BC
   - run PPO (scratch and/or BC warm-start)
   - evaluate against baselines
   - register a trained checkpoint
   - run it through API
   - replay it in the UI
2. Add or finalize a DVC pipeline that captures the major stages:
   - scenario pack build
   - dataset build
   - BC
   - PPO
   - evaluation
   - report generation
3. Add convenient scripts/commands such as:
   - `make phase2-smoke`
   - `make phase2-train`
   - `make phase2-eval`
   - `make phase2-demo`
4. Add determinism / reproducibility checks where practical:
   - stable scenario/dataset fingerprints
   - checkpoint manifests
   - evaluation-report manifests
5. Write `docs/phase2/phase2-completion-report.md` that includes:
   - what was implemented
   - final artifact graph
   - scenario families and split strategy
   - baselines implemented
   - model architectures and training flows
   - evaluation methodology
   - known limitations
   - direct next steps for Phase 3
6. Tighten rough edges:
   - naming consistency
   - configs
   - onboarding docs
   - command ergonomics
   - default local demo paths

Constraints:
- Keep Phase 2 scoped to orbital RL.
- Do not hide failing metrics or unresolved training issues; document them honestly.
- Optimize for a strong engineering demo, not fake benchmark glory.
- Preserve compatibility with the existing Phase 1 platform.

Deliverables:
- end-to-end reproducible Phase 2 workflow
- DVC pipeline and helper commands
- completion report
- final docs polish
- validation / smoke scripts

Acceptance criteria:
- a new engineer can run a small Phase 2 training/eval flow locally
- trained checkpoints can be loaded through the planner API and replayed in the UI
- the repo now feels like a real RL systems project, not disconnected experiments
- the completion report is honest, detailed, and implementation-grounded

At the end:
- run the smallest practical end-to-end validation path
- summarize the final Phase 2 state and any remaining debt
```

---

## Notes on execution strategy

- Keep everything deterministic unless a prompt explicitly calls for optional external tracking.
- Prefer manifests, fingerprints, and config-driven builds over manual one-off scripts.
- Use the existing replay/event pipeline as the backbone of trust.
- Do not let the RL layer fork the system into a notebook-only side project.
- The prestige signal in Phase 2 is not just “I trained PPO.” It is:
  - strong baselines
  - structured policy architecture
  - reproducible offline + online training
  - honest evaluation
  - and trained policies that plug into the same mission-control product surface.
