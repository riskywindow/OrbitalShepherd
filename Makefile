PYTHON ?= python3
COREPACK_HOME ?= /tmp/corepack
PYTHONPATH := apps/api/src:packages/contracts/python/src:packages/core/src:packages/ephemeris/src:packages/scenario_engine/src:packages/opportunity_builder/src:packages/env_runtime/src:packages/benchmark/src:packages/policy_models/src:packages/training/src

export PYTHONPATH

PHASE2_DATASET_BUILD_ID ?= offbuild:phase2:bc-bootstrap-smoke-v1
PHASE2_BC_CONFIG ?= training/configs/bc/phase2_bc_smoke.yaml
PHASE2_PPO_CONFIG ?= training/configs/ppo/phase2_ppo_bc_warmstart_smoke.yaml
PHASE2_BC_ALIAS ?= data/training/manifests/phase2-bc-latest.json
PHASE2_PPO_ALIAS ?= data/training/manifests/phase2-ppo-latest.json
PHASE2_POLICY_LABEL ?= ppo_from_bc_smoke
PHASE2_EVAL_RUN_ID ?= evalrun--phase2-heldout-smoke-v1
PHASE2_EVAL_LIMIT ?= 1
PHASE2_BASELINE_ID ?= urgency_greedy

PHASE3_ROUTING_COMPOSE_FILE ?= infra/compose/phase3-routing.compose.yaml
PHASE3_ROUTING_DSN ?= postgresql://orbital:orbital@127.0.0.1:55432/orbital_shepherd_phase3

.PHONY: lint typecheck test contracts-validate scenario-pack-build scenario-pack-validate benchmark-run api-dev web-install web-dev web-build web-lint web-format-check phase1-prepare phase1-verify phase2-config-validate phase2-split-build phase2-pack-build phase2-pack-validate phase2-dataset-build phase2-bc-train phase2-ppo-train phase2-smoke phase2-train phase2-eval phase2-demo phase2-demo-prepare phase3-routing-up phase3-routing-down phase3-routing-bootstrap phase3-routing-ingest-fixture phase3-routing-smoke phase3-tactical-pack-build phase3-tactical-pack-validate demo quickstart check

lint:
	ruff check apps/api/src packages tests

typecheck:
	mypy

test:
	pytest

contracts-validate:
	$(PYTHON) -m orbital_shepherd_contracts validate

scenario-pack-build:
	$(PYTHON) scripts/build_scenario_pack.py

scenario-pack-validate:
	$(PYTHON) scripts/validate_scenario_pack.py

benchmark-run:
	$(PYTHON) scripts/run_phase1_benchmark.py $(BENCH_ARGS)

api-dev:
	$(PYTHON) scripts/run_phase1_api.py --host 127.0.0.1 --port 8000

web-install:
	COREPACK_HOME=$(COREPACK_HOME) pnpm install

web-dev:
	COREPACK_HOME=$(COREPACK_HOME) pnpm --dir apps/web dev --host 127.0.0.1 --port 3000

web-build:
	COREPACK_HOME=$(COREPACK_HOME) pnpm --dir apps/web build

web-lint:
	COREPACK_HOME=$(COREPACK_HOME) pnpm --dir apps/web lint

web-format-check:
	COREPACK_HOME=$(COREPACK_HOME) pnpm --dir apps/web format:check

phase1-prepare:
	$(PYTHON) scripts/phase1_demo.py prepare

phase1-verify:
	$(PYTHON) scripts/verify_phase1_stack.py

phase2-config-validate:
	$(PYTHON) scripts/phase2_training.py validate-configs

phase2-split-build:
	$(PYTHON) scripts/phase2_training.py build-split-registry

phase2-pack-build:
	$(PYTHON) scripts/phase2_training.py build-training-pack

phase2-pack-validate:
	$(PYTHON) scripts/phase2_training.py validate-training-pack

phase2-dataset-build:
	$(PYTHON) scripts/phase2_training.py build-offline-dataset --training-pack-manifest data/training/manifests/phase2-training-pack-manifest.json --split-registry training/configs/curriculum/phase2_splits.yaml --output-root data/training/datasets/osbench-phase2-foundation-v1 --manifest-root data/training/manifests --planner urgency_greedy --split train --split val --top-k 64 --limit-bundles-per-split 1 --build-id $(PHASE2_DATASET_BUILD_ID)

phase2-bc-train:
	$(PYTHON) scripts/phase2_training.py train-bc --config $(PHASE2_BC_CONFIG) --best-checkpoint-alias $(PHASE2_BC_ALIAS)

phase2-ppo-train:
	$(PYTHON) scripts/phase2_training.py train-ppo --config $(PHASE2_PPO_CONFIG) --latest-checkpoint-alias $(PHASE2_PPO_ALIAS)

phase2-smoke:
	$(PYTHON) scripts/verify_phase2_stack.py

phase2-train: phase2-pack-build phase2-pack-validate phase2-dataset-build phase2-bc-train phase2-ppo-train

phase2-eval:
	$(PYTHON) scripts/phase2_demo.py evaluate --checkpoint-manifest $(PHASE2_PPO_ALIAS) --policy-label $(PHASE2_POLICY_LABEL) --run-id $(PHASE2_EVAL_RUN_ID) --limit-bundles-per-split $(PHASE2_EVAL_LIMIT)

phase2-demo-prepare:
	$(PYTHON) scripts/phase2_demo.py prepare --checkpoint-manifest $(PHASE2_PPO_ALIAS) --baseline-id $(PHASE2_BASELINE_ID)

phase2-demo:
	$(PYTHON) scripts/phase2_demo.py serve --checkpoint-manifest $(PHASE2_PPO_ALIAS) --baseline-id $(PHASE2_BASELINE_ID)

phase3-routing-up:
	docker compose -f $(PHASE3_ROUTING_COMPOSE_FILE) up -d

phase3-routing-down:
	docker compose -f $(PHASE3_ROUTING_COMPOSE_FILE) down

phase3-routing-bootstrap:
	$(PYTHON) scripts/phase3_routing.py --dsn $(PHASE3_ROUTING_DSN) bootstrap-db

phase3-routing-ingest-fixture:
	$(PYTHON) scripts/phase3_routing.py --dsn $(PHASE3_ROUTING_DSN) ingest-bundle data/fixtures/region_builder/compiled/fixture_micro_region_bundle.json

phase3-routing-smoke:
	$(PYTHON) scripts/phase3_routing.py --dsn $(PHASE3_ROUTING_DSN) smoke

phase3-tactical-pack-build:
	$(PYTHON) scripts/build_tactical_scenario_pack.py

phase3-tactical-pack-validate:
	$(PYTHON) scripts/validate_tactical_scenario_pack.py

demo quickstart:
	$(PYTHON) scripts/phase1_demo.py serve

check: lint typecheck test contracts-validate web-build
