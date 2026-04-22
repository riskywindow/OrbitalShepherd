# Phase 1 Engineering Foundation

## Purpose

This document records the implementation choices made for the first Phase 1 scaffold. The Phase 0 documents at the repo root remain the contract source of truth; this foundation only creates the structure and deterministic utilities required to begin implementation.

## Foundation decisions

### Modular package layout

The repository is organized so that later prompts can add functionality without refactoring the top-level shape:

- `apps/api` for the future planner, replay, and metrics service
- `apps/web` for the Cesium-facing mission-control UI shell
- `packages/contracts` for references to the Phase 0 schemas and examples
- `packages/core` for deterministic shared helpers
- `packages/scenario_engine`, `packages/opportunity_builder`, `packages/env_runtime`, and `packages/benchmark` for the first implementation slices named in the architecture docs
- `data/fixtures`, `data/scenarios`, and `data/replays` for local file-backed assets

### Determinism-first utilities

`packages/core` is the first reusable package because Phase 0 requires deterministic, replayable artifacts. The utilities included here are intentionally small and dependency-light:

- canonical JSON serialization for stable artifact bytes
- SHA-256 fingerprints for bundle and event identity work
- strict UTC parsing and formatting for contract compliance
- seeded RNG helpers for reproducible scenario generation
- stable ID helpers for namespaced identifiers

### Tooling

The backend uses a single root `pyproject.toml` to keep linting, typing, and tests consistent across the early workspace. The frontend stays isolated under `apps/web` with Vite tooling.

Selected tools:

- Python: `ruff`, `mypy`, `pytest`
- Frontend: React, TypeScript, Vite, ESLint, Prettier

### Storage

Phase 1 still prefers local file-backed storage. The scaffold therefore creates only directories and helpers for:

- fixtures
- scenario bundles
- replay logs

There is no database container or background service yet.

### No docker compose

`docker-compose.yml` was intentionally not added. At this stage the repository has no service dependency that benefits from container orchestration, and introducing it now would add friction without improving determinism or developer velocity.

## Current API status

`apps/api` contains a placeholder HTTP service that exposes the foundation metadata and confirms the Phase 0 contract expectations. It exists to validate the repo shape and give later prompts a stable entrypoint, not to implement planner logic.

## Next implementation steps

The scaffold is prepared for the next sequence already described in the Phase 1 plan:

1. contract normalization and schema completion
2. scenario engine and validation
3. opportunity generation
4. environment runtime and replay writing
5. baselines, benchmark runner, and UI integration
