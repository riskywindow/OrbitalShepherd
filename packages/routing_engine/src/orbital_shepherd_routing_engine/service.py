from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

from orbital_shepherd_contracts import RegionBundle, load_json
from orbital_shepherd_routing_engine.config import RoutingEngineConfig
from orbital_shepherd_routing_engine.memory_backend import InMemoryRoutingBackend
from orbital_shepherd_routing_engine.models import (
    EtaMatrixRequest,
    EtaMatrixResult,
    OverlayRegistrationResult,
    OverlaySpecType,
    RegionIngestResult,
    ServiceAreaRequest,
    ServiceAreaResult,
    ShortestPathRequest,
    ShortestPathResult,
)
from orbital_shepherd_routing_engine.postgres_backend import PostgresRoutingBackend


class RoutingBackend(Protocol):
    def apply_migrations(self) -> list[str]: ...

    def ingest_region_bundle(
        self,
        bundle: RegionBundle | Mapping[str, object],
    ) -> RegionIngestResult: ...

    def register_overlay(
        self,
        overlay: OverlaySpecType | Mapping[str, object],
    ) -> OverlayRegistrationResult: ...

    def shortest_path(self, request: ShortestPathRequest) -> ShortestPathResult: ...

    def eta_matrix(self, request: EtaMatrixRequest) -> EtaMatrixResult: ...

    def service_area(self, request: ServiceAreaRequest) -> ServiceAreaResult: ...


class RoutingEngineService:
    def __init__(self, backend: RoutingBackend) -> None:
        self.backend = backend

    @classmethod
    def in_memory(cls) -> RoutingEngineService:
        return cls(InMemoryRoutingBackend())

    @classmethod
    def postgres(cls, config: RoutingEngineConfig | None = None) -> RoutingEngineService:
        return cls(PostgresRoutingBackend(config))

    def bootstrap_database(self) -> list[str]:
        return self.backend.apply_migrations()

    def ingest_region_bundle(
        self,
        bundle: RegionBundle | Mapping[str, object],
    ) -> RegionIngestResult:
        return self.backend.ingest_region_bundle(bundle)

    def ingest_region_bundle_path(self, bundle_path: Path | str) -> RegionIngestResult:
        bundle = load_json(Path(bundle_path))
        return self.ingest_region_bundle(bundle)

    def register_overlay(
        self,
        overlay: OverlaySpecType | Mapping[str, object],
    ) -> OverlayRegistrationResult:
        return self.backend.register_overlay(overlay)

    def shortest_path(self, request: ShortestPathRequest) -> ShortestPathResult:
        return self.backend.shortest_path(request)

    def eta_matrix(self, request: EtaMatrixRequest) -> EtaMatrixResult:
        return self.backend.eta_matrix(request)

    def service_area(self, request: ServiceAreaRequest) -> ServiceAreaResult:
        return self.backend.service_area(request)


def load_region_bundle(source: Path | str | Mapping[str, Any]) -> RegionBundle:
    if isinstance(source, Mapping):
        return RegionBundle.model_validate(source)
    return RegionBundle.model_validate(load_json(Path(source)))
