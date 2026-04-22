"""FastAPI application package for the Orbital Shepherd Phase 1 API."""

from orbital_shepherd_api.app import create_app


def build_status_document() -> dict[str, object]:
    return {
        "status": "ready-for-phase1-implementation",
        "service": "orbital-shepherd-api",
        "implementation": "fastapi",
    }


__all__ = ["build_status_document", "create_app"]
