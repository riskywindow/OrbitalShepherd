"""Phase 3 geo artifact exports."""

from orbital_shepherd_geo_artifacts.exporters import export_region_bundle

PACKAGE_NAME = "geo_artifacts"
PACKAGE_PURPOSE = (
    "Provide spatial artifact helpers and portable exports for "
    "deterministic tactical ingest workflows."
)

__all__ = ["PACKAGE_NAME", "PACKAGE_PURPOSE", "export_region_bundle"]
