"""Phase 3 region builder."""

from orbital_shepherd_region_builder.compiler import (
    RegionBuildRecord,
    compile_manifest_path,
    compile_manifest_to_bundle,
    region_bundle_fingerprint,
    region_bundle_id_from_manifest_id,
)
from orbital_shepherd_region_builder.config import RegionBuilderConfig

PACKAGE_NAME = "region_builder"
PACKAGE_PURPOSE = (
    "Compile deterministic RegionBundle artifacts from manifest recipes "
    "and spatial ingest sources."
)

__all__ = [
    "PACKAGE_NAME",
    "PACKAGE_PURPOSE",
    "RegionBuildRecord",
    "RegionBuilderConfig",
    "compile_manifest_path",
    "compile_manifest_to_bundle",
    "region_bundle_fingerprint",
    "region_bundle_id_from_manifest_id",
]
