"""Shared Earth Engine initialization for all data_pipeline scripts.

Reuses the GCP project already linked to this machine's GEE credentials
(see Trial v1's gee_extract.py, which authenticated against isro-bah-2026).
"""

import ee

DEFAULT_PROJECT = "isro-bah-2026"

_initialized = False


def init_ee(project: str = DEFAULT_PROJECT) -> None:
    global _initialized
    if _initialized:
        return
    ee.Initialize(project=project)
    _initialized = True
