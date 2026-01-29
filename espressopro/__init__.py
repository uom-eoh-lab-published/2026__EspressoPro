# -*- coding: utf-8 -*-
"""EspressoPro — modular cell type annotation pipeline."""

from __future__ import annotations

# ---------------------------- model_loading ----------------------------

from .model_loading import (
    load_models,
    get_package_data_path,
    get_default_models_path,
    get_default_data_path,
    download_models,
    ensure_models_available,
)

# ------------------------- prediction / scoring ------------------------
# Prediction API varies across package versions; import defensively.

_HAS_STACK_PREDICTION = False
_HAS_GENERATE_PREDICTIONS = False
_HAS_AUDIT_FEATURE_OVERLAP = False
_HAS_BEST_LOCALISED = False
_HAS_CONSENSUS = False
_HAS_CONSENSUS_SAMPLE = False
_HAS_UNWEIGHTED_AVG = False

try:
    from .prediction import stack_prediction  # noqa: F401
    _HAS_STACK_PREDICTION = True
except Exception:
    pass

try:
    from .prediction import generate_predictions  # noqa: F401
    _HAS_GENERATE_PREDICTIONS = True
except Exception:
    pass

try:
    from .prediction import audit_feature_overlap  # noqa: F401
    _HAS_AUDIT_FEATURE_OVERLAP = True
except Exception:
    pass

try:
    from .prediction import add_best_localised_tracks  # noqa: F401
    _HAS_BEST_LOCALISED = True
except Exception:
    pass

try:
    from .prediction import add_consensus_weighted_tracks  # noqa: F401
    _HAS_CONSENSUS = True
except Exception:
    pass

try:
    from .prediction import add_consensus_weighted_tracks_sample  # noqa: F401
    _HAS_CONSENSUS_SAMPLE = True
except Exception:
    pass

try:
    from .prediction import add_unweighted_average_tracks, add_unweighted_average_tracks_sample  # noqa: F401
    _HAS_UNWEIGHTED_AVG = True
except Exception:
    pass

# ------------------------------ annotation ----------------------------

from .annotation import (
    voting_annotator,
    Broad_Annotation,
    Simplified_Annotation,
    Detailed_Annotation,
    annotate_data,
    mark_small_clusters,
    mark_mixed_clusters,
    refine_labels_by_knn_consensus,
    clear_annotation,
    score_mixed_clusters,
)

# Optional: annotate_counts_matrix (only if defined)
_HAS_ANNOTATE_COUNTS = False
try:
    from .annotation import annotate_counts_matrix  # type: ignore
    _HAS_ANNOTATE_COUNTS = True
except Exception:
    pass

# ------------------------ protein preprocessing ------------------------

from .protein_preprocessing import (
    Normalise_protein_data,
    Scale_protein_data,
)

# ------------------------------ missionbio ----------------------------

from .missionbio import (
    suggest_cluster_celltype_identity,
    reassign_disconnected_cells,
    print_disconnected_summary,
    print_cluster_suggestions,
)

# ------------------------------- markers ------------------------------
# (updated: pyUCell/GMM-based add_signature_annotation + add_mast_annotation live here)

from .markers import add_mast_annotation, add_signature_annotation

# ------------------------------ constants -----------------------------

from .constants import (
    SIMPLIFIED_CLASSES,
    DETAILED_CLASSES,
    SIMPLIFIED_PARENT_MAP,
    DETAILED_PARENT_MAP,
)

__version__ = "1.0.0"

__all__ = [
    # model_loading
    "load_models",
    "get_package_data_path",
    "get_default_models_path",
    "get_default_data_path",
    "download_models",
    "ensure_models_available",

    # Annotation
    "voting_annotator",
    "Broad_Annotation",
    "Simplified_Annotation",
    "Detailed_Annotation",
    "annotate_data",
    "Normalise_protein_data",
    "Scale_protein_data",
    "mark_small_clusters",
    "refine_labels_by_knn_consensus",
    "clear_annotation",
    "score_mixed_clusters",
    "mark_mixed_clusters",

    # MissionBio
    "suggest_cluster_celltype_identity",
    "reassign_disconnected_cells",
    "print_disconnected_summary",
    "print_cluster_suggestions",

    # Markers
    "add_mast_annotation",
    "add_signature_annotation",

    # Constants
    "SIMPLIFIED_CLASSES",
    "DETAILED_CLASSES",
    "SIMPLIFIED_PARENT_MAP",
    "DETAILED_PARENT_MAP",
]

# Prediction exports: only if import succeeded
if _HAS_STACK_PREDICTION:
    __all__.append("stack_prediction")
if _HAS_GENERATE_PREDICTIONS:
    __all__.append("generate_predictions")
if _HAS_AUDIT_FEATURE_OVERLAP:
    __all__.append("audit_feature_overlap")
if _HAS_BEST_LOCALISED:
    __all__.append("add_best_localised_tracks")
if _HAS_CONSENSUS:
    __all__.append("add_consensus_weighted_tracks")
if _HAS_CONSENSUS_SAMPLE:
    __all__.append("add_consensus_weighted_tracks_sample")
if _HAS_UNWEIGHTED_AVG:
    __all__.extend(["add_unweighted_average_tracks", "add_unweighted_average_tracks_sample"])

if _HAS_ANNOTATE_COUNTS:
    __all__.append("annotate_counts_matrix")
