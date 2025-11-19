from typing import Union, Any, List, Optional, Dict, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import scanpy as sc
from anndata import AnnData
import pyucell as uc  # pip install pyucell


# -------------------------------------------------------------------
# Generic helpers: set/get vectors & lengths
# -------------------------------------------------------------------

def _is_mosaic_sample(obj: Any) -> bool:
    """Duck-typed check for a MissionBio Mosaic Sample-like object."""
    return (
        hasattr(obj, "protein")
        and hasattr(obj.protein, "row_attrs")
        and hasattr(obj.protein, "col_attrs")
        and ("barcode" in getattr(obj.protein.row_attrs, "keys", lambda: obj.protein.row_attrs.keys())())
    )


def _set_vec(obj: Any, name: str, values: np.ndarray) -> None:
    """
    Store a per-cell vector on the object.

    - AnnData:   adata.obs[name]
    - Mosaic:    sample.protein.row_attrs[name]
    """
    arr = np.asarray(values)

    if isinstance(obj, AnnData):
        obj.obs[name] = arr
    elif _is_mosaic_sample(obj):
        obj.protein.row_attrs[name] = arr
    else:
        raise TypeError(
            f"_set_vec: unsupported object type {type(obj)}. "
            "Expected AnnData or a MissionBio Sample-like object."
        )


def _get_vec(obj: Any, name: str) -> Optional[np.ndarray]:
    """
    Get a per-cell vector from the object.

    Returns None if the field is not present.
    """
    if isinstance(obj, AnnData):
        if name in obj.obs.columns:
            return obj.obs[name].to_numpy()
        return None

    if _is_mosaic_sample(obj):
        keys = getattr(obj.protein.row_attrs, "keys", lambda: obj.protein.row_attrs.keys())()
        if name in keys:
            return np.asarray(obj.protein.row_attrs[name])
        return None

    raise TypeError(
        f"_get_vec: unsupported object type {type(obj)}. "
        "Expected AnnData or a MissionBio Sample-like object."
    )


def _get_len(obj: Any) -> int:
    """Number of cells."""
    if isinstance(obj, AnnData):
        return obj.n_obs
    if _is_mosaic_sample(obj):
        return len(obj.protein.row_attrs["barcode"])
    raise TypeError(
        f"_get_len: unsupported object type {type(obj)}. "
        "Expected AnnData or a MissionBio Sample-like object."
    )


# -------------------------------------------------------------------
# Thresholding helper: simple 1D Otsu
# -------------------------------------------------------------------

def _otsu_1d(x: np.ndarray, nbins: int = 256) -> float:
    """
    Very simple 1D Otsu thresholding on a 1D array.

    Returns a scalar threshold.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0

    hist, bin_edges = np.histogram(x, bins=nbins)
    hist = hist.astype(float)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    weight1 = np.cumsum(hist)
    weight2 = np.cumsum(hist[::-1])[::-1]

    # avoid division by zero
    nonzero_mask = (weight1 > 0) & (weight2 > 0)
    if not np.any(nonzero_mask):
        return np.median(x)

    mean1 = np.cumsum(hist * bin_centers) / np.maximum(weight1, 1e-12)
    mean2 = (np.cumsum((hist * bin_centers)[::-1]) / np.maximum(weight2[::-1], 1e-12))[::-1]

    inter_class_var = weight1 * weight2 * (mean1 - mean2) ** 2
    idx = np.argmax(inter_class_var * nonzero_mask)
    return bin_centers[idx]


# -------------------------------------------------------------------
# UMAP helper for plotting
# -------------------------------------------------------------------

def _ensure_umap_for_plot(
    obj: Any, k_neighbors: int = 15
) -> Tuple[AnnData, str]:
    """
    Return (adata_for_plot, backend), where backend is:
      - "adata"  if obj is AnnData
      - "sample" if obj is a Mosaic Sample (we build an AnnData from protein)

    Ensures that `X_umap` is present (computes neighbors + UMAP if needed).
    """
    if isinstance(obj, AnnData):
        adata = obj
        backend = "adata"
    elif _is_mosaic_sample(obj):
        from copy import deepcopy
        # Build a fresh AnnData from protein for plotting
        adata = _sample_protein_to_adata(obj, layer="Normalized_reads")
        backend = "sample"
    else:
        raise TypeError(
            f"_ensure_umap_for_plot: unsupported object type {type(obj)}. "
            "Expected AnnData or a MissionBio Sample-like object."
        )

    if "X_umap" not in adata.obsm_keys():
        # Quick UMAP using default settings
        sc.pp.neighbors(adata, n_neighbors=k_neighbors, use_rep=None)
        sc.tl.umap(adata)

    return adata, backend


# -------------------------------------------------------------------
# Helper: build AnnData from MissionBio protein assay
# -------------------------------------------------------------------

def _sample_protein_to_adata(sample, layer: str = "Normalized_reads") -> AnnData:
    """
    Convert MissionBio Mosaic `sample.protein` assay to an AnnData object
    suitable for pyUCell.

    - X: sample.protein.layers[layer]  (cells × antibodies)
    - obs.index: barcodes
    - var.index: antibody IDs (sample.protein.col_attrs['id'])
    """
    protein = sample.protein

    available_layers = getattr(protein.layers, "keys", lambda: protein.layers.keys())()
    if layer not in available_layers:
        raise KeyError(
            f"Protein layer {layer!r} not found. "
            f"Available layers: {list(available_layers)}"
        )

    X = protein.layers[layer]
    barcodes = np.asarray(protein.row_attrs["barcode"])
    ids = np.asarray(protein.col_attrs["id"])

    obs = pd.DataFrame(index=barcodes)
    var = pd.DataFrame(index=ids)

    adata = AnnData(X=X, obs=obs, var=var)
    return adata


# -------------------------------------------------------------------
# Helper: find the correct obs column after pyUCell
# -------------------------------------------------------------------

def _get_ucell_obs_column(
    adata: AnnData, cell_type_label: str, signatures: Dict[str, List[str]]
) -> str:
    """
    Try to find which obs column pyUCell wrote the scores to.

    Preference:
      1) cell_type_label
      2) f"{cell_type_label}_UCell"
      3) for each key in signatures: key, f"{key}_UCell"

    Raise KeyError if nothing matches.
    """
    cols = list(adata.obs.columns)

    # 1) direct match
    if cell_type_label and cell_type_label in cols:
        return cell_type_label

    # 2) label + '_UCell'
    candidate = f"{cell_type_label}_UCell"
    if cell_type_label and candidate in cols:
        return candidate

    # 3) check all signature keys
    candidates = []
    for k in signatures.keys():
        if k in cols:
            candidates.append(k)
        k_ucell = f"{k}_UCell"
        if k_ucell in cols:
            candidates.append(k_ucell)

    candidates = list(dict.fromkeys(candidates))  # unique

    if len(candidates) == 1:
        return candidates[0]

    raise KeyError(
        f"Could not find UCell score column for label '{cell_type_label}'. "
        f"Tried '{cell_type_label}', '{cell_type_label}_UCell', and keys from signatures. "
        f"Found candidate columns: {candidates or 'none'}. "
        f"Available obs columns include (first 20): {cols[:20]}"
    )


# -------------------------------------------------------------------
# Core: add_signature_annotation (UCell-based)
# -------------------------------------------------------------------

def add_signature_annotation(
    obj: Union[AnnData, Any],
    positive_markers: Optional[List[str]] = None,
    negative_markers: Optional[List[str]] = None,
    cell_type_label: str = "",
    *,
    signatures: Optional[Dict[str, List[str]]] = None,
    thresh: Optional[float] = None,
    q: float = 0.99,
    k_neighbors: int = 15,
    field_out: str = "CommonDetailed.Celltype.Refined",
    signature_key: Optional[str] = None,
    show_plots: bool = False,
) -> Union[AnnData, Any]:
    """
    Label cells matching a marker signature and write to `field_out`.

    This version ALWAYS uses UCell for the signature score.

    Two ways to specify the signature:

    1) Explicit UCell signatures dict:
        signatures = {"Mast": ["CD2", "CD25", "CD117", "CD3-", "CD19-"]}

    2) Separate positive/negative lists:
        positive_markers = ["CD2", "CD25", "CD117"]
        negative_markers = ["CD3", "CD19"]
        -> internally converted to signatures = {"<cell_type_label>": [...]}

    Behaviour:
    - If `obj` is an AnnData: run pyUCell directly on it
      (max_rank = min(800, n_features)).
    - If `obj` is a MissionBio Mosaic Sample (has `.protein.col_attrs['id']`):
      build an AnnData from sample.protein and run pyUCell
      (max_rank = round(#antibodies / 2)).

    The resulting UCell scores (0–1) are used as `sig` for thresholding.
    """
    signature_key = signature_key or f"{cell_type_label}_signature"

    # ------------------------------------------------------------------
    # Build or validate `signatures` dict
    # ------------------------------------------------------------------
    if signatures is None:
        if cell_type_label == "":
            raise ValueError(
                "cell_type_label must be provided when `signatures` is None."
            )
        if positive_markers is None:
            positive_markers = []
        if negative_markers is None:
            negative_markers = []

        sig_genes = list(positive_markers) + [f"{g}-" for g in negative_markers]
        signatures = {cell_type_label: sig_genes}
    else:
        if cell_type_label == "":
            if len(signatures) != 1:
                raise ValueError(
                    "When `cell_type_label` is empty, `signatures` must have exactly "
                    "one key to infer the label."
                )
            cell_type_label = next(iter(signatures.keys()))

    # ------------------------------------------------------------------
    # Decide how to get an AnnData view for UCell & run pyUCell
    # ------------------------------------------------------------------
    if isinstance(obj, AnnData):
        adata = obj
        max_rank = int(min(800, adata.n_vars))
        uc.compute_ucell_scores(adata, signatures=signatures, max_rank=max_rank)
    else:
        # MissionBio Sample?
        if not _is_mosaic_sample(obj):
            raise TypeError(
                "add_signature_annotation currently supports AnnData objects "
                "and MissionBio Mosaic Samples (with .protein.row_attrs['barcode'])."
            )
        sample = obj
        adata = _sample_protein_to_adata(sample, layer="Normalized_reads")
        n_features = len(sample.protein.col_attrs["id"])
        max_rank = int(round(n_features / 2))
        uc.compute_ucell_scores(adata, signatures=signatures, max_rank=max_rank)

    # ------------------------------------------------------------------
    # Extract the correct UCell score column
    # ------------------------------------------------------------------
    score_col = _get_ucell_obs_column(adata, cell_type_label, signatures)
    sig = adata.obs[score_col].to_numpy()

    # Store the UCell signature scores on the original object
    _set_vec(obj, signature_key, sig)

    # ------------------------------------------------------------------
    # Thresholding to call positives
    # ------------------------------------------------------------------
    if thresh is None:
        if np.unique(sig).size > 50:
            thresh = np.quantile(sig, q)
        else:
            thresh = _otsu_1d(sig)

    mask = sig > thresh
    print(f"[{cell_type_label}] threshold {thresh:.3f} → {int(mask.sum())} cells")

    # ------------------------------------------------------------------
    # Write labels to field_out on the *original* object
    # ------------------------------------------------------------------
    n = _get_len(obj)
    existing = _get_vec(obj, field_out)
    if existing is None:
        fallback = _get_vec(obj, "CommonDetailed.Celltype")
        existing = (
            fallback.astype(object, copy=True)
            if fallback is not None
            else np.array(["Unknown"] * n, dtype=object)
        )
    else:
        existing = existing.astype(object, copy=True)

    existing[mask] = cell_type_label
    _set_vec(obj, field_out, existing)

    # If this is an AnnData, drop any existing color map for this field
    if isinstance(obj, AnnData):
        obj.uns.pop(f"{field_out}_colors", None)

    # ------------------------------------------------------------------
    # Optional plots
    # ------------------------------------------------------------------
    if show_plots:
        plt.figure(figsize=(6, 3))
        plt.hist(sig, bins=60)
        plt.axvline(thresh, color="red", ls="--")
        plt.title(f"{cell_type_label} UCell signature")
        plt.tight_layout()
        plt.show()

        a_plot, backend = _ensure_umap_for_plot(obj, k_neighbors=k_neighbors)
        if backend == "sample":
            # rebuild vectors for plotting if needed
            a_plot.obs[field_out] = _get_vec(obj, field_out)
            a_plot.obs[signature_key] = _get_vec(obj, signature_key)

        sc.pl.umap(
            a_plot,
            color=[field_out, signature_key],
            cmap="coolwarm",
            wspace=0.35,
            na_color="lightgrey",
            show=True,
        )

    return obj


# -------------------------------------------------------------------
# Mast-cell specific helpers
# -------------------------------------------------------------------

MAST_SIGNATURE: Dict[str, List[str]] = {
    "Mast": ["CD2", "CD25", "CD30", "CD33", "CD117", "FcεRIα"]
}


def add_mast_annotation(
    obj: Union[AnnData, Any],
    *,
    thresh: Optional[float] = None,
    q: float = 0.99,
    k_neighbors: int = 15,
    field_out: str = "CommonDetailed.Celltype.Refined",
    show_plots: bool = False,
) -> Union[AnnData, Any]:
    """Add a 'Mast' label using the predefined UCell mast signature."""
    return add_signature_annotation(
        obj=obj,
        cell_type_label="Mast",
        signatures=MAST_SIGNATURE,
        thresh=thresh,
        q=q,
        k_neighbors=k_neighbors,
        field_out=field_out,
        show_plots=show_plots,
    )
