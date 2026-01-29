# -*- coding: utf-8 -*-
"""Marker-based cell type detection and annotation (AnnData + MissionBio Sample)."""

from __future__ import annotations

from typing import Any, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc

try:
    from anndata import AnnData
except Exception:  # soft import
    AnnData = object  # type: ignore[misc,assignment]

from .constants import MAST_NEG, MAST_POS


# --------------------------------- I/O helpers ---------------------------------

def _is_sample(x: Any) -> bool:
    return hasattr(x, "protein") and hasattr(x.protein, "row_attrs") and hasattr(x.protein, "get_attribute")


def _get_feature_matrix_and_names(obj: Union[AnnData, Any]) -> Tuple[pd.DataFrame, pd.Index]:
    """
    Return features as a DataFrame and its column index.
    AnnData: .X / .var_names
    Sample : protein.get_attribute('Scaled_reads', 'row+col')
    """
    if isinstance(obj, AnnData):
        X = obj.X.A if hasattr(obj.X, "A") else (obj.X.toarray() if hasattr(obj.X, "toarray") else obj.X)
        df = pd.DataFrame(X, index=obj.obs_names.astype(str), columns=obj.var_names.astype(str))
        return df, df.columns

    if _is_sample(obj):
        df = obj.protein.get_attribute("Scaled_reads", constraint="row+col")
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)
        df.index = df.index.astype(str)
        df.columns = df.columns.astype(str)
        df = df.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return df, df.columns

    raise TypeError("Expected AnnData or missionbio.mosaic.sample.Sample")


def _get_len(obj: Union[AnnData, Any]) -> int:
    if isinstance(obj, AnnData):
        return obj.n_obs
    if _is_sample(obj):
        df, _ = _get_feature_matrix_and_names(obj)
        return df.shape[0]
    raise TypeError("Expected AnnData or missionbio.mosaic.sample.Sample")


def _get_vec(obj: Union[AnnData, Any], key: str) -> Optional[np.ndarray]:
    if isinstance(obj, AnnData):
        return obj.obs[key].to_numpy() if key in obj.obs.columns else None
    if _is_sample(obj):
        return np.asarray(obj.protein.row_attrs[key]).reshape(-1) if key in obj.protein.row_attrs else None
    return None


def _set_vec(obj: Union[AnnData, Any], key: str, vec: np.ndarray) -> None:
    if isinstance(obj, AnnData):
        obj.obs[key] = vec
        return
    if _is_sample(obj):
        obj.protein.row_attrs[key] = np.asarray(vec)
        return
    raise TypeError("Expected AnnData or missionbio.mosaic.sample.Sample")


def _ensure_umap_for_plot(obj: Union[AnnData, Any], k_neighbors: int = 15) -> Tuple["AnnData", str]:
    """
    Ensure a 2D UMAP exists and return an AnnData to plot plus backend flag ('anndata'|'sample').
    """
    if isinstance(obj, AnnData):
        if "X_umap" not in obj.obsm:
            sc.pp.neighbors(obj, n_neighbors=k_neighbors)
            sc.tl.umap(obj)
        return obj, "anndata"

    if _is_sample(obj):
        df, _ = _get_feature_matrix_and_names(obj)
        idx = df.index.astype(str)
        meta = pd.DataFrame(index=idx)
        for k, v in obj.protein.row_attrs.items():
            arr = np.asarray(v)
            if arr.ndim == 1 and arr.shape[0] == len(idx):
                meta[k] = arr
        a = sc.AnnData(X=np.zeros((len(idx), 0), dtype=float), obs=meta, var=pd.DataFrame(index=[]))
        if "umap" not in obj.protein.row_attrs:
            raise KeyError("row_attrs['umap'] not found for plotting")
        U = np.asarray(obj.protein.row_attrs["umap"])
        if U.ndim != 2 or U.shape[1] < 2:
            raise ValueError("row_attrs['umap'] must be (n_cells, ≥2)")
        a.obsm["X_umap"] = U[:, :2]
        return a, "sample"

    raise TypeError("Expected AnnData or missionbio.mosaic.sample.Sample")


# --------------------------------- core logic ----------------------------------

def _mean_marker_score(obj: Union[AnnData, Any], genes: List[str], key: str) -> np.ndarray:
    """Mean of selected features per cell; stores to `key`."""
    df, names = _get_feature_matrix_and_names(obj)
    keep = [g for g in genes if g in names]
    out = np.zeros(df.shape[0], dtype=float) if not keep else df[keep].mean(axis=1).to_numpy()
    _set_vec(obj, key, out)
    return out


def _otsu_1d(x: np.ndarray) -> float:
    """Otsu threshold for 1D data."""
    h, bins = np.histogram(x, 256)
    mids = (bins[:-1] + bins[1:]) / 2
    w1 = np.cumsum(h)
    w2 = np.cumsum(h[::-1])[::-1]
    m1 = np.cumsum(h * mids) / (w1 + 1e-9)
    m2 = (np.cumsum((h * mids)[::-1]) / (w2 + 1e-9))[::-1]
    var = w1[:-1] * w2[1:] * (m1[:-1] - m2[1:]) ** 2
    return mids[np.argmax(var)]


def add_signature_annotation(
    obj: Union[AnnData, Any],
    positive_markers: List[str],
    negative_markers: List[str],
    cell_type_label: str,
    *,
    thresh: Optional[float] = None,
    q: float = 0.99,
    k_neighbors: int = 15,
    field_out: str = "CommonDetailed.Celltype.Refined",
    signature_key: Optional[str] = None,
    show_plots: bool = False,
) -> Union[AnnData, Any]:
    """
    Label cells matching a marker signature and write to `field_out`.
    Signature = mean(positive_markers) - mean(negative_markers).
    """
    signature_key = signature_key or f"{cell_type_label}_signature"

    pos = _mean_marker_score(obj, positive_markers, f"{cell_type_label}_pos")
    neg = _mean_marker_score(obj, negative_markers, f"{cell_type_label}_neg")
    sig = pos - neg
    _set_vec(obj, signature_key, sig)

    if thresh is None:
        thresh = np.quantile(sig, q) if np.unique(sig).size > 50 else _otsu_1d(sig)

    mask = sig > thresh
    print(f"[{cell_type_label}] threshold {thresh:.3f} → {int(mask.sum())} cells")

    n = _get_len(obj)
    existing = _get_vec(obj, field_out)
    if existing is None:
        fallback = _get_vec(obj, "CommonDetailed.Celltype")
        existing = (fallback.astype(object, copy=True) if fallback is not None
                    else np.array(["Unknown"] * n, dtype=object))
    else:
        existing = existing.astype(object, copy=True)

    existing[mask] = cell_type_label
    _set_vec(obj, field_out, existing)

    if isinstance(obj, AnnData):
        obj.uns.pop(f"{field_out}_colors", None)

    if show_plots:
        plt.figure(figsize=(6, 3))
        plt.hist(sig, bins=60)
        plt.axvline(thresh, color="red", ls="--")
        plt.title(f"{cell_type_label} signature")
        plt.tight_layout()
        plt.show()

        a_plot, backend = _ensure_umap_for_plot(obj, k_neighbors=k_neighbors)
        if backend == "sample":
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


def add_mast_annotation(
    obj: Union[AnnData, Any],
    *,
    thresh: Optional[float] = None,
    q: float = 0.99,
    k_neighbors: int = 15,
    field_out: str = "CommonDetailed.Celltype.Refined",
    show_plots: bool = False,
) -> Union[AnnData, Any]:
    """Add a 'Mast' label using predefined marker sets."""
    return add_signature_annotation(
        obj=obj,
        positive_markers=MAST_POS,
        negative_markers=MAST_NEG,
        cell_type_label="Mast",
        thresh=thresh,
        q=q,
        k_neighbors=k_neighbors,
        field_out=field_out,
        show_plots=show_plots,
    )
