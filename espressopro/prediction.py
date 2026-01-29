# -*- coding: utf-8 -*-
"""Prediction/scoring utilities (with scaling) and average (mean/weighted) atlas blending.

UPDATED to support EspressoPro multiclass package structure:
- Multiclass_models.joblib is a dict with keys: {'atlas','depth','panel_name','class_names','heads','temp_scaler'}
- Each head is an OvR binary classifier bundle; probabilities are generated per-class (preferring Platt),
  then optionally temperature-scaled via temp_scaler, then row-normalized.

Retains support for:
- legacy OvR heads stored per label under models[atlas][depth][label] = bundle
- older '__MULTICLASS__' estimator-style bundles with predict_proba()

New features (v2.0):
- Fixed binary temperature scaling for Broad depth (2-class problems)
- Improved agreement calculation with hybrid MAD+range method
- Automated average filtering (Averaged.Consensus.* tracks) for high-confidence predictions
- Better handling of __BUNDLE__ storage format
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import warnings

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata import AnnData
from sklearn.ensemble import StackingClassifier
from sklearn.neighbors import KDTree
from sklearn.pipeline import Pipeline

from .model_loading import (
    ensure_models_available,
    get_default_data_path,
    get_default_models_path,
    load_models,
)
from .constants import SIMPLIFIED_CLASSES, _DETAILED_LABELS


# ----------------------------- DEBUG SWITCHES -----------------------------
DEBUG_PRED = False         # top-level prints per atlas/depth
DEBUG_KEYS_MAX = 30        # show at most N keys when printing dict keys
DEBUG_HEAD_FAILS_MAX = 5   # show at most N head failures per atlas/depth
DEBUG_PRINT_SKIPS = True   # print reasons for skipping
# ------------------------------------------------------------------------


# ----------------------------- helpers -----------------------------

def _dense(arr):
    return arr.toarray() if sp.issparse(arr) else np.asarray(arr)


def _mosaic_feature_names(sample) -> List[str]:
    """Return preferred feature names for a Mosaic sample."""
    prefs = ("target", "antibody", "feature_name", "name", "channel", "ids")
    for key in prefs:
        try:
            vals = sample.protein.col_attrs[key]
            if vals is None:
                continue
            names = [str(x) for x in np.asarray(vals)]
            looks_like_targets = sum(n.upper().startswith("CD") or "HLA" in n.upper() for n in names)
            if looks_like_targets >= max(5, 0.2 * len(names)):
                return names
            if key in ("target", "antibody", "feature_name"):
                return names
        except Exception:
            pass
    return [f"feat_{i}" for i in range(sample.protein.shape[1])]


def _mosaic_cell_index(sample) -> List[str]:
    for key in ("cell_barcode", "barcode", "ids", "cell_id"):
        try:
            vals = sample.protein.row_attrs[key]
            if vals is not None:
                return [str(x) for x in np.asarray(vals)]
        except Exception:
            pass
    return [f"cell_{i}" for i in range(sample.protein.shape[0])]


def _looks_like_estimator(x: object) -> bool:
    return hasattr(x, "predict_proba") or hasattr(x, "decision_function") or hasattr(x, "predict")


def _make_query_df(obj: Any, mosaic_layer: str = "Normalized_reads", **kwargs) -> pd.DataFrame:
    """Build a feature matrix from AnnData / Mosaic Sample / DataFrame."""
    layer = kwargs.get("base_layer", mosaic_layer)

    if isinstance(obj, AnnData):
        X = _dense(obj.X)
        df = pd.DataFrame(X, index=obj.obs_names.astype(str), columns=obj.var_names.astype(str))
        return df.apply(pd.to_numeric, errors="coerce")

    if hasattr(obj, "protein") and hasattr(obj.protein, "get_attribute"):
        try:
            df = obj.protein.get_attribute(layer, constraint="row+col")
            if not isinstance(df, pd.DataFrame):
                df = pd.DataFrame(df)
            df.index = df.index.astype(str)
            df.columns = df.columns.astype(str)
            return df.apply(pd.to_numeric, errors="coerce")
        except Exception:
            X = _dense(obj.protein.layers[layer])
            try:
                cols = _mosaic_feature_names(obj)
            except Exception:
                cols = [f"feat_{i}" for i in range(X.shape[1])]
            try:
                idx = _mosaic_cell_index(obj)
            except Exception:
                idx = [f"cell_{i}" for i in range(X.shape[0])]
            if len(cols) != X.shape[1]:
                cols = [f"feat_{i}" for i in range(X.shape[1])]
            if len(idx) != X.shape[0]:
                idx = [f"cell_{i}" for i in range(X.shape[0])]
            df = pd.DataFrame(X, index=idx, columns=cols)
            return df.apply(pd.to_numeric, errors="coerce")

    if isinstance(obj, pd.DataFrame):
        return obj.copy()

    raise TypeError("obj must be an AnnData, a missionbio.mosaic.sample.Sample, or a pandas DataFrame")


def _canon_feature_names(names: Sequence[str]) -> List[str]:
    return [str(n).strip().replace(" ", "_").replace("/", "_").replace("-", "_") for n in names]


def _unwrap_estimator(m):
    return getattr(m, "estimator", None) or getattr(m, "base_estimator", None) or m


def _load_feature_names_sidecar(models_path: Path, atlas: str, depth: str, label: str) -> Optional[List[str]]:
    lab_piece = str(label).replace(" ", "_")
    depth_piece = str(depth)
    patterns = [
        f"*{depth_piece}_{lab_piece}*/feature_names.joblib",
        f"*{lab_piece}*/feature_names.joblib",
        "*/feature_names.joblib",
    ]
    tried = []
    for pat in patterns:
        for cand in models_path.rglob(pat):
            tried.append(str(cand))
            try:
                cols = joblib.load(cand)
                if isinstance(cols, (list, tuple)) and all(isinstance(c, str) for c in cols):
                    print(f"[generate_predictions] Loaded feature names for '{label}' from: {cand}")
                    return list(cols)
            except Exception:
                continue
    if tried and DEBUG_PRED:
        print("[generate_predictions] Tried sidecar feature_names at:\n  - " + "\n  - ".join(tried[:50]))
    return None


def _get_shared_features_atlas_only(atlas: str, data_path: Union[str, Path], cache: Dict[str, List[str]]) -> List[str]:
    if atlas in cache:
        return cache[atlas]
    fp = Path(data_path) / atlas / "Shared_Features.csv"
    if not fp.exists():
        raise FileNotFoundError(f"No atlas-level Shared_Features.csv for atlas='{atlas}'. Tried: {fp}")
    feats = (pd.read_csv(fp, header=None).squeeze("columns").astype(str).str.strip().tolist())
    cache[atlas] = feats
    return feats


def _get_group_scaler(cell_dict: Dict[str, Any]) -> Tuple[Optional[Any], bool]:
    """Return (scaler, is_consistent) across labels inside a (atlas, depth) dict."""
    scalers: Dict[str, Any] = {}
    for lab, b in cell_dict.items():
        if not isinstance(b, dict) or str(lab).startswith("__"):
            continue
        s = b.get("scaler", None)
        if s is not None:
            scalers[str(lab)] = s
    if not scalers:
        return None, True
    vals = list(scalers.values())
    first = vals[0]
    if all(id(first) == id(s) for s in vals[1:]):
        return first, True
    m0 = getattr(first, "mean_", None)
    r0 = getattr(first, "scale_", None)
    if (m0 is None) or (r0 is None):
        return first, False
    for s in vals[1:]:
        m = getattr(s, "mean_", None)
        r = getattr(s, "scale_", None)
        if (m is None) or (r is None) or (not np.allclose(m0, m, equal_nan=True)) or (not np.allclose(r0, r, equal_nan=True)):
            return first, False
    return first, True


def _maybe_load_temp_scaler(models_for_atlas: Mapping, depth: str, atlas: str, data_path: str):
    """Legacy helper for on-disk temp scaler sidecars (not used for package dicts)."""
    if depth in models_for_atlas and isinstance(models_for_atlas[depth], dict):
        d = models_for_atlas[depth]
        if "__TEMP_SCALER__" in d:
            return d["__TEMP_SCALER__"]
    if "__TEMP_SCALER__" in models_for_atlas:
        return models_for_atlas["__TEMP_SCALER__"]
    cand = [
        Path(data_path) / atlas / f"{depth}_multiclass_temp_scaler.joblib",
        Path(data_path) / atlas / "multiclass_temp_scaler.joblib",
    ]
    for fp in cand:
        if fp.exists():
            try:
                return joblib.load(fp)
            except Exception:
                pass
    return None


def _any_base_needs_names(est) -> bool:
    base = getattr(est, "estimator", est)
    if hasattr(base, "feature_names_in_"):
        return True
    if isinstance(base, StackingClassifier):
        for e in getattr(base, "estimators_", []):
            last = e[-1] if isinstance(e, Pipeline) else e
            if hasattr(last, "feature_names_in_"):
                return True
    return False


def stack_prediction(model, X_df: pd.DataFrame) -> np.ndarray:
    """Return positive-class prob (binary) or max prob (multiclass)."""
    use_df = _any_base_needs_names(model)
    X_in = X_df if use_df else (X_df.to_numpy() if hasattr(X_df, "to_numpy") else X_df)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r"X has feature names, but .* was fitted without feature names")
        proba = model.predict_proba(X_in)
    return proba[:, 1] if proba.shape[1] == 2 else proba.max(axis=1)


def _resolve_training_columns(
    models_path: Path,
    atlas: str,
    depth: str,
    label: str,
    bundle: Dict[str, Any],
    est_base: Any,
    shared_feats_cache: Dict[str, List[str]],
    data_path: Union[str, Path],
) -> List[str]:
    """Prefer estimator.feature_names_in_ → bundle['columns'] → sidecar → atlas Shared_Features."""
    if hasattr(est_base, "feature_names_in_"):
        return [str(c) for c in est_base.feature_names_in_]
    cols = bundle.get("columns") or bundle.get("cols")
    if cols is not None:
        return list(map(str, cols))
    cols = _load_feature_names_sidecar(models_path, atlas, depth, label)
    if cols is not None:
        return cols
    return _get_shared_features_atlas_only(atlas, data_path, shared_feats_cache)


def _row_normalize(M: np.ndarray) -> np.ndarray:
    M = np.clip(np.asarray(M, dtype=float), 0.0, 1.0)
    rs = M.sum(axis=1, keepdims=True)
    rs[rs <= 0.0] = 1.0
    return M / rs


def _is_multiclass_package(d: Any) -> bool:
    """Return True if dict looks like a Multiclass_models.joblib package."""
    return (
        isinstance(d, dict)
        and "heads" in d
        and "class_names" in d
        and isinstance(d.get("heads"), dict)
        and isinstance(d.get("class_names"), (list, tuple))
    )


def _pick_head_model(head_bundle: Dict[str, Any]) -> Any:
    """Prefer Platt-calibrated head if available; otherwise fall back to raw; accept legacy keys."""
    return (
        head_bundle.get("model_platt")
        or head_bundle.get("model")
        or head_bundle.get("Stacked")
        or head_bundle.get("model_raw")
        or head_bundle.get("estimator")
        or head_bundle.get("est")
    )


def _head_predict_proba_ovr1(
    head_bundle: Dict[str, Any],
    query_df: pd.DataFrame,
    *,
    base_layer: str,
) -> np.ndarray:
    """
    Produce OvR positive-class probabilities for one class head.
    Uses head_bundle['columns'] + optional head_bundle['scaler'].
    """
    model = _pick_head_model(head_bundle)
    if model is None or not _looks_like_estimator(model):
        raise ValueError("Head bundle has no usable estimator (expected model_platt/model_raw/etc.)")

    cols = head_bundle.get("columns") or head_bundle.get("cols")
    if cols is None:
        raise KeyError("Head bundle missing 'columns' (panel features)")

    cols = list(map(str, cols))
    X = query_df.reindex(columns=cols).fillna(0.0)

    scaler = head_bundle.get("scaler")
    if base_layer != "Scaled_reads" and scaler is not None:
        scaler_cols = list(map(str, getattr(scaler, "feature_names_in_", cols)))
        Xs = query_df.reindex(columns=scaler_cols)

        if hasattr(scaler, "mean_") and len(getattr(scaler, "mean_", [])) == len(scaler_cols):
            mu = pd.Series(scaler.mean_, index=scaler_cols)
            Xs = Xs.fillna(mu)
        else:
            Xs = Xs.fillna(0.0)

        Xs_tr = pd.DataFrame(scaler.transform(Xs.values), index=Xs.index, columns=scaler_cols)
        X = Xs_tr.reindex(columns=cols, fill_value=0.0)

    est_base = _unwrap_estimator(model)
    if hasattr(est_base, "feature_names_in_"):
        order = [str(c) for c in est_base.feature_names_in_]
        X = X.reindex(columns=order, fill_value=0.0)

    use_df = _any_base_needs_names(model)
    X_in = X if use_df else X.to_numpy()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r"X has feature names, but .* was fitted without feature names")
        P = model.predict_proba(X_in)

    P = np.asarray(P)
    if P.ndim != 2 or P.shape[1] < 2:
        raise ValueError(f"Head predict_proba returned shape {P.shape}, expected (n,2+)")

    return np.clip(P[:, 1].astype(float), 0.0, 1.0)


def _short_keys(d: Mapping, max_n: int = DEBUG_KEYS_MAX) -> List[str]:
    try:
        ks = list(map(str, d.keys()))
    except Exception:
        return ["<unprintable keys>"]
    if len(ks) > max_n:
        return ks[:max_n] + [f"... (+{len(ks)-max_n} more)"]
    return ks


def _extract_multiclass_container(cell_dict: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Try to find the multiclass package dict in multiple common placements:
      - under cell_dict["__BUNDLE__"] (primary location)
      - under cell_dict["__MULTICLASS__"] (alternative)
      - directly at cell_dict (cell_dict itself is the package)
    Returns the candidate dict if found; else None.
    """
    bundle = cell_dict.get("__BUNDLE__")
    if isinstance(bundle, dict) and _is_multiclass_package(bundle):
        return bundle
    
    mc = cell_dict.get("__MULTICLASS__")
    if isinstance(mc, dict) and _is_multiclass_package(mc):
        return mc
    
    if _is_multiclass_package(cell_dict):
        return cell_dict
    
    for k in ("Multiclass_models", "Multiclass_models.joblib", "MULTICLASS", "multiclass"):
        v = cell_dict.get(k)
        if isinstance(v, dict) and _is_multiclass_package(v):
            return v
    return None


# -------------------- Improved Agreement Calculation --------------------

def _robust_agreement(M: np.ndarray, method: str = "hybrid") -> np.ndarray:
    """
    Robust agreement calculation with multiple options.
    
    Parameters
    ----------
    M : np.ndarray, shape (n_cells, n_atlases)
        Probability scores from different atlases
    method : str
        - 'mad': Current method (Median Absolute Deviation)
        - 'hybrid': Combine MAD and range (RECOMMENDED)
    
    Returns
    -------
    agree : np.ndarray, shape (n_cells,)
        Agreement scores in [0, 1] where 1 = perfect agreement
    """
    n, A = M.shape
    eps = 1e-9
    
    if A == 0:
        return np.zeros(n)
    if A == 1:
        return np.ones(n)
    
    if method == "mad":
        # Current method
        med = np.median(M, axis=1)
        mad = np.median(np.abs(M - med[:, None]), axis=1)
        return np.clip(1.0 - 2.0 * mad, 0.0, 1.0)
    
    elif method == "hybrid":
        # Combine MAD and range for robustness
        # MAD component
        med = np.median(M, axis=1)
        mad = np.median(np.abs(M - med[:, None]), axis=1)
        nmad = mad / (med + 0.1)
        mad_agree = np.exp(-3.0 * nmad)
        
        # Range component
        ptp = M.ptp(axis=1)
        range_agree = 1.0 - np.clip(ptp, 0.0, 1.0)
        
        # Geometric mean (both must be high for high agreement)
        agree = np.sqrt(mad_agree * range_agree)
        
        return np.clip(agree, 0.0, 1.0)
    
    else:
        raise ValueError(f"Unknown method: {method}")


# ------------------------ feature overlap audit ------------------------

def audit_feature_overlap(
    obj: Union["AnnData", Any],
    models_path: Optional[Union[str, Path]] = None,
    data_path: Optional[Union[str, Path]] = None,
    *,
    base_layer: str = "Normalized_reads",
    show: int = 20,
    write_dir: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """Inspect feature-name overlap between the query object and every OvR head / multiclass bundle."""
    if models_path is None:
        models_path = str(get_default_models_path())
    if data_path is None:
        data_path = str(get_default_data_path())
    models = load_models(models_path)
    models_path = Path(models_path)
    data_path = Path(data_path)

    is_sample = hasattr(obj, "protein") and hasattr(obj.protein, "get_attribute")
    if is_sample:
        try:
            query_df = _make_query_df(obj, mosaic_layer=base_layer)
        except KeyError:
            query_df = _make_query_df(obj, mosaic_layer="Normalized_reads")
    else:
        query_df = _make_query_df(obj)

    query_df = (query_df.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0))
    qcols = list(query_df.columns)
    qset = set(_canon_feature_names(qcols))

    print(f"[audit] query_n={len(qcols)}")
    print(f"[audit] first {min(show, len(qcols))} query features:", qcols[:show])

    rows = []
    for atlas, depth_map in models.items():
        if not isinstance(depth_map, dict):
            continue
        for depth, cell_dict in depth_map.items():
            if not isinstance(cell_dict, dict):
                continue

            pkg = _extract_multiclass_container(cell_dict)
            if isinstance(pkg, dict) and _is_multiclass_package(pkg):
                mc_labels = [str(x) for x in pkg.get("class_names", [])]
                heads = pkg.get("heads", {})
                for cls in mc_labels:
                    hb = heads.get(cls)
                    if not isinstance(hb, dict):
                        continue
                    train_cols_raw = list(map(str, (hb.get("columns") or hb.get("cols") or [])))
                    train_cols = _canon_feature_names(train_cols_raw)
                    tset = set(train_cols)
                    overlap_n = len(qset & tset)
                    rows.append({
                        "atlas": str(atlas),
                        "depth": str(depth),
                        "label": str(cls),
                        "query_n": len(qcols),
                        "train_n": len(train_cols),
                        "overlap_n": overlap_n,
                        "overlap_pct": (overlap_n / max(1, len(train_cols))) * 100.0,
                    })
                continue

            if "__MULTICLASS__" in cell_dict and isinstance(cell_dict["__MULTICLASS__"], dict):
                b = cell_dict["__MULTICLASS__"]
                est = (b.get("model") or b.get("Stacked") or b.get("estimator") or b.get("est"))
                if est is not None and _looks_like_estimator(est):
                    est_base = _unwrap_estimator(est)
                    train_cols_raw = _resolve_training_columns(
                        models_path, str(atlas), str(depth), "__MULTICLASS__", b, est_base, {}, data_path
                    )
                    train_cols = _canon_feature_names(train_cols_raw)
                    tset = set(train_cols)
                    overlap_n = len(qset & tset)
                    rows.append({
                        "atlas": str(atlas),
                        "depth": str(depth),
                        "label": "__MULTICLASS__",
                        "query_n": len(qcols),
                        "train_n": len(train_cols),
                        "overlap_n": overlap_n,
                        "overlap_pct": (overlap_n / max(1, len(train_cols))) * 100.0,
                    })

            for label, bundle in cell_dict.items():
                if not isinstance(bundle, dict) or str(label).startswith("__"):
                    continue
                est = (bundle.get("model") or bundle.get("Stacked") or bundle.get("estimator") or bundle.get("est"))
                if est is None or not _looks_like_estimator(est):
                    continue
                est_base = _unwrap_estimator(est)
                train_cols_raw = _resolve_training_columns(
                    models_path, str(atlas), str(depth), str(label), bundle, est_base, {}, data_path
                )
                train_cols = _canon_feature_names(train_cols_raw)
                tset = set(train_cols)
                overlap_n = len(qset & tset)
                rows.append({
                    "atlas": str(atlas),
                    "depth": str(depth),
                    "label": str(label),
                    "query_n": len(qcols),
                    "train_n": len(train_cols),
                    "overlap_n": overlap_n,
                    "overlap_pct": (overlap_n / max(1, len(train_cols))) * 100.0,
                })

    df = pd.DataFrame(rows)
    if len(df) == 0:
        print("[audit] No models found to audit.")
        return df
    df = df.sort_values(["overlap_pct", "atlas", "depth", "label"])
    print("\n[audit] worst overlaps:\n", df.head(min(10, len(df))).to_string(index=False))

    if write_dir is not None:
        write_dir = Path(write_dir)
        write_dir.mkdir(parents=True, exist_ok=True)
        out = write_dir / "feature_overlap_audit.csv"
        df.to_csv(out, index=False)
        print(f"[audit] wrote: {out}")

    return df


def _preview_query_df(df: pd.DataFrame, *, n_rows: int = 5, n_cols: int = 12, tag: str = "query_df") -> None:
    n_cells, n_feats = df.shape
    print(f"[generate_predictions] {tag} shape: {n_cells} cells x {n_feats} features")
    cols_preview = list(df.columns[:min(n_cols, n_feats)])
    try:
        print(df.loc[df.index[:n_rows], cols_preview].to_string())
    except Exception:
        print(df.head(n_rows).to_string())


_ATLAS_NAMES = ("Hao", "Triana", "Zhang", "Luecken")


def _prune_existing_tracks(
    obj: Union[AnnData, Any],
    *,
    keep_atlases: Optional[Sequence[str]] = None,
    drop_averaged: bool = True,
    drop_best: bool = True,
) -> None:
    keep = set(keep_atlases or [])
    atlas_prefixes_to_drop = []
    for a in _ATLAS_NAMES:
        if keep and (a in keep):
            continue
        atlas_prefixes_to_drop.append(f"{a}.")

    def _should_drop_key(k: str) -> bool:
        if k.startswith("Atlas."):
            return True
        if any(k.startswith(p) for p in atlas_prefixes_to_drop):
            return True
        if drop_averaged and (k.startswith("Averaged.") or k.startswith("Averaged.Unweighted.")):
            return True
        if drop_best and (k.startswith("BestBroad.") or k.startswith("BestSimplified.") or k.startswith("BestDetailed.")):
            return True
        return False

    if isinstance(obj, AnnData):
        drop_cols = [c for c in list(obj.obs.columns) if _should_drop_key(c)]
        if drop_cols:
            obj.obs.drop(columns=drop_cols, inplace=True, errors="ignore")
        drop_obsm = []
        for k in list(obj.obsm.keys()):
            if any(k.startswith(p) for p in atlas_prefixes_to_drop) or k.startswith("Atlas."):
                drop_obsm.append(k)
            if drop_averaged and k.startswith("Averaged."):
                drop_obsm.append(k)
        for k in drop_obsm:
            obj.obsm.pop(k, None)

    elif hasattr(obj, "protein") and hasattr(obj.protein, "row_attrs"):
        to_delete = [k for k in list(obj.protein.row_attrs.keys()) if _should_drop_key(k)]
        for k in to_delete:
            try:
                del obj.protein.row_attrs[k]
            except Exception:
                pass


# ------------------------ main prediction entry ------------------------

def generate_predictions(
    obj: Union["AnnData", Any],
    models_path: Optional[Union[str, Path]] = None,
    data_path: Optional[Union[str, Path]] = None,
    *,
    base_layer: str = "Normalized_reads",
    preview: bool = True,
    preview_rows: int = 5,
    preview_cols: int = 12,
    add_average: bool = True,
    average_prefix: str = "Averaged.",
    average_normalize: bool = True,
    agreement_calculation_method: str = "hybrid",
    add_average_consensus: bool = True,
    average_consensus_agreement_threshold: float = 0.7,
    average_consensus_score_threshold: float = 0.3,
    use_atlases: Optional[Union[str, Sequence[str]]] = None,
    apply_exclusions: Optional[bool] = True,
) -> Union["AnnData", Any]:
    """
    Predict probabilities with optional scaling; write per-atlas probabilities.
    
    NEW in v2.0: Automatically creates high-confidence average tracks.
    
    Parameters
    ----------
    obj : AnnData or Sample
        Query data object
    models_path : str or Path, optional
        Path to models directory
    data_path : str or Path, optional
        Path to data directory
    base_layer : str, default="Normalized_reads"
        Layer to use for predictions
    preview : bool, default=True
        Print preview of query data
    add_average : bool, default=True
        Create Averaged.* average tracks
    average_mode : str, default="plain"
        average calculation mode: 'plain', 'global', 'row', 'hybrid'
    agreement_calculation_method : str, default="hybrid"
        Agreement calculation: 'mad' or 'hybrid' (recommended)
    add_average_consensus : bool, default=True
        Create Averaged.Consensus.* high-confidence tracks
    average_consensus_agreement_threshold : float, default=0.7
        Minimum agreement (0-1) to include in average consensus
    average_consensus_score_threshold : float, default=0.3
        Minimum score (0-1) to include in average consensus
    use_atlases : str or list, optional
        Restrict to specific atlases
    apply_exclusions : bool, default=True
        Apply atlas exclusions for problematic label/atlas combinations
    
    Returns
    -------
    obj : AnnData or Sample
        Input object with added prediction tracks
    
    Writes Tracks
    -------------
    Per-atlas:
        {Atlas}.{Depth}.{Label}.predscore
        {Atlas}.{Depth}.pred / .conf / .Celltype / .Celltype.TopScore
    average (if add_average=True):
        Averaged.{Depth}.{Label}.predscore
        Averaged.{Depth}.{Label}.agreement
        Averaged.{Depth}.{Label}.n_atlases
    High-Confidence average (if add_average_consensus=True):
        Averaged.Consensus.{Depth}.{Label}.predscore (filtered)
        Averaged.Consensus.{Depth}.pred (or "Uncertain")
        Averaged.Consensus.{Depth}.conf
    """
    if models_path is None:
        models_path = str(get_default_models_path())
        print(f"[generate_predictions] Using default models path: {models_path}")
    if data_path is None:
        data_path = str(get_default_data_path())
        print(f"[generate_predictions] Using default data path: {data_path}")

    print("[generate_predictions] Ensuring models are available...")
    try:
        ensure_models_available()
    except Exception as e:
        print(f"[generate_predictions] Warning: Could not ensure models available: {e}")

    if use_atlases is None:
        requested: Optional[List[str]] = None
    elif isinstance(use_atlases, str):
        requested = [use_atlases]
    else:
        requested = list(use_atlases)

    valid_names = {"Hao", "Triana", "Zhang", "Luecken"}
    if requested is not None:
        bad = [a for a in requested if a not in valid_names]
        if bad:
            raise ValueError(f"Unknown atlas names in use_atlases: {bad} (valid: {sorted(valid_names)})")

    if requested is None:
        models = load_models(models_path)
        print("[generate_predictions] Using all atlases: Hao, Triana, Zhang, Luecken")
    else:
        models = load_models(models_path, model_names=tuple(requested))
        print(f"[generate_predictions] Restricting to atlases: {requested}")

    models_path = Path(models_path)
    data_path = Path(data_path)

    is_anndata = isinstance(obj, AnnData)
    is_sample = hasattr(obj, "protein") and hasattr(obj.protein, "get_attribute")

    _prune_existing_tracks(
        obj,
        keep_atlases=(requested if requested is not None else _ATLAS_NAMES),
        drop_averaged=False,
        drop_best=True,
    )

    if is_sample:
        try:
            query_df = _make_query_df(obj, mosaic_layer=base_layer)
        except KeyError:
            query_df = _make_query_df(obj, mosaic_layer="Normalized_reads")
    else:
        query_df = _make_query_df(obj)

    query_df = (
        query_df.apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )
    query_df.columns = _canon_feature_names(list(query_df.columns))
    row_index = query_df.index

    if preview:
        _preview_query_df(query_df, n_rows=preview_rows, n_cols=preview_cols, tag="query_df")

    shared_feats_cache: Dict[str, List[str]] = {}
    probs_store: Dict[Tuple[str, str], Tuple[np.ndarray, List[str]]] = {}

    for atlas, depth_map in models.items():
        if not isinstance(depth_map, dict):
            continue
        if requested is not None and atlas not in requested:
            continue

        for depth, cell_dict in depth_map.items():
            if not isinstance(cell_dict, dict):
                continue

            if DEBUG_PRED:
                print(f"\n[generate_predictions] Processing {atlas}.{depth}")
                print(f"  cell_dict keys: {_short_keys(cell_dict)}")

            pkg = None
            
            if '__BUNDLE__' in cell_dict:
                candidate = cell_dict['__BUNDLE__']
                if isinstance(candidate, dict) and _is_multiclass_package(candidate):
                    pkg = candidate
                    if DEBUG_PRED:
                        print(f"  → Found package in '__BUNDLE__'")
            
            if pkg is None and '__MULTICLASS__' in cell_dict:
                candidate = cell_dict['__MULTICLASS__']
                if isinstance(candidate, dict) and _is_multiclass_package(candidate):
                    pkg = candidate
                    if DEBUG_PRED:
                        print(f"  → Found package in '__MULTICLASS__'")
            
            if pkg is None and _is_multiclass_package(cell_dict):
                pkg = cell_dict
                if DEBUG_PRED:
                    print(f"  → cell_dict itself is the package")

            if pkg is not None:
                mc_labels = [str(x) for x in pkg.get("class_names", [])]
                heads = pkg.get("heads", {})
                
                if DEBUG_PRED:
                    print(f"  → Package has {len(mc_labels)} classes: {mc_labels[:10]}{'...' if len(mc_labels) > 10 else ''}")
                    print(f"  → heads dict has {len(heads)} entries")
                
                if not mc_labels or not isinstance(heads, dict):
                    print(f"[generate_predictions][WARN] {atlas}.{depth}: malformed package (no class_names or heads); skipping.")
                    continue

                P_ovr = np.zeros((query_df.shape[0], len(mc_labels)), dtype=float)
                ok_any = False
                failed_heads = []

                for j, cls in enumerate(mc_labels):
                    hb = heads.get(cls)
                    if not isinstance(hb, dict):
                        failed_heads.append(f"{cls}: not a dict")
                        continue
                    
                    try:
                        if DEBUG_PRED and j < 3:
                            cols = hb.get("columns") or hb.get("cols")
                            print(f"  Attempting head '{cls}': has {len(cols) if cols else 0} features")
                        
                        P_ovr[:, j] = _head_predict_proba_ovr1(hb, query_df, base_layer=base_layer)
                        ok_any = True
                        
                        if DEBUG_PRED and j < 3:
                            print(f"  ✓ Head '{cls}' succeeded (mean prob: {P_ovr[:, j].mean():.3f})")
                            
                    except Exception as e:
                        failed_heads.append(f"{cls}: {str(e)[:80]}")
                        if DEBUG_PRED and len(failed_heads) <= DEBUG_HEAD_FAILS_MAX:
                            print(f"  ✗ Head '{cls}' failed: {e}")

                if failed_heads:
                    if DEBUG_PRED:
                        print(f"  Summary: {len(failed_heads)}/{len(mc_labels)} heads failed")
                        if len(failed_heads) <= 5:
                            for fail in failed_heads:
                                print(f"    - {fail}")

                if not ok_any:
                    print(f"[generate_predictions][WARN] {atlas}.{depth}: no heads produced probabilities; skipping.")
                    continue

                ts_scaler = pkg.get("temp_scaler", None)
                if ts_scaler is not None:
                    try:
                        if DEBUG_PRED:
                            print(f"  Applying temp_scaler (type: {type(ts_scaler).__name__})")
                            print(f"    Before: P_ovr sum per row (first 3): {P_ovr.sum(axis=1)[:3]}")
                        
                        is_binary = len(mc_labels) == 2
                        
                        if is_binary:
                            P_input = P_ovr[:, 1].reshape(-1, 1)
                            
                            if DEBUG_PRED:
                                print(f"    Binary mode: using positive class only, shape {P_input.shape}")
                            
                            P_calibrated = ts_scaler.transform(P_input)
                            P_calibrated = np.asarray(P_calibrated).reshape(-1)
                            P_mc = np.column_stack([1.0 - P_calibrated, P_calibrated])
                            
                            if DEBUG_PRED:
                                print(f"    After calibration: P_mc shape {P_mc.shape}, sums {P_mc.sum(axis=1)[:3]}")
                        
                        else:
                            P_mc = ts_scaler.transform(P_ovr)
                            P_mc = np.asarray(P_mc)
                            
                            if DEBUG_PRED:
                                print(f"    After: P_mc shape {P_mc.shape}, sum per row (first 3): {P_mc.sum(axis=1)[:3]}")
                        
                        if P_mc.ndim == 1:
                            P_mc = P_mc.reshape(-1, 1)
                        
                        if P_mc.shape[1] != len(mc_labels):
                            if DEBUG_PRED:
                                print(f"    Shape mismatch ({P_mc.shape[1]} != {len(mc_labels)}), using row normalization")
                            P_mc = _row_normalize(P_ovr)
                        else:
                            P_mc = _row_normalize(P_mc)
                            
                    except Exception as e:
                        if DEBUG_PRED:
                            import traceback
                            print(f"[generate_predictions][WARN] {atlas}.{depth}: temp_scaler failed ({e})")
                            print(f"    Traceback: {traceback.format_exc()[:300]}")
                        print(f"[generate_predictions][WARN] {atlas}.{depth}: temp_scaler failed ({e}); using row-normalized OvR.")
                        P_mc = _row_normalize(P_ovr)
                else:
                    if DEBUG_PRED:
                        print(f"  No temp_scaler, using row normalization")
                    P_mc = _row_normalize(P_ovr)

                if DEBUG_PRED:
                    print(f"  Final P_mc: shape={P_mc.shape}, row sums (first 3): {P_mc.sum(axis=1)[:3]}")
                    print(f"  ✓ Storing to probs_store[({atlas}, {depth})]")

                probs_store[(atlas, depth)] = (P_mc, mc_labels)
                continue
            
            if DEBUG_PRED:
                print(f"  No multiclass package found, checking older estimator format...")
            
            mc_bundle = cell_dict.get("__MULTICLASS__") if isinstance(cell_dict.get("__MULTICLASS__"), dict) else None
            if isinstance(mc_bundle, dict):
                mc_est = (mc_bundle.get("model") or mc_bundle.get("Stacked") or mc_bundle.get("estimator") or mc_bundle.get("est"))
                if mc_est is not None and _looks_like_estimator(mc_est):
                    mc_est_base = _unwrap_estimator(mc_est)

                    mc_labels = (
                        mc_bundle.get("class_names")
                        or mc_bundle.get("classes")
                        or (list(getattr(mc_est, "classes_", [])) if hasattr(mc_est, "classes_") else None)
                    )
                    if not mc_labels:
                        if DEBUG_PRED:
                            print(f"  Older estimator found but no class labels; skipping.")
                        continue
                    mc_labels = [str(x) for x in mc_labels]

                    train_cols = _resolve_training_columns(
                        models_path=models_path,
                        atlas=str(atlas),
                        depth=str(depth),
                        label="__MULTICLASS__",
                        bundle=mc_bundle,
                        est_base=mc_est_base,
                        shared_feats_cache=shared_feats_cache,
                        data_path=data_path,
                    )
                    train_cols = list(map(str, train_cols))

                    scaler = None
                    if base_layer != "Scaled_reads":
                        scaler = mc_bundle.get("scaler", None)

                    if scaler is not None:
                        scaler_cols = (
                            list(getattr(scaler, "feature_names_in_", []))
                            or mc_bundle.get("scaler_columns")
                            or mc_bundle.get("columns_full")
                            or train_cols
                        )
                        scaler_cols = list(map(str, scaler_cols))

                        Xs = query_df.reindex(columns=scaler_cols)
                        if hasattr(scaler, "mean_") and len(getattr(scaler, "mean_", [])) == len(scaler_cols):
                            mu = pd.Series(scaler.mean_, index=scaler_cols)
                            Xs = Xs.fillna(mu)
                        else:
                            Xs = Xs.fillna(0.0)

                        Xs_tr = pd.DataFrame(scaler.transform(Xs.values), index=Xs.index, columns=scaler_cols)
                        X_tr = Xs_tr.reindex(columns=train_cols, fill_value=0.0)
                    else:
                        X_tr = query_df.reindex(columns=train_cols).fillna(0.0)

                    if hasattr(mc_est_base, "feature_names_in_"):
                        order = [str(c) for c in mc_est_base.feature_names_in_]
                        X_tr = X_tr.reindex(columns=order, fill_value=0.0)

                    use_df = _any_base_needs_names(mc_est)
                    X_in = X_tr if use_df else X_tr.to_numpy()
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", message=r"X has feature names, but .* was fitted without feature names")
                        P = mc_est.predict_proba(X_in)

                    P = np.asarray(P)
                    if P.ndim != 2 or P.shape[1] != len(mc_labels):
                        if DEBUG_PRED:
                            print(f"  Estimator proba shape {P.shape} != {len(mc_labels)} classes; skipping.")
                        continue

                    probs_store[(atlas, depth)] = (_row_normalize(P), mc_labels)
                    continue

            if DEBUG_PRED:
                print(f"  No estimator found, checking legacy OvR heads...")
            
            group_scaler, ok_group = _get_group_scaler(cell_dict)
            cell_labels: List[str] = []
            ovr_cols: List[np.ndarray] = []

            for label, bundle in cell_dict.items():
                if not isinstance(bundle, dict) or str(label).startswith("__"):
                    continue

                est = (bundle.get("model") or bundle.get("Stacked") or bundle.get("estimator") or bundle.get("est"))
                if est is None or not _looks_like_estimator(est):
                    continue
                est_base = _unwrap_estimator(est)

                train_cols = _resolve_training_columns(
                    models_path,
                    atlas=str(atlas),
                    depth=str(depth),
                    label=str(label),
                    bundle=bundle,
                    est_base=est_base,
                    shared_feats_cache=shared_feats_cache,
                    data_path=data_path,
                )

                scaler = None
                if base_layer != "Scaled_reads":
                    scaler = group_scaler if (group_scaler is not None and ok_group) else bundle.get("scaler")

                if scaler is not None:
                    scaler_cols = (
                        list(getattr(scaler, "feature_names_in_", []))
                        or bundle.get("scaler_columns")
                        or bundle.get("columns_full")
                        or train_cols
                    )
                    scaler_cols = list(map(str, scaler_cols))

                    Xs = query_df.reindex(columns=scaler_cols)
                    if hasattr(scaler, "mean_") and len(getattr(scaler, "mean_", [])) == len(scaler_cols):
                        mu = pd.Series(scaler.mean_, index=scaler_cols)
                        Xs = Xs.fillna(mu)
                    else:
                        Xs = Xs.fillna(0.0)

                    Xs_tr = pd.DataFrame(scaler.transform(Xs.values), index=Xs.index, columns=scaler_cols)
                    Xh_tr = Xs_tr.reindex(columns=train_cols, fill_value=0.0)
                else:
                    Xh_tr = query_df.reindex(columns=train_cols).fillna(0.0)

                if hasattr(est_base, "feature_names_in_"):
                    order = [str(c) for c in est_base.feature_names_in_]
                    Xh_tr = Xh_tr.reindex(columns=order, fill_value=0.0)

                score = stack_prediction(est, Xh_tr)
                score = np.clip(score, 0.0, 1.0)

                ovr_cols.append(np.asarray(score, dtype=float).reshape(-1, 1))
                cell_labels.append(str(label))

            if cell_labels:
                P_raw = np.hstack(ovr_cols)
                probs_store[(atlas, depth)] = (P_raw, cell_labels)

    for (atlas, depth), (P_raw, cell_labels) in probs_store.items():
        P_mc = _row_normalize(P_raw)

        if is_anndata:
            for j, lab in enumerate(cell_labels):
                obj.obs[f"{atlas}.{depth}.{lab}.predscore"] = pd.Series(P_mc[:, j], index=row_index, dtype=float)
        else:
            for j, lab in enumerate(cell_labels):
                obj.protein.row_attrs[f"{atlas}.{depth}.{lab}.predscore"] = P_mc[:, j].astype(float)

        winner_idx = np.asarray(P_mc).argmax(axis=1)
        winner_lab = np.array([cell_labels[i] for i in winner_idx], dtype=object)
        winner_conf = P_mc[np.arange(P_mc.shape[0]), winner_idx]

        if is_anndata:
            obj.obsm[f"{atlas}.{depth}.probs"] = pd.DataFrame(P_mc, index=row_index, columns=cell_labels)
            obj.obs[f"{atlas}.{depth}.pred"] = pd.Categorical(winner_lab, categories=cell_labels)
            obj.obs[f"{atlas}.{depth}.conf"] = winner_conf.astype(float)
            obj.obs[f"{atlas}.{depth}.Celltype"] = obj.obs[f"{atlas}.{depth}.pred"].astype(object)
            obj.obs[f"{atlas}.{depth}.Celltype.TopScore"] = obj.obs[f"{atlas}.{depth}.conf"].astype(float)
        else:
            obj.protein.row_attrs[f"{atlas}.{depth}.pred"] = winner_lab.astype(str)
            obj.protein.row_attrs[f"{atlas}.{depth}.conf"] = winner_conf.astype(float)
            obj.protein.row_attrs[f"{atlas}.{depth}.Celltype"] = winner_lab.astype(str)
            obj.protein.row_attrs[f"{atlas}.{depth}.Celltype.TopScore"] = winner_conf.astype(float)

    if add_average:
        used_atlases = sorted({atl for (atl, _depth) in probs_store.keys()})
        if not used_atlases:
            print("[generate_predictions] No atlases produced predictions; skipping averaging.")
            return obj

        use_exclusions = (len(used_atlases) > 1) if apply_exclusions is None else bool(apply_exclusions)

        if add_average:
            if is_anndata:
                add_averaged_tracks(
                    obj,
                    atlases=used_atlases,
                    normalize_multiclass=average_normalize,
                    depths=("Broad", "Simplified", "Detailed"),
                    out_prefix=average_prefix,
                    apply_exclusions=use_exclusions,
                    agreement_calculation_method=agreement_calculation_method,
                )
            else:
                add_averaged_tracks_sample(
                    obj,
                    atlases=used_atlases,
                    normalize_multiclass=average_normalize,
                    depths=("Broad", "Simplified", "Detailed"),
                    out_prefix=average_prefix,
                    apply_exclusions=use_exclusions,
                    agreement_calculation_method=agreement_calculation_method,
                )
    
    if add_average_consensus and add_average:
        print("[generate_predictions] Creating average consensus tracks...")
        add_averaged_consensus_tracks(
            obj,
            atlases=used_atlases,
            depths=("Broad", "Simplified", "Detailed"),
            agreement_threshold=average_consensus_agreement_threshold,
            score_threshold=average_consensus_score_threshold,
            average_prefix="Averaged.Consensus.",
            source_prefix=average_prefix,
            agreement_calculation_method=agreement_calculation_method,
            apply_exclusions=use_exclusions,
        )

    return obj

# OPTIONAL (but recommended): add these two tiny helpers near your other helpers
# to make debugging even clearer when you suspect a structure mismatch.

def _debug_summarize_models_tree(models: Mapping[str, Any], *, max_depths: int = 10) -> None:
    """Print a compact summary of what load_models() returned."""
    print("[generate_predictions][DEBUG] models tree summary:")
    for atlas, depth_map in models.items():
        if not isinstance(depth_map, dict):
            print(f"  - {atlas}: type={type(depth_map)}")
            continue
        depths = list(depth_map.keys())
        print(f"  - {atlas}: depths={depths[:max_depths]}{'...' if len(depths) > max_depths else ''}")
        for depth in depths[:max_depths]:
            d = depth_map.get(depth)
            if not isinstance(d, dict):
                print(f"      * {depth}: type={type(d)}")
                continue
            if _is_multiclass_package(d):
                pkg = d
                n_cls = len(pkg.get('class_names', []) or [])
                n_heads = len(pkg.get('heads', {}) or {}) if isinstance(pkg.get('heads', {}), dict) else 'NA'
                has_ts = pkg.get("temp_scaler") is not None
                print(f"      * {depth}: PACKAGE n_classes={n_cls} n_heads={n_heads} temp_scaler={'yes' if has_ts else 'no'}")
            elif "__MULTICLASS__" in d:
                print(f"      * {depth}: __MULTICLASS__ keys={list(d['__MULTICLASS__'].keys()) if isinstance(d.get('__MULTICLASS__'), dict) else type(d.get('__MULTICLASS__'))}")
            else:
                # legacy OvR heads
                head_keys = [k for k, v in d.items() if isinstance(v, dict) and not str(k).startswith("__")]
                print(f"      * {depth}: legacy_heads={len(head_keys)} (sample={head_keys[:5]})")


def _debug_compare_query_to_expected(query_df: pd.DataFrame, expected_cols: Sequence[str], *, title: str = "") -> None:
    """Print overlap/missing between query features and expected training features."""
    qset = set(map(str, query_df.columns))
    exp = list(map(str, expected_cols))
    eset = set(exp)
    inter = sorted(qset & eset)
    missing = [c for c in exp if c not in qset]
    extra = sorted(qset - eset)
    if title:
        print(f"[generate_predictions][DEBUG] {title}")
    print(f"  overlap: {len(inter)}/{len(exp)} ({(100.0*len(inter)/max(1,len(exp))):.1f}%)")
    print(f"  missing expected: {len(missing)} (first 20): {missing[:20]}")
    print(f"  extra in query: {len(extra)} (first 20): {extra[:20]}")


# -------------------- best-localised tracks (public) --------------------

def _local_dispersion(vals: np.ndarray, coords: np.ndarray, k: int = 15, p: int = 2, eps: float = 1e-9) -> np.ndarray:
    tree = KDTree(coords, metric="minkowski", p=p)
    dist, idx = tree.query(coords, k=k + 1)
    weights = 1 / (dist + eps)
    neighb_means = (weights * vals[idx]).sum(1) / weights.sum(1)
    deviation_abs = np.abs(vals[idx] - neighb_means[:, None])
    return (weights * deviation_abs).sum(1) / weights.sum(1)


def _best_localised_score(
    adata: AnnData,
    depth: str,
    label: str,
    atlases: Sequence[str],
    q: float = 0.90,
    k: int = 15,
    p: int = 2,
) -> Tuple[Optional[str], Optional[np.ndarray]]:
    coords = adata.obsm["X_umap"]
    best_med = np.inf
    best_vec = best_atl = None

    for atl in atlases:
        col = f"{atl}.{depth}.{label}.predscore"
        if col not in adata.obs:
            continue
        x = adata.obs[col].to_numpy()
        if x.ptp() == 0:
            continue
        x = (x - x.min()) / x.ptp()
        hi_mask = x >= np.quantile(x, q)
        if hi_mask.sum() < k + 1:
            continue
        disp = _local_dispersion(x, coords, k=k, p=p)
        med = np.median(disp[hi_mask])
        if med < best_med:
            best_med, best_vec, best_atl = med, x, atl
    return best_atl, best_vec


def add_best_localised_tracks(
    obj: Union["AnnData", Any],
    *,
    atlases: Sequence[str],
    depths: Sequence[str] = ("Broad",),
    labels: Optional[Sequence[str]] = None,
    q: float = 0.90,
    k: int = 15,
    p: int = 2,
    prefix: str = "Best",
) -> None:
    """Pick, per label, the atlas whose high-score cells are most spatially localized on UMAP."""
    depth_to_labels: Dict[str, Sequence[str]] = {
        "Broad": ["Immature", "Mature"],
        "Simplified": list(SIMPLIFIED_CLASSES.keys()),
        "Detailed": _DETAILED_LABELS,
    }

    is_anndata = isinstance(obj, AnnData)
    is_sample = hasattr(obj, "protein") and hasattr(obj.protein, "row_attrs")

    if is_anndata:
        if "X_umap" not in obj.obsm:
            raise ValueError("add_best_localised_tracks requires adata.obsm['X_umap']")
        for depth in depths:
            lbls = list(labels) if labels is not None else list(depth_to_labels.get(depth, []))
            for lbl in lbls:
                best_atl, _ = _best_localised_score(obj, depth, lbl, atlases, q=q, k=k, p=p)
                if not best_atl:
                    continue
                src = f"{best_atl}.{depth}.{lbl}.predscore"
                if src in obj.obs:
                    out = f"{prefix}{depth}.{lbl}.predscore"
                    obj.obs[out] = obj.obs[src].astype(float).to_numpy()

    elif is_sample:
        coords = None
        for key in ("umap", "X_umap", "umap_coords"):
            if key in obj.protein.row_attrs:
                coords = np.asarray(obj.protein.row_attrs[key])
                break
        if coords is None:
            raise ValueError("add_best_localised_tracks requires sample.protein.row_attrs['umap'] (or 'X_umap')")

        n = int(coords.shape[0])
        adata = AnnData(X=np.zeros((n, 1)))
        adata.obsm["X_umap"] = coords

        for depth in depths:
            lbls = list(labels) if labels is not None else list(depth_to_labels.get(depth, []))
            for lbl in lbls:
                for atl in atlases:
                    key = f"{atl}.{depth}.{lbl}.predscore"
                    if key in obj.protein.row_attrs:
                        adata.obs[key] = np.asarray(obj.protein.row_attrs[key], dtype=float)
            for lbl in lbls:
                best_atl, _ = _best_localised_score(adata, depth, lbl, atlases, q=q, k=k, p=p)
                if not best_atl:
                    continue
                src = f"{best_atl}.{depth}.{lbl}.predscore"
                if src in obj.protein.row_attrs:
                    out = f"{prefix}{depth}.{lbl}.predscore"
                    obj.protein.row_attrs[out] = np.asarray(obj.protein.row_attrs[src], dtype=float)
    else:
        raise ValueError("Unsupported object type for add_best_localised_tracks")


# -------------------- atlas exclusions --------------------

EXCLUDE_ATLAS: Dict[str, Dict[str, set]] = {
    "Simplified": {
        "CD4_T": {"Luecken", "Zhang", "Triana"},
        "CD8_T": {"Hao"},
        "Other_T": {"Luecken"},
        "Erythroid": {"Triana", "Luecken"},
        "HSPC": {"Zhang", "Luecken", "Triana"},
        "Monocyte": {"Zhang", "Triana"},
        "Myeloid": {"Zhang", "Luecken"},
        "NK": {"Zhang", "Hao"},
        "cDC": {"Luecken"},
        "B": {"Triana"},
        "Plasma": {"Zhang", "Triana"},
    },
    "Detailed": {
        "B_Memory": {"Triana"},
        "B_Naive": {"Zhang"},
        "CD14_Mono": {"Luecken", "Hao"},
        "CD16_Mono": {""},
        "CD4_CTL": {""},
        "CD4_T_Memory": {"Triana", "Zhang", "Luecken"},
        "CD4_T_Naive":  {"Triana", "Zhang", "Luecken"},
        "CD8_T_Memory": {"Luecken"},
        "CD8_T_Naive": {"Triana"},
        "EoBaMaP": {"Zhang", "Luecken", "Hao"},
        "ErP": {"Triana", "Hao"},
        "Erythroblast": {"Hao", "Triana", "Zhang"},
        "GMP": {"Zhang"},
        "GdT": {"Hao", "Zhang", "Luecken"},
        "HSC_MPP": {"Zhang", "Luecken"},
        "Immature_B": {"Hao", "Zhang"},
        "LMPP": {"Zhang", "Luecken", "Hao"},
        "MAIT": {"Luecken", "Triana"},
        "MEP": {"Hao", "Triana"},
        "MkP": {""},
        "Myeloid_progenitor": {"Hao", "Luecken"},
        "NK_CD56_bright": {"Zhang"},
        "NK_CD56_dim": {"Zhang", "Luecken", "Hao"},
        "Plasma": {"Zhang"},
        "Pre-B": {"Hao", "Luecken", "Zhang"},
        "Pre-Pro-B": {"Hao", "Luecken"},
        "Pro-B": {"Hao", "Luecken"},
        "Treg": {""},
        "cDC1": {"Hao", "Zhang"},
        "cDC2": {"Zhang", "Luecken", "Triana"},
        "pDC": {"Triana", "Zhang"},
    },
}


def _filter_atlases_for_label(atlases: Sequence[str], depth: str, label: str, *, apply_exclusions: bool) -> List[str]:
    if not apply_exclusions:
        return list(atlases)
    banned = EXCLUDE_ATLAS.get(depth, {}).get(label, set())
    return [a for a in atlases if a not in banned]


def _stack_label_scores(
    adata: AnnData,
    depth: str,
    label: str,
    atlases: Sequence[str],
    *,
    apply_exclusions: bool = True,
) -> Tuple[np.ndarray, List[str]]:
    allowed = _filter_atlases_for_label(atlases, depth, label, apply_exclusions=apply_exclusions)
    cols = []
    used = []
    for atl in allowed:
        col = f"{atl}.{depth}.{label}.predscore"
        if col in adata.obs:
            cols.append(col)
            used.append(atl)
    if not cols:
        return np.zeros((adata.n_obs, 0), dtype=float), []
    M = np.column_stack([adata.obs[c].astype(float).to_numpy() for c in cols])
    return M, used


# -------------------- weighting helpers --------------------

def _global_agreement_weights(M: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    A = M.shape[1]
    if A <= 1:
        return np.ones(A, dtype=float)
    X = M - M.mean(axis=0, keepdims=True)
    denom = np.sqrt((X**2).sum(axis=0, keepdims=True)) + eps
    Xn = X / denom
    C = Xn.T @ Xn
    np.fill_diagonal(C, 0.0)
    C = np.clip(C, 0.0, 1.0)
    w = C.mean(axis=1)
    if float(w.sum()) <= eps:
        return np.ones(A, dtype=float) / A
    return (w / w.sum()).astype(float)


def _row_robust_weights(row: np.ndarray, eps: float = 1e-9, alpha: float = 2.0, hard_tau: float = 4.0) -> np.ndarray:
    A = row.size
    if A <= 1:
        return np.ones(A, dtype=float)
    med = np.median(row)
    dev = np.abs(row - med)
    mad = np.median(dev) + eps
    soft = 1.0 / (1.0 + (dev / mad) ** alpha)
    if hard_tau is not None and hard_tau > 0:
        soft[dev > (hard_tau * mad)] = eps
    s = soft.sum()
    return soft if s == 0 else (soft / s)

def _average_blend(
    M: np.ndarray,
    eps: float = 1e-9,
    mode: str = "plain",          # 'plain' | 'global' | 'row' | 'hybrid'
    lambda_global: float = 0.5,   # only for 'hybrid'
) -> Tuple[np.ndarray, np.ndarray]:
    n, A = M.shape
    if A == 0:
        return np.zeros(n, dtype=float), np.zeros(n, dtype=float)
    if A == 1:
        cons = M[:, 0].astype(float).copy()
        agree = np.ones(n, dtype=float)
        return cons, agree

    M = np.clip(M.astype(float), 0.0, 1.0)

    if mode == "plain":
        cons = M.mean(axis=1)
    elif mode == "global":
        wg = _global_agreement_weights(M, eps=eps)
        cons = (M * wg[None, :]).sum(axis=1)
    elif mode == "row":
        cons = np.empty(n, dtype=float)
        for i in range(n):
            wr = _row_robust_weights(M[i, :], eps=eps)
            cons[i] = float((wr * M[i, :]).sum())
    elif mode == "hybrid":
        wg = _global_agreement_weights(M, eps=eps)
        lam = float(np.clip(lambda_global, 0.0, 1.0))
        cons = np.empty(n, dtype=float)
        for i in range(n):
            wr = _row_robust_weights(M[i, :], eps=eps)
            w = lam * wg + (1.0 - lam) * wr
            ws = w.sum()
            cons[i] = float((w * M[i, :]).sum() / (ws if ws > 0 else 1.0))
    else:
        raise ValueError("mode must be one of: 'plain', 'global', 'row', 'hybrid'")

    med = np.median(M, axis=1)
    mad = np.median(np.abs(M - med[:, None]), axis=1)
    agree = np.clip(1.0 - 2.0 * mad, 0.0, 1.0)

    return cons.astype(float), agree.astype(float)

# Replace these two functions in your prediction.py file

def add_averaged_tracks(
    adata: AnnData,
    *,
    atlases: Sequence[str],
    normalize_multiclass: bool = False,
    depths: Sequence[str] = ("Broad", "Simplified", "Detailed"),
    out_prefix: str = "Averaged.",
    apply_exclusions: bool = True,
    agreement_calculation_method: str = "hybrid",
) -> None:
    """
    AnnData variant of average Averaged.* tracks.
    Writes:
      {out_prefix}{depth}.{label}.predscore
      {out_prefix}{depth}.{label}.agreement
      {out_prefix}{depth}.{label}.n_atlases
    """
    depth_to_labels: Dict[str, Sequence[str]] = {
        "Broad": ["Immature", "Mature"],
        "Simplified": list(SIMPLIFIED_CLASSES.keys()),
        "Detailed": _DETAILED_LABELS,
    }

    single_passthrough = (len(atlases) == 1 and not apply_exclusions)
    src_atl = atlases[0] if single_passthrough else None

    for depth in depths:
        labels = depth_to_labels.get(depth, [])
        if not labels:
            continue

        created_cols: List[str] = []

        for lbl in labels:
            pred_col = f"{out_prefix}{depth}.{lbl}.predscore"
            agr_col  = f"{out_prefix}{depth}.{lbl}.agreement"
            nat_col  = f"{out_prefix}{depth}.{lbl}.n_atlases"

            if single_passthrough:
                src = f"{src_atl}.{depth}.{lbl}.predscore"
                if src in adata.obs:
                    adata.obs[pred_col] = adata.obs[src].astype(float).to_numpy()
                    adata.obs[agr_col]  = 1.0
                    adata.obs[nat_col]  = 1
                    created_cols.append(pred_col)
                continue

            M, _used = _stack_label_scores(adata, depth, lbl, atlases, apply_exclusions=apply_exclusions)
            if M.shape[1] == 0:
                continue

            M = np.clip(M, 0.0, 1.0)

            cons = M.mean(axis=1)

            # Calculate agreement using improved method
            agree = _robust_agreement(M, method=agreement_calculation_method)

            adata.obs[pred_col] = cons.astype(float)
            adata.obs[agr_col]  = agree.astype(float)
            adata.obs[nat_col]  = int(M.shape[1])
            created_cols.append(pred_col)

        # Optional normalization across labels (not for Broad)
        if depth != "Broad" and normalize_multiclass and created_cols:
            X = adata.obs[created_cols].to_numpy(dtype=float)
            X = np.clip(X, 0.0, 1.0)
            rs = X.sum(axis=1, keepdims=True)
            rs[rs <= 0.0] = 1.0
            adata.obs.loc[:, created_cols] = X / rs


def add_averaged_tracks_sample(
    sample,
    *,
    atlases: Sequence[str],
    normalize_multiclass: bool = False,
    depths: Sequence[str] = ("Broad", "Simplified", "Detailed"),
    out_prefix: str = "Averaged.",
    apply_exclusions: bool = True,
    agreement_calculation_method: str = "hybrid",  # NEW PARAMETER
) -> None:
    """
    Sample (Mosaic) variant of average Averaged.* tracks.
    Writes:
      {out_prefix}{depth}.{label}.predscore
      {out_prefix}{depth}.{label}.agreement
      {out_prefix}{depth}.{label}.n_atlases
    """
    depth_to_labels: Dict[str, Sequence[str]] = {
        "Broad": ["Immature", "Mature"],
        "Simplified": list(SIMPLIFIED_CLASSES.keys()),
        "Detailed": _DETAILED_LABELS,
    }

    single_passthrough = (len(atlases) == 1 and not apply_exclusions)
    src_atl = atlases[0] if single_passthrough else None

    for depth in depths:
        labels = depth_to_labels.get(depth, [])
        if not labels:
            continue

        created_keys: List[str] = []
        label_vecs: List[np.ndarray] = []

        for lbl in labels:
            pred_key = f"{out_prefix}{depth}.{lbl}.predscore"
            agr_key  = f"{out_prefix}{depth}.{lbl}.agreement"
            nat_key  = f"{out_prefix}{depth}.{lbl}.n_atlases"

            if single_passthrough:
                src = f"{src_atl}.{depth}.{lbl}.predscore"
                if src in sample.protein.row_attrs:
                    vec = np.asarray(sample.protein.row_attrs[src], dtype=float).reshape(-1)
                    sample.protein.row_attrs[pred_key] = vec
                    sample.protein.row_attrs[agr_key]  = np.ones_like(vec)
                    sample.protein.row_attrs[nat_key]  = 1
                    created_keys.append(pred_key)
                    label_vecs.append(vec)
                continue

            # Stack scores from allowed atlases
            allowed = _filter_atlases_for_label(atlases, depth, lbl, apply_exclusions=apply_exclusions)
            cols = []
            for atl in allowed:
                key = f"{atl}.{depth}.{lbl}.predscore"
                if key in sample.protein.row_attrs:
                    cols.append(np.asarray(sample.protein.row_attrs[key], dtype=float).reshape(-1))

            if not cols:
                continue

            M = np.clip(np.column_stack(cols), 0.0, 1.0)

            cons = M.mean(axis=1)

            # Calculate agreement using improved method
            agree = _robust_agreement(M, method=agreement_calculation_method)

            sample.protein.row_attrs[pred_key] = cons.astype(float)
            sample.protein.row_attrs[agr_key]  = agree.astype(float)
            sample.protein.row_attrs[nat_key]  = int(M.shape[1])

            created_keys.append(pred_key)
            label_vecs.append(cons.reshape(-1))

        # Normalize across labels (not for Broad)
        if depth != "Broad" and normalize_multiclass and created_keys:
            X = np.clip(np.column_stack(label_vecs).astype(float), 0.0, 1.0)
            rs = X.sum(axis=1, keepdims=True)
            rs[rs <= 0.0] = 1.0
            Xn = X / rs
            for j, key in enumerate(created_keys):
                sample.protein.row_attrs[key] = Xn[:, j].astype(float)

def add_averaged_consensus_tracks(
    obj: Union["AnnData", Any],
    *,
    atlases: Sequence[str],
    depths: Sequence[str] = ("Broad", "Simplified", "Detailed"),
    agreement_threshold: float = 0.7,
    score_threshold: float = 0.3,
    average_prefix: str = "Averaged.Consensus.",
    source_prefix: str = "Averaged.",
    agreement_calculation_method: str = "hybrid",
    apply_exclusions: bool = True,
    uncertainty_penalty: float = 0.5,
) -> None:
    """
    Create average consensus predictions with agreement-based score modulation.
    
    KEY BEHAVIOR:
    - High agreement + low score → Keep low score (datasets agree it's negative)
    - High agreement + high score → Keep high score (datasets agree it's positive)  
    - Low agreement → Push toward uncertainty_penalty (datasets disagree)
    - All probabilities are ROW-NORMALIZED so they sum to 1 per cell
    
    Parameters
    ----------
    uncertainty_penalty : float, default=0.5
        When agreement is low, push scores toward this "uncertain" value.
        Should be in (0, 1), typically 0.5 for maximum uncertainty.
    
    Writes:
      Averaged.Consensus.{depth}.{label}.predscore  (normalized modulated scores)
      Averaged.Consensus.{depth}.pred                (average call or "Uncertain")
      Averaged.Consensus.{depth}.conf                (confidence)
      Averaged.Consensus.{depth}.probs               (probability matrix)
    """
    depth_to_labels: Dict[str, Sequence[str]] = {
        "Broad": ["Immature", "Mature"],
        "Simplified": list(SIMPLIFIED_CLASSES.keys()),
        "Detailed": _DETAILED_LABELS,
    }
    
    is_anndata = isinstance(obj, AnnData)
    is_sample = hasattr(obj, "protein") and hasattr(obj.protein, "row_attrs")
    
    if not is_anndata and not is_sample:
        raise ValueError("obj must be AnnData or Sample")
    
    for depth in depths:
        labels = depth_to_labels.get(depth, [])
        if not labels:
            continue
        
        if is_anndata:
            n_cells = obj.n_obs
        else:
            n_cells = obj.protein.shape[0]
        
        # Initialize modulated probability matrix (before normalization)
        average_probs_raw = np.zeros((n_cells, len(labels)), dtype=float)
        n_high_agreement = 0
        
        for j, lbl in enumerate(labels):
            score_key = f"{source_prefix}{depth}.{lbl}.predscore"
            agree_key = f"{source_prefix}{depth}.{lbl}.agreement"
            
            # Get scores and agreement
            if is_anndata:
                if score_key not in obj.obs or agree_key not in obj.obs:
                    continue
                scores = obj.obs[score_key].to_numpy()
                agrees = obj.obs[agree_key].to_numpy()
            else:
                if score_key not in obj.protein.row_attrs or agree_key not in obj.protein.row_attrs:
                    continue
                scores = np.asarray(obj.protein.row_attrs[score_key])
                agrees = np.asarray(obj.protein.row_attrs[agree_key])
            
            # Agreement-modulated scores (BEFORE normalization)
            # High agreement → trust the score
            # Low agreement → blend toward uncertainty_penalty
            blend_factor = np.clip(agrees, 0.0, 1.0)
            modulated_scores = (
                blend_factor * scores +
                (1 - blend_factor) * uncertainty_penalty
            )
            
            average_probs_raw[:, j] = modulated_scores
            
            # Track high-agreement cells
            high_agreement_mask = (agrees >= agreement_threshold)
            n_high_agreement += high_agreement_mask.sum()
        
        # ROW NORMALIZE: Make probabilities sum to 1 per cell
        # This is CRITICAL - just like individual atlas predictions
        row_sums = average_probs_raw.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        average_probs_norm = average_probs_raw / row_sums
        
        # Store NORMALIZED scores per label
        for j, lbl in enumerate(labels):
            out_key = f"{average_prefix}{depth}.{lbl}.predscore"
            if is_anndata:
                obj.obs[out_key] = average_probs_norm[:, j].astype(float)
            else:
                obj.protein.row_attrs[out_key] = average_probs_norm[:, j].astype(float)
        
        # Average high-agreement count across labels
        n_high_agreement = n_high_agreement // max(1, len(labels))
        
        # Make average calls using NORMALIZED probabilities
        winner_idx = average_probs_norm.argmax(axis=1)
        winner_conf = average_probs_norm[np.arange(n_cells), winner_idx]
        winner_lab = np.array([labels[i] for i in winner_idx], dtype=object)
        
        # Mark cells as "Uncertain" if winning score is below threshold
        uncertain_mask = (winner_conf < score_threshold)
        winner_lab[uncertain_mask] = "Uncertain"
        winner_conf[uncertain_mask] = 0.0
        
        # Store average predictions
        if is_anndata:
            obj.obs[f"{average_prefix}{depth}.pred"] = pd.Categorical(
                winner_lab, categories=list(labels) + ["Uncertain"]
            )
            obj.obs[f"{average_prefix}{depth}.conf"] = winner_conf.astype(float)
            obj.obsm[f"{average_prefix}{depth}.probs"] = pd.DataFrame(
                average_probs_norm, index=obj.obs_names, columns=labels
            )
        else:
            obj.protein.row_attrs[f"{average_prefix}{depth}.pred"] = winner_lab.astype(str)
            obj.protein.row_attrs[f"{average_prefix}{depth}.conf"] = winner_conf.astype(float)
        
        n_certain = (winner_lab != "Uncertain").sum()
        print(f"[average_consensus] {depth}: {n_certain}/{n_cells} cells have confident predictions")
        print(f"  (~{n_high_agreement} cells avg with agreement≥{agreement_threshold}; final confident after score≥{score_threshold}: {n_certain})")


# ============================================================================
# EXAMPLE: How normalization affects the results
# ============================================================================
"""
SCENARIO: Simplified depth with 3 classes [B, NK, Monocyte]

Cell 1: High agreement on all classes
  Raw Averaged scores:     B=0.80, NK=0.05, Mono=0.10  (already sum to ~0.95)
  Agreement:               B=0.95, NK=0.95, Mono=0.95  (all high)
  
  Modulated (before norm): B=0.95*0.80 + 0.05*0.5 = 0.785
                          NK=0.95*0.05 + 0.05*0.5 = 0.0725
                          Mono=0.95*0.10 + 0.05*0.5 = 0.12
                          Sum = 0.9775
  
  After ROW NORM:         B=0.785/0.9775 = 0.803
                          NK=0.0725/0.9775 = 0.074
                          Mono=0.12/0.9775 = 0.123
                          Sum = 1.000 ✓

Cell 2: Low agreement on all classes (datasets disagree)
  Raw Averaged scores:     B=0.30, NK=0.25, Mono=0.35
  Agreement:               B=0.20, NK=0.25, Mono=0.30  (all low)
  
  Modulated (before norm): B=0.20*0.30 + 0.80*0.5 = 0.46
                          NK=0.25*0.25 + 0.75*0.5 = 0.4375
                          Mono=0.30*0.35 + 0.70*0.5 = 0.455
                          Sum = 1.3525
  
  After ROW NORM:         B=0.46/1.3525 = 0.340
                          NK=0.4375/1.3525 = 0.323
                          Mono=0.455/1.3525 = 0.336
                          Sum = 1.000 ✓
  
  → All classes pushed toward ~0.33 (uniform) due to disagreement ✓

Cell 3: Mixed - high agreement that B is negative, low agreement on others
  Raw Averaged scores:     B=0.05, NK=0.40, Mono=0.45
  Agreement:               B=0.90, NK=0.30, Mono=0.35
  
  Modulated (before norm): B=0.90*0.05 + 0.10*0.5 = 0.095
                          NK=0.30*0.40 + 0.70*0.5 = 0.47
                          Mono=0.35*0.45 + 0.65*0.5 = 0.4825
                          Sum = 1.0475
  
  After ROW NORM:         B=0.095/1.0475 = 0.091  (stays low - high agreement)
                          NK=0.47/1.0475 = 0.449
                          Mono=0.4825/1.0475 = 0.461
                          Sum = 1.000 ✓
  
  → B stays low (high agreement), NK/Mono uncertain (low agreement) ✓
"""