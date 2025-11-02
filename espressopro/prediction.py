# -*- coding: utf-8 -*-
"""Prediction/scoring utilities (with scaling) and consensus (mean/weighted) atlas blending."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import warnings

import joblib
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from anndata import AnnData
from sklearn.ensemble import StackingClassifier
from sklearn.neighbors import KDTree
from sklearn.pipeline import Pipeline

from .core import (
    ensure_models_available,
    get_default_data_path,
    get_default_models_path,
    load_models,
)
from .constants import SIMPLIFIED_CLASSES, _DETAILED_LABELS

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
    if tried:
        print("[generate_predictions] Tried sidecar feature_names at:\n  - " + "\n  - ".join(tried))
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
    """Inspect feature-name overlap between the query object and every OvR head."""
    if models_path is None:
        models_path = str(get_default_models_path())
    if data_path is None:
        data_path = str(get_default_data_path())
    models = load_models(models_path)
    models_path = Path(models_path)
    data_path   = Path(data_path)

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
    qset  = set(qcols)

    cd_like  = sum(c.upper().startswith("CD") for c in qcols)
    hla_like = sum("HLA" in c.upper() for c in qcols)
    print(f"[audit] query_n={len(qcols)} ; CD-like={cd_like} ; HLA-like={hla_like}")
    print(f"[audit] first {min(show, len(qcols))} query features:", qcols[:show])

    rows = []
    for atlas, depth_map in models.items():
        if not isinstance(depth_map, dict):
            continue
        for depth, cell_dict in depth_map.items():
            if not isinstance(cell_dict, dict):
                continue
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
                overlap = sorted(qset & tset)
                rows.append({
                    "atlas": str(atlas),
                    "depth": str(depth),
                    "label": str(label),
                    "query_n": len(qcols),
                    "train_n": len(train_cols),
                    "overlap_n": len(overlap),
                    "overlap_pct": (len(overlap) / max(1, len(train_cols))) * 100.0,
                })

    df = pd.DataFrame(rows).sort_values(["overlap_pct", "atlas", "depth", "label"])
    print("\n[audit] worst overlaps:\n", df.head(min(10, len(df))).to_string(index=False))
    return df

def _preview_query_df(df: pd.DataFrame, *, n_rows: int = 5, n_cols: int = 12, tag: str = "query_df") -> None:
    """Lightweight preview of the query matrix."""
    n_cells, n_feats = df.shape
    print(f"[generate_predictions] {tag} shape: {n_cells} cells x {n_feats} features")
    cols_preview = list(df.columns[:min(n_cols, n_feats)])
    try:
        print(df.loc[df.index[:n_rows], cols_preview].to_string())
    except Exception:
        print(df.head(n_rows).to_string())
    nan_ct = int(np.isnan(df.values).sum())
    inf_ct = int(np.isinf(df.values).sum())
    if nan_ct or inf_ct:
        print(f"[generate_predictions][WARN] found NaN={nan_ct}, inf={inf_ct} in {tag}")

def _preview_rescale_pair(
    X_before: pd.DataFrame,
    X_after: pd.DataFrame,
    atlas: str,
    depth: str,
    label: str,
    *,
    n_cols: int = 6
) -> None:
    """Show mean/std shift on a few columns after scaling."""
    if X_before.shape[1] == 0:
        print(f"[generate_predictions][{atlas}.{depth}.{label}] (no columns to preview)")
        return
    cols = list(X_before.columns[:min(n_cols, X_before.shape[1])])
    b_mean = X_before[cols].mean().round(3)
    b_std  = X_before[cols].std(ddof=0).round(3)
    a_mean = X_after[cols].mean().round(3)
    a_std  = X_after[cols].std(ddof=0).round(3)
    print(f"[generate_predictions][{atlas}.{depth}.{label}] scaling preview (first {len(cols)} cols):")
    print("  means  before → after:", dict(zip(cols, zip(b_mean.tolist(), a_mean.tolist()))))
    print("  stddev before → after:", dict(zip(cols, zip(b_std.tolist(),  a_std.tolist()))))

_ATLAS_NAMES = ("Hao", "Triana", "Zhang", "Luecken")

def _prune_existing_tracks(
    obj: Union[AnnData, Any],
    *,
    keep_atlases: Optional[Sequence[str]] = None,
    drop_averaged: bool = True,
    drop_best: bool = True,
) -> None:
    """Remove columns from prior runs not matching current selection."""
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
    show_rescaling: bool = True,
    add_consensus: bool = True,
    consensus_prefix: str = "Averaged.",
    consensus_normalize: bool = False,    # keep plain scores unless you need multiclass norm
    consensus_mode: str = "plain",        # 'plain' | 'global' | 'row' | 'hybrid'
    lambda_global: float = 0.5,           # for 'hybrid'
    add_unweighted: bool = False,
    unweighted_prefix: str = "Averaged.Unweighted.",
    unweighted_normalize: bool = True,
    use_atlases: Optional[Union[str, Sequence[str]]] = None,
    apply_exclusions: Optional[bool] = True,
) -> Union["AnnData", Any]:
    """
    Predict OvR scores with optional scaling; write per-atlas probabilities (no 'Atlas.' prefix).
    If multiple atlases are used, also write Averaged.* (consensus blend; default = row-wise plain mean).
    Additionally creates per-atlas summary columns:
        <Atlas>.<Depth>.pred
        <Atlas>.<Depth>.conf
        <Atlas>.<Depth>.Celltype
        <Atlas>.<Depth>.Celltype.TopScore
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
    data_path   = Path(data_path)

    is_anndata = (AnnData is not None and isinstance(obj, AnnData))
    is_sample  = hasattr(obj, "protein") and hasattr(obj.protein, "get_attribute")

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
        ts_scaler = _maybe_load_temp_scaler(models.get(atlas, {}), depth, atlas, str(data_path))

        def _row_normalize(M: np.ndarray) -> np.ndarray:
            M = np.clip(M, 0.0, 1.0)
            rs = M.sum(axis=1, keepdims=True)
            rs[rs <= 0.0] = 1.0
            return M / rs

        if ts_scaler is not None:
            try:
                P_mc = ts_scaler.transform(P_raw)
                P_mc = np.asarray(P_mc)
                if P_mc.ndim == 1:
                    P_mc = P_mc.reshape(-1, 1)
                if P_mc.shape[1] != len(cell_labels):
                    P_mc = _row_normalize(P_raw)
                else:
                    P_mc = _row_normalize(P_mc)
            except Exception:
                P_mc = _row_normalize(P_raw)
        else:
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

    if add_consensus or add_unweighted:
        used_atlases = sorted({atl for (atl, _depth) in probs_store.keys()})
        if not used_atlases:
            print("[generate_predictions] No atlases produced predictions; skipping averaging.")
            return obj

        use_exclusions = (len(used_atlases) > 1) if apply_exclusions is None else bool(apply_exclusions)

        if add_consensus:
            if is_anndata:
                add_consensus_weighted_tracks(
                    obj,
                    atlases=used_atlases,
                    normalize_multiclass=consensus_normalize,
                    depths=("Broad", "Simplified", "Detailed"),
                    out_prefix=consensus_prefix,
                    apply_exclusions=use_exclusions,
                    consensus_mode=consensus_mode,
                    lambda_global=lambda_global,
                )
            else:
                add_consensus_weighted_tracks_sample(
                    obj,
                    atlases=used_atlases,
                    normalize_multiclass=consensus_normalize,
                    depths=("Broad", "Simplified", "Detailed"),
                    out_prefix=consensus_prefix,
                    apply_exclusions=use_exclusions,
                    consensus_mode=consensus_mode,
                    lambda_global=lambda_global,
                )

        if add_unweighted:
            if is_anndata:
                add_unweighted_average_tracks(
                    obj,
                    atlases=used_atlases,
                    normalize_multiclass=unweighted_normalize,
                    depths=("Broad", "Simplified", "Detailed"),
                    out_prefix=unweighted_prefix,
                    apply_exclusions=use_exclusions,
                )
            else:
                add_unweighted_average_tracks_sample(
                    obj,
                    atlases=used_atlases,
                    normalize_multiclass=unweighted_normalize,
                    depths=("Broad", "Simplified", "Detailed"),
                    out_prefix=unweighted_prefix,
                    apply_exclusions=use_exclusions,
                )

    return obj

# -------------------- locality score (unchanged) --------------------

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

# -------------------- public: add best-localised tracks --------------------

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
    """Pick, per label, the atlas whose high-score cells are most spatially localized on UMAP.

    Writes columns like 'Best<Depth>.<Label>.predscore' copied from the best atlas' predscore.
    Requires obj to have UMAP coords at .obsm['X_umap'] (for AnnData), or for MissionBio samples,
    coords in sample.protein.row_attrs['umap'].
    """
    depth_to_labels: Dict[str, Sequence[str]] = {
        "Broad": ["Immature", "Mature"],
        "Simplified": list(SIMPLIFIED_CLASSES.keys()),
        "Detailed": _DETAILED_LABELS,
    }

    is_anndata = (AnnData is not None and isinstance(obj, AnnData))
    is_sample  = hasattr(obj, "protein") and hasattr(obj.protein, "row_attrs")

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
        # Build a minimal AnnData view if UMAP coords are available in the sample
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
        # mirror per-atlas columns into adata.obs for scoring
        for depth in depths:
            lbls = list(labels) if labels is not None else list(depth_to_labels.get(depth, []))
            for lbl in lbls:
                for atl in atlases:
                    key = f"{atl}.{depth}.{lbl}.predscore"
                    if key in obj.protein.row_attrs:
                        adata.obs[key] = np.asarray(obj.protein.row_attrs[key], dtype=float)
            # choose best and write back
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

# -------------------- atlas exclusions (Broad has none) --------------------

EXCLUDE_ATLAS: Dict[str, Dict[str, set]] = {
    "Simplified": {
        "CD4_T":   {"Hao", "Zhang"},
        "CD8_T":   {"Hao", "Zhang"},
        "Other_T":   {"Luecken"},
        "Erythroid": {"Triana", "Hao"},
        "HSPC":    {"Zhang", "Luecken", "Hao"},
        "Monocyte": {"Zhang"},
        "Myeloid": {"Zhang"},
        "NK":      {"Zhang"},
        "cDC":     {"Luecken", "Zhang"},
        "B":       {"Hao", "Triana"},
        "Plasma":  {"Hao", "Zhang", "Triana"},
    },
    "Detailed": {
        "Erythroblast": {"Hao", "Triana", "Zhang"},
        "ErP":          {"Triana"},
        "cDC2":         {"Zhang", "Luecken"},
        "cDC1":         {"Hao", "Zhang"},
        "pDC":          {"Luecken", "Zhang"},
        "MEP":          {"Zhang"},
        "MkP":          {"Zhang"},
        "HSC_MPP":      {"Zhang", "Luecken"},
        "LMPP":         {"Zhang"},
        "GMP":          {"Zhang"},
        "Pre-Pro-B":    {"Zhang"},
        "Pre-B":        {"Triana"},
        "CD14_Mono":   {"Zhang"},
        "CD16_Mono":   {"Zhang"},
        "CD4_CTL":   {"Hao"},
        "CD4_T_Memory":   {"Hao"},
        "CD4_T_Naive":   {"Hao"},
        "CD8_T_Naive":   {"Luecken"},
        "Immature_B":   {"Triana", "Hao", "Zhang"},
        "B_Naive":   {"Zhang", "Luecken"},
        "B_Memory":   {"Zhang", "Triana", "Luecken"},
        "Plasma":       {"Zhang"},
        "MAIT":             {"Triana", "Zhang", "Luecken"},
        "NK_CD56_bright":   {"Zhang", "Zhang", "Luecken"},
        "Treg":             {"Luecken"},
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
    """Stack per-atlas predscore columns for (depth, label)."""
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
    """
    Atlas-level reliability weights from cross-atlas agreement.
    Returns a length-A vector (A = #atlases) with sum=1.
    """
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
    """
    Per-cell robust weights against outlier atlases based on MAD from the row median.
    Returns a length-A vector with sum=1 (unless row is constant → uniform).
    """
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

# -------------------- consensus blender + writers --------------------

def _consensus_blend(
    M: np.ndarray,
    eps: float = 1e-9,
    mode: str = "plain",          # 'plain' | 'global' | 'row' | 'hybrid'
    lambda_global: float = 0.5,   # only for 'hybrid'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Blend atlas scores across columns to a single consensus per cell.
    - mode='plain':   simple row-wise mean (default; matches naive /A).
    - mode='global':  same weights for all cells from cross-atlas agreement.
    - mode='row':     per-cell robust weights (down-weight outliers per row).
    - mode='hybrid':  convex combo of global and row weights.
    Returns:
      consensus: (n,) vector
      agreement: MAD-based agreement score in [0,1] (higher is better agreement)
    """
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

    # agreement: 1 - 2*MAD(row) clipped to [0,1]
    med = np.median(M, axis=1)
    mad = np.median(np.abs(M - med[:, None]), axis=1)
    agree = np.clip(1.0 - 2.0 * mad, 0.0, 1.0)

    return cons.astype(float), agree.astype(float)

def add_consensus_weighted_tracks(
    adata: AnnData,
    *,
    atlases: Sequence[str],
    normalize_multiclass: bool = False,  # default False to preserve the mean
    depths: Sequence[str] = ("Broad", "Simplified", "Detailed"),
    out_prefix: str = "Averaged.",
    apply_exclusions: bool = True,
    consensus_mode: str = "plain",        # NEW
    lambda_global: float = 0.5,           # NEW
) -> None:
    """Consensus Averaged.* tracks (plain/weighted) across atlases, respecting exclusions."""
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
                    vec = adata.obs[src].astype(float).to_numpy()
                    adata.obs[pred_col] = vec
                    adata.obs[agr_col]  = 1.0
                    adata.obs[nat_col]  = 1
                    created_cols.append(pred_col)
                continue

            M, _used = _stack_label_scores(adata, depth, lbl, atlases, apply_exclusions=apply_exclusions)
            if M.shape[1] == 0:
                continue
            M = np.clip(M, 0.0, 1.0)
            # Force unweighted mean specifically for Averaged.Broad.Immature.predscore
            if depth == "Broad" and lbl == "Immature":
                cons, agree = _consensus_blend(M, mode="plain", lambda_global=lambda_global)
            else:
                cons, agree = _consensus_blend(M, mode=consensus_mode, lambda_global=lambda_global)

            adata.obs[pred_col] = cons
            adata.obs[agr_col]  = agree
            adata.obs[nat_col]  = int(M.shape[1])
            created_cols.append(pred_col)

        # For Broad, keep plain means (no renormalization)
        if depth != "Broad" and normalize_multiclass and created_cols:
            X = adata.obs[created_cols].to_numpy(dtype=float)
            X = np.clip(X, 0.0, 1.0)
            rs = X.sum(axis=1, keepdims=True)
            rs[rs <= 0.0] = 1.0
            adata.obs.loc[:, created_cols] = X / rs

def add_consensus_weighted_tracks_sample(
    sample,
    *,
    atlases: Sequence[str],
    normalize_multiclass: bool = False,
    depths: Sequence[str] = ("Broad", "Simplified", "Detailed"),
    out_prefix: str = "Averaged.",
    apply_exclusions: bool = True,
    consensus_mode: str = "plain",        # NEW
    lambda_global: float = 0.5,           # NEW
) -> None:
    """Sample variant of consensus Averaged.* tracks."""
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

        created, label_vecs = [], []

        for lbl in labels:
            pred_key = f"{out_prefix}{depth}.{lbl}.predscore"
            agr_key  = f"{out_prefix}{depth}.{lbl}.agreement"
            nat_key  = f"{out_prefix}{depth}.{lbl}.n_atlases"

            if single_passthrough:
                src = f"{src_atl}.{depth}.{lbl}.predscore"
                if src in sample.protein.row_attrs:
                    vec = np.asarray(sample.protein.row_attrs[src], dtype=float).reshape(-1)
                    sample.protein.row_attrs[pred_key] = vec
                    sample.protein.row_attrs[agr_key]  = np.ones_like(vec, dtype=float)
                    sample.protein.row_attrs[nat_key]  = 1
                    created.append(pred_key)
                    label_vecs.append(vec)
                continue

            allowed = _filter_atlases_for_label(atlases, depth, lbl, apply_exclusions=apply_exclusions)
            cols = []
            for atl in allowed:
                key = f"{atl}.{depth}.{lbl}.predscore"
                if key in sample.protein.row_attrs:
                    cols.append(np.asarray(sample.protein.row_attrs[key], dtype=float).reshape(-1))
            if not cols:
                continue
            M = np.clip(np.column_stack(cols), 0.0, 1.0)
            # Force unweighted mean specifically for Averaged.Broad.Immature.predscore
            if depth == "Broad" and lbl == "Immature":
                cons, agree = _consensus_blend(M, mode="plain", lambda_global=lambda_global)
            else:
                cons, agree = _consensus_blend(M, mode=consensus_mode, lambda_global=lambda_global)

            sample.protein.row_attrs[pred_key] = cons.astype(float)
            sample.protein.row_attrs[agr_key]  = agree.astype(float)
            sample.protein.row_attrs[nat_key]  = int(M.shape[1])
            created.append(pred_key)
            label_vecs.append(cons.reshape(-1))

    if normalize_multiclass and created and depth != "Broad":
            X = np.clip(np.column_stack(label_vecs).astype(float), 0.0, 1.0)
            rs = X.sum(axis=1, keepdims=True)
            rs[rs <= 0.0] = 1.0
            Xn = X / rs
            for j, key in enumerate(created):
                sample.protein.row_attrs[key] = Xn[:, j].astype(float)

# -------------------- legacy "unweighted" (kept) --------------------

def add_unweighted_average_tracks(
    adata: AnnData,
    *,
    atlases: Sequence[str],
    normalize_multiclass: bool = True,
    depths: Sequence[str] = ("Broad", "Simplified", "Detailed"),
    out_prefix: str = "Averaged.Unweighted.",
    apply_exclusions: bool = True,
) -> None:
    """Unweighted Averaged.* tracks (row-wise mean); retained for backward compatibility."""
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
            nat_col  = f"{out_prefix}{depth}.{lbl}.n_atlases"

            if single_passthrough:
                src = f"{src_atl}.{depth}.{lbl}.predscore"
                if src in adata.obs:
                    adata.obs[pred_col] = adata.obs[src].astype(float).to_numpy()
                    adata.obs[nat_col]  = 1
                    created_cols.append(pred_col)
                continue

            M, _used = _stack_label_scores(adata, depth, lbl, atlases, apply_exclusions=apply_exclusions)
            if M.shape[1] == 0:
                continue
            M = np.clip(M, 0.0, 1.0)
            cons = M.mean(axis=1)

            adata.obs[pred_col] = cons.astype(float)
            adata.obs[nat_col]  = int(M.shape[1])
            created_cols.append(pred_col)

    # For Broad, keep plain means (no renormalization)
    if depth != "Broad" and normalize_multiclass and created_cols:
            X = adata.obs[created_cols].to_numpy(dtype=float)
            X = np.clip(X, 0.0, 1.0)
            rs = X.sum(axis=1, keepdims=True)
            rs[rs <= 0.0] = 1.0
            adata.obs.loc[:, created_cols] = X / rs

def add_unweighted_average_tracks_sample(
    sample,
    *,
    atlases: Sequence[str],
    normalize_multiclass: bool = True,
    depths: Sequence[str] = ("Broad", "Simplified", "Detailed"),
    out_prefix: str = "Averaged.Unweighted.",
    apply_exclusions: bool = True,
) -> None:
    """Sample variant of unweighted averaging (row-wise mean); retained for backward compatibility."""
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
            nat_key  = f"{out_prefix}{depth}.{lbl}.n_atlases"

            if single_passthrough:
                src = f"{src_atl}.{depth}.{lbl}.predscore"
                if src in sample.protein.row_attrs:
                    vec = np.asarray(sample.protein.row_attrs[src], dtype=float).reshape(-1)
                    sample.protein.row_attrs[pred_key] = vec
                    sample.protein.row_attrs[nat_key]  = 1
                    created_keys.append(pred_key)
                    label_vecs.append(vec)
                continue

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

            sample.protein.row_attrs[pred_key] = cons.astype(float)
            sample.protein.row_attrs[nat_key]  = int(M.shape[1])

            created_keys.append(pred_key)
            label_vecs.append(cons.reshape(-1))

        # For Broad, keep plain means (no renormalization)
        if depth != "Broad" and normalize_multiclass and created_keys:
            X = np.clip(np.column_stack(label_vecs).astype(float), 0.0, 1.0)
            rs = X.sum(axis=1, keepdims=True)
            rs[rs <= 0.0] = 1.0
            Xn = X / rs
            for j, key in enumerate(created_keys):
                sample.protein.row_attrs[key] = Xn[:, j].astype(float)
