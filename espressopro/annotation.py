# -*- coding: utf-8 -*-
"""Cell type annotation workflows and voting systems with Averaged + per-atlas outputs.

What this module produces
-------------------------
Averaged winners (pred/conf) and constrained labels:
    Averaged.Broad.pred / conf
    Averaged.Simplified.pred / conf
    Averaged.Detailed.pred / conf
    Averaged.Broad.Celltype (+ TopScore/LowConf)
    Averaged.Simplified.Celltype (+ TopScore/LowConf)
    Averaged.Detailed.Celltype (+ TopScore/LowConf)

Per-atlas (no 'Atlas.' prefix) hierarchy-constrained labels:
    <Atlas>.Broad.Celltype (+ TopScore/LowConf)
    <Atlas>.Simplified.Celltype (+ TopScore/LowConf)
    <Atlas>.Detailed.Celltype (+ TopScore/LowConf)

Housekeeping:
    - Legacy 'Atlas.<name>.*' are converted to '<name>.*' (if needed) and then removed.
    - 'Averaged.Unweighted.*' tracks are removed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional, Any, Sequence, Mapping

import copy
import re
import warnings

import numpy as np
import pandas as pd
import anndata as ad
from anndata import AnnData
from scipy.sparse import isspmatrix, issparse
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples
from sklearn.neighbors import NearestNeighbors

from .prediction import generate_predictions, add_best_localised_tracks
from .constants import (
    SIMPLIFIED_CLASSES,
    DETAILED_CLASSES,
    SIMPLIFIED_PARENT_MAP,
    DETAILED_PARENT_MAP,
)

# near the other imports in annotation.py
from .prediction import EXCLUDE_ATLAS

def _filter_atlases_for_label_local(atlases, depth, label, *, apply_exclusions: bool = True):
    if not apply_exclusions:
        return list(atlases)
    banned = EXCLUDE_ATLAS.get(depth, {}).get(label, set())
    return [a for a in atlases if a not in banned]


def _is_mosaic_sample(x: Any) -> bool:
    return hasattr(x, "protein") and hasattr(x.protein, "row_attrs") and hasattr(x.protein, "get_attribute")


def _class_from_key(k: str) -> str:
    parts = str(k).split(".")
    return parts[-2] if len(parts) >= 3 and parts[-1] == "predscore" else ""


def _clr_fallback(X: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    X[~np.isfinite(X)] = 0.0
    logs = np.log(X + eps)
    return logs - logs.mean(axis=1, keepdims=True)


def Normalise_protein_data(
    data,
    inplace: bool = True,
    jitter: float = 0.45,
    random_state: int = 42,
    scale: Optional[float] = None,
):
    """NSP normalization for protein counts with CLR fallback."""
    is_mosaic_sample = hasattr(data, "protein") and hasattr(getattr(data, "protein"), "layers")
    if is_mosaic_sample and "read_counts" in data.protein.layers:
        counts = data.protein.layers["read_counts"]
        X = counts.toarray() if isspmatrix(counts) or issparse(counts) else np.asarray(counts, dtype=float)
        try:
            from missionbio.demultiplex.protein.nsp import NSP
            nsp = NSP(jitter=jitter, random_state=random_state)
            Xn = nsp.transform(X, scale=scale)
            print("[Normalise_protein_data] Applied MissionBio NSP normalization")
        except ImportError:
            warnings.warn("MissionBio package not available. Falling back to CLR normalization.")
            Xn = _clr_fallback(X)
            print("[Normalise_protein_data] Applied CLR normalization (fallback)")
        except Exception as e:
            warnings.warn(f"NSP normalization failed ({e}). Falling back to CLR normalization.")
            Xn = _clr_fallback(X)
            print("[Normalise_protein_data] Applied CLR normalization (fallback)")

        if inplace:
            data.protein.layers["Normalized_reads"] = Xn
            return None
        sample_copy = data.copy() if hasattr(data, "copy") else copy.deepcopy(data)
        sample_copy.protein.layers["Normalized_reads"] = Xn
        return sample_copy

    is_anndata = isinstance(data, AnnData)

    if is_anndata:
        adata = data if inplace else data.copy()
        X = adata.X.toarray() if issparse(adata.X) or isspmatrix(adata.X) else np.asarray(adata.X, dtype=float)
        try:
            from missionbio.demultiplex.protein.nsp import NSP
            nsp = NSP(jitter=jitter, random_state=random_state)
            Xn = nsp.transform(X, scale=scale)
            print("[Normalise_protein_data] Applied MissionBio NSP normalization")
        except ImportError:
            warnings.warn("MissionBio package not available. Falling back to CLR normalization.")
            Xn = _clr_fallback(X)
            print("[Normalise_protein_data] Applied CLR normalization (fallback)")
        except Exception as e:
            warnings.warn(f"NSP normalization failed ({e}). Falling back to CLR normalization.")
            Xn = _clr_fallback(X)
            print("[Normalise_protein_data] Applied CLR normalization (fallback)")

        adata.X = Xn
        return None if inplace else adata

    if isinstance(data, pd.DataFrame):
        X = data.values.astype(float)
        return _nsp_then_clr(X, jitter, random_state, scale)

    if isinstance(data, np.ndarray) or issparse(data) or isspmatrix(data):
        X = data.toarray() if issparse(data) or isspmatrix(data) else np.asarray(data, dtype=float)
        return _nsp_then_clr(X, jitter, random_state, scale)

    raise ValueError(
        "Input must be a MissionBio Sample (protein.layers['read_counts']), AnnData, numpy array, DataFrame, or sparse."
    )


def _nsp_then_clr(X: np.ndarray, jitter: float, random_state: int, scale: Optional[float]) -> np.ndarray:
    try:
        from missionbio.demultiplex.protein.nsp import NSP
        nsp = NSP(jitter=jitter, random_state=random_state)
        Xn = nsp.transform(X, scale=scale)
        print("[Normalise_protein_data] Applied MissionBio NSP normalization")
        return Xn
    except ImportError:
        warnings.warn("MissionBio package not available. Falling back to CLR normalization.")
    except Exception as e:
        warnings.warn(f"NSP normalization failed ({e}). Falling back to CLR normalization.")
    Xn = _clr_fallback(X)
    print("[Normalise_protein_data] Applied CLR normalization (fallback)")
    return Xn


def Scale_protein_data(data, inplace: bool = True):
    """StandardScaler on normalized protein data."""
    try:
        from missionbio.mosaic.sample import Sample
    except ImportError:
        Sample = None

    if Sample is not None and isinstance(data, Sample):
        if "Normalized_reads" not in data.protein.layers:
            raise ValueError("No 'Normalized_reads' layer found in Sample.protein.layers")
        x_array = np.array(data.protein.layers["Normalized_reads"], dtype=np.float64)
        from sklearn import preprocessing
        scaler = preprocessing.StandardScaler()
        data.protein.layers["Scaled_reads"] = scaler.fit_transform(x_array)
        print("[Scale_protein_data] Scaled 'Normalized_reads' -> saved as 'Scaled_reads'")
        return None

    if isinstance(data, AnnData):
        adata = data if inplace else data.copy()
        x = adata.X
    else:
        if isinstance(data, (pd.DataFrame, np.ndarray)) or issparse(data) or isspmatrix(data):
            x = data
        else:
            raise ValueError("Input must be AnnData, MissionBio Sample, numpy array, DataFrame, or sparse matrix")

    if isinstance(x, pd.DataFrame):
        x_array = x.values.astype(np.float64)
        is_dataframe = True
        df_index, df_columns = x.index, x.columns
    elif issparse(x) or isspmatrix(x):
        x_array = x.toarray().astype(np.float64)
        is_dataframe = False
    else:
        x_array = np.array(x, dtype=np.float64)
        is_dataframe = False

    from sklearn import preprocessing
    scaler = preprocessing.StandardScaler()
    scaled = scaler.fit_transform(x_array)

    if isinstance(data, AnnData):
        data.X = scaled
        return None if inplace else data
    if is_dataframe:
        return pd.DataFrame(scaled, index=df_index, columns=df_columns)
    return scaled


def _make_meta_from_row_attrs(sample, needed_cols):
    df_scaled = sample.protein.get_attribute("Scaled_reads", constraint="row+col")
    if not isinstance(df_scaled, pd.DataFrame):
        df_scaled = pd.DataFrame(df_scaled)
    idx = df_scaled.index.astype(str)

    obs = pd.DataFrame(index=idx)
    for col in needed_cols:
        if col in sample.protein.row_attrs:
            v = np.asarray(sample.protein.row_attrs[col]).reshape(-1)
            if v.shape[0] == len(idx):
                obs[col] = v
    return ad.AnnData(X=np.zeros((len(idx), 0), dtype=float), obs=obs, var=pd.DataFrame(index=[]))


def _write_obs_to_row_attrs(sample, adata_meta, cols):
    for c in cols:
        if c not in adata_meta.obs.columns:
            continue
        vals = adata_meta.obs[c]
        arr = vals.astype(str).to_numpy() if pd.api.types.is_categorical_dtype(vals) else np.asarray(vals.to_numpy())
        sample.protein.row_attrs[c] = arr


def _runtime_class_map(class_map: Dict[str, List[str]], cols_available: set) -> Dict[str, List[str]]:
    out = {lbl: [c for c in cols if c in cols_available] for lbl, cols in class_map.items()}
    out = {lbl: cols for lbl, cols in out.items() if cols}
    if not any(out.values()):
        raise KeyError("No matching predscore columns found for any class in the provided class_map.")
    return out


def _runtime_parent_subset(parent_map: Dict[str, List[str]], cols_available: set) -> Dict[str, List[str]]:
    return {p: [c for c in cols if c in cols_available] for p, cols in parent_map.items()}


def voting_annotator(
    obj: Union["AnnData", object],
    level_name: str,
    class_to_sources: Dict[str, List[str]],
    parent_field: Optional[str] = None,
    parent_to_subset: Optional[Dict[str, List[str]]] = None,
    conf_threshold: float = 0.75,
    *,
    normalize: bool = False,
) -> None:
    """Aggregate sources → per-class predscores, optional masking, and final labels."""
    is_anndata = isinstance(obj, AnnData)
    is_sample = _is_mosaic_sample(obj)
    if not (is_anndata or is_sample):
        raise TypeError("voting_annotator expects an AnnData or a missionbio.mosaic.sample.Sample")

    if is_anndata:
        n = obj.n_obs

        def has_key(k: str) -> bool: return k in obj.obs.columns
        def get_vec(k: str) -> np.ndarray: return obj.obs[k].to_numpy()
        def set_vec(k: str, v: np.ndarray): obj.obs[k] = v
        def list_keys() -> List[str]: return list(obj.obs.columns)
    else:
        df_scaled = obj.protein.get_attribute("Scaled_reads", constraint="row+col")
        if not isinstance(df_scaled, pd.DataFrame):
            df_scaled = pd.DataFrame(df_scaled)
        n = len(df_scaled.index)

        def has_key(k: str) -> bool: return k in obj.protein.row_attrs
        def get_vec(k: str) -> np.ndarray: return np.asarray(obj.protein.row_attrs[k]).reshape(-1)
        def set_vec(k: str, v: np.ndarray): obj.protein.row_attrs[k] = np.asarray(v)
        def list_keys() -> List[str]: return list(obj.protein.row_attrs.keys())

    all_single = True
    for _, cols in class_to_sources.items():
        present = [c for c in cols if has_key(c)]
        if len(present) != 1:
            all_single = False
            break

    if all_single:
        for out_cls, cols in class_to_sources.items():
            src = [c for c in cols if has_key(c)][0]
            set_vec(f"{level_name}.{out_cls}.predscore", get_vec(src))
    else:
        for out_cls, cols in class_to_sources.items():
            present = [c for c in cols if has_key(c)]
            if present:
                mats = [get_vec(c) for c in present]
                mats = [m.reshape(-1) for m in mats if m.shape[0] == n]
                if mats:
                    avg = np.mean(np.vstack(mats), axis=0)
                    set_vec(f"{level_name}.{out_cls}.predscore", avg)

    score_cols = [k for k in list_keys() if k.startswith(f"{level_name}.") and k.endswith(".predscore")]
    if not score_cols:
        return

    rows = []
    for k in score_cols:
        v = get_vec(k)
        vv = v if v.shape[0] == n else np.zeros(n, dtype=float)
        rows.append(vv.reshape(1, -1))
    M = np.vstack(rows).T

    mask = np.ones_like(M, dtype=bool)
    if parent_field and parent_to_subset and (parent_field in list_keys()):
        parents = get_vec(parent_field).astype(str)

        def _classes_from_fullcols(cols_list):
            out = set()
            for s in cols_list:
                parts = str(s).split(".")
                if len(parts) >= 3 and parts[-1] == "predscore":
                    out.add(parts[-2])
            return out

        allowed_classes = {p: _classes_from_fullcols(cols) for p, cols in parent_to_subset.items()}
        out_class_names = [_class_from_key(c) for c in score_cols]

        for i in range(n):
            p = parents[i]
            if p in allowed_classes:
                keep = allowed_classes[p]
                if len(keep) == 0:
                    mask[i, :] = False
                else:
                    for j, cls in enumerate(out_class_names):
                        mask[i, j] = (cls in keep)

    if normalize:
        M_masked = np.where(mask, M, -np.inf)
        Mmax = np.max(M_masked, axis=1, keepdims=True)
        M_stable = M_masked - Mmax
        Z = np.exp(M_stable)
        denom = Z.sum(axis=1, keepdims=True)
        denom[denom == 0.0] = 1.0
        P = Z / denom
    else:
        P = np.where(mask, M, 0.0)
        P = np.clip(P, 0.0, 1.0)

    for j, k in enumerate(score_cols):
        set_vec(k, P[:, j])

    winner_idx = np.argmax(P, axis=1)
    winner_scores = P[np.arange(P.shape[0]), winner_idx]
    winner_cols = np.array(score_cols, dtype=object)[winner_idx]
    winners = np.array([_class_from_key(c) for c in winner_cols], dtype=object)

    set_vec(f"{level_name}.Celltype", winners)
    set_vec(f"{level_name}.Celltype.TopScore", winner_scores)
    set_vec(f"{level_name}.Celltype.LowConf", (winner_scores < conf_threshold).astype(bool))


def Broad_Annotation(adata_or_sample, conf_threshold: float = 0.75):
    """Broad (Mature/Immature) from Averaged.Broad.*."""
    level_name = "Averaged.Broad"
    REF = {
        "Mature":   [f"{level_name}.Mature.predscore"],
        "Immature": [f"{level_name}.Immature.predscore"],
    }

    if _is_mosaic_sample(adata_or_sample):
        sample = adata_or_sample
        needed = [f"{level_name}.Mature.predscore", f"{level_name}.Immature.predscore"]
        meta = _make_meta_from_row_attrs(sample, needed)
        for k in needed:
            if k not in meta.obs:
                raise KeyError(f"[Broad] Missing required column in row_attrs: {k}")
        voting_annotator(meta, level_name, REF, conf_threshold=conf_threshold)
        _write_obs_to_row_attrs(
            sample, meta,
            [f"{level_name}.Celltype", f"{level_name}.Celltype.TopScore", f"{level_name}.Celltype.LowConf"]
        )
        return sample

    adata = adata_or_sample
    for k in (f"{level_name}.Mature.predscore", f"{level_name}.Immature.predscore"):
        if k not in adata.obs:
            raise KeyError(f"[Broad] Missing required column in adata.obs: {k}")
    voting_annotator(adata, level_name, REF, conf_threshold=conf_threshold)
    return adata


def Simplified_Annotation(adata_or_sample, conf_threshold: float = 0.75):
    """Constrained Simplified from Averaged.Simplified.* gated by Averaged.Broad.Celltype."""
    level_name = "Averaged.Simplified"

    if _is_mosaic_sample(adata_or_sample):
        sample = adata_or_sample
        if f"Averaged.Broad.Celltype" not in sample.protein.row_attrs:
            Broad_Annotation(sample, conf_threshold=conf_threshold)

        meta = _make_meta_from_row_attrs(sample, list(sample.protein.row_attrs.keys()))
        if "Averaged.Broad.Celltype" in sample.protein.row_attrs:
            meta.obs["Averaged.Broad.Celltype"] = np.asarray(sample.protein.row_attrs["Averaged.Broad.Celltype"])

        available = set(meta.obs.columns)
        rt_classes = _runtime_class_map(SIMPLIFIED_CLASSES, available)
        rt_parents = _runtime_parent_subset(SIMPLIFIED_PARENT_MAP, available)

        voting_annotator(
            meta, level_name, rt_classes,
            parent_field="Averaged.Broad.Celltype",
            parent_to_subset=rt_parents,
            conf_threshold=conf_threshold,
        )

        _write_obs_to_row_attrs(
            sample, meta,
            [c for c in meta.obs.columns if c.startswith(f"{level_name}.") and c.endswith(".predscore")]
            + [f"{level_name}.Celltype", f"{level_name}.Celltype.TopScore", f"{level_name}.Celltype.LowConf"]
        )
        return sample

    adata = adata_or_sample
    if "Averaged.Broad.Celltype" not in adata.obs:
        Broad_Annotation(adata, conf_threshold=conf_threshold)

    available = set(adata.obs.columns)
    rt_classes = _runtime_class_map(SIMPLIFIED_CLASSES, available)
    rt_parents = _runtime_parent_subset(SIMPLIFIED_PARENT_MAP, available)

    voting_annotator(
        adata, level_name, rt_classes,
        parent_field="Averaged.Broad.Celltype",
        parent_to_subset=rt_parents,
        conf_threshold=conf_threshold,
    )
    return adata


def Detailed_Annotation(adata_or_sample, conf_threshold: float = 0.6):
    """Constrained Detailed from Averaged.Detailed.* gated by Averaged.Simplified.Celltype."""
    level_name = "Averaged.Detailed"

    if _is_mosaic_sample(adata_or_sample):
        sample = adata_or_sample
        if f"Averaged.Simplified.Celltype" not in sample.protein.row_attrs:
            Simplified_Annotation(sample, conf_threshold=0.75)

        meta = _make_meta_from_row_attrs(sample, list(sample.protein.row_attrs.keys()))
        if "Averaged.Simplified.Celltype" in sample.protein.row_attrs:
            meta.obs["Averaged.Simplified.Celltype"] = np.asarray(sample.protein.row_attrs["Averaged.Simplified.Celltype"])

        available = set(meta.obs.columns)
        rt_classes = _runtime_class_map(DETAILED_CLASSES, available)
        rt_parents = _runtime_parent_subset(DETAILED_PARENT_MAP, available)

        voting_annotator(
            meta, level_name, rt_classes,
            parent_field="Averaged.Simplified.Celltype",
            parent_to_subset=rt_parents,
            conf_threshold=conf_threshold,
        )

        _write_obs_to_row_attrs(
            sample, meta,
            [c for c in meta.obs.columns if c.startswith(f"{level_name}.") and c.endswith(".predscore")]
            + [f"{level_name}.Celltype", f"{level_name}.Celltype.TopScore", f"{level_name}.Celltype.LowConf"]
        )
        return sample

    adata = adata_or_sample
    if "Averaged.Simplified.Celltype" not in adata.obs:
        Simplified_Annotation(adata, conf_threshold=0.75)

    available = set(adata.obs.columns)
    rt_classes = _runtime_class_map(DETAILED_CLASSES, available)
    rt_parents = _runtime_parent_subset(DETAILED_PARENT_MAP, available)

    voting_annotator(
        adata, level_name, rt_classes,
        parent_field="Averaged.Simplified.Celltype",
        parent_to_subset=rt_parents,
        conf_threshold=conf_threshold,
    )
    return adata


_SIMPLIFIED_NAMES: List[str] = list(SIMPLIFIED_CLASSES.keys())
_DETAILED_NAMES:   List[str] = list(DETAILED_CLASSES.keys())


def _build_simplified_classmap_for(atlas: str) -> dict[str, list[str]]:
    return {lbl: [f"{atlas}.Simplified.{lbl}.predscore"] for lbl in _SIMPLIFIED_NAMES}


def _build_detailed_classmap_for(atlas: str) -> dict[str, list[str]]:
    return {lbl: [f"{atlas}.Detailed.{lbl}.predscore"] for lbl in _DETAILED_NAMES}


def _build_simplified_parentmap_for(atlas: str) -> dict[str, list[str]]:
    return {
        "Immature": [f"{atlas}.Simplified.HSPC.predscore"],
        "Mature":   [f"{atlas}.Simplified.{l}.predscore" for l in _SIMPLIFIED_NAMES if l != "HSPC"],
    }


def _build_detailed_parentmap_for(atlas: str) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for parent, allowed in DETAILED_PARENT_MAP.items():
        cls_names = []
        for key in allowed:
            parts = str(key).split(".")
            if len(parts) >= 4 and parts[-1] == "predscore":
                cls_names.append(parts[-2])
        out[parent] = [f"{atlas}.Detailed.{c}.predscore" for c in cls_names]
    return out


def Atlas_Broad_Annotation(adata_or_sample, atlas: str, conf_threshold: float = 0.75):
    """Broad (Mature/Immature) from <Atlas>.Broad.*."""
    level_name = f"{atlas}.Broad"
    REF = {
        "Mature":   [f"{atlas}.Broad.Mature.predscore"],
        "Immature": [f"{atlas}.Broad.Immature.predscore"],
    }

    if _is_mosaic_sample(adata_or_sample):
        sample = adata_or_sample
        needed = [f"{atlas}.Broad.Mature.predscore", f"{atlas}.Broad.Immature.predscore"]
        meta = _make_meta_from_row_attrs(sample, needed)
        for k in needed:
            if k not in meta.obs:
                raise KeyError(f"[{level_name}] Missing required column in row_attrs: {k}")
        voting_annotator(meta, level_name, REF, conf_threshold=conf_threshold)
        _write_obs_to_row_attrs(sample, meta, [
            f"{level_name}.Celltype", f"{level_name}.Celltype.TopScore", f"{level_name}.Celltype.LowConf"
        ])
        return sample

    adata = adata_or_sample
    for k in (f"{atlas}.Broad.Mature.predscore", f"{atlas}.Broad.Immature.predscore"):
        if k not in adata.obs: raise KeyError(f"[{level_name}] Missing required column in adata.obs: {k}")
    voting_annotator(adata, level_name, REF, conf_threshold=conf_threshold)
    return adata


def Atlas_Simplified_Annotation(adata_or_sample, atlas: str, conf_threshold: float = 0.75):
    """Constrained Simplified from <Atlas>.Simplified.* gated by <Atlas>.Broad.Celltype."""
    con_level = f"{atlas}.Constrained.Simplified"

    if _is_mosaic_sample(adata_or_sample):
        sample = adata_or_sample
        if f"{atlas}.Broad.Celltype" not in sample.protein.row_attrs:
            Atlas_Broad_Annotation(sample, atlas, conf_threshold=conf_threshold)

        meta = _make_meta_from_row_attrs(sample, list(sample.protein.row_attrs.keys()))
        if f"{atlas}.Broad.Celltype" in sample.protein.row_attrs:
            meta.obs[f"{atlas}.Broad.Celltype"] = np.asarray(sample.protein.row_attrs[f"{atlas}.Broad.Celltype"])

        avail = set(meta.obs.columns)
        classes = _runtime_class_map(_build_simplified_classmap_for(atlas), avail)
        parents = _runtime_parent_subset(_build_simplified_parentmap_for(atlas), avail)

        voting_annotator(
            meta, con_level, classes,
            parent_field=f"{atlas}.Broad.Celltype",
            parent_to_subset=parents,
            conf_threshold=conf_threshold,
        )

        meta.obs[f"{atlas}.Simplified.Celltype"] = meta.obs[f"{con_level}.Celltype"].astype(object)
        meta.obs[f"{atlas}.Simplified.Celltype.TopScore"] = meta.obs[f"{con_level}.Celltype.TopScore"].astype(float)
        meta.obs[f"{atlas}.Simplified.Celltype.LowConf"]  = meta.obs[f"{con_level}.Celltype.LowConf"].astype(bool)

        _write_obs_to_row_attrs(
            sample, meta,
            [c for c in meta.obs.columns if c.startswith(f"{con_level}.") and c.endswith(".predscore")]
            + [f"{atlas}.Simplified.Celltype", f"{atlas}.Simplified.Celltype.TopScore", f"{atlas}.Simplified.Celltype.LowConf"]
        )
        return sample

    adata = adata_or_sample
    if f"{atlas}.Broad.Celltype" not in adata.obs:
        Atlas_Broad_Annotation(adata, atlas, conf_threshold=conf_threshold)

    avail = set(adata.obs.columns)
    classes = _runtime_class_map(_build_simplified_classmap_for(atlas), avail)
    parents = _runtime_parent_subset(_build_simplified_parentmap_for(atlas), avail)

    voting_annotator(
        adata, con_level, classes,
        parent_field=f"{atlas}.Broad.Celltype",
        parent_to_subset=parents,
        conf_threshold=conf_threshold,
    )

    adata.obs[f"{atlas}.Simplified.Celltype"] = adata.obs[f"{con_level}.Celltype"].astype(object)
    adata.obs[f"{atlas}.Simplified.Celltype.TopScore"] = adata.obs[f"{con_level}.Celltype.TopScore"].astype(float)
    adata.obs[f"{atlas}.Simplified.Celltype.LowConf"]  = adata.obs[f"{con_level}.Celltype.LowConf"].astype(bool)
    return adata


def Atlas_Detailed_Annotation(adata_or_sample, atlas: str, conf_threshold: float = 0.6):
    """Constrained Detailed from <Atlas>.Detailed.* gated by <Atlas>.Simplified.Celltype."""
    con_level = f"{atlas}.Constrained.Detailed"

    if _is_mosaic_sample(adata_or_sample):
        sample = adata_or_sample
        if f"{atlas}.Simplified.Celltype" not in sample.protein.row_attrs:
            Atlas_Simplified_Annotation(sample, atlas, conf_threshold=0.75)

        meta = _make_meta_from_row_attrs(sample, list(sample.protein.row_attrs.keys()))
        if f"{atlas}.Simplified.Celltype" in sample.protein.row_attrs:
            meta.obs[f"{atlas}.Simplified.Celltype"] = np.asarray(sample.protein.row_attrs[f"{atlas}.Simplified.Celltype"])

        avail = set(meta.obs.columns)
        classes = _runtime_class_map(_build_detailed_classmap_for(atlas), avail)
        parents = _runtime_parent_subset(_build_detailed_parentmap_for(atlas), avail)

        voting_annotator(
            meta, con_level, classes,
            parent_field=f"{atlas}.Simplified.Celltype",
            parent_to_subset=parents,
            conf_threshold=conf_threshold,
        )

        meta.obs[f"{atlas}.Detailed.Celltype"] = meta.obs[f"{con_level}.Celltype"].astype(object)
        meta.obs[f"{atlas}.Detailed.Celltype.TopScore"] = meta.obs[f"{con_level}.Celltype.TopScore"].astype(float)
        meta.obs[f"{atlas}.Detailed.Celltype.LowConf"]  = meta.obs[f"{con_level}.Celltype.LowConf"].astype(bool)

        _write_obs_to_row_attrs(
            sample, meta,
            [c for c in meta.obs.columns if c.startswith(f"{con_level}.") and c.endswith(".predscore")]
            + [f"{atlas}.Detailed.Celltype", f"{atlas}.Detailed.Celltype.TopScore", f"{atlas}.Detailed.Celltype.LowConf"]
        )
        return sample

    adata = adata_or_sample
    if f"{atlas}.Simplified.Celltype" not in adata.obs:
        Atlas_Simplified_Annotation(adata, atlas, conf_threshold=0.75)

    avail = set(adata.obs.columns)
    classes = _runtime_class_map(_build_detailed_classmap_for(atlas), avail)
    parents = _runtime_parent_subset(_build_detailed_parentmap_for(atlas), avail)

    voting_annotator(
        adata, con_level, classes,
        parent_field=f"{atlas}.Simplified.Celltype",
        parent_to_subset=parents,
        conf_threshold=conf_threshold,
    )

    adata.obs[f"{atlas}.Detailed.Celltype"] = adata.obs[f"{con_level}.Celltype"].astype(object)
    adata.obs[f"{atlas}.Detailed.Celltype.TopScore"] = adata.obs[f"{con_level}.Celltype.TopScore"].astype(float)
    adata.obs[f"{atlas}.Detailed.Celltype.LowConf"]  = adata.obs[f"{con_level}.Celltype.LowConf"].astype(bool)
    return adata


_ATLAS_RE = re.compile(r"^(?P<atlas>[^.]+)\.(?P<level>Broad|Simplified|Detailed)\.(?P<class>[^.]+)\.predscore$")


def add_Averaged_tracks(
    obj: Union["AnnData", object],
    atlases: Sequence[str] = ("Hao", "Zhang", "Triana", "Luecken"),
    *,
    levels: Sequence[str] = ("Broad", "Simplified", "Detailed"),
    weights: Optional[Mapping[str, float]] = None,
    out_prefix: str = "Averaged.",
    write_atlas_name: bool = True,
    atlas_name_value: str = "avg",
    apply_exclusions: bool = True,   # NEW
) -> None:
    """Average per-atlas predscores into Averaged.* tracks (now respecting EXCLUDE_ATLAS)."""
    is_anndata = isinstance(obj, AnnData)
    is_sample = _is_mosaic_sample(obj)
    if not (is_anndata or is_sample):
        raise TypeError("add_Averaged_tracks expects an AnnData or a missionbio.mosaic.sample.Sample")

    if is_anndata:
        pending: dict[str, np.ndarray] = {}
        index = obj.obs.index
        def has_key(k: str) -> bool: return (k in obj.obs.columns) or (k in pending)
        def get_vec(k: str) -> np.ndarray: return pending.get(k, obj.obs[k].to_numpy())
        def set_vec(k: str, v: np.ndarray): pending[k] = np.asarray(v).ravel()
        def list_keys() -> List[str]:
            base = [c for c in obj.obs.columns if not str(c).startswith("Atlas.")]
            extra = [c for c in pending.keys() if not str(c).startswith("Atlas.")]
            return list(set(base + extra))
    else:
        df_scaled = obj.protein.get_attribute("Scaled_reads", constraint="row+col")
        if not isinstance(df_scaled, pd.DataFrame):
            df_scaled = pd.DataFrame(df_scaled)
        index = df_scaled.index.astype(str)
        def has_key(k: str) -> bool: return k in obj.protein.row_attrs
        def get_vec(k: str) -> np.ndarray: return np.asarray(obj.protein.row_attrs[k]).reshape(-1)
        def set_vec(k: str, v: np.ndarray): obj.protein.row_attrs[k] = np.asarray(v).ravel()
        def list_keys() -> List[str]: return [c for c in obj.protein.row_attrs.keys() if not str(c).startswith("Atlas.")]

    n = len(index)
    cols = list_keys()
    by_level_class: Dict[tuple, Dict[str, str]] = {}
    for c in cols:
        m = _ATLAS_RE.match(str(c))
        if not m:
            continue
        atlas = m.group("atlas")
        level = m.group("level")
        klass = m.group("class")
        if atlas in atlases and level in levels:
            by_level_class.setdefault((level, klass), {})[atlas] = c

    if weights is None:
        weights = {}
    else:
        weights = {k: float(v) for k, v in weights.items() if k in atlases}

    for (level, klass), atlas_cols in by_level_class.items():
        # apply exclusions here
        allowed = _filter_atlases_for_label_local(atlases, level, klass, apply_exclusions=apply_exclusions)
        present = [a for a in allowed if a in atlas_cols]
        if not present:
            continue

        mats = []
        for a in present:
            key = atlas_cols[a]
            if not has_key(key):
                continue
            v = get_vec(key)
            if v.shape[0] != n:
                print(f"[add_Averaged_tracks] Skipping {key}: length {v.shape[0]} != {n}")
                continue
            mats.append(v.reshape(1, -1))

        if not mats:
            continue
        M = np.vstack(mats)

        if weights:
            w = np.array([weights.get(a, 0.0) for a in present], dtype=float)
            if w.sum() <= 0:
                w = np.ones(len(present), dtype=float)
        else:
            w = np.ones(len(present), dtype=float)
        w = w / w.sum()

        avg = (w[:, None] * M).sum(axis=0)

        out_col = f"{out_prefix}{level}.{klass}.predscore"
        set_vec(out_col, avg)

        if write_atlas_name:
            atlas_col = f"{out_prefix}{level}.{klass}.atlas"
            set_vec(atlas_col, np.array([atlas_name_value] * n, dtype=object))

    for level in levels:
        patt = re.compile(rf"^{re.escape(out_prefix)}{level}\.(.+)\.predscore$")
        class_names: List[str] = []
        rows: List[np.ndarray] = []

        if is_anndata:
            keys_now = [c for c in list(set(obj.obs.columns).union(pending.keys())) if not str(c).startswith("Atlas.")]
        else:
            keys_now = [c for c in obj.protein.row_attrs.keys() if not str(c).startswith("Atlas.")]

        for k in keys_now:
            m = patt.match(str(k))
            if not m:
                continue
            klass = m.group(1)
            v = get_vec(k)
            if v.shape[0] != n:
                continue
            class_names.append(klass)
            rows.append(v.reshape(1, -1))

        if not rows:
            continue

        mat = np.vstack(rows)
        argmax = mat.argmax(axis=0)
        pred = np.array(class_names, dtype=object)[argmax]
        conf = mat.max(axis=0)

        set_vec(f"{out_prefix}{level}.pred", pred)
        set_vec(f"{out_prefix}{level}.conf", conf)

    if is_anndata and pending:
        _join_obs_cols(obj, pending)


_LEGACY_ATLAS_COL = re.compile(r"^Atlas\.([^.]+)\.(Broad|Simplified|Detailed)\.([^.]+)\.predscore$")
_UNWEIGHTED_AVG   = re.compile(r"^Averaged\.Unweighted\.")


def _coerce_legacy_atlas_prefix(obj: Union[AnnData, Any]) -> None:
    """
    If columns exist as 'Atlas.<name>.<level>.<class>.predscore', copy them to
    '<name>.<level>.<class>.predscore' when the new name is missing, then drop the 'Atlas.*' ones.
    """
    if isinstance(obj, AnnData):
        keys = list(obj.obs.columns)
        for k in keys:
            m = _LEGACY_ATLAS_COL.match(str(k))
            if not m:
                continue
            atlas, level, klass = m.group(1), m.group(2), m.group(3)
            newk = f"{atlas}.{level}.{klass}.predscore"
            if newk not in obj.obs.columns:
                obj.obs[newk] = obj.obs[k].to_numpy()
        drop = [c for c in obj.obs.columns if str(c).startswith("Atlas.")]
        if drop:
            obj.obs.drop(columns=drop, inplace=True, errors="ignore")
    elif _is_mosaic_sample(obj):
        keys = list(obj.protein.row_attrs.keys())
        for k in keys:
            m = _LEGACY_ATLAS_COL.match(str(k))
            if not m:
                continue
            atlas, level, klass = m.group(1), m.group(2), m.group(3)
            newk = f"{atlas}.{level}.{klass}.predscore"
            if newk not in obj.protein.row_attrs:
                obj.protein.row_attrs[newk] = np.asarray(obj.protein.row_attrs[k])
        for k in [c for c in list(obj.protein.row_attrs.keys()) if str(c).startswith("Atlas.")]:
            try:
                del obj.protein.row_attrs[k]
            except Exception:
                pass


def _drop_unweighted_averaged(obj: Union[AnnData, Any]) -> None:
    """Remove all 'Averaged.Unweighted.*' tracks."""
    if isinstance(obj, AnnData):
        drop = [c for c in obj.obs.columns if _UNWEIGHTED_AVG.match(str(c))]
        if drop:
            obj.obs.drop(columns=drop, inplace=True, errors="ignore")
    elif _is_mosaic_sample(obj):
        for k in [c for c in list(obj.protein.row_attrs.keys()) if _UNWEIGHTED_AVG.match(str(c))]:
            try:
                del obj.protein.row_attrs[k]
            except Exception:
                pass


def _ensure_averaged_level_preds(obj, levels=("Broad", "Simplified", "Detailed")) -> None:
    """
    Create Averaged.<level>.pred and Averaged.<level>.conf from existing
    Averaged.<level>.*.predscore columns.
    """
    is_adata = isinstance(obj, AnnData)
    get_all_keys = (lambda: list(obj.obs.columns)) if is_adata else (lambda: list(obj.protein.row_attrs.keys()))
    get_vec = (lambda k: obj.obs[k].to_numpy()) if is_adata else (lambda k: np.asarray(obj.protein.row_attrs[k]))
    set_vec = (lambda k, v: obj.obs.__setitem__(k, v)) if is_adata else (lambda k, v: obj.protein.row_attrs.__setitem__(k, np.asarray(v)))

    keys = get_all_keys()
    for level in levels:
        patt = re.compile(rf"^Averaged\.{level}\.([^.]+)\.predscore$")
        cls_names, mats = [], []
        for k in keys:
            m = patt.match(str(k))
            if not m:
                continue
            cls_names.append(m.group(1))
            mats.append(get_vec(k).reshape(-1, 1))
        if not mats:
            continue
        M = np.hstack(mats)
        argmax = M.argmax(axis=1)
        pred = np.array(cls_names, dtype=object)[argmax]
        conf = M.max(axis=1)
        set_vec(f"Averaged.{level}.pred", pred)
        set_vec(f"Averaged.{level}.conf", conf)


_ATLAS_SCORE_RE = re.compile(r"^[^.]+\.(Broad|Simplified|Detailed)\.[^.]+\.predscore$")
_AVG_SCORE_RE   = re.compile(r"^Averaged\.(Broad|Simplified|Detailed)\.[^.]+\.predscore$")


def annotate_data(
    obj,
    models_path: Optional[Union[str, Path]] = None,
    data_path: Optional[Union[str, Path]] = None,
    *,
    atlases: Sequence[str] = ("Hao", "Zhang", "Triana", "Luecken"),
):
    """
    Ensure predictions exist and normalized naming is used, then compute:
      - Averaged.<level>.pred/conf and constrained Celltype fields
      - Per-atlas hierarchy-constrained <Atlas>.<level>.Celltype (when per-atlas predscores exist)

    Also removes legacy 'Atlas.*' and 'Averaged.Unweighted.*' tracks.
    """
    if isinstance(obj, AnnData):
        keys = list(obj.obs.columns)
    elif _is_mosaic_sample(obj):
        keys = list(obj.protein.row_attrs.keys())
    else:
        raise TypeError("annotate_data expects an AnnData or a missionbio.mosaic.sample.Sample")

    has_atlas_scores = any(_ATLAS_SCORE_RE.match(str(k)) for k in keys)
    has_avg_scores   = any(_AVG_SCORE_RE.match(str(k)) for k in keys)

    if not (has_atlas_scores or has_avg_scores):
        print("[annotate_data] No prediction tracks detected. Running generate_predictions...")
        from .core import get_default_models_path, get_default_data_path, ensure_models_available
        try:
            ensure_models_available()
        except Exception as e:
            print(f"[annotate_data] Warning: ensure_models_available failed: {e}")

        if models_path is None:
            models_path = str(get_default_models_path())
            print(f"[annotate_data] Using default models path: {models_path}")
        if data_path is None:
            data_path = str(get_default_data_path())
            print(f"[annotate_data] Using default data path: {data_path}")

        obj = generate_predictions(obj, models_path=models_path, data_path=data_path)

    _coerce_legacy_atlas_prefix(obj)
    _drop_unweighted_averaged(obj)

    now_keys = list(obj.obs.columns) if isinstance(obj, AnnData) else list(obj.protein.row_attrs.keys())
    if not any(_AVG_SCORE_RE.match(str(k)) for k in now_keys):
        print("[annotate_data] Averaged.* predscores missing; creating via add_Averaged_tracks...")
        add_Averaged_tracks(obj, atlases=atlases, apply_exclusions=True)  # pass flag


    _ensure_averaged_level_preds(obj, levels=("Broad", "Simplified", "Detailed"))

    Broad_Annotation(obj)
    Simplified_Annotation(obj)
    Detailed_Annotation(obj)

    present_keys = set(list(obj.obs.columns)) if isinstance(obj, AnnData) else set(list(obj.protein.row_attrs.keys()))
    for atlas in atlases:
        if f"{atlas}.Broad.Mature.predscore" in present_keys and f"{atlas}.Broad.Immature.predscore" in present_keys:
            Atlas_Broad_Annotation(obj, atlas, conf_threshold=0.75)
        if f"{atlas}.Simplified.NK.predscore" in present_keys or any(k.startswith(f"{atlas}.Simplified.") and k.endswith(".predscore") for k in present_keys):
            Atlas_Simplified_Annotation(obj, atlas, conf_threshold=0.75)
        if f"{atlas}.Detailed.CD14_Mono.predscore" in present_keys or any(k.startswith(f"{atlas}.Detailed.") and k.endswith(".predscore") for k in present_keys):
            Atlas_Detailed_Annotation(obj, atlas, conf_threshold=0.60)

    return obj


def _get_matrix_for_embedding(
    obj: Union["AnnData", Any],
    embedding_key: str = "X_umap",
    n_pca: int = 20,
) -> np.ndarray:
    """Return features for QC metrics: UMAP if present, else PCA of scaled data."""
    if isinstance(obj, AnnData):
        if hasattr(obj, "obsm") and embedding_key in obj.obsm:
            return np.asarray(obj.obsm[embedding_key])
        X = obj.X.A if hasattr(obj.X, "A") else (obj.X.toarray() if hasattr(obj.X, "toarray") else obj.X)
        X = np.asarray(X, dtype=float)
        n_comp = min(n_pca, X.shape[1]) if X.ndim == 2 and X.shape[1] > 1 else 1
        return X.reshape(-1, 1) if n_comp <= 1 else PCA(n_components=n_comp, random_state=0).fit_transform(X)
    elif _is_mosaic_sample(obj):
        if "umap" in obj.protein.row_attrs:
            U = np.asarray(obj.protein.row_attrs["umap"])
            return U[:, :2] if U.ndim == 2 and U.shape[1] >= 2 else U.reshape(-1, 1)
        df = obj.protein.get_attribute("Scaled_reads", constraint="row+col")
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)
        X = df.values.astype(float)
        n_comp = min(n_pca, X.shape[1]) if X.shape[1] > 1 else 1
        return X.reshape(-1, 1) if n_comp <= 1 else PCA(n_components=n_comp, random_state=0).fit_transform(X)
    else:
        raise TypeError("Expected AnnData or missionbio.mosaic.sample.Sample")


def clear_annotation(obj: Union[AnnData, Any]) -> Union[AnnData, Any]:
    """Remove annotation-related columns and extraneous prob matrices."""
    patterns = [
        re.compile(r"\.pred$", re.IGNORECASE),
        re.compile(r"\.predscore$", re.IGNORECASE),
        re.compile(r"\.score$", re.IGNORECASE),
        re.compile(r"\.triple$", re.IGNORECASE),
        re.compile(r"\.lowconf$", re.IGNORECASE),
        re.compile(r"\.atlas$", re.IGNORECASE),
        re.compile(r"\.conf$", re.IGNORECASE),
        re.compile(r"__centroid_dist$", re.IGNORECASE),
        re.compile(r"__centroid_robust_z$", re.IGNORECASE),
        re.compile(r"_pos$", re.IGNORECASE),
        re.compile(r"_neg$", re.IGNORECASE),
        re.compile(r"_signature$", re.IGNORECASE),
    ]

    def matches(k: str) -> bool:
        return any(p.search(k) for p in patterns)

    keep_embeddings = {"X_pca", "X_harmony", "X_umap"}

    if isinstance(obj, AnnData):
        drop_cols = [c for c in obj.obs.columns if matches(c)]
        if drop_cols:
            obj.obs.drop(columns=drop_cols, inplace=True)

        drop_obsm = [k for k in list(obj.obsm.keys())
                     if (matches(k) or k.endswith(".probs")) and k not in keep_embeddings]
        for k in drop_obsm:
            del obj.obsm[k]

    elif _is_mosaic_sample(obj):
        drop_attrs = [k for k in list(obj.protein.row_attrs.keys()) if matches(k)]
        for k in drop_attrs:
            del obj.protein.row_attrs[k]
    else:
        raise TypeError("Expected AnnData or missionbio.mosaic.sample.Sample")

    return obj


def mark_mixed_clusters(
    obj: Union["AnnData", any],
    label_col: str,
    *,
    cluster_col: Optional[str] = None,
    min_frequency_threshold: float = 0.3,
    mixed_label: str = "Mixed",
) -> Union["AnnData", any]:
    """Relabel clusters with no dominant label as 'Mixed'."""
    is_anndata = isinstance(obj, AnnData)
    is_sample  = _is_mosaic_sample(obj)

    if not (is_anndata or is_sample):
        raise TypeError("Expected AnnData or missionbio.mosaic.sample.Sample")

    if is_anndata:
        if cluster_col is None:
            for cand in ("leiden", "louvain", "clusters", "cluster"):
                if cand in getattr(obj, "obs", pd.DataFrame()).columns:
                    cluster_col = cand
                    break
        if cluster_col is None or cluster_col not in obj.obs.columns:
            raise KeyError("Cluster column not found. Provide cluster_col (e.g. 'leiden').")
        clusters = obj.obs[cluster_col].astype(str).to_numpy()
    else:
        clusters = np.asarray(obj.protein.get_labels()).astype(str)

    if is_anndata:
        if label_col not in obj.obs.columns:
            raise KeyError(f"'{label_col}' not in adata.obs")
        labels = obj.obs[label_col]
        was_cat = pd.api.types.is_categorical_dtype(labels)
        labels_s = labels.astype(str)
    else:
        if label_col not in obj.protein.row_attrs:
            raise KeyError(f"'{label_col}' not in Sample.protein.row_attrs")
        labels_arr = np.asarray(obj.protein.row_attrs[label_col])
        labels_s = pd.Series(labels_arr.astype(str))
        was_cat = False

    df = pd.DataFrame({"cluster": clusters, "label": labels_s.values})
    counts = df.groupby(["cluster", "label"]).size().rename("count").reset_index()
    totals = df.groupby("cluster").size().rename("total").reset_index()
    freq = counts.merge(totals, on="cluster", how="left")
    freq["frequency"] = freq["count"] / freq["total"].replace(0, np.nan)

    top = freq.loc[freq.groupby("cluster")["frequency"].idxmax(), ["cluster", "label", "frequency"]]
    mixed_clusters = top.loc[top["frequency"] < float(min_frequency_threshold), "cluster"].astype(str).tolist()

    if not mixed_clusters:
        return obj

    mixed_mask = pd.Series(clusters).isin(mixed_clusters).to_numpy()
    updated = labels_s.where(~mixed_mask, other=mixed_label)

    if is_anndata:
        if was_cat:
            cats = pd.Index(labels.cat.categories).astype(str)
            if mixed_label not in cats:
                cats = cats.append(pd.Index([mixed_label]))
            obj.obs[label_col] = pd.Categorical(updated, categories=cats)
        else:
            obj.obs[label_col] = updated
        return obj

    obj.protein.row_attrs[label_col] = np.asarray(updated.values, dtype=object)
    return obj


def refine_labels_by_knn_consensus(
    obj: Union["AnnData", any],
    label_col: str = "Averaged.Detailed.Celltype",
    embedding_key: str = "X_umap",
    *,
    k_neighbors: int = 30,
    min_neighbor_frac: float = 0.70,
    max_self_frac: float = 0.30,
    outlier_z: float = 2.0,
    min_label_size: int = 30,
    per_label_cap: float = 0.10,
    global_cap: float = 0.05,
    out_col: Optional[str] = None,
    debug_cols: bool = True,
):
    """Safer relabeling via local KNN consensus + outlier checks."""
    is_anndata = isinstance(obj, AnnData)
    is_sample  = _is_mosaic_sample(obj)
    if not (is_anndata or is_sample):
        raise TypeError("Expected AnnData or missionbio.mosaic.sample.Sample")

    out_col = out_col or f"{label_col}_refined_consensus"

    if is_anndata:
        if embedding_key not in obj.obsm:
            raise KeyError(f"Embedding '{embedding_key}' not found in adata.obsm")
        X = np.asarray(obj.obsm[embedding_key])
        if label_col not in obj.obs.columns:
            raise KeyError(f"'{label_col}' not in adata.obs")
        labels = obj.obs[label_col]
        was_cat = pd.api.types.is_categorical_dtype(labels)
        labels_s = labels.astype(str)
        idx_like = obj.obs_names
    else:
        if "umap" not in obj.protein.row_attrs:
            raise KeyError("Expected 'umap' in Sample.protein.row_attrs")
        X = np.asarray(obj.protein.row_attrs["umap"])
        if X.ndim != 2 or X.shape[1] < 2:
            raise ValueError("'umap' must be 2D with ≥2 columns")
        if label_col not in obj.protein.row_attrs:
            raise KeyError(f"'{label_col}' not in Sample.protein.row_attrs")
        labels_s = pd.Series(np.asarray(obj.protein.row_attrs[label_col]).astype(str))
        was_cat = False
        idx_like = pd.Index(range(X.shape[0]))

    n = X.shape[0]
    k = max(2, min(k_neighbors, n - 1))

    counts = labels_s.value_counts()
    frequent = set(counts[counts >= min_label_size].index)

    z = np.full(n, np.nan, dtype=float)
    for lab, idx_lab in labels_s.groupby(labels_s).groups.items():
        idx_arr = np.fromiter(idx_lab, dtype=int)
        if idx_arr.size < 3:
            continue
        mu = X[idx_arr].mean(axis=0)
        d = np.linalg.norm(X[idx_arr] - mu, axis=1)
        med = np.median(d)
        mad = np.median(np.abs(d - med)) + 1e-12
        z_vals = (d - med) / (1.4826 * mad)
        z[idx_arr] = z_vals

    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    nn.fit(X)
    dists, neigh = nn.kneighbors(X, return_distance=True)
    neigh = neigh[:, 1:]
    neigh = neigh[:, :k]

    neigh_labels = labels_s.iloc[neigh.reshape(-1)].to_numpy().reshape(n, -1)
    top_label = []
    top_frac = np.zeros(n, dtype=float)
    self_frac = np.zeros(n, dtype=float)

    for i in range(n):
        vals, ct = np.unique(neigh_labels[i], return_counts=True)
        order = np.argsort(-ct)
        vals, ct = vals[order], ct[order]
        top_label.append(vals[0])
        top_frac[i] = ct[0] / float(neigh_labels.shape[1])
        cur = labels_s.iloc[i]
        self_ct = ct[vals == cur].sum() if np.any(vals == cur) else 0
        self_frac[i] = self_ct / float(neigh_labels.shape[1])

    top_label = np.asarray(top_label, dtype=object)
    margin = top_frac - self_frac

    cur_labels = labels_s.to_numpy()
    candidate = top_label
    flip_mask = (
        (candidate != cur_labels) &
        (top_frac >= float(min_neighbor_frac)) &
        ((self_frac <= float(max_self_frac)) | (z > float(outlier_z))) &
        (np.isin(candidate, list(frequent)))
    )

    to_flip_idx = np.where(flip_mask)[0]
    if to_flip_idx.size:
        order = np.argsort(-margin[to_flip_idx])
        to_flip_idx = to_flip_idx[order]

        final_idx = []
        per_src_used: Dict[str, int] = {}
        per_src_cap: Dict[str, int] = {lab: int(np.floor(c * float(per_label_cap))) for lab, c in counts.items()}

        for i in to_flip_idx:
            src = cur_labels[i]
            if per_src_cap.get(src, 0) <= 0:
                continue
            used = per_src_used.get(src, 0)
            if used < per_src_cap[src]:
                per_src_used[src] = used + 1
                final_idx.append(i)

        gcap = int(np.floor(n * float(global_cap)))
        if len(final_idx) > gcap:
            final_idx = final_idx[:gcap]
    else:
        final_idx = []

    new_labels = labels_s.copy()
    if final_idx:
        new_labels.iloc[final_idx] = candidate[final_idx]

    if is_anndata:
        if was_cat:
            cats = pd.Index(labels.cat.categories).astype(str)
            extra = np.setdiff1d(np.unique(new_labels), cats)
            cats = cats.append(pd.Index(extra))
            obj.obs[out_col] = pd.Categorical(new_labels, categories=cats)
        else:
            obj.obs[out_col] = new_labels
        if debug_cols:
            obj.obs[f"{out_col}__knn_top_frac"] = top_frac
            obj.obs[f"{out_col}__knn_self_frac"] = self_frac
            obj.obs[f"{out_col}__centroid_robust_z"] = z
            obj.obs[f"{out_col}__flip"] = False
            obj.obs.loc[idx_like[np.asarray(final_idx, dtype=int)], f"{out_col}__flip"] = True

        per_label_changes = {}
        if final_idx:
            src_series = pd.Series(cur_labels[np.asarray(final_idx, dtype=int)])
            per_label_changes = src_series.value_counts().astype(int).to_dict()
        return int(len(final_idx)), obj.obs[out_col], obj, per_label_changes

    obj.protein.row_attrs[out_col] = np.asarray(new_labels.values, dtype=object)
    if debug_cols:
        obj.protein.row_attrs[f"{out_col}__knn_top_frac"] = top_frac.astype(float)
        obj.protein.row_attrs[f"{out_col}__knn_self_frac"] = self_frac.astype(float)
        obj.protein.row_attrs[f"{out_col}__centroid_robust_z"] = z.astype(float)
        flip_vec = np.zeros(n, dtype=bool)
        if final_idx:
            flip_vec[np.asarray(final_idx, dtype=int)] = True
        obj.protein.row_attrs[f"{out_col}__flip"] = flip_vec
    return obj


def score_mixed_clusters(
    obj: Union["AnnData", Any],
    clusters: Union[str, np.ndarray, pd.Series, list],
    labels: Union[str, np.ndarray, pd.Series, list],
    *,
    embedding_key: str = "X_umap",
    n_pca: int = 20,
    weights: Dict[str, float] = None,
    min_cells_for_silhouette: int = 2,
) -> pd.DataFrame:
    """Quantify cluster mixing via label entropy and embedding cohesion."""

    def _ensure_1d(arr_or_name) -> np.ndarray:
        if isinstance(arr_or_name, str):
            if isinstance(obj, AnnData):
                if arr_or_name not in obj.obs.columns:
                    raise KeyError(f"'{arr_or_name}' not in adata.obs")
                return obj.obs[arr_or_name].to_numpy()
            elif _is_mosaic_sample(obj):
                if arr_or_name not in obj.protein.row_attrs:
                    raise KeyError(f"'{arr_or_name}' not in sample.protein.row_attrs")
                return np.asarray(obj.protein.row_attrs[arr_or_name])
        return np.asarray(arr_or_name)

    cvec = _ensure_1d(clusters).astype(str)
    lvec = _ensure_1d(labels).astype(str)
    if cvec.shape[0] != lvec.shape[0]:
        raise ValueError("clusters and labels must have the same length")

    Z = _get_matrix_for_embedding(obj, embedding_key=embedding_key, n_pca=n_pca)
    if Z.shape[0] != cvec.shape[0]:
        raise ValueError("Feature/embedding length does not match clusters/labels length")

    sil = np.full(Z.shape[0], np.nan, dtype=float)
    uniq_clusters = np.unique(cvec)
    if uniq_clusters.size >= 2 and all((cvec == u).sum() >= min_cells_for_silhouette for u in uniq_clusters):
        try:
            sil = silhouette_samples(Z, cvec, metric="euclidean")
        except Exception:
            pass

    rows = []
    for cl in uniq_clusters:
        idx = (cvec == cl)
        n = int(idx.sum())
        labs = lvec[idx]
        _, counts = np.unique(labs, return_counts=True)
        p = counts / counts.sum()

        majority_frac = float(p.max())
        if len(p) > 1:
            H = -np.sum(p * np.log(p + 1e-12))
            Hmax = np.log(len(p))
            entropy_norm = float(H / Hmax)
        else:
            entropy_norm = 0.0

        s = np.nanmean(sil[idx]) if np.any(idx) else np.nan
        mix_from_entropy = entropy_norm
        mix_from_sil = 0.5 if np.isnan(s) else (1.0 - s) * 0.5

        rows.append({
            "cluster": cl,
            "n_cells": n,
            "majority_frac": majority_frac,
            "entropy_norm": entropy_norm,
            "silhouette_mean": s,
            "mix_from_entropy": mix_from_entropy,
            "mix_from_sil": mix_from_sil,
        })

    df = pd.DataFrame(rows).set_index("cluster")

    if weights is None:
        weights = {"entropy": 0.6, "silhouette": 0.4}
    w_e = float(weights.get("entropy", 0.6))
    w_s = float(weights.get("silhouette", 0.4))
    w_sum = w_e + w_s if (w_e + w_s) > 0 else 1.0
    w_e, w_s = w_e / w_sum, w_s / w_sum

    df["mixed_likelihood"] = w_e * df["mix_from_entropy"].values + w_s * df["mix_from_sil"].values
    return df.sort_values("mixed_likelihood", ascending=False)


def mark_small_clusters(
    obj: Union["AnnData", Any],
    label_col: str,
    *,
    min_cells: int = 3,
    small_label: str = "Small",
):
    """Relabel clusters with < min_cells counts as 'Small'."""
    is_anndata = isinstance(obj, AnnData)
    is_sample = _is_mosaic_sample(obj)

    if not (is_anndata or is_sample):
        raise TypeError("Expected AnnData or missionbio.mosaic.sample.Sample")

    if is_anndata:
        if label_col not in obj.obs.columns:
            raise KeyError(f"'{label_col}' not found in adata.obs")
        labels = obj.obs[label_col]
        was_categorical = pd.api.types.is_categorical_dtype(labels)
        labels_s = labels.astype(str)
    else:
        if label_col not in obj.protein.row_attrs:
            raise KeyError(f"'{label_col}' not found in Sample.protein.row_attrs")
        arr = np.asarray(obj.protein.row_attrs[label_col])
        labels_s = pd.Series(arr.astype(str))
        was_categorical = False

    counts = labels_s.value_counts()
    small = counts[counts < min_cells].index
    updated = labels_s.replace(dict.fromkeys(small, small_label))

    if is_anndata:
        if was_categorical:
            cats = pd.Index(labels.cat.categories).astype(str)
            if small_label not in cats:
                cats = cats.append(pd.Index([small_label]))
            obj.obs[label_col] = pd.Categorical(updated, categories=cats)
        else:
            obj.obs[label_col] = updated
        return obj

    obj.protein.row_attrs[label_col] = np.asarray(updated.values, dtype=object)
    return obj
