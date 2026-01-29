# -*- coding: utf-8 -*-
"""
Cell type annotation workflows (Averaged + per-atlas outputs) with hierarchy constraints.

What this module produces
-------------------------
Averaged winners (pred/conf) and constrained labels:
    Averaged.Broad.Celltype (+ TopScore/LowConf)
    Averaged.Simplified.Celltype (+ TopScore/LowConf)
    Averaged.Detailed.Celltype (+ TopScore/LowConf)

Per-atlas hierarchy-constrained labels (no 'Atlas.' prefix):
    <Atlas>.Broad.Celltype (+ TopScore/LowConf)
    <Atlas>.Simplified.Celltype (+ TopScore/LowConf)
    <Atlas>.Detailed.Celltype (+ TopScore/LowConf)

Constrained probability tracks:
    Averaged.Simplified.<cls>.predscore.constrained
    Averaged.Detailed.<cls>.predscore.constrained
    <Atlas>.Simplified.<cls>.predscore.constrained
    <Atlas>.Detailed.<cls>.predscore.constrained

Housekeeping:
    - Legacy 'Atlas.<name>.*' converted to '<name>.*' (if needed) and then removed.
    - 'Averaged.Unweighted.*' tracks removed.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import anndata as ad
from anndata import AnnData

from .constants import (
    SIMPLIFIED_CLASSES,
    DETAILED_CLASSES,
    SIMPLIFIED_PARENT_MAP,
    DETAILED_PARENT_MAP,
)
from .prediction import EXCLUDE_ATLAS


# -----------------------------------------------------------------------------
# Core helpers
# -----------------------------------------------------------------------------

def _filter_atlases_for_label_local(
    atlases: Sequence[str],
    depth: str,
    label: str,
    *,
    apply_exclusions: bool = True,
) -> List[str]:
    if not apply_exclusions:
        return list(atlases)
    banned = EXCLUDE_ATLAS.get(depth, {}).get(label, set())
    return [a for a in atlases if a not in banned]

def _is_mosaic_sample(x: Any) -> bool:
    """
    MissionBio Sample-ish object detector.
    We require row_attrs + get_attribute (used throughout for feature access).
    """
    return hasattr(x, "protein") and hasattr(x.protein, "row_attrs") and hasattr(x.protein, "get_attribute")

def _class_from_key(k: str) -> str:
    # expects something like: "<prefix>.<Class>.predscore"  -> returns "<Class>"
    parts = str(k).split(".")
    return parts[-2] if len(parts) >= 3 and parts[-1] == "predscore" else ""


def _make_meta_from_row_attrs(sample: Any, needed_cols: Sequence[str]) -> AnnData:
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


def _write_obs_to_row_attrs(sample: Any, adata_meta: AnnData, cols: Sequence[str]) -> None:
    for c in cols:
        if c not in adata_meta.obs.columns:
            continue
        vals = adata_meta.obs[c]
        arr = (
            vals.astype(str).to_numpy()
            if pd.api.types.is_categorical_dtype(vals)
            else np.asarray(vals.to_numpy())
        )
        sample.protein.row_attrs[c] = arr


def _runtime_class_map(class_map: Dict[str, List[str]], cols_available: set) -> Dict[str, List[str]]:
    out = {lbl: [c for c in cols if c in cols_available] for lbl, cols in class_map.items()}
    out = {lbl: cols for lbl, cols in out.items() if cols}
    if not any(out.values()):
        raise KeyError("No matching predscore columns found for any class in the provided class_map.")
    return out


def _runtime_parent_subset(parent_map: Dict[str, List[str]], cols_available: set) -> Dict[str, List[str]]:
    return {p: [c for c in cols if c in cols_available] for p, cols in parent_map.items()}


# -----------------------------------------------------------------------------
# Voting annotator (core engine)
# -----------------------------------------------------------------------------

def voting_annotator(
    obj: Union[AnnData, Any],
    level_name: str,
    class_to_sources: Dict[str, List[str]],
    parent_field: Optional[str] = None,
    parent_to_subset: Optional[Dict[str, List[str]]] = None,
    conf_threshold: float = 0.75,
    *,
    normalize: bool = False,
) -> None:
    """
    Aggregate sources → per-class predscores, optional hierarchy masking, optional renormalization,
    and final labels.

    Writes:
        <level_name>.<Class>.predscore (one per Class)
        <level_name>.Celltype
        <level_name>.Celltype.TopScore
        <level_name>.Celltype.LowConf

    Important behavior:
        - If `normalize=True`, this performs *probability renormalization* (divide by row-sum of
          allowed classes), NOT a softmax. This prevents "forced 1/0" collapse when the allowed
          set effectively has only one available sibling (common when some classes are missing).
        - Additionally, if a row has <2 allowed classes, we do NOT renormalize it (we keep the
          masked probabilities as-is) to avoid turning any nonzero into 1.0.
    """
    is_anndata = isinstance(obj, AnnData)
    is_sample = _is_mosaic_sample(obj)
    if not (is_anndata or is_sample):
        raise TypeError("voting_annotator expects an AnnData or a missionbio.mosaic.sample.Sample")

    if is_anndata:
        n = obj.n_obs

        def has_key(k: str) -> bool:
            return k in obj.obs.columns

        def get_vec(k: str) -> np.ndarray:
            return obj.obs[k].to_numpy()

        def set_vec(k: str, v: np.ndarray) -> None:
            obj.obs[k] = v

        def list_keys() -> List[str]:
            return list(obj.obs.columns)

    else:
        df_scaled = obj.protein.get_attribute("Scaled_reads", constraint="row+col")
        if not isinstance(df_scaled, pd.DataFrame):
            df_scaled = pd.DataFrame(df_scaled)
        n = len(df_scaled.index)

        def has_key(k: str) -> bool:
            return k in obj.protein.row_attrs

        def get_vec(k: str) -> np.ndarray:
            return np.asarray(obj.protein.row_attrs[k]).reshape(-1)

        def set_vec(k: str, v: np.ndarray) -> None:
            obj.protein.row_attrs[k] = np.asarray(v)

        def list_keys() -> List[str]:
            return list(obj.protein.row_attrs.keys())

    # -------------------------------------------------------------------------
    # Build <level_name>.<Class>.predscore columns by averaging/copying sources
    # -------------------------------------------------------------------------

    # Fast path: every class has exactly one present source column
    all_single = True
    for _, cols in class_to_sources.items():
        present = [c for c in cols if has_key(c)]
        if len(present) != 1:
            all_single = False
            break

    if all_single:
        for out_cls, cols in class_to_sources.items():
            src = [c for c in cols if has_key(c)][0]
            v = get_vec(src)
            if v.shape[0] != n:
                raise ValueError(f"Source column '{src}' length {v.shape[0]} != n_obs {n}")
            set_vec(f"{level_name}.{out_cls}.predscore", v)
    else:
        for out_cls, cols in class_to_sources.items():
            present = [c for c in cols if has_key(c)]
            if not present:
                continue
            mats = []
            for c in present:
                v = get_vec(c).reshape(-1)
                if v.shape[0] != n:
                    continue
                mats.append(v)
            if mats:
                avg = np.mean(np.vstack(mats), axis=0)
                set_vec(f"{level_name}.{out_cls}.predscore", avg)

    # Collect score columns for this level
    score_cols = [k for k in list_keys() if str(k).startswith(f"{level_name}.") and str(k).endswith(".predscore")]
    if not score_cols:
        return

    # Matrix of scores (N, C)
    rows = []
    for k in score_cols:
        v = get_vec(k).reshape(-1)
        if v.shape[0] != n:
            v = np.zeros(n, dtype=float)
        rows.append(v.reshape(1, -1))
    M = np.vstack(rows).T  # (N, C)

    # -------------------------------------------------------------------------
    # Optional hierarchy mask
    # parent_to_subset values can be:
    #   - full predscore column names, e.g. "Averaged.Detailed.X.predscore"
    #   - OR bare class names, e.g. "X"
    # -------------------------------------------------------------------------
    mask = np.ones_like(M, dtype=bool)

    if parent_field and parent_to_subset and (parent_field in set(list_keys())):
        parents = get_vec(parent_field).astype(str)

        def _to_class_set(items: List[str]) -> set:
            out = set()
            for s in items:
                s = str(s)
                # full column name
                if s.endswith(".predscore"):
                    out.add(_class_from_key(s))
                else:
                    # bare class name
                    out.add(s)
            out.discard("")
            return out

        allowed_classes = {p: _to_class_set(cols) for p, cols in parent_to_subset.items()}
        out_class_names = [_class_from_key(c) for c in score_cols]

        for i in range(n):
            p = parents[i]
            if p not in allowed_classes:
                continue
            keep = allowed_classes[p]
            # If keep is empty, do not constrain this row (leave mask all True).
            if not keep:
                continue
            for j, cls in enumerate(out_class_names):
                mask[i, j] = (cls in keep)

    # Apply mask (disallowed -> 0), and clip
    P = np.where(mask, M, 0.0)
    P = np.clip(P, 0.0, 1.0)

    # -------------------------------------------------------------------------
    # Renormalize among allowed classes (probability renormalization, not softmax)
    # Key bugfix:
    #   - If only one class is allowed for a row, do NOT renormalize to 1.0.
    #     Keep the original masked probability magnitude.
    # -------------------------------------------------------------------------
    if normalize:
        allowed_ct = mask.sum(axis=1)
        renorm_rows = allowed_ct >= 2
        if np.any(renorm_rows):
            denom = P[renorm_rows].sum(axis=1, keepdims=True)
            denom[denom == 0.0] = 1.0
            P[renorm_rows] = P[renorm_rows] / denom

    # Write back predscores (now masked + optionally renormalized)
    for j, k in enumerate(score_cols):
        set_vec(k, P[:, j])

    # Winner fields
    winner_idx = np.argmax(P, axis=1)
    winner_scores = P[np.arange(P.shape[0]), winner_idx]
    winner_cols = np.array(score_cols, dtype=object)[winner_idx]
    winners = np.array([_class_from_key(c) for c in winner_cols], dtype=object)

    set_vec(f"{level_name}.Celltype", winners)
    set_vec(f"{level_name}.Celltype.TopScore", winner_scores)
    set_vec(f"{level_name}.Celltype.LowConf", (winner_scores < float(conf_threshold)).astype(bool))

# -----------------------------------------------------------------------------
# Averaged hierarchical annotation (Broad → Simplified → Detailed)
# -----------------------------------------------------------------------------

def Broad_Annotation(adata_or_sample: Union[AnnData, Any], conf_threshold: float = 0.75):
    """Broad (Mature/Immature) from Averaged.Broad.*."""
    level_name = "Averaged.Broad"
    REF = {
        "Mature": [f"{level_name}.Mature.predscore"],
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
            sample,
            meta,
            [f"{level_name}.Celltype", f"{level_name}.Celltype.TopScore", f"{level_name}.Celltype.LowConf"],
        )
        return sample

    adata = adata_or_sample
    for k in (f"{level_name}.Mature.predscore", f"{level_name}.Immature.predscore"):
        if k not in adata.obs:
            raise KeyError(f"[Broad] Missing required column in adata.obs: {k}")
    voting_annotator(adata, level_name, REF, conf_threshold=conf_threshold)
    return adata


def Simplified_Annotation(adata_or_sample: Union[AnnData, Any], conf_threshold: float = 0.75):
    """Constrained Simplified from Averaged.Simplified.* gated by Averaged.Broad.Celltype."""
    level_name = "Averaged.Simplified"

    if _is_mosaic_sample(adata_or_sample):
        sample = adata_or_sample
        if "Averaged.Broad.Celltype" not in sample.protein.row_attrs:
            Broad_Annotation(sample, conf_threshold=conf_threshold)

        meta = _make_meta_from_row_attrs(sample, list(sample.protein.row_attrs.keys()))
        meta.obs["Averaged.Broad.Celltype"] = np.asarray(sample.protein.row_attrs["Averaged.Broad.Celltype"])

        available = set(meta.obs.columns)
        rt_classes = _runtime_class_map(SIMPLIFIED_CLASSES, available)
        rt_parents = _runtime_parent_subset(SIMPLIFIED_PARENT_MAP, available)

        voting_annotator(
            meta,
            level_name,
            rt_classes,
            parent_field="Averaged.Broad.Celltype",
            parent_to_subset=rt_parents,
            conf_threshold=conf_threshold,
            normalize=True,
        )

        # expose constrained predscores as *.predscore.constrained
        for lbl in SIMPLIFIED_CLASSES.keys():
            src = f"{level_name}.{lbl}.predscore"
            if src in meta.obs.columns:
                meta.obs[f"{level_name}.{lbl}.predscore.constrained"] = meta.obs[src].astype(float)

        _write_obs_to_row_attrs(
            sample,
            meta,
            [c for c in meta.obs.columns if c.startswith(f"{level_name}.") and (c.endswith(".predscore") or c.endswith(".predscore.constrained"))]
            + [f"{level_name}.Celltype", f"{level_name}.Celltype.TopScore", f"{level_name}.Celltype.LowConf"],
        )
        return sample

    adata = adata_or_sample
    if "Averaged.Broad.Celltype" not in adata.obs:
        Broad_Annotation(adata, conf_threshold=conf_threshold)

    available = set(adata.obs.columns)
    rt_classes = _runtime_class_map(SIMPLIFIED_CLASSES, available)
    rt_parents = _runtime_parent_subset(SIMPLIFIED_PARENT_MAP, available)

    voting_annotator(
        adata,
        level_name,
        rt_classes,
        parent_field="Averaged.Broad.Celltype",
        parent_to_subset=rt_parents,
        conf_threshold=conf_threshold,
        normalize=True,
    )

    for lbl in SIMPLIFIED_CLASSES.keys():
        src = f"{level_name}.{lbl}.predscore"
        if src in adata.obs.columns:
            adata.obs[f"{level_name}.{lbl}.predscore.constrained"] = adata.obs[src].astype(float)

    return adata


def Detailed_Annotation(adata_or_sample: Union[AnnData, Any], conf_threshold: float = 0.6):
    """Constrained Detailed from Averaged.Detailed.* gated by Averaged.Simplified.Celltype."""
    level_name = "Averaged.Detailed"

    if _is_mosaic_sample(adata_or_sample):
        sample = adata_or_sample
        if "Averaged.Simplified.Celltype" not in sample.protein.row_attrs:
            Simplified_Annotation(sample, conf_threshold=0.75)

        meta = _make_meta_from_row_attrs(sample, list(sample.protein.row_attrs.keys()))
        meta.obs["Averaged.Simplified.Celltype"] = np.asarray(sample.protein.row_attrs["Averaged.Simplified.Celltype"])

        available = set(meta.obs.columns)
        rt_classes = _runtime_class_map(DETAILED_CLASSES, available)
        rt_parents = _runtime_parent_subset(DETAILED_PARENT_MAP, available)

        voting_annotator(
            meta,
            level_name,
            rt_classes,
            parent_field="Averaged.Simplified.Celltype",
            parent_to_subset=rt_parents,
            conf_threshold=conf_threshold,
            normalize=True,
        )

        # expose constrained predscores as *.predscore.constrained
        for lbl in DETAILED_CLASSES.keys():
            src = f"{level_name}.{lbl}.predscore"
            if src in meta.obs.columns:
                meta.obs[f"{level_name}.{lbl}.predscore.constrained"] = meta.obs[src].astype(float)

        _write_obs_to_row_attrs(
            sample,
            meta,
            [c for c in meta.obs.columns if c.startswith(f"{level_name}.") and (c.endswith(".predscore") or c.endswith(".predscore.constrained"))]
            + [f"{level_name}.Celltype", f"{level_name}.Celltype.TopScore", f"{level_name}.Celltype.LowConf"],
        )
        return sample

    adata = adata_or_sample
    if "Averaged.Simplified.Celltype" not in adata.obs:
        Simplified_Annotation(adata, conf_threshold=0.75)

    available = set(adata.obs.columns)
    rt_classes = _runtime_class_map(DETAILED_CLASSES, available)
    rt_parents = _runtime_parent_subset(DETAILED_PARENT_MAP, available)

    voting_annotator(
        adata,
        level_name,
        rt_classes,
        parent_field="Averaged.Simplified.Celltype",
        parent_to_subset=rt_parents,
        conf_threshold=conf_threshold,
        normalize=True,
    )

    for lbl in DETAILED_CLASSES.keys():
        src = f"{level_name}.{lbl}.predscore"
        if src in adata.obs.columns:
            adata.obs[f"{level_name}.{lbl}.predscore.constrained"] = adata.obs[src].astype(float)

    return adata


# -----------------------------------------------------------------------------
# Atlas constrained annotation (Broad → Simplified → Detailed)
# Produces per-atlas *.predscore.constrained tracks
# -----------------------------------------------------------------------------

_SIMPLIFIED_NAMES: List[str] = list(SIMPLIFIED_CLASSES.keys())
_DETAILED_NAMES: List[str] = list(DETAILED_CLASSES.keys())


def _build_simplified_classmap_for(atlas: str) -> Dict[str, List[str]]:
    return {lbl: [f"{atlas}.Simplified.{lbl}.predscore"] for lbl in _SIMPLIFIED_NAMES}


def _build_detailed_classmap_for(atlas: str) -> Dict[str, List[str]]:
    return {lbl: [f"{atlas}.Detailed.{lbl}.predscore"] for lbl in _DETAILED_NAMES}


def _build_simplified_parentmap_for(atlas: str) -> Dict[str, List[str]]:
    return {
        "Immature": [f"{atlas}.Simplified.HSPC.predscore"],
        "Mature": [f"{atlas}.Simplified.{l}.predscore" for l in _SIMPLIFIED_NAMES if l != "HSPC"],
    }


def _build_detailed_parentmap_for(atlas: str) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for parent, allowed in DETAILED_PARENT_MAP.items():
        cls_names: List[str] = []
        for key in allowed:
            parts = str(key).split(".")
            if len(parts) >= 4 and parts[-1] == "predscore":
                cls_names.append(parts[-2])
        out[parent] = [f"{atlas}.Detailed.{c}.predscore" for c in cls_names]
    return out


def Atlas_Broad_Annotation(adata_or_sample: Union[AnnData, Any], atlas: str, conf_threshold: float = 0.75):
    """Broad (Mature/Immature) from <Atlas>.Broad.*."""
    level_name = f"{atlas}.Broad"
    REF = {
        "Mature": [f"{atlas}.Broad.Mature.predscore"],
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
        _write_obs_to_row_attrs(
            sample,
            meta,
            [f"{level_name}.Celltype", f"{level_name}.Celltype.TopScore", f"{level_name}.Celltype.LowConf"],
        )
        return sample

    adata = adata_or_sample
    for k in (f"{atlas}.Broad.Mature.predscore", f"{atlas}.Broad.Immature.predscore"):
        if k not in adata.obs:
            raise KeyError(f"[{level_name}] Missing required column in adata.obs: {k}")
    voting_annotator(adata, level_name, REF, conf_threshold=conf_threshold)
    return adata


def Atlas_Simplified_Annotation(adata_or_sample: Union[AnnData, Any], atlas: str, conf_threshold: float = 0.75):
    """Constrained Simplified from <Atlas>.Simplified.* gated by <Atlas>.Broad.Celltype."""
    con_level = f"{atlas}.Constrained.Simplified"

    if _is_mosaic_sample(adata_or_sample):
        sample = adata_or_sample
        if f"{atlas}.Broad.Celltype" not in sample.protein.row_attrs:
            Atlas_Broad_Annotation(sample, atlas, conf_threshold=conf_threshold)

        meta = _make_meta_from_row_attrs(sample, list(sample.protein.row_attrs.keys()))
        meta.obs[f"{atlas}.Broad.Celltype"] = np.asarray(sample.protein.row_attrs[f"{atlas}.Broad.Celltype"])

        avail = set(meta.obs.columns)
        classes = _runtime_class_map(_build_simplified_classmap_for(atlas), avail)
        parents = _runtime_parent_subset(_build_simplified_parentmap_for(atlas), avail)

        voting_annotator(
            meta,
            con_level,
            classes,
            parent_field=f"{atlas}.Broad.Celltype",
            parent_to_subset=parents,
            conf_threshold=conf_threshold,
            normalize=True,
        )

        # expose constrained predscores as <atlas>.Simplified.<lbl>.predscore.constrained
        for lbl in _SIMPLIFIED_NAMES:
            src = f"{con_level}.{lbl}.predscore"
            if src in meta.obs.columns:
                meta.obs[f"{atlas}.Simplified.{lbl}.predscore.constrained"] = meta.obs[src].astype(float)

        # promote winners to <atlas>.Simplified.Celltype*
        meta.obs[f"{atlas}.Simplified.Celltype"] = meta.obs[f"{con_level}.Celltype"].astype(object)
        meta.obs[f"{atlas}.Simplified.Celltype.TopScore"] = meta.obs[f"{con_level}.Celltype.TopScore"].astype(float)
        meta.obs[f"{atlas}.Simplified.Celltype.LowConf"] = meta.obs[f"{con_level}.Celltype.LowConf"].astype(bool)

        _write_obs_to_row_attrs(
            sample,
            meta,
            [c for c in meta.obs.columns if c.startswith(f"{con_level}.") and c.endswith(".predscore")]
            + [c for c in meta.obs.columns if c.startswith(f"{atlas}.Simplified.") and c.endswith(".predscore.constrained")]
            + [f"{atlas}.Simplified.Celltype", f"{atlas}.Simplified.Celltype.TopScore", f"{atlas}.Simplified.Celltype.LowConf"],
        )
        return sample

    adata = adata_or_sample
    if f"{atlas}.Broad.Celltype" not in adata.obs:
        Atlas_Broad_Annotation(adata, atlas, conf_threshold=conf_threshold)

    avail = set(adata.obs.columns)
    classes = _runtime_class_map(_build_simplified_classmap_for(atlas), avail)
    parents = _runtime_parent_subset(_build_simplified_parentmap_for(atlas), avail)

    voting_annotator(
        adata,
        con_level,
        classes,
        parent_field=f"{atlas}.Broad.Celltype",
        parent_to_subset=parents,
        conf_threshold=conf_threshold,
        normalize=True,
    )

    for lbl in _SIMPLIFIED_NAMES:
        src = f"{con_level}.{lbl}.predscore"
        if src in adata.obs.columns:
            adata.obs[f"{atlas}.Simplified.{lbl}.predscore.constrained"] = adata.obs[src].astype(float)

    adata.obs[f"{atlas}.Simplified.Celltype"] = adata.obs[f"{con_level}.Celltype"].astype(object)
    adata.obs[f"{atlas}.Simplified.Celltype.TopScore"] = adata.obs[f"{con_level}.Celltype.TopScore"].astype(float)
    adata.obs[f"{atlas}.Simplified.Celltype.LowConf"] = adata.obs[f"{con_level}.Celltype.LowConf"].astype(bool)
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
            meta,
            con_level,
            classes,
            parent_field=f"{atlas}.Simplified.Celltype",
            parent_to_subset=parents,
            conf_threshold=conf_threshold,
        )

        meta.obs[f"{atlas}.Detailed.Celltype"] = meta.obs[f"{con_level}.Celltype"].astype(object)
        meta.obs[f"{atlas}.Detailed.Celltype.TopScore"] = meta.obs[f"{con_level}.Celltype.TopScore"].astype(float)
        meta.obs[f"{atlas}.Detailed.Celltype.LowConf"] = meta.obs[f"{con_level}.Celltype.LowConf"].astype(bool)

        _write_obs_to_row_attrs(
            sample,
            meta,
            [c for c in meta.obs.columns if c.startswith(f"{con_level}.") and c.endswith(".predscore")]
            + [f"{atlas}.Detailed.Celltype", f"{atlas}.Detailed.Celltype.TopScore", f"{atlas}.Detailed.Celltype.LowConf"],
        )
        return sample

    adata = adata_or_sample
    if f"{atlas}.Simplified.Celltype" not in adata.obs:
        Atlas_Simplified_Annotation(adata, atlas, conf_threshold=0.75)

    avail = set(adata.obs.columns)
    classes = _runtime_class_map(_build_detailed_classmap_for(atlas), avail)
    parents = _runtime_parent_subset(_build_detailed_parentmap_for(atlas), avail)

    voting_annotator(
        adata,
        con_level,
        classes,
        parent_field=f"{atlas}.Simplified.Celltype",
        parent_to_subset=parents,
        conf_threshold=conf_threshold,
    )

    adata.obs[f"{atlas}.Detailed.Celltype"] = adata.obs[f"{con_level}.Celltype"].astype(object)
    adata.obs[f"{atlas}.Detailed.Celltype.TopScore"] = adata.obs[f"{con_level}.Celltype.TopScore"].astype(float)
    adata.obs[f"{atlas}.Detailed.Celltype.LowConf"] = adata.obs[f"{con_level}.Celltype.LowConf"].astype(bool)
    return adata


_ATLAS_RE = re.compile(r"^(?P<atlas>[^.]+)\.(?P<level>Broad|Simplified|Detailed)\.(?P<class>[^.]+)\.predscore$")
_LEGACY_ATLAS_COL = re.compile(r"^Atlas\.([^.]+)\.(Broad|Simplified|Detailed)\.([^.]+)\.predscore$")
_UNWEIGHTED_AVG = re.compile(r"^Averaged\.Unweighted\.")
_ATLAS_SCORE_RE = re.compile(r"^[^.]+\.(Broad|Simplified|Detailed)\.[^.]+\.predscore$")
_AVG_SCORE_RE = re.compile(r"^Averaged\.(Broad|Simplified|Detailed)\.[^.]+\.predscore$")


def add_Averaged_tracks(
    obj: Union[AnnData, object],
    atlases: Sequence[str] = ("Hao", "Zhang", "Triana", "Luecken"),
    *,
    levels: Sequence[str] = ("Broad", "Simplified", "Detailed"),
    weights: Optional[Mapping[str, float]] = None,
    out_prefix: str = "Averaged.",
    write_atlas_name: bool = True,
    atlas_name_value: str = "avg",
    apply_exclusions: bool = True,
) -> None:
    """Average per-atlas predscores into Averaged.* tracks (respecting EXCLUDE_ATLAS + per-class reweighting)."""
    is_anndata = isinstance(obj, AnnData)
    is_sample = _is_mosaic_sample(obj)
    if not (is_anndata or is_sample):
        raise TypeError("add_Averaged_tracks expects an AnnData or a missionbio.mosaic.sample.Sample")

    if is_anndata:
        pending: dict[str, np.ndarray] = {}
        index = obj.obs.index

        def has_key(k: str) -> bool:
            return (k in obj.obs.columns) or (k in pending)

        def get_vec(k: str) -> np.ndarray:
            return pending.get(k, obj.obs[k].to_numpy())

        def set_vec(k: str, v: np.ndarray):
            pending[k] = np.asarray(v).ravel()

        def list_keys() -> List[str]:
            base = [c for c in obj.obs.columns if not str(c).startswith("Atlas.")]
            extra = [c for c in pending.keys() if not str(c).startswith("Atlas.")]
            return list(set(base + extra))

    else:
        df_scaled = obj.protein.get_attribute("Scaled_reads", constraint="row+col")
        if not isinstance(df_scaled, pd.DataFrame):
            df_scaled = pd.DataFrame(df_scaled)
        index = df_scaled.index.astype(str)

        def has_key(k: str) -> bool:
            return k in obj.protein.row_attrs

        def get_vec(k: str) -> np.ndarray:
            return np.asarray(obj.protein.row_attrs[k]).reshape(-1)

        def set_vec(k: str, v: np.ndarray):
            obj.protein.row_attrs[k] = np.asarray(v).ravel()

        def list_keys() -> List[str]:
            return [c for c in obj.protein.row_attrs.keys() if not str(c).startswith("Atlas.")]

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

    # Normalize weights (atlas priors)
    if weights is None:
        weights = {}
    else:
        weights = {str(k): float(v) for k, v in weights.items() if k in atlases}

    for (level, klass), atlas_cols in by_level_class.items():
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
        M = np.vstack(mats)  # (#present, N)

        # weights over remaining atlases ONLY
        if weights:
            w = np.array([max(0.0, float(weights.get(a, 0.0))) for a in present], dtype=float)
            if not np.isfinite(w).all():
                w = np.nan_to_num(w, nan=0.0, neginf=0.0, posinf=0.0)
            if w.sum() <= 0.0:
                w = np.ones(len(present), dtype=float)
        else:
            w = np.ones(len(present), dtype=float)
        w = w / w.sum()

        avg = (w[:, None] * M).sum(axis=0)  # (N,)
        label_out = f"{out_prefix}{level}.{klass}.predscore"
        set_vec(label_out, avg)

        if write_atlas_name:
            atlas_col = f"{out_prefix}{level}.{klass}.atlas"
            set_vec(atlas_col, np.array([atlas_name_value] * n, dtype=object))

    # Write Averaged.<level>.pred/conf from newly created predscores
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
            v = pending.get(k, (obj.obs[k].to_numpy() if is_anndata else np.asarray(obj.protein.row_attrs[k])))
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

    if is_anndata and "pending" in locals() and pending:
        _join_obs_cols(obj, pending)


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


# -*- coding: utf-8 -*-
"""
Updated annotate_data function for hierarchical cell type annotation.

This function:
1. Looks at Broad depth → assigns highest Averaged.Broad predscore as identity
2. Looks at Simplified depth → constrains by Broad ontology, renormalizes, assigns highest
3. Looks at Detailed depth → constrains by Simplified ontology, renormalizes, assigns highest
"""

def annotate_data(
    obj: Union[AnnData, Any],
    models_path: Optional[Union[str, Path]] = None,
    data_path: Optional[Union[str, Path]] = None,
    *,
    use_consensus: bool = False,  # Use Averaged.Consensus.* instead of Averaged.*
    source_prefix: str = "Averaged.",  # Will be "Averaged.Consensus." if use_consensus=True
) -> Union[AnnData, Any]:
    """
    Hierarchical cell type annotation using Averaged prediction scores.
    
    Process:
    1. Broad depth: Assign highest Averaged.Broad.{label}.predscore as Broad identity
    2. Simplified depth: 
       - Constrain predscores based on Broad identity (ontology alignment)
       - Renormalize so constrained predscores sum to 1
       - Assign highest as Simplified identity
    3. Detailed depth:
       - Constrain predscores based on Simplified identity (ontology alignment)
       - Renormalize so constrained predscores sum to 1
       - Assign highest as Detailed identity
    
    Parameters
    ----------
    obj : AnnData or Sample
        Object with prediction scores
    models_path : str or Path, optional
        Path to models (used if predictions don't exist)
    data_path : str or Path, optional
        Path to data (used if predictions don't exist)
    use_consensus : bool, default=False
        If True, use Averaged.Consensus.* tracks instead of Averaged.*
    source_prefix : str, default="Averaged."
        Prefix for source tracks (automatically set to "Averaged.Consensus." if use_consensus=True)
    
    Returns
    -------
    obj : AnnData or Sample
        Object with added annotation columns:
        - Averaged.Broad.Celltype
        - Averaged.Simplified.Celltype  
        - Averaged.Detailed.Celltype
    """
    
    # Handle use_consensus flag
    if use_consensus:
        source_prefix = "Averaged.Consensus."
    
    is_anndata = isinstance(obj, AnnData)
    is_sample = _is_mosaic_sample(obj)
    
    if not (is_anndata or is_sample):
        raise TypeError("annotate_data expects an AnnData or a missionbio.mosaic.sample.Sample")
    
    # Import these here to avoid circular imports
    from .constants import SIMPLIFIED_CLASSES, _DETAILED_LABELS
    
    # Define ontology mappings
    # Broad -> Simplified
    BROAD_TO_SIMPLIFIED = {
        "Immature": ["HSPC"],
        "Mature": [k for k in SIMPLIFIED_CLASSES.keys() if k != "HSPC"]
    }
    
    # Simplified -> Detailed (from constants)
    SIMPLIFIED_TO_DETAILED = {
        "B": ["Pre-Pro-B", "Pro-B", "Pre-B", "Immature_B", "B_Naive", "B_Memory"],
        "Plasma": ["Plasma"],
        "CD4_T": ["CD4_T_Naive", "CD4_T_Memory", "CD4_CTL", "Treg"],
        "CD8_T": ["CD8_T_Naive", "CD8_T_Memory", "MAIT"],
        "Other_T": ["GdT", "MAIT"],
        "NK": ["NK_CD56_bright", "NK_CD56_dim"],
        "Monocyte": ["CD14_Mono", "CD16_Mono", "cDC1", "cDC2"],
        "Myeloid": ["Myeloid_progenitor"],
        "cDC": ["cDC1", "cDC2"],
        "pDC": ["pDC"],
        "Erythroid": ["Erythroblast", "ErP"],
        "HSPC": ["HSC_MPP", "LMPP", "GMP", "MEP", "MkP", "EoBaMaP", "Pre-Pro-B", "Pro-B"],
    }
    
    # Helper functions for getting/setting values
    if is_anndata:
        def has_col(col: str) -> bool:
            return col in obj.obs.columns
        
        def get_col(col: str) -> np.ndarray:
            return obj.obs[col].to_numpy()
        
        def set_col(col: str, vals: np.ndarray):
            obj.obs[col] = vals
        
        def list_cols() -> List[str]:
            return list(obj.obs.columns)
        
        n_cells = obj.n_obs
    else:
        def has_col(col: str) -> bool:
            return col in obj.protein.row_attrs
        
        def get_col(col: str) -> np.ndarray:
            return np.asarray(obj.protein.row_attrs[col]).reshape(-1)
        
        def set_col(col: str, vals: np.ndarray):
            obj.protein.row_attrs[col] = np.asarray(vals)
        
        def list_cols() -> List[str]:
            return list(obj.protein.row_attrs.keys())
        
        # Get number of cells from scaled data
        df_scaled = obj.protein.get_attribute("Scaled_reads", constraint="row+col")
        if not isinstance(df_scaled, pd.DataFrame):
            df_scaled = pd.DataFrame(df_scaled)
        n_cells = len(df_scaled.index)
    
    # Check if predictions exist
    broad_cols = [f"{source_prefix}Broad.Mature.predscore", f"{source_prefix}Broad.Immature.predscore"]
    has_predictions = all(has_col(c) for c in broad_cols)
    
    if not has_predictions:
        print(f"[annotate_data] No {source_prefix}* prediction tracks found. Running generate_predictions...")
        from .prediction import generate_predictions
        from .model_loading import ensure_models_available, get_default_models_path, get_default_data_path
        
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
    
    # ============================================================================
    # STEP 1: BROAD ANNOTATION (Immature vs Mature)
    # ============================================================================
    print("[annotate_data] Step 1: Broad annotation...")
    
    broad_labels = ["Immature", "Mature"]
    broad_scores = np.zeros((n_cells, len(broad_labels)), dtype=float)
    
    for i, label in enumerate(broad_labels):
        col = f"{source_prefix}Broad.{label}.predscore"
        if has_col(col):
            broad_scores[:, i] = get_col(col)
        else:
            raise KeyError(f"Missing required column: {col}")
    
    # Assign highest score as Broad identity
    broad_winner_idx = broad_scores.argmax(axis=1)
    broad_identity = np.array([broad_labels[i] for i in broad_winner_idx], dtype=object)
    broad_conf = broad_scores[np.arange(n_cells), broad_winner_idx]
    
    set_col(f"{source_prefix}Broad.Celltype", broad_identity)
    set_col(f"{source_prefix}Broad.Celltype.TopScore", broad_conf)
    
    print(f"  ✓ Broad annotation complete. Distribution:")
    unique, counts = np.unique(broad_identity, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"    {u}: {c} cells ({100*c/n_cells:.1f}%)")
    
    # ============================================================================
    # STEP 2: SIMPLIFIED ANNOTATION (constrained by Broad)
    # ============================================================================
    print("[annotate_data] Step 2: Simplified annotation (constrained by Broad)...")
    
    simplified_labels = list(SIMPLIFIED_CLASSES.keys())
    simplified_scores = np.zeros((n_cells, len(simplified_labels)), dtype=float)
    
    # Get raw scores
    for i, label in enumerate(simplified_labels):
        col = f"{source_prefix}Simplified.{label}.predscore"
        if has_col(col):
            simplified_scores[:, i] = get_col(col)
    
    # Apply Broad constraints and renormalize
    constrained_simplified = np.zeros_like(simplified_scores)
    
    for cell_idx in range(n_cells):
        broad_label = broad_identity[cell_idx]
        allowed_simplified = BROAD_TO_SIMPLIFIED.get(broad_label, [])
        
        # Set scores to 0 for classes not allowed by ontology
        for i, simp_label in enumerate(simplified_labels):
            if simp_label in allowed_simplified:
                constrained_simplified[cell_idx, i] = simplified_scores[cell_idx, i]
            else:
                constrained_simplified[cell_idx, i] = 0.0
        
        # Renormalize so they sum to 1
        row_sum = constrained_simplified[cell_idx].sum()
        if row_sum > 0:
            constrained_simplified[cell_idx] = constrained_simplified[cell_idx] / row_sum
        else:
            # If all zeros (shouldn't happen), distribute uniformly among allowed
            if allowed_simplified:
                for i, simp_label in enumerate(simplified_labels):
                    if simp_label in allowed_simplified:
                        constrained_simplified[cell_idx, i] = 1.0 / len(allowed_simplified)
    
    # Store constrained predscores
    for i, label in enumerate(simplified_labels):
        col = f"{source_prefix}Simplified.{label}.predscore.constrained"
        set_col(col, constrained_simplified[:, i])
    
    # Assign highest constrained score as Simplified identity
    simplified_winner_idx = constrained_simplified.argmax(axis=1)
    simplified_identity = np.array([simplified_labels[i] for i in simplified_winner_idx], dtype=object)
    simplified_conf = constrained_simplified[np.arange(n_cells), simplified_winner_idx]
    
    set_col(f"{source_prefix}Simplified.Celltype", simplified_identity)
    set_col(f"{source_prefix}Simplified.Celltype.TopScore", simplified_conf)
    
    print(f"  ✓ Simplified annotation complete. Top 5 classes:")
    unique, counts = np.unique(simplified_identity, return_counts=True)
    sorted_idx = np.argsort(-counts)
    for idx in sorted_idx[:5]:
        print(f"    {unique[idx]}: {counts[idx]} cells ({100*counts[idx]/n_cells:.1f}%)")
    
    # ============================================================================
    # STEP 3: DETAILED ANNOTATION (constrained by Simplified)
    # ============================================================================
    print("[annotate_data] Step 3: Detailed annotation (constrained by Simplified)...")
    
    detailed_labels = _DETAILED_LABELS
    detailed_scores = np.zeros((n_cells, len(detailed_labels)), dtype=float)
    
    # Get raw scores
    for i, label in enumerate(detailed_labels):
        col = f"{source_prefix}Detailed.{label}.predscore"
        if has_col(col):
            detailed_scores[:, i] = get_col(col)
    
    # Apply Simplified constraints and renormalize
    constrained_detailed = np.zeros_like(detailed_scores)
    
    for cell_idx in range(n_cells):
        simp_label = simplified_identity[cell_idx]
        allowed_detailed = SIMPLIFIED_TO_DETAILED.get(simp_label, [])
        
        # Set scores to 0 for classes not allowed by ontology
        for i, det_label in enumerate(detailed_labels):
            if det_label in allowed_detailed:
                constrained_detailed[cell_idx, i] = detailed_scores[cell_idx, i]
            else:
                constrained_detailed[cell_idx, i] = 0.0
        
        # Renormalize so they sum to 1
        row_sum = constrained_detailed[cell_idx].sum()
        if row_sum > 0:
            constrained_detailed[cell_idx] = constrained_detailed[cell_idx] / row_sum
        else:
            # If all zeros (shouldn't happen), distribute uniformly among allowed
            if allowed_detailed:
                for i, det_label in enumerate(detailed_labels):
                    if det_label in allowed_detailed:
                        constrained_detailed[cell_idx, i] = 1.0 / len(allowed_detailed)
    
    # Store constrained predscores
    for i, label in enumerate(detailed_labels):
        col = f"{source_prefix}Detailed.{label}.predscore.constrained"
        set_col(col, constrained_detailed[:, i])
    
    # Assign highest constrained score as Detailed identity
    detailed_winner_idx = constrained_detailed.argmax(axis=1)
    detailed_identity = np.array([detailed_labels[i] for i in detailed_winner_idx], dtype=object)
    detailed_conf = constrained_detailed[np.arange(n_cells), detailed_winner_idx]
    
    set_col(f"{source_prefix}Detailed.Celltype", detailed_identity)
    set_col(f"{source_prefix}Detailed.Celltype.TopScore", detailed_conf)
    
    print(f"  ✓ Detailed annotation complete. Top 10 classes:")
    unique, counts = np.unique(detailed_identity, return_counts=True)
    sorted_idx = np.argsort(-counts)
    for idx in sorted_idx[:10]:
        print(f"    {unique[idx]}: {counts[idx]} cells ({100*counts[idx]/n_cells:.1f}%)")
    
    print("[annotate_data] ✓ Hierarchical annotation complete!")
    
    return obj

# Example usage documentation:
"""
# For standard Averaged tracks:
PBMC_HD01 = annotate_data(PBMC_HD01)

# For consensus tracks (high-confidence only):
PBMC_HD01 = annotate_data(PBMC_HD01, use_consensus=True)

# Output columns created:
# - Averaged.Broad.Celltype
# - Averaged.Broad.Celltype.TopScore
# - Averaged.Simplified.Celltype
# - Averaged.Simplified.Celltype.TopScore
# - Averaged.Simplified.{label}.predscore.constrained (for each label)
# - Averaged.Detailed.Celltype
# - Averaged.Detailed.Celltype.TopScore
# - Averaged.Detailed.{label}.predscore.constrained (for each label)

# If use_consensus=True:
# - Averaged.Consensus.Broad.Celltype
# - Averaged.Consensus.Simplified.Celltype
# - Averaged.Consensus.Detailed.Celltype
# (and corresponding .TopScore and .predscore.constrained columns)
"""


def _get_matrix_for_embedding(
    obj: Union[AnnData, Any],
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

    if _is_mosaic_sample(obj):
        if "umap" in obj.protein.row_attrs:
            U = np.asarray(obj.protein.row_attrs["umap"])
            return U[:, :2] if U.ndim == 2 and U.shape[1] >= 2 else U.reshape(-1, 1)

        df = obj.protein.get_attribute("Scaled_reads", constraint="row+col")
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)
        X = df.values.astype(float)
        n_comp = min(n_pca, X.shape[1]) if X.shape[1] > 1 else 1
        return X.reshape(-1, 1) if n_comp <= 1 else PCA(n_components=n_comp, random_state=0).fit_transform(X)

    raise TypeError("Expected AnnData or missionbio.mosaic.sample")

import re
from typing import Any, Union

def clear_annotation(obj: Any) -> Any:
    """
    For missionbio.mosaic Sample-like objects:
    - Keep only Averaged.(Broad|Simplified|Detailed).Celltype* keys (and their derived variants)
    - Keep a small set of core non-annotation fields (barcode/umap/etc)
    - Delete everything else from sample.protein.row_attrs
    """
    if not _is_mosaic_sample(obj):
        raise TypeError("Expected a missionbio.mosaic Sample-like object")

    ra = obj.protein.row_attrs

    # --- Always keep these non-annotation fields (edit as needed)
    keep_core = {
        "barcode",
        "label",
        "sample_name",
        "umap",
        "pca",
        "Clusters",   # your cluster IDs
    }

    # --- Keep ONLY these annotation keys
    keep_re = re.compile(
        r"^Averaged\.(Broad|Simplified|Detailed)\.Celltype(\..*)?$"
    )

    keep_keys = {k for k in ra.keys() if (k in keep_core) or keep_re.match(str(k))}
    drop_keys = [k for k in list(ra.keys()) if k not in keep_keys]

    for k in drop_keys:
        try:
            del ra[k]
        except Exception:
            pass

    return obj

from typing import Any, Optional, Union

import numpy as np
import pandas as pd

try:
    from anndata import AnnData
except Exception:
    AnnData = object  # type: ignore[misc,assignment]

def mark_mixed_clusters(
    obj: Union["AnnData", Any],
    label_in: str,
    *,
    cluster_col: Optional[str] = None,
    min_frequency_threshold: float = 0.3,
    mixed_label: str = "Mixed",
    label_out: Optional[str] = None,
) -> Union["AnnData", Any]:
    """
    Relabel clusters with no dominant label as 'Mixed' (frequency-only),
    using the same logic as suggest_cluster_celltype_identity().

    If label_out is provided:
        - copy label_in -> label_out
        - apply Mixed relabeling to label_out
    Else:
        - modify label_in in place
    """
    is_anndata = isinstance(obj, AnnData)
    is_sample = _is_mosaic_sample(obj)

    if not (is_anndata or is_sample):
        raise TypeError("Expected AnnData or a missionbio.mosaic Sample-like object")

    # -------------------------
    # clusters
    # -------------------------
    if is_anndata:
        if cluster_col is None:
            for cand in ("leiden", "louvain", "Clusters", "clusters", "cluster"):
                if cand in obj.obs.columns:
                    cluster_col = cand
                    break
        if cluster_col is None or cluster_col not in obj.obs.columns:
            raise KeyError("Cluster column not found. Provide cluster_col.")
        clusters = obj.obs[cluster_col].astype(str).to_numpy()
    else:
        clusters = np.asarray(obj.protein.get_labels()).astype(str)

    # -------------------------
    # labels
    # -------------------------
    if is_anndata:
        if label_in not in obj.obs.columns:
            raise KeyError(f"'{label_in}' not in adata.obs")
        labels = obj.obs[label_in]
        was_cat = isinstance(labels.dtype, pd.CategoricalDtype)
        labels_arr = labels.to_numpy(dtype=object)
    else:
        if label_in not in obj.protein.row_attrs:
            raise KeyError(f"'{label_in}' not in Sample.protein.row_attrs")
        labels_arr = np.asarray(obj.protein.row_attrs[label_in], dtype=object)
        was_cat = False

    # -------------------------
    # choose output column
    # -------------------------
    target_col = label_out or label_in
    updated = labels_arr.copy()

    # -------------------------
    # compute mixed clusters
    # -------------------------
    mixed_clusters = set()

    for cl in pd.unique(clusters):
        idx = np.where(clusters == cl)[0]
        labs = updated[idx]
        labs_valid = labs[pd.notna(labs)]

        if labs_valid.size == 0:
            mixed_clusters.add(str(cl))
            continue

        _, cnt = np.unique(labs_valid.astype(str), return_counts=True)
        top_freq = float(cnt.max() / cnt.sum())

        if top_freq < float(min_frequency_threshold):
            mixed_clusters.add(str(cl))

    if not mixed_clusters:
        return obj

    # -------------------------
    # apply Mixed label
    # -------------------------
    mixed_mask = pd.Series(clusters).isin(mixed_clusters).to_numpy()
    updated[mixed_mask] = mixed_label

    if is_anndata:
        if was_cat:
            cats = pd.Index(obj.obs[label_in].cat.categories).astype(str)
            if mixed_label not in cats:
                cats = cats.append(pd.Index([mixed_label]))
            obj.obs[target_col] = pd.Categorical(updated.astype(str), categories=cats)
        else:
            obj.obs[target_col] = updated
    else:
        obj.protein.row_attrs[target_col] = updated

    return obj

def score_mixed_clusters(
    obj: Union["AnnData", Any],
    clusters: Union[str, np.ndarray, pd.Series, list],
    labels: Union[str, np.ndarray, pd.Series, list],
    *,
    embedding_key: str = "X_umap",
    n_pca: int = 20,
    weights: Optional[Dict[str, float]] = None,
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

    if uniq_clusters.size >= 2 and all((cvec == u).sum() >= int(min_cells_for_silhouette) for u in uniq_clusters):
        try:
            sil = silhouette_samples(Z, cvec, metric="euclidean")
        except Exception:
            pass

    rows = []
    for cl in uniq_clusters:
        idx = (cvec == cl)
        n = int(idx.sum())
        labs = lvec[idx]
        _, cts = np.unique(labs, return_counts=True)
        p = cts / cts.sum()

        majority_frac = float(p.max())
        if len(p) > 1:
            H = -np.sum(p * np.log(p + 1e-12))
            Hmax = np.log(len(p))
            entropy_norm = float(H / Hmax)
        else:
            entropy_norm = 0.0

        s = float(np.nanmean(sil[idx])) if np.any(idx) else np.nan
        mix_from_entropy = entropy_norm
        mix_from_sil = 0.5 if np.isnan(s) else (1.0 - s) * 0.5

        rows.append(
            {
                "cluster": cl,
                "n_cells": n,
                "majority_frac": majority_frac,
                "entropy_norm": entropy_norm,
                "silhouette_mean": s,
                "mix_from_entropy": mix_from_entropy,
                "mix_from_sil": mix_from_sil,
            }
        )

    df = pd.DataFrame(rows).set_index("cluster")

    if weights is None:
        weights = {"entropy": 0.6, "silhouette": 0.4}

    w_e = float(weights.get("entropy", 0.6))
    w_s = float(weights.get("silhouette", 0.4))
    w_sum = (w_e + w_s) if (w_e + w_s) > 0 else 1.0
    w_e, w_s = w_e / w_sum, w_s / w_sum

    df["mixed_likelihood"] = w_e * df["mix_from_entropy"].values + w_s * df["mix_from_sil"].values
    return df.sort_values("mixed_likelihood", ascending=False)

def mark_small_clusters(
    obj: Union["AnnData", Any],
    label_in: str,
    *,
    cluster_col: Optional[str] = None,
    min_cells: int = 3,
    small_label: str = "Small",
    label_out: Optional[str] = None,
) -> Union["AnnData", Any]:
    """
    Relabel clusters with fewer than `min_cells` cells as `small_label`.

    If label_out is provided:
        - copy label_in -> label_out
        - apply relabeling to label_out
    Else:
        - modify label_in in place
    """
    is_anndata = isinstance(obj, AnnData)
    is_sample = _is_mosaic_sample(obj)

    if not (is_anndata or is_sample):
        raise TypeError("Expected AnnData or a missionbio.mosaic Sample-like object")

    # -------------------------
    # clusters
    # -------------------------
    if is_anndata:
        if cluster_col is None:
            for cand in ("leiden", "louvain", "Clusters", "clusters", "cluster"):
                if cand in obj.obs.columns:
                    cluster_col = cand
                    break
        if cluster_col is None or cluster_col not in obj.obs.columns:
            raise KeyError("Cluster column not found. Provide cluster_col.")
        clusters = obj.obs[cluster_col].astype(str).to_numpy()

    else:
        # IMPORTANT: use row_attrs[cluster_col] if provided
        if cluster_col is not None:
            if cluster_col not in obj.protein.row_attrs:
                raise KeyError(f"'{cluster_col}' not found in sample.protein.row_attrs")
            clusters = np.asarray(obj.protein.row_attrs[cluster_col], dtype=object).astype(str)
        else:
            # fallback only if cluster_col not provided
            clusters = np.asarray(obj.protein.get_labels(), dtype=object).astype(str)

    # -------------------------
    # labels
    # -------------------------
    if is_anndata:
        if label_in not in obj.obs.columns:
            raise KeyError(f"'{label_in}' not found in adata.obs")
        labels = obj.obs[label_in]
        was_cat = isinstance(labels.dtype, pd.CategoricalDtype)
        labels_arr = labels.to_numpy(dtype=object)
    else:
        if label_in not in obj.protein.row_attrs:
            raise KeyError(f"'{label_in}' not in Sample.protein.row_attrs")
        labels_arr = np.asarray(obj.protein.row_attrs[label_in], dtype=object)
        was_cat = False

    # -------------------------
    # output column
    # -------------------------
    target_col = label_out or label_in
    updated = labels_arr.copy()

    # -------------------------
    # identify small clusters
    # -------------------------
    cluster_sizes = pd.Series(clusters).value_counts()
    small_clusters = cluster_sizes[cluster_sizes < int(min_cells)].index.astype(str)

    if len(small_clusters) == 0:
        # still write label_out copy if requested
        if label_out is not None:
            if is_anndata:
                obj.obs[target_col] = updated
            else:
                obj.protein.row_attrs[target_col] = updated
        return obj

    small_mask = pd.Series(clusters).isin(small_clusters).to_numpy()
    updated[small_mask] = small_label

    # -------------------------
    # write back
    # -------------------------
    if is_anndata:
        if was_cat:
            cats = pd.Index(obj.obs[label_in].cat.categories).astype(str)
            if small_label not in cats:
                cats = cats.append(pd.Index([small_label]))
            obj.obs[target_col] = pd.Categorical(updated.astype(str), categories=cats)
        else:
            obj.obs[target_col] = updated
    else:
        obj.protein.row_attrs[target_col] = updated

    return obj

def _join_obs_cols(adata: AnnData, pending: Mapping[str, np.ndarray]) -> None:
    for k, v in pending.items():
        adata.obs[k] = v
