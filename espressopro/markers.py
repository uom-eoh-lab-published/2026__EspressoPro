# -*- coding: utf-8 -*-
"""pyUCell-based signature annotation with unimodal bulk + right-tail component calling.

Key idea
--------
Assume a single bulk/background population (unimodal) + rare positives in the extreme right tail.
We therefore:
  1) compute a pyUCell signature score (pos - neg, or pos only),
  2) fit a 2-component GMM ONLY on the upper-tail subset (score >= quantile(tail_q)),
  3) define positives as cells IN the tail whose posterior for the highest-mean component >= posterior_threshold.

Defaults
--------
posterior_threshold = 0.90  (requested)
tail_q              = 0.80
random_state        = 42
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import scanpy as sc
import pyucell as uc
from sklearn.mixture import GaussianMixture

try:
    from anndata import AnnData
except Exception:  # soft import
    AnnData = object  # type: ignore[misc,assignment]

from .constants import MAST_POS, MAST_NEG


# ------------------------------- shared helpers -------------------------------

def _is_sample(x: Any) -> bool:
    return hasattr(x, "protein") and hasattr(x.protein, "row_attrs") and hasattr(x.protein, "get_attribute")


def _get_df(obj: Union[AnnData, Any], layer: str) -> pd.DataFrame:
    """Return cells x features DataFrame from AnnData or MissionBio Sample."""
    if isinstance(obj, AnnData):
        X = obj.X.A if hasattr(obj.X, "A") else (obj.X.toarray() if hasattr(obj.X, "toarray") else obj.X)
        df = pd.DataFrame(X, index=obj.obs_names.astype(str), columns=obj.var_names.astype(str))
        return df

    if _is_sample(obj):
        df = obj.protein.get_attribute(layer, constraint="row+col")
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)
        df.index = df.index.astype(str)
        df.columns = df.columns.astype(str)
        df = df.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return df

    raise TypeError("Expected AnnData or missionbio.mosaic.sample.Sample-like object")


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
    raise TypeError("Expected AnnData or missionbio.mosaic.sample.Sample-like object")


# -------------------------- tail-only calling (core) ---------------------------

def _right_tail_mask_via_tail_gmm(
    x: np.ndarray,
    *,
    tail_q: float = 0.80,
    posterior_threshold: float = 0.90,  # requested default
    random_state: int = 42,
    min_tail_n: int = 10,
) -> Tuple[np.ndarray, dict]:
    """
    Call positives using ONLY the right-tail component.

    Steps
    -----
    - Compute tail cutoff: tail_cut = quantile(x, tail_q) on finite values
    - Fit 2-component GMM on tail values only: x_tail = x[x >= tail_cut]
    - Choose right-most component: hi = argmax(mean)
    - Compute posterior P(hi) for ALL finite points under tail-trained model
    - Call positives only if:
        (x in tail) AND (P(hi) >= posterior_threshold)

    Returns
    -------
    mask : bool array, same length as x
    info : dict with fitted parameters + counts
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    finite = np.isfinite(x)
    mask = np.zeros_like(x, dtype=bool)

    if finite.sum() < max(5, min_tail_n) or np.nanstd(x[finite]) == 0:
        return mask, {"status": "degenerate"}

    s = x[finite]
    tail_cut = float(np.quantile(s, tail_q))
    in_tail_local = s >= tail_cut

    if in_tail_local.sum() < min_tail_n:
        return mask, {
            "status": "tail_too_small",
            "tail_q": float(tail_q),
            "tail_cut": tail_cut,
            "n_tail": int(in_tail_local.sum()),
            "min_tail_n": int(min_tail_n),
        }

    X_tail = s[in_tail_local].reshape(-1, 1)
    gmm = GaussianMixture(n_components=2, random_state=random_state).fit(X_tail)

    means = gmm.means_.ravel()
    vars_ = gmm.covariances_.ravel()
    weights = gmm.weights_.ravel()

    hi = int(np.argmax(means))  # right-most mean

    # posterior for ALL finite points under tail-trained model
    post_hi = gmm.predict_proba(s.reshape(-1, 1))[:, hi]
    called_local = in_tail_local & (post_hi >= posterior_threshold)

    # map back to full-length mask
    finite_idx = np.where(finite)[0]
    mask[finite_idx] = called_local

    return mask, {
        "status": "ok",
        "tail_q": float(tail_q),
        "tail_cut": tail_cut,
        "posterior_threshold": float(posterior_threshold),
        "means_tail": [float(m) for m in means],
        "vars_tail": [float(v) for v in vars_],
        "weights_tail": [float(w) for w in weights],
        "hi_component": hi,
        "n_called": int(mask.sum()),
        "n_finite": int(finite.sum()),
        "n_tail": int(in_tail_local.sum()),
    }


# -------------------------- generic pyUCell signature --------------------------

def add_signature_annotation(
    obj: Union[AnnData, Any],
    positive_markers: List[str],
    negative_markers: Optional[List[str]],
    cell_type_label: str,
    *,
    layer: str = "Normalized_reads",
    max_rank: Optional[int] = None,              # default: round(n_features/2)
    score_key: Optional[str] = None,             # where to store signature score
    label_in: str = "Averaged.Detailed.Celltype",
    label_out: str = "Averaged.Detailed.Celltype.Refined",
    tail_q: float = 0.80,
    posterior_threshold: float = 0.90,           # requested default
    gmm_random_state: int = 42,
    min_tail_n: int = 10,
    verbose: bool = False,
) -> Union[AnnData, Any]:
    """
    Compute a pyUCell-based signature score and refine labels using tail-only GMM calling.

    Signature score:
      score = UCell(positive_markers) - UCell(negative_markers)   (if negative_markers provided)
      score = UCell(positive_markers)                             (if negative_markers is None/empty)

    Calling:
      - tail_cut = quantile(score, tail_q)
      - fit 2-GMM on score[score >= tail_cut]
      - hi component = highest mean
      - call positives = (score in tail) & (posterior_hi >= posterior_threshold)

    Writes:
      - score_key (default: f"{cell_type_label}_signature_score")
      - label_out (default: "Averaged.Detailed.Celltype.Refined")
    """
    df = _get_df(obj, layer=layer)
    n_cells, n_feats = df.shape

    # max_rank default: round(n_features/2)
    if max_rank is None:
        max_rank = int(np.round(n_feats / 2.0))
    max_rank = max(1, min(int(max_rank), n_feats))

    # Temporary AnnData for pyUCell
    ad = sc.AnnData(X=df.to_numpy(dtype=float))
    ad.obs_names = df.index.astype(str)
    ad.var_names = df.columns.astype(str)

    # Filter markers to present features
    pos_keep = [m for m in (positive_markers or []) if m in ad.var_names]
    neg_keep = [m for m in (negative_markers or []) if m in ad.var_names]

    if len(pos_keep) == 0:
        raise ValueError(f"[{cell_type_label}] None of the positive markers were found in features.")

    # Compute UCell scores
    uc.compute_ucell_scores(ad, signatures={f"{cell_type_label}_pos": pos_keep}, max_rank=max_rank)
    pos_cols = [c for c in ad.obs.columns if f"{cell_type_label}_pos" in c and "UCell" in c]
    pos_col = pos_cols[-1]  # robust to naming variations
    pos_score = ad.obs[pos_col].to_numpy(dtype=float)

    if len(neg_keep) > 0:
        uc.compute_ucell_scores(ad, signatures={f"{cell_type_label}_neg": neg_keep}, max_rank=max_rank)
        neg_cols = [c for c in ad.obs.columns if f"{cell_type_label}_neg" in c and "UCell" in c]
        neg_col = neg_cols[-1]
        neg_score = ad.obs[neg_col].to_numpy(dtype=float)
        score = pos_score - neg_score
    else:
        score = pos_score

    # store score
    score_key = score_key or f"{cell_type_label}_signature_score"
    _set_vec(obj, score_key, score)

    # tail-only calling
    mask, info = _right_tail_mask_via_tail_gmm(
        score,
        tail_q=tail_q,
        posterior_threshold=posterior_threshold,
        random_state=gmm_random_state,
        min_tail_n=min_tail_n,
    )

    # base labels
    base = _get_vec(obj, label_in)
    if base is None:
        base = _get_vec(obj, "CommonDetailed.Celltype")
        if base is None:
            base = np.array(["Unknown"] * n_cells, dtype=object)

    refined = np.asarray(base, dtype=object).copy()
    refined[mask] = cell_type_label
    _set_vec(obj, label_out, refined)

    if verbose:
        print(
            f"[{cell_type_label}] max_rank={max_rank} tail_q={tail_q:.2f} "
            f"post_thr={posterior_threshold:.2f} score_key='{score_key}' -> '{label_out}'"
        )
        if info.get("status") == "ok":
            print(
                f"[{cell_type_label}] tail_cut={info['tail_cut']:.3f} "
                f"means_tail={info['means_tail']} hi={info['hi_component']} "
                f"called={info['n_called']}/{info['n_finite']}"
            )
        else:
            print(f"[{cell_type_label}] calling status={info.get('status')} ({info})")

    return obj


# ------------------------------- Mast-specific API ------------------------------

def add_mast_annotation(
    obj: Union[AnnData, Any],
    *,
    layer: str = "Normalized_reads",
    max_rank: Optional[int] = None,
    score_key: str = "Mast_signature_score",
    label_in: str = "Averaged.Detailed.Celltype",
    label_out: str = "Averaged.Detailed.Celltype.Refined",
    tail_q: float = 0.80,
    posterior_threshold: float = 0.90,  # requested default
    gmm_random_state: int = 42,
    min_tail_n: int = 10,
    verbose: bool = False,
) -> Union[AnnData, Any]:
    """
    Mast-specific wrapper using predefined MAST_POS/MAST_NEG marker sets.

    Writes:
      - signature score: score_key (default 'Mast_signature_score')
      - refined labels:  label_out (default 'Averaged.Detailed.Celltype.Refined')
    """
    return add_signature_annotation(
        obj=obj,
        positive_markers=MAST_POS,
        negative_markers=MAST_NEG,
        cell_type_label="Mast",
        layer=layer,
        max_rank=max_rank,
        score_key=score_key,
        label_in=label_in,
        label_out=label_out,
        tail_q=tail_q,
        posterior_threshold=posterior_threshold,
        gmm_random_state=gmm_random_state,
        min_tail_n=min_tail_n,
        verbose=verbose,
    )
