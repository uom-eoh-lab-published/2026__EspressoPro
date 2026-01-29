# -*- coding: utf-8 -*-
"""pyUCell-based signature annotation + adaptive GMM plotting."""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import scanpy as sc
import pyucell as uc
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from scipy.stats import norm as sp_norm  # IMPORTANT: avoid shadowing

try:
    from anndata import AnnData
except Exception:
    AnnData = object  # type: ignore[misc,assignment]

from .constants import MAST_POS, MAST_NEG


# ------------------------------- helpers -------------------------------

def _is_sample(x: Any) -> bool:
    return hasattr(x, "protein") and hasattr(x.protein, "row_attrs") and hasattr(x.protein, "get_attribute")


def _get_df(obj: Union[AnnData, Any], layer: str) -> pd.DataFrame:
    """Return cells x features DataFrame from AnnData or MissionBio Sample."""
    if isinstance(obj, AnnData):
        X = obj.X.A if hasattr(obj.X, "A") else (obj.X.toarray() if hasattr(obj.X, "toarray") else obj.X)
        return pd.DataFrame(X, index=obj.obs_names.astype(str), columns=obj.var_names.astype(str))

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


# ------------------------- adaptive caller + plot -------------------------

def call_tail_or_bimodal_gmm(
    scores_all: np.ndarray,
    *,
    random_state: int = 42,
    posterior_threshold_bimodal: float = 0.90,
    posterior_threshold_tail: float = 0.90,
    tail_q: float = 0.80,
    sep_threshold: float = 1.25,
    bins: int = 60,
    title: str = "Signature score — adaptive GMM",
    figsize=(4.35, 4.35),
    show: bool = True,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Adaptive GMM caller + plotted distributions.

    If bimodal (sep >= sep_threshold):
      - Fit 2-GMM on all scores
      - Positive = higher-mean component, posterior >= posterior_threshold_bimodal
      - Plot BOTH components

    Else (tail/unimodal):
      - Define tail: score >= quantile(tail_q)
      - Fit 2-GMM on tail only
      - Positive = right-most tail component, posterior >= posterior_threshold_tail, AND within tail
      - Plot:
          * Bulk Gaussian (fit to scores below tail_cut)
          * Right-tail component only (from tail GMM)
    """
    scores_all = np.asarray(scores_all, dtype=float).reshape(-1)
    finite = np.isfinite(scores_all)
    s = scores_all[finite]

    mask = np.zeros_like(scores_all, dtype=bool)
    info: Dict[str, Any] = {"status": "ok"}

    if s.size < 10 or float(np.nanstd(s)) == 0.0:
        info.update({"status": "degenerate", "method": "none"})
        if show:
            fig, ax = plt.subplots(figsize=figsize)
            ax.hist(s, bins=bins, density=True, color="lightgrey", edgecolor="black", alpha=0.8)
            ax.set_title(title + " (degenerate)", fontweight="bold")
            ax.set_xlabel("Signature score"); ax.set_ylabel("Density")
            plt.tight_layout(); plt.show()
        return mask, info

    X = s.reshape(-1, 1)
    xs = np.linspace(float(s.min()), float(s.max()), 2000)

    # --- full fit for sep decision ---
    gmm_full = GaussianMixture(n_components=2, random_state=random_state).fit(X)
    means = gmm_full.means_.ravel()
    vars_ = gmm_full.covariances_.ravel()
    weights = gmm_full.weights_.ravel()

    sep = float(np.abs(means[1] - means[0]) / (np.sqrt(vars_[0] + vars_[1]) + 1e-12))
    use_bimodal = sep >= sep_threshold

    if use_bimodal:
        hi = int(np.argmax(means))
        post_hi = gmm_full.predict_proba(X)[:, hi]
        called_local = post_hi >= posterior_threshold_bimodal

        # compute cut for annotation line (first xs where posterior crosses threshold)
        post_grid = gmm_full.predict_proba(xs.reshape(-1, 1))[:, hi]
        cross = np.where(post_grid >= posterior_threshold_bimodal)[0]
        cut = float(xs[cross[0]]) if cross.size else np.nan

        mask[np.where(finite)[0]] = called_local

        info.update({
            "method": f"full 2-GMM (sep={sep:.2f} ≥ {sep_threshold})",
            "sep": sep,
            "means": [float(m) for m in means],
            "hi_component": hi,
            "posterior_threshold": float(posterior_threshold_bimodal),
            "cut": cut,
            "n_called": int(mask.sum()),
            "n_total_finite": int(s.size),
        })

        if show:
            fig, ax = plt.subplots(figsize=figsize)
            ax.hist(s, bins=bins, density=True, color="lightgrey", edgecolor="black", alpha=0.8)
            for i in range(2):
                ax.plot(xs, weights[i] * sp_norm.pdf(xs, means[i], np.sqrt(vars_[i])),
                        linewidth=2, label=f"Component {i} (μ={means[i]:.3f})")
            if np.isfinite(cut):
                ax.axvline(cut, color="black", linestyle="--", linewidth=1,
                           label=f"P(hi)≥{posterior_threshold_bimodal:.2f} @ {cut:.3f}")
            ax.text(
                0.02, 0.98,
                f"{info['method']}\ncalled={called_local.sum()}/{s.size} ({100*called_local.mean():.2f}%)",
                transform=ax.transAxes, ha="left", va="top", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85),
            )
            ax.set_title(title, fontweight="bold")
            ax.set_xlabel("Signature score"); ax.set_ylabel("Density")
            ax.legend(frameon=False, fontsize=8)
            plt.tight_layout(); plt.show()

        return mask, info

    # ---- unimodal bulk + right-tail calling ----
    tail_cut = float(np.quantile(s, tail_q))
    in_tail = s >= tail_cut

    if in_tail.sum() < 10 or (~in_tail).sum() < 10:
        info.update({
            "status": "tail_or_bulk_too_small",
            "method": f"tail (sep={sep:.2f} < {sep_threshold})",
            "sep": sep,
            "tail_q": float(tail_q),
            "tail_cut": tail_cut,
            "n_tail": int(in_tail.sum()),
            "n_bulk": int((~in_tail).sum()),
        })
        if show:
            fig, ax = plt.subplots(figsize=figsize)
            ax.hist(s, bins=bins, density=True, color="lightgrey", edgecolor="black", alpha=0.8)
            ax.axvline(tail_cut, color="black", linestyle=":", linewidth=1,
                       label=f"tail_q={tail_q:.2f} @ {tail_cut:.3f}")
            ax.set_title(title + " (tail/bulk too small)", fontweight="bold")
            ax.set_xlabel("Signature score"); ax.set_ylabel("Density")
            ax.legend(frameon=False, fontsize=8)
            plt.tight_layout(); plt.show()
        return mask, info

    bulk = s[~in_tail]
    bulk_mu = float(np.mean(bulk))
    bulk_sd = float(np.std(bulk) + 1e-12)

    gmm_tail = GaussianMixture(n_components=2, random_state=random_state).fit(s[in_tail].reshape(-1, 1))
    means_t = gmm_tail.means_.ravel()
    vars_t = gmm_tail.covariances_.ravel()
    weights_t = gmm_tail.weights_.ravel()
    hi = int(np.argmax(means_t))  # RIGHT-most tail component

    post_hi_all = gmm_tail.predict_proba(X)[:, hi]
    called_local = in_tail & (post_hi_all >= posterior_threshold_tail)

    post_grid = gmm_tail.predict_proba(xs.reshape(-1, 1))[:, hi]
    cross = np.where((xs >= tail_cut) & (post_grid >= posterior_threshold_tail))[0]
    cut = float(xs[cross[0]]) if cross.size else np.nan

    mask[np.where(finite)[0]] = called_local

    info.update({
        "method": f"tail 2-GMM (sep={sep:.2f} < {sep_threshold})",
        "sep": sep,
        "tail_q": float(tail_q),
        "tail_cut": tail_cut,
        "means_tail": [float(m) for m in means_t],
        "hi_tail_component": hi,
        "posterior_threshold": float(posterior_threshold_tail),
        "cut": cut,
        "n_called": int(mask.sum()),
        "n_total_finite": int(s.size),
        "n_tail": int(in_tail.sum()),
    })

    if show:
        fig, ax = plt.subplots(figsize=figsize)
        ax.hist(s, bins=bins, density=True, color="lightgrey", edgecolor="black", alpha=0.8)

        # Bulk Gaussian
        ax.plot(xs, sp_norm.pdf(xs, bulk_mu, bulk_sd),
                linewidth=2, label=f"Bulk Gaussian (μ={bulk_mu:.3f})")

        # Right-tail component only
        ax.plot(xs, weights_t[hi] * sp_norm.pdf(xs, means_t[hi], np.sqrt(vars_t[hi])),
                linewidth=2, label=f"Right tail comp (μ={means_t[hi]:.3f})")

        ax.axvline(tail_cut, color="black", linestyle=":", linewidth=1,
                   label=f"tail_q={tail_q:.2f} @ {tail_cut:.3f}")

        if np.isfinite(cut):
            ax.axvline(cut, color="black", linestyle="--", linewidth=1,
                       label=f"P(hi)≥{posterior_threshold_tail:.2f} @ {cut:.3f}")

        ax.text(
            0.02, 0.98,
            f"{info['method']}\ncalled={called_local.sum()}/{s.size} ({100*called_local.mean():.2f}%)",
            transform=ax.transAxes, ha="left", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85),
        )

        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Signature score"); ax.set_ylabel("Density")
        ax.legend(frameon=False, fontsize=8)
        plt.tight_layout(); plt.show()

    return mask, info


# ---------------------- pyUCell scoring + label refinement ----------------------

def add_signature_annotation(
    obj: Union[AnnData, Any],
    *,
    layer: str = "Normalized_reads",
    positive_markers: List[str],
    negative_markers: Optional[Union[List[str], str]] = None,
    cell_type_label: str,
    score_key: Optional[str] = None,
    label_in: str = "Averaged.Detailed.Celltype",
    label_out: str = "Averaged.Detailed.Celltype.Refined",
    max_rank: Optional[int] = None,  # default: round(n_features/2)
    # --- adaptive GMM params ---
    sep_threshold: float = 1.25,
    posterior_threshold_bimodal: float = 0.90,
    posterior_threshold_tail: float = 0.90,
    tail_q: float = 0.80,
    gmm_random_state: int = 42,
    # --- plotting ---
    plot_gmm: bool = False,
    plot_title: Optional[str] = None,
    figsize=(4.35, 4.35),
    bins: int = 60,
    verbose: bool = False,
) -> Union[AnnData, Any]:
    """
    1) Compute pyUCell score:
         score = UCell(pos) - UCell(neg)   (if neg provided/non-empty)
         score = UCell(pos)                (if neg empty/None)
    2) Call positives with adaptive GMM (bimodal vs unimodal+right-tail)
    3) Write score + refined labels
    4) Optionally plot the GMM fit
    """
    df = _get_df(obj, layer=layer)
    n_cells, n_feats = df.shape

    if max_rank is None:
        max_rank = int(np.round(n_feats / 2.0))
    max_rank = max(1, min(int(max_rank), n_feats))

    # Temporary AnnData for pyUCell
    ad = sc.AnnData(X=df.to_numpy(dtype=float))
    ad.obs_names = df.index.astype(str)
    ad.var_names = df.columns.astype(str)

    # normalize negative_markers input
    if negative_markers is None:
        negative_markers_list: List[str] = []
    elif isinstance(negative_markers, str):
        negative_markers_list = [m for m in negative_markers.split(",") if m.strip()]
    else:
        negative_markers_list = list(negative_markers)

    pos_keep = [m for m in (positive_markers or []) if m in ad.var_names]
    neg_keep = [m for m in (negative_markers_list or []) if m in ad.var_names]

    if len(pos_keep) == 0:
        raise ValueError(f"[{cell_type_label}] None of the positive markers were found in features.")

    uc.compute_ucell_scores(ad, signatures={f"{cell_type_label}_pos": pos_keep}, max_rank=max_rank)
    pos_col = [c for c in ad.obs.columns if f"{cell_type_label}_pos" in c and "UCell" in c]
    pos_col = pos_col[-1]
    pos_score = ad.obs[pos_col].to_numpy(dtype=float)

    if len(neg_keep) > 0:
        uc.compute_ucell_scores(ad, signatures={f"{cell_type_label}_neg": neg_keep}, max_rank=max_rank)
        neg_col = [c for c in ad.obs.columns if f"{cell_type_label}_neg" in c and "UCell" in c]
        neg_col = neg_col[-1]
        neg_score = ad.obs[neg_col].to_numpy(dtype=float)
        score = pos_score - neg_score
    else:
        score = pos_score

    score_key = score_key or f"{cell_type_label}_signature_score"
    _set_vec(obj, score_key, score)

    # adaptive calling (+ optional plot)
    mask, info = call_tail_or_bimodal_gmm(
        score,
        random_state=gmm_random_state,
        posterior_threshold_bimodal=posterior_threshold_bimodal,
        posterior_threshold_tail=posterior_threshold_tail,
        tail_q=tail_q,
        sep_threshold=sep_threshold,
        bins=bins,
        title=plot_title or f"{cell_type_label} signature score — adaptive GMM",
        figsize=figsize,
        show=plot_gmm,
    )

    base = _get_vec(obj, label_in)
    if base is None:
        base = _get_vec(obj, "CommonDetailed.Celltype")
        if base is None:
            base = np.array(["Unknown"] * n_cells, dtype=object)

    refined = np.asarray(base, dtype=object).copy()
    refined[mask] = cell_type_label
    _set_vec(obj, label_out, refined)

    if verbose:
        print(f"[{cell_type_label}] max_rank={max_rank} wrote score='{score_key}' labels='{label_out}'")
        print(f"[{cell_type_label}] {info}")

    return obj

def add_mast_annotation(
    obj: Union[AnnData, Any],
    *,
    layer: str = "Normalized_reads",
    max_rank: Optional[int] = None,
    score_key: str = "Mast_signature_score",
    label_in: str = "Averaged.Detailed.Celltype",
    label_out: str = "Averaged.Detailed.Celltype.Refined",
    sep_threshold: float = 1.25,
    posterior_threshold_bimodal: float = 0.90,
    posterior_threshold_tail: float = 0.90,
    tail_q: float = 0.80,
    gmm_random_state: int = 42,
    plot_gmm: bool = False,
    plot_title: Optional[str] = None,     # <-- NEW
    figsize=(4.35, 4.35),
    bins: int = 60,
    verbose: bool = False,
) -> Union[AnnData, Any]:
    """Mast wrapper around add_signature_annotation (with optional GMM plot)."""
    return add_signature_annotation(
        obj=obj,
        layer=layer,
        positive_markers=MAST_POS,
        negative_markers=MAST_NEG,
        cell_type_label="Mast",
        score_key=score_key,
        label_in=label_in,
        label_out=label_out,
        max_rank=max_rank,
        sep_threshold=sep_threshold,
        posterior_threshold_bimodal=posterior_threshold_bimodal,
        posterior_threshold_tail=posterior_threshold_tail,
        tail_q=tail_q,
        gmm_random_state=gmm_random_state,
        plot_gmm=plot_gmm,
        plot_title=plot_title or "Mast signature score — adaptive GMM",  # <-- NEW
        figsize=figsize,
        bins=bins,
        verbose=verbose,
    )