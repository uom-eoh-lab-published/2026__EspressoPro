# -*- coding: utf-8 -*-
from __future__ import annotations

import copy
from typing import Literal, Optional, Union

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.sparse import issparse, isspmatrix, csr_matrix, csc_matrix
from scipy.stats import gmean
from warnings import warn

try:
    from scipy.sparse import csr_array, csc_array  # newer scipy
except Exception:  # pragma: no cover
    csr_array = ()
    csc_array = ()

try:
    from sklearn import preprocessing
except Exception:  # pragma: no cover
    preprocessing = None


def _asarray_dense(z):
    # robust conversion: works for scipy sparse outputs, numpy.matrix, and ndarrays
    if hasattr(z, "toarray"):
        return z.toarray()
    if hasattr(z, "A"):
        return z.A
    return np.asarray(z)

def clr_transform_matrix(
    X,
    axis: int = 0,
    flavor: Literal["seurat", "stoeckius", "standard"] = "seurat",
):
    if axis not in (0, 1):
        raise ValueError("axis must be 0 or 1")

    x = X

    if flavor == "seurat":
        if issparse(x) or isspmatrix(x):
            # ensure format
            if axis == 0 and not isinstance(x, (csc_matrix, csc_array) if csc_array else (csc_matrix,)):
                warn("X is sparse but not CSC. Converting to CSC for axis=0.")
                x = x.tocsc()
            elif axis == 1 and not isinstance(x, (csr_matrix, csr_array) if csr_array else (csr_matrix,)):
                warn("X is sparse but not CSR. Converting to CSR for axis=1.")
                x = x.tocsr()

            m = np.log1p(x).mean(axis=axis)
            scale = np.exp(_asarray_dense(m))  # <-- FIX

            x = x.copy()
            x.data /= np.repeat(scale, x.getnnz(axis=axis))
            np.log1p(x.data, out=x.data)
            return x

        # dense path
        x = np.asarray(x, dtype=float)
        np.log1p(x / np.exp(np.log1p(x).mean(axis=axis, keepdims=True)), out=x)
        return x

    elif flavor in ("stoeckius", "standard"):
        x = x.toarray() if (issparse(x) or isspmatrix(x)) else np.asarray(x, dtype=float)
        if flavor == "stoeckius":
            x = x + 1.0
        np.log(x / gmean(x, axis=axis, keepdims=True), out=x)
        return x

    else:
        raise ValueError(f"Unknown flavor '{flavor}'")

def Normalise_protein_data(
    data,
    inplace: bool = True,
    axis: int = 1,
    flavor: Literal["seurat", "stoeckius", "standard"] = "seurat",
    input_layer: str = "read_counts",
    output_layer: str = "Normalized_reads",
):
    """
    CLR-only normalization for protein counts.

    Supports:
    - MissionBio Sample with data.protein.layers[input_layer]
    - AnnData (adata.X)
    - pandas DataFrame
    - numpy array or sparse matrix
    """
    # MissionBio Sample
    is_mosaic_sample = hasattr(data, "protein") and hasattr(getattr(data, "protein"), "layers")
    if is_mosaic_sample and input_layer in data.protein.layers:
        X = data.protein.layers[input_layer]
        Xn = clr_transform_matrix(X, axis=axis, flavor=flavor)

        if inplace:
            data.protein.layers[output_layer] = Xn
            return None

        sample_copy = data.copy() if hasattr(data, "copy") else copy.deepcopy(data)
        sample_copy.protein.layers[output_layer] = Xn
        return sample_copy

    # AnnData
    if isinstance(data, AnnData):
        adata = data if inplace else data.copy()
        adata.X = clr_transform_matrix(adata.X, axis=axis, flavor=flavor)
        return None if inplace else adata

    # DataFrame
    if isinstance(data, pd.DataFrame):
        Xn = clr_transform_matrix(data.values.astype(float), axis=axis, flavor=flavor)
        return pd.DataFrame(Xn, index=data.index, columns=data.columns)

    # ndarray / sparse
    if isinstance(data, np.ndarray) or issparse(data) or isspmatrix(data):
        return clr_transform_matrix(data, axis=axis, flavor=flavor)

    raise ValueError(
        "Input must be a MissionBio Sample (protein.layers[input_layer]), AnnData, numpy array, DataFrame, or sparse."
    )


def Scale_protein_data(
    data,
    inplace: bool = True,
    input_layer: str = "Normalized_reads",
    output_layer: str = "Scaled_reads",
    with_mean: bool = True,
    with_std: bool = True,
):
    """
    StandardScaler on normalized protein data.

    - MissionBio Sample: reads protein.layers[input_layer] and writes output_layer.
    - AnnData: scales adata.X.
    - DataFrame/ndarray/sparse: returns scaled dense output (and DataFrame preserved if input was DataFrame).
    """
    if preprocessing is None:
        raise ImportError("scikit-learn is required for Scale_protein_data (StandardScaler).")

    scaler = preprocessing.StandardScaler(with_mean=with_mean, with_std=with_std)

    # MissionBio Sample
    is_mosaic_sample = hasattr(data, "protein") and hasattr(getattr(data, "protein"), "layers")
    if is_mosaic_sample:
        if input_layer not in data.protein.layers:
            raise ValueError(f"No '{input_layer}' layer found in Sample.protein.layers")
        X = data.protein.layers[input_layer]
        X = X.toarray() if (issparse(X) or isspmatrix(X)) else np.asarray(X, dtype=np.float64)
        scaled = scaler.fit_transform(X)

        if inplace:
            data.protein.layers[output_layer] = scaled
            return None

        sample_copy = data.copy() if hasattr(data, "copy") else copy.deepcopy(data)
        sample_copy.protein.layers[output_layer] = scaled
        return sample_copy

    # AnnData
    if isinstance(data, AnnData):
        adata = data if inplace else data.copy()
        X = adata.X
        X = X.toarray() if (issparse(X) or isspmatrix(X)) else np.asarray(X, dtype=np.float64)
        adata.X = scaler.fit_transform(X)
        return None if inplace else adata

    # DataFrame
    if isinstance(data, pd.DataFrame):
        X = data.values.astype(np.float64)
        scaled = scaler.fit_transform(X)
        return pd.DataFrame(scaled, index=data.index, columns=data.columns)

    # ndarray / sparse
    if isinstance(data, np.ndarray) or issparse(data) or isspmatrix(data):
        X = data.toarray() if (issparse(data) or isspmatrix(data)) else np.asarray(data, dtype=np.float64)
        return scaler.fit_transform(X)

    raise ValueError("Input must be AnnData, MissionBio Sample, numpy array, DataFrame, or sparse matrix")
