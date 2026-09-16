import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_samples
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

def reassign_disconnected_cells(
    sample,
    annotation,
    n_neighbors: int = 15,
    min_group_size: int = 10,
    reassignment_method: str = "nearest_neighbors",
    n_neighbors_reassign: int = 30,
    embedding_key: str = "pca",
    *,
    rewrite: bool = False,
    label_out: str = "spatially_refined_labels",
    verbose: bool = False,
):
    """
    Identify spatially disconnected cell groups and reassign small isolated clusters.
    
    This function finds cells that are labeled with a cell type but are spatially
    distant from the main group of that cell type (e.g., isolated CD14 monocytes
    far from the bulk CD14 monocyte population).
    
    Parameters
    ----------
    sample : missionbio.mosaic.Sample
        Sample with a protein assay.
    annotation : array-like or str
        Per-cell labels or the name of a row_attrs field.
    n_neighbors : int
        Number of neighbors to determine connectivity within same cell type.
        Default is 15.
    min_group_size : int
        Minimum size of a connected component to keep its original label.
        Smaller disconnected groups will be reassigned. Default is 10.
    reassignment_method : str
        Method for reassigning isolated cells:
        - "nearest_neighbors": Assign to most common label among nearest neighbors
        - "remove": Mark as "Uncertain" or similar
        Default is "nearest_neighbors".
    n_neighbors_reassign : int
        Number of neighbors to consider for reassignment (across all cell types).
        Default is 30.
    embedding_key : str
        Key in sample.protein.row_attrs for the embedding to use.
        Options: 'pca', 'umap'. Default is 'pca'.
    rewrite : bool
        If True, write reassigned labels to sample.protein.row_attrs[label_out].
    label_out : str
        Row attribute name used when rewrite=True.
    verbose : bool
        If True, print detailed information about reassignments.
        
    Returns
    -------
    np.ndarray or missionbio.mosaic.Sample
        If rewrite=False: returns (reassigned_labels, disconnected_info) tuple
        If rewrite=True: returns the modified sample object
    dict
        Information about disconnected components (only when rewrite=False)
    """
    # Get cluster labels
    if not hasattr(sample, "protein") or not hasattr(sample.protein, "get_labels"):
        raise TypeError("Expected a MissionBio Sample with protein.get_labels().")
    
    # Resolve per-cell annotation
    if isinstance(annotation, str):
        if not hasattr(sample.protein, "row_attrs") or annotation not in sample.protein.row_attrs:
            raise KeyError(f"'{annotation}' not found in sample.protein.row_attrs")
        ann_vec = np.asarray(sample.protein.row_attrs[annotation])
    else:
        ann_vec = np.asarray(annotation)
    
    # Get embedding from row_attrs
    if not hasattr(sample.protein, "row_attrs") or embedding_key not in sample.protein.row_attrs:
        raise KeyError(f"'{embedding_key}' not found in sample.protein.row_attrs")
    
    embedding = np.asarray(sample.protein.row_attrs[embedding_key])
    
    if verbose:
        print(f"Using '{embedding_key}' embedding with shape {embedding.shape}")
    
    # Copy annotations for modification
    reassigned_labels = ann_vec.copy()
    
    # Track disconnected components
    disconnected_info = {}
    total_reassigned = 0
    
    # For each unique cell type, find disconnected components
    unique_celltypes = np.unique(ann_vec)
    
    for celltype in unique_celltypes:
        # Get indices of cells with this label
        celltype_mask = ann_vec == celltype
        celltype_indices = np.where(celltype_mask)[0]
        n_cells = len(celltype_indices)
        
        if n_cells == 0:
            continue
        
        # Build connectivity graph for this cell type
        # Two cells are connected if they are within k-nearest neighbors of each other
        celltype_embedding = embedding[celltype_indices]
        
        # Use fewer neighbors than available cells
        k = min(n_neighbors + 1, n_cells)
        nbrs = NearestNeighbors(n_neighbors=k, metric='euclidean')
        nbrs.fit(celltype_embedding)
        
        # Get adjacency matrix (connectivity graph)
        adjacency = nbrs.kneighbors_graph(celltype_embedding, mode='connectivity')
        
        # Find connected components
        n_components, component_labels = connected_components(
            csgraph=adjacency, directed=False, return_labels=True
        )
        
        if n_components == 1:
            # All cells are connected
            if verbose:
                print(f"{celltype}: All {n_cells} cells are connected")
            continue
        
        # Identify component sizes
        component_sizes = np.bincount(component_labels)
        largest_component = np.argmax(component_sizes)
        
        # Find small disconnected components
        small_components = []
        for comp_id in range(n_components):
            if comp_id != largest_component and component_sizes[comp_id] < min_group_size:
                small_components.append(comp_id)
        
        if len(small_components) == 0:
            if verbose:
                print(f"{celltype}: {n_components} components, but all significant groups > {min_group_size} cells")
            continue
        
        # Get indices of cells in small disconnected components
        small_component_mask = np.isin(component_labels, small_components)
        cells_to_reassign = celltype_indices[small_component_mask]
        
        if verbose:
            print(f"\n{celltype}:")
            print(f"  Total cells: {n_cells}")
            print(f"  Connected components: {n_components}")
            print(f"  Largest component: {component_sizes[largest_component]} cells")
            print(f"  Small disconnected groups: {len(small_components)}")
            print(f"  Cells to reassign: {len(cells_to_reassign)}")
        
        disconnected_info[celltype] = {
            "total_cells": n_cells,
            "n_components": n_components,
            "largest_component_size": int(component_sizes[largest_component]),
            "cells_to_reassign": len(cells_to_reassign),
            "component_sizes": component_sizes.tolist(),
        }
        
        # Reassign these cells
        if reassignment_method == "nearest_neighbors":
            # Build nearest neighbors on full dataset
            nbrs_global = NearestNeighbors(
                n_neighbors=min(n_neighbors_reassign + 1, len(embedding)),
                metric='euclidean'
            )
            nbrs_global.fit(embedding)
            
            for idx in cells_to_reassign:
                # Find nearest neighbors across all cells
                distances, indices = nbrs_global.kneighbors([embedding[idx]])
                neighbor_indices = indices[0][1:]  # Exclude self
                
                # Get labels of neighbors
                neighbor_labels = ann_vec[neighbor_indices]
                
                # Find most common label (excluding current celltype ideally)
                unique_labels, counts = np.unique(neighbor_labels, return_counts=True)
                
                # Prefer labels different from current
                if len(unique_labels) > 1:
                    # Remove current celltype if it appears
                    mask = unique_labels != celltype
                    if mask.any():
                        unique_labels = unique_labels[mask]
                        counts = counts[mask]
                
                # Assign to most common
                most_common_idx = np.argmax(counts)
                new_label = unique_labels[most_common_idx]
                reassigned_labels[idx] = new_label
                
                if verbose and len(cells_to_reassign) <= 20:
                    print(f"    Cell {idx}: {celltype} → {new_label}")
            
            total_reassigned += len(cells_to_reassign)
            
        elif reassignment_method == "remove":
            # Mark as uncertain
            reassigned_labels[cells_to_reassign] = "Uncertain"
            total_reassigned += len(cells_to_reassign)
    
    if verbose:
        print(f"\nTotal cells reassigned: {total_reassigned} ({100 * total_reassigned / len(ann_vec):.2f}%)")
    
    # Optionally write back
    if not rewrite:
        return reassigned_labels, disconnected_info
    
    # Write to sample and return the modified sample
    sample.protein.row_attrs[label_out] = reassigned_labels
    return sample


def print_disconnected_summary(disconnected_info: dict) -> None:
    """Pretty-print the disconnected components information."""
    print("\n" + "=" * 70)
    print("SPATIALLY DISCONNECTED CELL GROUPS")
    print("=" * 70)
    
    for celltype, info in disconnected_info.items():
        print(f"\n{celltype}:")
        print(f"  Total cells: {info['total_cells']}")
        print(f"  Connected components found: {info['n_components']}")
        print(f"  Largest connected group: {info['largest_component_size']} cells "
              f"({100 * info['largest_component_size'] / info['total_cells']:.1f}%)")
        print(f"  Cells reassigned: {info['cells_to_reassign']}")
        
        # Show component size distribution
        sizes = sorted(info['component_sizes'], reverse=True)
        if len(sizes) > 1:
            print(f"  Component sizes: {sizes}")
    
    print("\n" + "=" * 70)

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


import numpy as np
import pandas as pd


def suggest_cluster_celltype_identity(
    sample,
    label_in: str = "Averaged.Detailed.Celltype",
    cluster_col: str = "cluster",
    dominance_threshold: float = 0.55,
    label_out: str = "annotated_clusters",
    rewrite: bool = True,
    verbose: bool = False,
):
    """
    Simple cluster -> celltype assignment (frequency-only).

    Rules
    -----
    For each cluster:
      - Compute label frequencies from row_attrs[label_in]
      - If top label frequency >= dominance_threshold:
            assign top label to all cells
        else:
            assign 'Mixed' to all cells

    No PCA, no kNN, no reclustering.
    """

    ra = sample.protein.row_attrs

    if label_in not in ra:
        raise KeyError(f"'{label_in}' not found in sample.protein.row_attrs")
    if cluster_col not in ra:
        raise KeyError(f"'{cluster_col}' not found in sample.protein.row_attrs")

    labels = np.asarray(ra[label_in], dtype=object)
    clusters = np.asarray(ra[cluster_col], dtype=object)

    n = len(labels)
    if len(clusters) != n:
        raise ValueError("annotation and cluster arrays must have same length")

    assigned = np.empty(n, dtype=object)
    summary = {}

    for cl in pd.unique(clusters):
        idx = np.where(clusters == cl)[0]
        labs = labels[idx]
        labs_valid = labs[pd.notna(labs)]

        if labs_valid.size == 0:
            suggested = "Unknown"
            top_label = None
            top_freq = np.nan
        else:
            uniq, cnt = np.unique(labs_valid.astype(str), return_counts=True)
            top_i = int(np.argmax(cnt))
            top_label = str(uniq[top_i])
            top_freq = float(cnt[top_i] / cnt.sum())

            suggested = (
                top_label
                if top_freq >= dominance_threshold
                else "Mixed"
            )

        assigned[idx] = suggested

        summary[cl] = {
            "suggested_celltype": suggested,
            "top_label": top_label,
            "top_freq": top_freq,
            "n_cells": int(idx.size),
        }

        if verbose:
            tf = f"{top_freq:.2%}" if np.isfinite(top_freq) else "NA"
            print(
                f"cluster={cl} n={idx.size} "
                f"top={top_label} ({tf}) -> {suggested}"
            )

    # Optional frequency table (cluster × label)
    freq_df = (
        pd.DataFrame({"cluster": clusters, "label": labels})
        .dropna(subset=["label"])
        .assign(label=lambda d: d["label"].astype(str))
        .groupby(["cluster", "label"])
        .size()
        .rename("count")
        .reset_index()
    )
    totals = freq_df.groupby("cluster")["count"].sum().rename("total").reset_index()
    freq_df = freq_df.merge(totals, on="cluster", how="left")
    freq_df["frequency"] = freq_df["count"] / freq_df["total"].replace(0, np.nan)

    pivot = (
        freq_df.pivot(index="cluster", columns="label", values="frequency")
        .fillna(0.0)
        .sort_index()
    )

    if rewrite:
        ra[label_out] = assigned
        return sample, summary, pivot

    return summary, pivot



def print_cluster_suggestions(summary: dict) -> None:
    """Pretty-print the mapping produced by suggest_cluster_celltype_identity()."""
    print("Cluster Cell Type Suggestions:")
    print("=" * 60)
    for cluster, info in summary.items():
        tf = info["top_freq"]
        tf_str = f"{tf:.2%}" if isinstance(tf, (float, np.floating)) and np.isfinite(tf) else "NA"
        cr = info["cross_label_edge_rate"]
        cr_str = f"{cr:.3f}" if isinstance(cr, (float, np.floating)) and np.isfinite(cr) else "NA"

        print(f"Cluster {cluster}:")
        print(f"  Suggested: {info['suggested_celltype']}")
        print(f"  Top label: {info['top_label']} ({tf_str})")
        print(f"  cross_rate: {cr_str}")
        print(f"  Total cells: {info['n_cells']}")
        print()
