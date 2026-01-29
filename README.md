# EspressoPro

[![PyPI version](https://badge.fury.io/py/espressopro.svg)](https://badge.fury.io/py/espressopro)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

**EspressoPro** is an automated, ontology-guided cell type annotator for single-cell surface protein (ADT) data, designed for Mission Bio Tapestri DNA+ADT assays and similar protein-only platforms. It uses pre-trained, per–cell type stacked models from four label-harmonised CITE-seq reference atlases of healthy blood and bone marrow cells. Predictions are combined across references and constrained by a cellular ontology to produce consistent immune and progenitor cell annotations, including rare populations.

## Install

**Latest (GitHub):**
```bash
pip install git+https://github.com/uom-eoh-lab-published/2026__EspressoPro.git
```

**Create a stable MissionBio Environment:**
```bash
conda create -n mosaic -c missionbio -c conda-forge   python=3.10 missionbio.mosaic-base=3.12.2 python-kaleido -y
conda activate mosaic
pip install git+https://github.com/uom-eoh-lab-published/2026__EspressoPro.git
```

In alternative used the same provided within https://github.com/uom-eoh-lab-published/2026__EspressoPro/blob/main/tutorials/environments/mosaic_2.yml  

## Quick start

### MissionBio Sample
```python
import missionbio.mosaic as ms
import espressopro as ep

# Load and preprocess data
sample = ms.load("sample.h5")  
ep.Normalise_protein_data(sample, inplace=True, axis=1, flavor="seurat")
ep.Scale_protein_data(sample, inplace=True)

# Dimensionality reduction and clustering
sample.protein.run_pca(
    attribute='Normalized_reads', 
    components=8, show_plot=False, random_state=42, svd_solver='randomized')  
sample.protein.run_umap(
    attribute='pca', 
    random_state=42, n_neighbors=50, min_dist=0.1, spread=8, n_components=2)  
sample.protein.cluster(
    attribute='umap', 
    method='graph-community', k=5, random_state=42)  

# Cell type annotation
sample = ep.generate_predictions(obj=sample)  
sample = ep.annotate_data(obj=sample)

# Optional: rename clusters with less than N cells to Small (potential noise)
sample = ep.mark_small_clusters(
    sample,
    label_in="Averaged.Detailed.Celltype",
    label_out="Averaged.Detailed.Celltype.Small",
    cluster_col="Clusters",
    min_cells=3,
)

# Optional: rename clusters with no single annotation as mixed according to label frequencies
sample = ep.mark_mixed_clusters(
    sample,
    label_in="Averaged.Detailed.Celltype",
    label_out="Averaged.Detailed.Celltype.Mixed",
    cluster_col="Clusters",
    min_frequency_threshold=0.35,
)

# Optional: expand cell type labels to clusters and rename mixed if not done earlier
sample, summary, pivot = ep.suggest_cluster_celltype_identity(
    sample=sample,
    dominance_threshold=0.35,
    label_in="Averaged.Detailed.Celltype",
    cluster_col="Clusters",
    label_out="Averaged.Detailed.Celltype.Refined"
)

# Optional: add predefined celltype-signature-based calls
sample = ep.add_mast_annotation(
    sample,
    layer="Normalized_reads",
    label_in='Averaged.Detailed.Celltype',
    label_out='Averaged.Detailed.Celltype.Custom',
)

# Optional: add custom celltype-signature-based calls
sample = ep.add_signature_annotation(
    sample,
    layer="Normalized_reads",
    label_in='Averaged.Detailed.Celltype',
    label_out='Averaged.Detailed.Celltype.Custom',
    positive_markers=["CD14", "CD33", "CD11b", "CD64"],
    negative_markers="",
    cell_type_label="CD14_Mono"
)

```

**Full tutorial:** [MissionBio_Tapestri.ipynb](https://github.com/uom-eoh-lab-published/2026__EspressoPro/blob/main/tutorials/MissionBio_Tapestri.ipynb)

## Repositories  

- Model: <https://huggingface.co/EspressoKris/EspressoPro>  

## Documentation

- Manuscript: <https://github.com/uom-eoh-lab-published/2026__EspressoPro_Manuscript>

## Citation
```bibtex
@software{espressopro,
  title   = {EspressoPro: An Automated Machine Learning Driven Protein Annotator For Tapestri Data},
  author  = {Gurashi, Kristian},
  year    = {2026},
  url     = {https://github.com/uom-eoh-lab-published/2026__EspressoPro},
  version = {1.0.0}
}
```

## License

CC © Kristian Gurashi. See [LICENSE](LICENSE).
