# `espressopro/` package layout

This folder is the `espressopro` Python package. This README is a map of what
each module does, so you don't have to open every file to remember where a
given piece of the pipeline lives. See the top-level project `README.md` for
installation and a runnable quick-start example.

## Files

### `__init__.py`
Package entry point. Re-exports the public API (`load_models`,
`download_models`, `annotate_data`, `Normalise_protein_data`,
`suggest_cluster_celltype_identity`, the class-map constants, etc.) so users
can do `import espressopro as ep` and call `ep.<function>` directly. Several
imports (from `prediction.py`) are wrapped in `try/except` and gated behind
`_HAS_*` flags, since not every function is guaranteed to exist across every
package version — the `__all__` list is built dynamically based on what
actually imported successfully. Also defines `__version__`.

### `model_loading.py`
Downloading, resolving, and loading the pre-trained model bundles that power
prediction. This is the module `ep.download_models()`, `ep.load_models()`,
`ep.ensure_models_available()`, `ep.get_default_models_path()`, and related
path-resolution helpers live in. Models are fetched from the
`EspressoKris/EspressoPro` Hugging Face repository as a `.tar.xz`/`.zip`
archive and extracted into `espressopro/data/Pre_trained_models/<ATLAS_NAME>/`.
`download_models(model_data=...)` accepts `"latest"` or a dated version
string (e.g. `"260714"`) to pick which release to fetch; `ATLAS_NAME`,
`DATA_VERSION`, and `DATA_ARCHIVE_NAME` are exported from here.

### `prediction.py`
Turns raw per-cell classifier outputs into the `*.predscore` columns that
`annotation.py` consumes. Loads the `Multiclass_models.joblib` bundles (via
`model_loading.py`), scores cells per atlas/depth (`generate_predictions`),
audits feature overlap between the query data and what a model was trained on
(`audit_feature_overlap`), and blends predictions across the four reference
atlases (Hao/Zhang/Triana/Luecken) into `Averaged.*` tracks — including
agreement-weighted blending (`add_averaged_tracks*`) and a stricter
high-confidence-only variant (`add_averaged_consensus_tracks`). Also home to
spatially-aware "best localised" tracks (`add_best_localised_tracks`) for
Xenium-style data.

### `annotation.py`
Consumes the `*.predscore` columns produced by `prediction.py` and turns them
into final cell-type calls, enforcing the Broad → Simplified → Detailed
ontology at each step (a cell called "Mature" at Broad can only win a mature
Simplified class, etc.). Contains the core voting engine
(`voting_annotator`), the per-level annotators (`Broad_Annotation`,
`Simplified_Annotation`, `Detailed_Annotation`, and their per-atlas
`Atlas_*_Annotation` counterparts), the higher-level `annotate_data()`
convenience wrapper, and post-hoc cluster cleanup helpers
(`mark_small_clusters`, `mark_mixed_clusters`, `score_mixed_clusters`,
`clear_annotation`).

### `constants.py`
Static lookup tables shared by `annotation.py` and `prediction.py`: the
Simplified/Detailed class → source-column maps (`SIMPLIFIED_CLASSES`,
`DETAILED_CLASSES`), the ontology parent → allowed-child maps used to
constrain votes (`SIMPLIFIED_PARENT_MAP`, `DETAILED_PARENT_MAP`), and the
mast-cell marker signature (`MAST_POS`/`MAST_NEG`) used by `markers.py`.

### `markers.py`
Marker/signature-based annotation as a complement to the model-based
pipeline — currently used for mast cell calling (`add_mast_annotation`) and
general gene/protein signature scoring via pyUCell (`add_signature_annotation`).
Includes an adaptive Gaussian-mixture-model caller
(`call_tail_or_bimodal_gmm`) that fits either a unimodal-tail or bimodal
distribution to a signature score to pick a positivity cutoff, with optional
diagnostic plotting.

### `protein_preprocessing.py`
ADT/protein-count preprocessing: CLR (centred log-ratio) normalization
(`clr_transform_matrix`, `Normalise_protein_data`) and scaling
(`Scale_protein_data`). Works across MissionBio `Sample` objects, `AnnData`,
plain `DataFrame`s, and raw (sparse or dense) arrays.

### `missionbio.py`
MissionBio Tapestri-specific spatial/cluster QC helpers that sit downstream
of annotation: `reassign_disconnected_cells` finds cells whose call is
spatially isolated from the rest of that cell type (via a kNN connectivity
graph) and reassigns them; `suggest_cluster_celltype_identity` and
`print_cluster_suggestions` summarize per-cluster label composition;
`print_disconnected_summary` reports on what `reassign_disconnected_cells`
changed.

### `core.py`
**Not imported by anything else in the package** — nothing in `__init__.py`
or any other module references it. It's an earlier version of
`model_loading.py`: same purpose (download/resolve/load pre-trained models),
but an older data layout (`<atlas>/Models/<Depth>_<CellLabel>/*_Stacked.joblib`
instead of the current `<atlas>/Release/<Depth>/Models/Multiclass_models.joblib`)
and an older default download URL. It looks like dead code left over from
before `model_loading.py` was rewritten. Worth deleting if you confirm you
don't need it — happy to do that in a follow-up if you'd like.
