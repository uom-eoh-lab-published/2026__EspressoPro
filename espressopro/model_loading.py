# -*- coding: utf-8 -*-
"""Core utilities for model loading and data management."""

from __future__ import annotations

import os
import sys
import shutil
import tarfile
import tempfile
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Union

import joblib
import pandas as pd

ATLAS_NAME = "TotalSeqD_Heme_Oncology_CAT399906"
MODELS_SUBPATH = Path("Pre_trained_models") / ATLAS_NAME


def download_models(
    *,
    force: bool = False,
    url: str = "https://huggingface.co/EspressoKris/EspressoPro/resolve/main/Pre_Trained_Models_For_TotalSeqD_Heme_Oncology_CAT399906.tar.xz",
    local_archive: Optional[str] = None,
) -> Path:
    """
    Download and extract pre-trained models under <pkg>/data/Pre_trained_models/<ATLAS_NAME>.
    Returns the package data directory.

    Notes
    -----
    - Downloads from a direct HTTPS link (Hugging Face).
    - Supports .tar, .tar.gz, .tar.xz, and .zip archives.
    - Safe extraction (prevents path traversal).
    """

    script_dir = Path(__file__).parent.resolve()
    data_dir = script_dir / "data"
    models_root = data_dir / MODELS_SUBPATH

    if not force and models_root.exists() and any(models_root.rglob("Multiclass_models.joblib")):
        print("[download_models] Models already present.")
        return data_dir

    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "Pre_trained_models").mkdir(parents=True, exist_ok=True)

    def _safe_extract_tar(tar: tarfile.TarFile, path: Path) -> None:
        base = path.resolve()
        for m in tar.getmembers():
            target = (path / m.name).resolve()
            if not str(target).startswith(str(base)):
                raise RuntimeError(f"Blocked path traversal in tar member: {m.name}")
        tar.extractall(path)

    def _safe_extract_zip(zf: zipfile.ZipFile, path: Path) -> None:
        base = path.resolve()
        for m in zf.infolist():
            target = (path / m.filename).resolve()
            if not str(target).startswith(str(base)):
                raise RuntimeError(f"Blocked path traversal in zip member: {m.filename}")
        zf.extractall(path)

    def _copy_dir(src: Path, dst: Path) -> None:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dst, dirs_exist_ok=True)

    def _merge_into_data(extracted_root: Path) -> None:
        """
        Merge extracted payload into <pkg>/data such that models end up at:
            <pkg>/data/Pre_trained_models/<ATLAS_NAME>

        Supports archives that contain any of:
          1) data/Pre_trained_models/<ATLAS_NAME>/...
          2) Pre_trained_models/<ATLAS_NAME>/...
          3) <ATLAS_NAME>/...
          4) (legacy) arbitrary folder layout that should be copied into data/
        """
        # If archive has a top-level "data" folder, treat that as the extraction root.
        candidates = [p for p in extracted_root.iterdir() if p.is_dir() and p.name.lower() == "data"]
        roots = candidates or [extracted_root]

        for root in roots:
            # Case 1: data/Pre_trained_models/<ATLAS_NAME>
            p1 = root / "Pre_trained_models" / ATLAS_NAME
            if p1.is_dir():
                _copy_dir(p1, data_dir / "Pre_trained_models" / ATLAS_NAME)
                continue

            # Case 2: Pre_trained_models/<ATLAS_NAME>
            p2 = root / "Pre_trained_models" / ATLAS_NAME
            if p2.is_dir():
                _copy_dir(p2, data_dir / "Pre_trained_models" / ATLAS_NAME)
                continue

            # Case 3: <ATLAS_NAME> at root
            p3 = root / ATLAS_NAME
            if p3.is_dir():
                _copy_dir(p3, data_dir / "Pre_trained_models" / ATLAS_NAME)
                continue

            # Fallback: copy children into data/ (legacy behavior)
            for child in root.iterdir():
                dest = data_dir / child.name
                if child.is_dir():
                    shutil.copytree(child, dest, dirs_exist_ok=True)
                else:
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(child, dest)

    def _extract_archive(archive_path: Path) -> None:
        print(f"[download_models] Extracting: {archive_path}")
        with tempfile.TemporaryDirectory() as tdir:
            out = Path(tdir) / "extract"
            out.mkdir(parents=True, exist_ok=True)
            try:
                if tarfile.is_tarfile(archive_path):
                    with tarfile.open(archive_path, "r:*") as tar:
                        _safe_extract_tar(tar, out)
                elif zipfile.is_zipfile(archive_path):
                    with zipfile.ZipFile(archive_path) as zf:
                        _safe_extract_zip(zf, out)
                else:
                    raise RuntimeError("Unknown archive format (not tar/zip).")
            except Exception as e:
                raise RuntimeError(f"Failed to extract archive: {e}") from e

            _merge_into_data(out)

    if local_archive:
        p = Path(local_archive)
        if not p.exists():
            raise FileNotFoundError(f"Local archive not found: {p}")
        _extract_archive(p)
    else:
        try:
            with tempfile.TemporaryDirectory() as tdir:
                tmpfile = Path(tdir) / "models.archive"
                print(f"[download_models] Downloading pre-trained models from {url} ...")
                with urllib.request.urlopen(url) as resp, open(tmpfile, "wb") as fh:
                    total = getattr(resp, "length", None) or 0
                    read = 0
                    block = 1024 * 1024
                    while True:
                        chunk = resp.read(block)
                        if not chunk:
                            break
                        fh.write(chunk)
                        read += len(chunk)
                        if total:
                            pct = 100 * read / total
                            sys.stdout.write(f"\r  {read/1e6:8.1f} MB / {total/1e6:8.1f} MB ({pct:5.1f}%)")
                            sys.stdout.flush()
                    if total:
                        sys.stdout.write("\n")

                if not tmpfile.exists() or tmpfile.stat().st_size == 0:
                    raise RuntimeError("Download produced an empty file.")

                _extract_archive(tmpfile)

        except Exception as e:
            print(f"[download_models] Failed to download models: {e}")
            print("[download_models] Place the extracted folder at:")
            print(f"  {models_root}")
            print("Or re-run with a local archive:")
            print("  download_models(local_archive='/abs/path/Pre_Trained_Models_For_TotalSeqD_Heme_Oncology_CAT399906.tar.xz')")

    if models_root.exists() and any(models_root.rglob("Multiclass_models.joblib")):
        print(f"[download_models] Models ready at {models_root}")
    else:
        print("[download_models] Models not found after extraction.")
    return data_dir


def _candidate_models_dirs() -> list[Path]:
    """Likely locations for …/Pre_trained_models/<ATLAS_NAME>."""
    here = Path(__file__).parent.resolve()
    pkg_data = here / "data"
    repo_data = here.parent / "data"
    repo_Data = here.parent / "Data"
    return [
        pkg_data / MODELS_SUBPATH,
        repo_data / MODELS_SUBPATH,
        repo_Data / MODELS_SUBPATH,
        Path.home() / ".espressopro" / MODELS_SUBPATH,
    ]


def ensure_models_available(*, local_archive: Optional[str] = None, force: bool = False) -> Path:
    """
    Ensure models exist; attempt to download if missing.

    Returns
    -------
    Path
        The *data directory* that contains Pre_trained_models/ (i.e., parent of Pre_trained_models).
    """
    # 1) Explicit models location
    env_models = os.environ.get("ESPRESSOPRO_MODELS")
    if env_models:
        p = Path(env_models).expanduser().resolve()

        # Accept:
        #   .../Pre_trained_models/<ATLAS_NAME>
        #   .../Pre_trained_models
        #   .../<data_dir>   (that contains Pre_trained_models/<ATLAS_NAME>)
        if p.is_dir():
            if p.name == ATLAS_NAME and p.parent.name == "Pre_trained_models":
                data_dir = p.parent.parent
                if (data_dir / MODELS_SUBPATH).exists():
                    return data_dir

            if p.name == "Pre_trained_models":
                data_dir = p.parent
                if (data_dir / MODELS_SUBPATH).exists():
                    return data_dir

            # If user points at a data dir
            if (p / MODELS_SUBPATH).exists():
                return p

        print(f"[ensure_models_available] ESPRESSOPRO_MODELS set but unusable: {p}")

    # 2) Look in common candidate locations
    for c in _candidate_models_dirs():
        if c.exists() and any(c.rglob("Multiclass_models.joblib")):
            return c.parent.parent  # .../<data_dir>

    # 3) Explicit data location
    env_data = os.environ.get("ESPRESSOPRO_DATA")
    if env_data:
        d = Path(env_data).expanduser().resolve()
        c = d / MODELS_SUBPATH
        if c.exists() and any(c.rglob("Multiclass_models.joblib")):
            return d
        else:
            print(f"[ensure_models_available] ESPRESSOPRO_DATA set but models not found under: {d}")

    # 4) Download into package data dir and re-check
    data_dir = download_models(local_archive=local_archive, force=force)

    for c in _candidate_models_dirs():
        if c.exists() and any(c.rglob("Multiclass_models.joblib")):
            return c.parent.parent

    raise FileNotFoundError(
        "Models directory not found.\n"
        "• Set ESPRESSOPRO_MODELS to …/Pre_trained_models/{ATLAS_NAME} (or …/Pre_trained_models or the data dir)\n"
        "• Or set ESPRESSOPRO_DATA to the parent data directory that contains Pre_trained_models/\n"
        "• Or pass explicit paths to generate_predictions(..., models_path=..., data_path=...)\n"
        "• Or use download_models(local_archive='…')"
    )


def get_default_models_path() -> Path:
    """Return …/data/Pre_trained_models/<ATLAS_NAME>."""
    data_dir = ensure_models_available()
    p = data_dir / MODELS_SUBPATH
    if not p.exists():
        raise FileNotFoundError(f"Expected models at {p} but not found.")
    return p


def get_default_data_path() -> Path:
    """Return the default *data directory* (parent that contains Pre_trained_models/)."""
    return ensure_models_available()


def get_package_data_path() -> Path:
    """
    Resolve the package data directory using:
      1) $ESPRESSOPRO_DATA
      2) importlib.resources
      3) pkg_resources
      4) ./data next to this file
      5) ensure_models_available()
    """
    env = os.getenv("ESPRESSOPRO_DATA")
    if env:
        p = Path(env).expanduser().resolve()
        if p.is_dir():
            return p

    try:
        import importlib.resources as resources
        p = Path(resources.files("espressopro") / "data")  # type: ignore[arg-type]
        if p.is_dir():
            return p
    except Exception:
        pass

    try:
        import pkg_resources  # noqa: F401
        p = Path(pkg_resources.resource_filename("espressopro", "data"))  # type: ignore[name-defined]
        if p.is_dir():
            return p
    except Exception:
        pass

    here_data = Path(__file__).resolve().parent / "data"
    if here_data.is_dir():
        return here_data

    package_root = Path(__file__).resolve().parent.parent
    repo_data = (package_root / "data").resolve()
    if repo_data.is_dir():
        return repo_data

    print("[get_package_data_path] Data directory not found, attempting download...")
    return ensure_models_available()


def load_models(
    models_path: Union[str, Path],
    model_names: Sequence[str] = ("Hao", "Zhang", "Triana", "Luecken"),
    annotation_depth: Sequence[str] = ("Broad", "Simplified", "Detailed"),
) -> Dict[str, Mapping]:
    """
    Load pre-trained models from the *TotalSeqD_Heme_Oncology_CAT399906* layout:

        <models_path>/<atlas>/Release/<Depth>/Models/Multiclass_models.joblib

    There is a single multiclass bundle per depth per atlas.
    """

    def _safe_load_joblib(path: Path):
        try:
            if (
                not path.is_file()
                or path.name.startswith("._")
                or path.name == ".DS_Store"
                or path.stat().st_size == 0
            ):
                return None
        except Exception:
            return None
        try:
            return joblib.load(path)
        except Exception as e:
            print(f"[load_models] failed to load {path}: {e}")
            return None

    def _normalize_bundle(bundle: object) -> dict:
        """
        Normalize the loaded bundle into a dict-like entry that downstream code
        can access consistently.
        """
        out: dict = {"__BUNDLE__": bundle}

        if isinstance(bundle, dict):
            # Common "model" carriers
            mdl = (
                bundle.get("model")
                or bundle.get("Stacked")
                or bundle.get("stacked")
                or bundle.get("clf")
                or bundle.get("classifier")
            )
            if mdl is not None:
                out["model"] = mdl
                out.setdefault("Stacked", mdl)

            # Optional metadata that may exist in your bundle
            if "class_names" in bundle and bundle["class_names"] is not None:
                out["class_names"] = list(map(str, bundle["class_names"]))
            if "excluded_classes" in bundle and bundle["excluded_classes"] is not None:
                out["excluded_classes"] = list(map(str, bundle["excluded_classes"]))

            # Temperature scaler often present either nested or as a key
            for k in ("multiclass_temp_scaler", "temp_scaler", "temperature_scaler", "temperature"):
                if k in bundle and bundle[k] is not None:
                    out["temp_scaler"] = bundle[k]
                    break
        else:
            # If joblib is not a dict, still expose as model
            out["model"] = bundle
            out.setdefault("Stacked", bundle)

        return out

    models: Dict[str, dict] = defaultdict(lambda: defaultdict(dict))
    root = Path(models_path)

    for atlas in model_names:
        atlas_root = root / atlas

        for depth in annotation_depth:
            # Your exact layout:
            # <atlas>/Release/<Depth>/Models/Multiclass_models.joblib
            bundle_path = atlas_root / "Release" / depth / "Models" / "Multiclass_models.joblib"

            if not bundle_path.exists():
                # Be explicit so users can immediately see what path was checked
                print(f"[load_models] missing bundle: {bundle_path}")
                continue

            bundle_obj = _safe_load_joblib(bundle_path)
            if bundle_obj is None:
                continue

            # Store under a stable key per depth, plus convenience projections
            models[atlas][depth]["__MULTICLASS__"] = _normalize_bundle(bundle_obj)
            models[atlas][depth].update(models[atlas][depth]["__MULTICLASS__"])

    return models
