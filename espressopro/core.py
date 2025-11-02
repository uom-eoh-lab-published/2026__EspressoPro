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
    url: str = "https://huggingface.co/EspressoKris/EspressoPro/resolve/main/models.tar.xz",
    local_archive: Optional[str] = None,
) -> Path:
    """
    Download and extract pre-trained models under <pkg>/data/Pre_trained_models/<ATLAS_NAME>.
    Returns the package data directory.

    Notes
    -----
    - Downloads from a direct HTTPS link (Hugging Face); no Google Drive dependency.
    - Supports .tar, .tar.gz, .tar.xz, and .zip archives.
    - Safe extraction (prevents path traversal).
    """

    script_dir = Path(__file__).parent.resolve()
    data_dir = script_dir / "data"
    models_root = data_dir / MODELS_SUBPATH

    if not force and models_root.exists() and any(models_root.rglob("*_Stacked.joblib")):
        print("[download_models] Models already present.")
        return data_dir

    data_dir.mkdir(parents=True, exist_ok=True)

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

    def _merge_into_data(extracted_root: Path) -> None:
        candidates = [p for p in extracted_root.iterdir() if p.is_dir() and p.name.lower() == "data"]
        roots = candidates or [extracted_root]
        for root in roots:
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
                    total = resp.length or 0
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
            print("  download_models(local_archive='/abs/path/models.tar.xz')")

    if models_root.exists() and any(models_root.rglob("*_Stacked.joblib")):
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
    """Ensure models exist; attempt to download if missing. Returns the package data directory."""
    env_models = os.environ.get("ESPRESSOPRO_MODELS")
    if env_models:
        p = Path(env_models).expanduser()
        if p.exists():
            return p.parent.parent if p.name == ATLAS_NAME else p.parent
        else:
            print(f"[ensure_models_available] ESPRESSOPRO_MODELS set but not found: {p}")

    for c in _candidate_models_dirs():
        if c.exists() and any(c.rglob("*_Stacked.joblib")):
            return c.parent.parent

    data_dir = download_models(local_archive=local_archive, force=force)

    for c in _candidate_models_dirs():
        if c.exists() and any(c.rglob("*_Stacked.joblib")):
            return c.parent.parent

    env_data = os.environ.get("ESPRESSOPRO_DATA")
    if env_data:
        d = Path(env_data).expanduser()
        c = d / MODELS_SUBPATH
        if c.exists():
            return d

    raise FileNotFoundError(
        "Models directory not found.\n"
        "• Set ESPRESSOPRO_MODELS to …/Pre_trained_models/{ATLAS_NAME}\n"
        "• Or set ESPRESSOPRO_DATA to the parent 'data' directory\n"
        "• Or pass explicit paths to generate_predictions(..., models_path=..., data_path=...)\n"
        "• Or make the Drive file public / use download_models(local_archive='…')"
    )


def get_default_models_path() -> Path:
    """Return …/data/Pre_trained_models/<ATLAS_NAME>."""
    data_dir = ensure_models_available()
    p = data_dir / MODELS_SUBPATH
    if not p.exists():
        raise FileNotFoundError(f"Expected models at {p} but not found.")
    return p


def get_default_data_path() -> Path:
    """Return the default data path (co-located with models)."""
    return get_default_models_path()


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
    Load pre-trained models from:
      <models_path>/<atlas>/Models/<Depth>_<CellLabel>/
    Also loads optional per-depth or atlas-wide temperature scalers.
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
            print(f"[load_models] failed to load {path.name}: {e}")
            return None

    def _is_good_dir(p: Path) -> bool:
        return p.is_dir() and not (p.name.startswith("._") or p.name in {"__MACOSX", ".DS_Store"})

    def _good_joblibs(glob_iter):
        for p in sorted(glob_iter):
            try:
                if p.is_file() and p.suffix == ".joblib" and not p.name.startswith("._") and p.name != ".DS_Store" and p.stat().st_size > 0:
                    yield p
            except Exception:
                continue

    def _merge_bundle(into: dict, bundle_obj):
        if isinstance(bundle_obj, dict):
            mdl = bundle_obj.get("model", bundle_obj.get("Stacked"))
            if mdl is not None:
                into["model"] = mdl
                into.setdefault("Stacked", mdl)
            if "Stacked" in bundle_obj and "Stacked" not in into:
                into["Stacked"] = bundle_obj["Stacked"]
            for k in ("scaler", "standardizer", "StandardScaler", "preproc", "preprocessor", "transformer"):
                if bundle_obj.get(k) is not None:
                    into["scaler"] = bundle_obj[k]
                    break
            cols = (
                bundle_obj.get("columns")
                or bundle_obj.get("cols")
                or bundle_obj.get("feature_names")
                or bundle_obj.get("feature_names_in_")
            )
            if cols is not None:
                into["columns"] = list(map(str, cols))
        else:
            into["model"] = bundle_obj
            into.setdefault("Stacked", bundle_obj)

    models: Dict[str, dict] = defaultdict(lambda: defaultdict(dict))
    root = Path(models_path)

    for atlas in model_names:
        atlas_root = root / atlas
        atlas_models_dir = atlas_root / "Models"
        if not atlas_models_dir.exists():
            print(f"[load_models] atlas folder missing: {atlas_models_dir}")
            continue

        for depth in annotation_depth:
            ts_path = atlas_root / f"{depth}_multiclass_temp_scaler.joblib"
            ts_obj = _safe_load_joblib(ts_path)
            if ts_obj is not None:
                models[atlas][depth]["__TEMP_SCALER__"] = ts_obj
        ts_all = atlas_root / "multiclass_temp_scaler.joblib"
        ts_obj = _safe_load_joblib(ts_all)
        if ts_obj is not None:
            models[atlas]["__TEMP_SCALER__"] = ts_obj

        for cell_dir in sorted(atlas_models_dir.iterdir()):
            if not _is_good_dir(cell_dir):
                continue
            try:
                depth, cell = cell_dir.name.split("_", 1)
            except ValueError:
                print(f"[load_models] bad dir name: {cell_dir.name} (expected '<Depth>_<CellLabel>')")
                continue
            if depth not in annotation_depth:
                continue

            entry = models[atlas][depth].setdefault(cell, {})

            used_bundle = False
            for bf in _good_joblibs(cell_dir.glob("*bundle*.joblib")):
                bobj = _safe_load_joblib(bf)
                if bobj is None:
                    continue
                _merge_bundle(entry, bobj)
                used_bundle = True
                break

            if not used_bundle:
                for jf in _good_joblibs(cell_dir.glob("*_Stacked.joblib")):
                    m = _safe_load_joblib(jf)
                    if m is not None:
                        entry["Stacked"] = m
                        entry.setdefault("model", m)
                        break
                for spath in _good_joblibs(list(cell_dir.glob("*_scaler.joblib")) + list(cell_dir.glob("*scaler*.joblib"))):
                    s = _safe_load_joblib(spath)
                    if s is not None:
                        entry["scaler"] = s
                        break
                cols = None
                for cpath in _good_joblibs(
                    list(cell_dir.glob("*_columns.joblib"))
                    + list(cell_dir.glob("feature_names.joblib"))
                    + list(cell_dir.glob("*feature*names*.joblib"))
                ):
                    c = _safe_load_joblib(cpath)
                    if c is not None:
                        cols = list(map(str, c))
                        break
                if cols is not None:
                    entry["columns"] = cols
                else:
                    csv = cell_dir / "columns.csv"
                    try:
                        if csv.is_file() and csv.stat().st_size > 0:
                            c = pd.read_csv(csv, header=None).squeeze("columns").astype(str).tolist()
                            entry["columns"] = c
                    except Exception:
                        pass

    return models
