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
DATA_VERSION = "260513"  # fallback version used when model_data is not given
DATA_ARCHIVE_NAME = f"{ATLAS_NAME}_{DATA_VERSION}.tar.xz"
HF_DATA_REPO = "https://huggingface.co/EspressoKris/EspressoPro/resolve/main"
DEFAULT_MODELS_URL = f"{HF_DATA_REPO}/{DATA_ARCHIVE_NAME}"
MODELS_SUBPATH = Path("Pre_trained_models") / ATLAS_NAME


def _resolve_data_archive_name(model_data: Optional[str]) -> str:
    """
    Map a ``model_data`` selector to the archive file name on Hugging Face.

    ``model_data`` may be:

      - ``None`` (default): use the fallback pinned ``DATA_VERSION``.
      - ``"latest"``: fetch the archive tagged as the latest release.
      - a dated version string, e.g. ``"260714"``: fetch that specific
        release.
    """
    version = model_data or DATA_VERSION
    return f"{ATLAS_NAME}_{version}.tar.xz"


def download_models(
    *,
    model_data: Optional[str] = None,
    force: bool = False,
    url: Optional[str] = None,
    local_archive: Optional[str] = None,
) -> Path:
    """
    Download and extract pre-trained models.

    Parameters
    ----------
    model_data
        Which data release to fetch from the EspressoPro Hugging Face
        repository (https://huggingface.co/EspressoKris/EspressoPro):

            ep.download_models(model_data="latest")
            # or ep.download_models(model_data="260714") to download a specified version

        Defaults to the pinned fallback release
        (``TotalSeqD_Heme_Oncology_CAT399906_260513.tar.xz``) when omitted.
        Ignored if ``url`` or ``local_archive`` is given.
    url
        Explicit archive URL to download instead of resolving one from
        ``model_data``. Takes precedence over ``model_data``.

    Models are expected to resolve to one of the following layouts:

        <pkg>/data/Pre_trained_models/<ATLAS_NAME>/
        <pkg>/data/<ATLAS_NAME>/

    The atlas root should contain:

        <ATLAS_NAME>/
            Hao/Release/Broad/Models/Multiclass_models.joblib
            Hao/Release/Simplified/Models/Multiclass_models.joblib
            Hao/Release/Detailed/Models/Multiclass_models.joblib
            ...

    Returns
    -------
    Path
        The package data directory.
    """

    script_dir = Path(__file__).parent.resolve()
    data_dir = script_dir / "data"
    models_root = data_dir / MODELS_SUBPATH

    if url is None and local_archive is None:
        url = f"{HF_DATA_REPO}/{_resolve_data_archive_name(model_data)}"

    if not force and any_existing_multiclass_bundle(models_root):
        print("[download_models] Models already present.")
        return data_dir

    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "Pre_trained_models").mkdir(parents=True, exist_ok=True)

    def _safe_extract_tar(tar: tarfile.TarFile, path: Path) -> None:
        base = path.resolve()
        for member in tar.getmembers():
            target = (path / member.name).resolve()
            if not str(target).startswith(str(base)):
                raise RuntimeError(f"Blocked path traversal in tar member: {member.name}")
        tar.extractall(path)

    def _safe_extract_zip(zf: zipfile.ZipFile, path: Path) -> None:
        base = path.resolve()
        for member in zf.infolist():
            target = (path / member.filename).resolve()
            if not str(target).startswith(str(base)):
                raise RuntimeError(f"Blocked path traversal in zip member: {member.filename}")
        zf.extractall(path)

    def _copy_dir(src: Path, dst: Path) -> None:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dst, dirs_exist_ok=True)

    def _merge_into_data(extracted_root: Path) -> None:
        """
        Merge extracted payload into <pkg>/data.

        Supports archives containing any of:

          1) data/Pre_trained_models/<ATLAS_NAME>/...
          2) Pre_trained_models/<ATLAS_NAME>/...
          3) <ATLAS_NAME>/...
          4) arbitrary legacy folder layout copied into data/
        """
        data_candidates = [
            p for p in extracted_root.iterdir()
            if p.is_dir() and p.name.lower() == "data"
        ]
        roots = data_candidates or [extracted_root]

        for root in roots:
            # Case 1 or 2: root/Pre_trained_models/<ATLAS_NAME>
            p1 = root / "Pre_trained_models" / ATLAS_NAME
            if p1.is_dir():
                _copy_dir(p1, data_dir / "Pre_trained_models" / ATLAS_NAME)
                continue

            # Case 3: root/<ATLAS_NAME>
            p2 = root / ATLAS_NAME
            if p2.is_dir():
                _copy_dir(p2, data_dir / "Pre_trained_models" / ATLAS_NAME)
                continue

            # Case 4: archive may itself unpack directly as atlas contents,
            # but without the atlas folder. Detect by looking for atlas folders.
            atlas_like = [
                root / atlas
                for atlas in ("Hao", "Zhang", "Triana", "Luecken")
                if (root / atlas / "Release").is_dir()
            ]
            if atlas_like:
                dst = data_dir / "Pre_trained_models" / ATLAS_NAME
                dst.mkdir(parents=True, exist_ok=True)
                for child in root.iterdir():
                    dest = dst / child.name
                    if child.is_dir():
                        shutil.copytree(child, dest, dirs_exist_ok=True)
                    else:
                        shutil.copy2(child, dest)
                continue

            # Fallback: copy children into data/
            for child in root.iterdir():
                dest = data_dir / child.name
                if child.is_dir():
                    shutil.copytree(child, dest, dirs_exist_ok=True)
                else:
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(child, dest)

    def _extract_archive(archive_path: Path) -> None:
        print(f"[download_models] Extracting: {archive_path}")
        with tempfile.TemporaryDirectory() as tmp_dir:
            out = Path(tmp_dir) / "extract"
            out.mkdir(parents=True, exist_ok=True)

            try:
                if tarfile.is_tarfile(archive_path):
                    with tarfile.open(archive_path, "r:*") as tar:
                        _safe_extract_tar(tar, out)
                elif zipfile.is_zipfile(archive_path):
                    with zipfile.ZipFile(archive_path) as zf:
                        _safe_extract_zip(zf, out)
                else:
                    raise RuntimeError("Unknown archive format: not tar or zip.")
            except Exception as exc:
                raise RuntimeError(f"Failed to extract archive: {exc}") from exc

            _merge_into_data(out)

    if local_archive:
        archive_path = Path(local_archive).expanduser().resolve()
        if not archive_path.exists():
            raise FileNotFoundError(f"Local archive not found: {archive_path}")
        _extract_archive(archive_path)

    else:
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmpfile = Path(tmp_dir) / "models.archive"
                print(f"[download_models] Downloading pre-trained models from {url} ...")

                with urllib.request.urlopen(url) as response, open(tmpfile, "wb") as handle:
                    total = getattr(response, "length", None) or 0
                    read = 0
                    block = 1024 * 1024

                    while True:
                        chunk = response.read(block)
                        if not chunk:
                            break

                        handle.write(chunk)
                        read += len(chunk)

                        if total:
                            pct = 100 * read / total
                            sys.stdout.write(
                                f"\r  {read / 1e6:8.1f} MB / {total / 1e6:8.1f} MB ({pct:5.1f}%)"
                            )
                            sys.stdout.flush()

                    if total:
                        sys.stdout.write("\n")

                if not tmpfile.exists() or tmpfile.stat().st_size == 0:
                    raise RuntimeError("Download produced an empty file.")

                _extract_archive(tmpfile)

        except Exception as exc:
            print(f"[download_models] Failed to download models: {exc}")
            print("[download_models] Place the extracted folder at one of:")
            print(f"  {data_dir / 'Pre_trained_models' / ATLAS_NAME}")
            print(f"  {data_dir / ATLAS_NAME}")
            print("Or re-run with a local archive:")
            print(
                "  download_models("
                f"local_archive='/abs/path/{DATA_ARCHIVE_NAME}'"
                ")"
            )

    if any_existing_multiclass_bundle(models_root) or any_existing_multiclass_bundle(data_dir / ATLAS_NAME):
        print(f"[download_models] Models ready under {data_dir}")
    else:
        print("[download_models] Models not found after extraction.")

    return data_dir


def any_existing_multiclass_bundle(path: Union[str, Path]) -> bool:
    """Return True if path exists and contains at least one Multiclass_models.joblib."""
    p = Path(path)
    return p.exists() and any(p.rglob("Multiclass_models.joblib"))


def _candidate_models_dirs() -> list[Path]:
    """
    Likely locations for the atlas root:

        TotalSeqD_Heme_Oncology_CAT399906/
            Hao/Release/Broad/Models/Multiclass_models.joblib
            ...

    Supports both:

        <data>/Pre_trained_models/<ATLAS_NAME>
        <data>/<ATLAS_NAME>
    """
    here = Path(__file__).parent.resolve()

    pkg_data = here / "data"
    repo_data = here.parent / "data"
    repo_Data = here.parent / "Data"
    home_data = Path.home() / ".espressopro"

    bases = [pkg_data, repo_data, repo_Data, home_data]

    candidates: list[Path] = []
    for base in bases:
        candidates.extend(
            [
                base / "Pre_trained_models" / ATLAS_NAME,
                base / ATLAS_NAME,
            ]
        )

    return candidates


def _resolve_models_root(models_path: Union[str, Path]) -> Path:
    """
    Resolve models_path to the actual atlas root:

        .../TotalSeqD_Heme_Oncology_CAT399906

    Accepts:

        .../TotalSeqD_Heme_Oncology_CAT399906
        .../Pre_trained_models
        .../data
        .../data/Pre_trained_models/TotalSeqD_Heme_Oncology_CAT399906
    """
    p = Path(models_path).expanduser().resolve()

    if not p.exists():
        raise FileNotFoundError(f"models_path does not exist: {p}")

    # Case 1: already points to the atlas root.
    if p.name == ATLAS_NAME and any_existing_multiclass_bundle(p):
        return p

    # Case 2: points to Pre_trained_models/.
    candidate = p / ATLAS_NAME
    if candidate.exists() and any_existing_multiclass_bundle(candidate):
        return candidate

    # Case 3: points to data/.
    candidate = p / "Pre_trained_models" / ATLAS_NAME
    if candidate.exists() and any_existing_multiclass_bundle(candidate):
        return candidate

    # Case 4: points to a parent containing the atlas folder somewhere shallow.
    matches = [m for m in p.glob(f"**/{ATLAS_NAME}") if m.is_dir()]
    matches = [m for m in matches if any_existing_multiclass_bundle(m)]
    if matches:
        matches = sorted(matches, key=lambda x: len(x.parts))
        return matches[0]

    # Case 5: user points to a directory that itself contains Hao/Zhang/etc.
    if any((p / atlas / "Release").is_dir() for atlas in ("Hao", "Zhang", "Triana", "Luecken")):
        if any_existing_multiclass_bundle(p):
            return p

    raise FileNotFoundError(
        "Could not resolve models_path to the atlas root.\n"
        f"Expected to find a directory named: {ATLAS_NAME}\n"
        "Expected layout:\n"
        f"  {ATLAS_NAME}/Hao/Release/Broad/Models/Multiclass_models.joblib\n"
        f"Received: {p}"
    )


def ensure_models_available(
    *,
    model_data: Optional[str] = None,
    local_archive: Optional[str] = None,
    force: bool = False,
) -> Path:
    """
    Ensure models exist; attempt to download if missing.

    Returns
    -------
    Path
        The data directory that contains either:

            Pre_trained_models/<ATLAS_NAME>/

        or:

            <ATLAS_NAME>/
    """
    # 1) Explicit models location.
    env_models = os.environ.get("ESPRESSOPRO_MODELS")
    if env_models:
        p = Path(env_models).expanduser().resolve()

        if p.is_dir():
            try:
                atlas_root = _resolve_models_root(p)

                # Return data directory if atlas root is:
                #   <data>/Pre_trained_models/<ATLAS_NAME>
                if atlas_root.name == ATLAS_NAME and atlas_root.parent.name == "Pre_trained_models":
                    return atlas_root.parent.parent

                # Return parent if atlas root is:
                #   <data>/<ATLAS_NAME>
                if atlas_root.name == ATLAS_NAME:
                    return atlas_root.parent

            except FileNotFoundError:
                pass

        print(f"[ensure_models_available] ESPRESSOPRO_MODELS set but unusable: {p}")

    # 2) Explicit data location.
    env_data = os.environ.get("ESPRESSOPRO_DATA")
    if env_data:
        d = Path(env_data).expanduser().resolve()
        candidates = [
            d / "Pre_trained_models" / ATLAS_NAME,
            d / ATLAS_NAME,
        ]
        for candidate in candidates:
            if any_existing_multiclass_bundle(candidate):
                return d

        print(f"[ensure_models_available] ESPRESSOPRO_DATA set but models not found under: {d}")

    # 3) Look in common candidate locations.
    for candidate in _candidate_models_dirs():
        if any_existing_multiclass_bundle(candidate):
            if candidate.parent.name == "Pre_trained_models":
                return candidate.parent.parent
            return candidate.parent

    # 4) Download into package data dir and re-check.
    data_dir = download_models(model_data=model_data, local_archive=local_archive, force=force)

    for candidate in [
        data_dir / "Pre_trained_models" / ATLAS_NAME,
        data_dir / ATLAS_NAME,
    ]:
        if any_existing_multiclass_bundle(candidate):
            return data_dir

    for candidate in _candidate_models_dirs():
        if any_existing_multiclass_bundle(candidate):
            if candidate.parent.name == "Pre_trained_models":
                return candidate.parent.parent
            return candidate.parent

    raise FileNotFoundError(
        "Models directory not found.\n"
        f"• Set ESPRESSOPRO_MODELS to …/Pre_trained_models/{ATLAS_NAME}, "
        f"…/{ATLAS_NAME}, …/Pre_trained_models, or the data directory.\n"
        "• Or set ESPRESSOPRO_DATA to the parent data directory that contains Pre_trained_models/.\n"
        "• Or pass explicit paths to generate_predictions(..., models_path=..., data_path=...).\n"
        "• Or use download_models(local_archive='…')."
    )


def get_default_models_path() -> Path:
    """
    Return the atlas root:

        .../TotalSeqD_Heme_Oncology_CAT399906

    This may be under either:

        data/Pre_trained_models/<ATLAS_NAME>

    or:

        data/<ATLAS_NAME>
    """
    data_dir = ensure_models_available()

    candidates = [
        data_dir / "Pre_trained_models" / ATLAS_NAME,
        data_dir / ATLAS_NAME,
    ]

    for candidate in candidates:
        if any_existing_multiclass_bundle(candidate):
            return candidate

    for candidate in _candidate_models_dirs():
        if any_existing_multiclass_bundle(candidate):
            return candidate

    raise FileNotFoundError(
        f"Expected models under {data_dir / 'Pre_trained_models' / ATLAS_NAME} "
        f"or {data_dir / ATLAS_NAME}, but no Multiclass_models.joblib files were found."
    )


def get_default_data_path() -> Path:
    """Return the default data directory."""
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
    Load pre-trained multiclass bundles from the expected release layout:

        <ATLAS_NAME>/<atlas>/Release/<Depth>/Models/Multiclass_models.joblib

    Example:

        TotalSeqD_Heme_Oncology_CAT399906/
            Hao/
                Release/
                    Broad/
                        Models/
                            Multiclass_models.joblib
                            class_names.csv
                            Temperature_scaler.joblib

    Parameters
    ----------
    models_path
        Path to one of:

            .../TotalSeqD_Heme_Oncology_CAT399906
            .../Pre_trained_models
            .../data
            .../data/Pre_trained_models/TotalSeqD_Heme_Oncology_CAT399906

    model_names
        Atlas names to load.

    annotation_depth
        Annotation resolutions to load.

    Returns
    -------
    dict
        Nested dictionary:

            models[atlas][depth]["__MULTICLASS__"]
            models[atlas][depth]["class_names"]
            models[atlas][depth]["temp_scaler"]
            models[atlas][depth]["heads"]

        Keys are added when present in the loaded bundle.
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
        except Exception as exc:
            print(f"[load_models] failed to load {path}: {exc}")
            return None

    def _read_class_names(models_dir: Path) -> Optional[list[str]]:
        csv_path = models_dir / "class_names.csv"
        if not csv_path.exists():
            return None

        try:
            df = pd.read_csv(csv_path)

            if df.empty:
                return None

            if df.shape[1] == 1:
                return df.iloc[:, 0].astype(str).tolist()

            for col in ("class_name", "class_names", "label", "labels", "celltype", "cell_type"):
                if col in df.columns:
                    return df[col].astype(str).tolist()

            return df.iloc[:, 0].astype(str).tolist()

        except Exception as exc:
            print(f"[load_models] failed to read class names from {csv_path}: {exc}")
            return None

    def _normalize_bundle(bundle: object, models_dir: Path) -> dict:
        """
        Normalize loaded Multiclass_models.joblib into a stable dictionary.

        Expected current bundle keys include:

            atlas
            depth
            panel_name
            class_names
            heads
            temp_scaler

        Older/simple formats are handled where possible.
        """
        out: dict = {"__BUNDLE__": bundle}

        if isinstance(bundle, dict):
            out.update(bundle)

            if "class_names" in bundle and bundle["class_names"] is not None:
                out["class_names"] = list(map(str, bundle["class_names"]))

            if "heads" in bundle and bundle["heads"] is not None:
                out["heads"] = bundle["heads"]

            for key in (
                "temp_scaler",
                "multiclass_temp_scaler",
                "temperature_scaler",
                "temperature",
            ):
                if key in bundle and bundle[key] is not None:
                    out["temp_scaler"] = bundle[key]
                    break

            # Compatibility with older/simpler bundle formats.
            model = (
                bundle.get("model")
                or bundle.get("Stacked")
                or bundle.get("stacked")
                or bundle.get("clf")
                or bundle.get("classifier")
            )
            if model is not None:
                out["model"] = model
                out.setdefault("Stacked", model)

            if "excluded_classes" in bundle and bundle["excluded_classes"] is not None:
                out["excluded_classes"] = list(map(str, bundle["excluded_classes"]))

        else:
            out["model"] = bundle
            out.setdefault("Stacked", bundle)

        # Fill class_names from CSV if not present in joblib.
        if "class_names" not in out or out["class_names"] is None:
            class_names = _read_class_names(models_dir)
            if class_names is not None:
                out["class_names"] = class_names

        # Load Temperature_scaler.joblib if not already inside Multiclass_models.joblib.
        if "temp_scaler" not in out or out["temp_scaler"] is None:
            temp_path = models_dir / "Temperature_scaler.joblib"
            if temp_path.exists():
                temp = _safe_load_joblib(temp_path)
                if temp is not None:
                    out["temp_scaler"] = temp

        return out

    root = _resolve_models_root(models_path)

    models: Dict[str, dict] = defaultdict(lambda: defaultdict(dict))

    for atlas in model_names:
        for depth in annotation_depth:
            models_dir = root / atlas / "Release" / depth / "Models"
            bundle_path = models_dir / "Multiclass_models.joblib"

            if not bundle_path.exists():
                print(f"[load_models] missing bundle: {bundle_path}")
                continue

            bundle_obj = _safe_load_joblib(bundle_path)
            if bundle_obj is None:
                continue

            normalized = _normalize_bundle(bundle_obj, models_dir)

            models[atlas][depth]["__MULTICLASS__"] = normalized
            models[atlas][depth].update(normalized)

            print(f"[load_models] loaded {atlas}/{depth}: {bundle_path}")

    return models