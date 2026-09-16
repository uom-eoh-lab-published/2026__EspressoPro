# -*- coding: utf-8 -*-
"""Core utilities for model loading and data management."""

from __future__ import annotations

import json
import os
import re
import sys
import shutil
import tarfile
import tempfile
import urllib.request
import zipfile
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Union

import joblib
import pandas as pd

ATLAS_NAME = "TotalSeqD_Heme_Oncology_CAT399906"
MODELS_SUBPATH = Path("Pre_trained_models") / ATLAS_NAME
MODELS_BASE_URL = "https://huggingface.co/EspressoKris/EspressoPro/resolve/main"

# Dated model releases use a six-digit YYMMDD suffix, for example:
# TotalSeqD_Heme_Oncology_CAT399906_260720.tar.xz
_MODEL_DATE_RE = re.compile(r"^\d{6}$")
_DATED_ATLAS_RE = re.compile(rf"^{re.escape(ATLAS_NAME)}_(\d{{6}})$")
_MODEL_DATE_MARKER_NAME = ".model_date"


def _normalize_model_date(model_date: Optional[str]) -> Optional[str]:
    """Validate and normalize a model release date in YYMMDD format."""
    if model_date is None:
        return None

    value = str(model_date).strip()
    if not value:
        return None
    if not _MODEL_DATE_RE.fullmatch(value):
        raise ValueError(
            f"model_date must use six-digit YYMMDD format, e.g. '260720'; got {model_date!r}"
        )

    yy, mm, dd = int(value[:2]), int(value[2:4]), int(value[4:6])
    try:
        date(2000 + yy, mm, dd)
    except ValueError as exc:
        raise ValueError(f"Invalid model_date {value!r}: {exc}") from exc

    return value


def _atlas_dir_name(model_date: Optional[str]) -> str:
    normalized = _normalize_model_date(model_date)
    return ATLAS_NAME if normalized is None else f"{ATLAS_NAME}_{normalized}"


def _model_date_from_name(name: Union[str, Path]) -> Optional[str]:
    """Extract a YYMMDD model date from an archive or atlas-directory name."""
    base = Path(str(name)).name
    for suffix in (".tar.xz", ".tar.gz", ".tgz", ".zip", ".tar"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    match = _DATED_ATLAS_RE.fullmatch(base)
    return _normalize_model_date(match.group(1)) if match else None


def _archive_filename(model_date: Optional[str]) -> str:
    """Build the archive filename for an optional YYMMDD release date."""
    return f"{_atlas_dir_name(model_date)}.tar.xz"


def _default_url_for_date(model_date: Optional[str]) -> str:
    return f"{MODELS_BASE_URL}/{_archive_filename(model_date)}"


_HF_REPO_ID = "EspressoKris/EspressoPro"
_HF_API_URL = f"https://huggingface.co/api/models/{_HF_REPO_ID}"
_LATEST_SENTINEL = "latest"


def _is_latest_sentinel(model_date: Optional[str]) -> bool:
    return isinstance(model_date, str) and model_date.strip().lower() == _LATEST_SENTINEL


def _list_remote_model_dates() -> list[str]:
    """Query the Hugging Face repo listing for all dated release archives."""
    req = urllib.request.Request(_HF_API_URL, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            info = json.load(resp)
    except Exception as exc:
        raise RuntimeError(
            f"Could not reach Hugging Face to resolve model_date='latest' "
            f"({_HF_API_URL}): {exc}"
        ) from exc

    filenames = [
        s.get("rfilename", "") for s in info.get("siblings", []) if isinstance(s, dict)
    ]
    dates = sorted({d for d in (_model_date_from_name(f) for f in filenames) if d})
    return dates


def _resolve_latest_model_date() -> str:
    """Return the newest dated release currently published on Hugging Face."""
    dates = _list_remote_model_dates()
    if not dates:
        raise RuntimeError(
            "model_date='latest' was requested but no dated releases "
            f"(<{ATLAS_NAME}>_YYMMDD.tar.xz) were found in {_HF_REPO_ID}."
        )
    return dates[-1]


def download_models(
    *,
    force: bool = False,
    model_date: Optional[str] = None,
    url: Optional[str] = None,
    local_archive: Optional[str] = None,
) -> Path:
    """
    Download and extract a pre-trained model release.

    Dated releases are stored side by side, so downloading one release never
    overwrites another release. For example:

        data/Pre_trained_models/TotalSeqD_Heme_Oncology_CAT399906_260513/
        data/Pre_trained_models/TotalSeqD_Heme_Oncology_CAT399906_260720/

    Parameters
    ----------
    force
        Re-download/re-extract the requested release even when it is present.
    model_date
        Six-digit release date in YYMMDD format, e.g. ``"260720"``.
        Pass ``"latest"`` to auto-resolve and download the newest dated
        release currently published on Hugging Face.
        When omitted, the unversioned archive ``<ATLAS_NAME>.tar.xz`` is used.
    url
        Explicit archive URL. If ``model_date`` is omitted and the URL filename
        contains a YYMMDD suffix, that date is inferred automatically.
    local_archive
        Path to an already-downloaded archive. Its YYMMDD suffix is inferred
        when ``model_date`` is omitted.

    Returns
    -------
    Path
        The package data directory.
    """
    if _is_latest_sentinel(model_date):
        resolved = _resolve_latest_model_date()
        print(f"[download_models] Resolved model_date='latest' -> {resolved}")
        model_date = resolved

    script_dir = Path(__file__).parent.resolve()
    data_dir = script_dir / "data"

    inferred_date: Optional[str] = None
    if model_date is None:
        if local_archive:
            inferred_date = _model_date_from_name(local_archive)
        elif url:
            inferred_date = _model_date_from_name(url)

    normalized_date = _normalize_model_date(model_date) or inferred_date
    target_name = _atlas_dir_name(normalized_date)
    models_root = data_dir / "Pre_trained_models" / target_name
    marker_path = models_root / _MODEL_DATE_MARKER_NAME
    release_label = normalized_date or "unversioned"

    if not force and any_existing_multiclass_bundle(models_root):
        print(f"[download_models] Models already present (release: {release_label}).")
        return data_dir

    if url is None:
        url = _default_url_for_date(normalized_date)

    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "Pre_trained_models").mkdir(parents=True, exist_ok=True)

    if force and models_root.exists():
        shutil.rmtree(models_root)

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

    def _looks_like_atlas_root(path: Path) -> bool:
        return any(
            (path / atlas / "Release").is_dir()
            for atlas in ("Hao", "Zhang", "Triana", "Luecken")
        )

    def _find_extracted_atlas_root(extracted_root: Path) -> Path:
        preferred_names = [target_name, ATLAS_NAME]
        candidates: list[Path] = []

        for name in preferred_names:
            candidates.extend(
                [
                    extracted_root / "data" / "Pre_trained_models" / name,
                    extracted_root / "Pre_trained_models" / name,
                    extracted_root / "data" / name,
                    extracted_root / name,
                ]
            )

        candidates.append(extracted_root)
        candidates.extend(p for p in extracted_root.rglob("*") if p.is_dir())

        valid = [p for p in candidates if p.is_dir() and _looks_like_atlas_root(p)]
        if not valid:
            raise RuntimeError(
                "Could not locate an atlas model root in the extracted archive. "
                "Expected Hao/Zhang/Triana/Luecken release directories."
            )
        return min(valid, key=lambda p: len(p.parts))

    def _extract_archive(archive_path: Path) -> None:
        print(f"[download_models] Extracting: {archive_path}")
        with tempfile.TemporaryDirectory() as tmp_dir:
            out = Path(tmp_dir) / "extract"
            out.mkdir(parents=True, exist_ok=True)

            if tarfile.is_tarfile(archive_path):
                with tarfile.open(archive_path, "r:*") as tar:
                    _safe_extract_tar(tar, out)
            elif zipfile.is_zipfile(archive_path):
                with zipfile.ZipFile(archive_path) as zf:
                    _safe_extract_zip(zf, out)
            else:
                raise RuntimeError("Unknown archive format: not tar or zip.")

            source_root = _find_extracted_atlas_root(out)
            models_root.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(source_root, models_root, dirs_exist_ok=True)

        if not any_existing_multiclass_bundle(models_root):
            raise RuntimeError(
                f"No Multiclass_models.joblib files were found after extraction into {models_root}"
            )

        marker_path.write_text(release_label)

    if local_archive:
        archive_path = Path(local_archive).expanduser().resolve()
        if not archive_path.exists():
            raise FileNotFoundError(f"Local archive not found: {archive_path}")
        _extract_archive(archive_path)
    else:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmpfile = Path(tmp_dir) / "models.archive"
            print(
                f"[download_models] Downloading pre-trained models "
                f"(release: {release_label}) from {url} ..."
            )

            try:
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
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to download model release {release_label!r} from {url}: {exc}"
                ) from exc

            if not tmpfile.exists() or tmpfile.stat().st_size == 0:
                raise RuntimeError("Download produced an empty file.")

            _extract_archive(tmpfile)

    print(f"[download_models] Models ready: {models_root} (release: {release_label})")
    return data_dir


def any_existing_multiclass_bundle(path: Union[str, Path]) -> bool:
    """Return True if path exists and contains at least one Multiclass_models.joblib."""
    p = Path(path)
    return p.exists() and any(p.rglob("Multiclass_models.joblib"))


def _candidate_data_bases() -> list[Path]:
    here = Path(__file__).parent.resolve()
    return [
        here / "data",
        here.parent / "data",
        here.parent / "Data",
        Path.home() / ".espressopro",
    ]


def _models_roots_under(base: Path) -> list[Path]:
    """Return valid dated and legacy atlas roots directly under a data-like base."""
    search_parents = [base, base / "Pre_trained_models"]
    roots: list[Path] = []

    for parent in search_parents:
        if not parent.is_dir():
            continue

        legacy = parent / ATLAS_NAME
        if any_existing_multiclass_bundle(legacy):
            roots.append(legacy)

        for candidate in parent.glob(f"{ATLAS_NAME}_[0-9][0-9][0-9][0-9][0-9][0-9]"):
            if candidate.is_dir() and _model_date_from_name(candidate.name):
                if any_existing_multiclass_bundle(candidate):
                    roots.append(candidate)

    unique: dict[str, Path] = {}
    for root in roots:
        unique[str(root.resolve())] = root.resolve()
    return list(unique.values())


def _select_models_root(
    roots: Sequence[Path],
    model_date: Optional[str] = None,
) -> Optional[Path]:
    """Select an exact requested release, or the newest installed dated release."""
    requested = _normalize_model_date(model_date)
    valid = [Path(root) for root in roots if any_existing_multiclass_bundle(root)]

    if requested is not None:
        exact = [root for root in valid if _model_date_from_name(root.name) == requested]
        return exact[0] if exact else None

    dated = [root for root in valid if _model_date_from_name(root.name) is not None]
    if dated:
        return max(dated, key=lambda root: _model_date_from_name(root.name) or "")

    legacy = [root for root in valid if root.name == ATLAS_NAME]
    return legacy[0] if legacy else None


def _candidate_models_dirs() -> list[Path]:
    """Return installed model roots, newest dated release first."""
    roots: list[Path] = []
    for base in _candidate_data_bases():
        roots.extend(_models_roots_under(base))

    unique: dict[str, Path] = {}
    for root in roots:
        unique[str(root.resolve())] = root.resolve()

    return sorted(
        unique.values(),
        key=lambda root: (
            _model_date_from_name(root.name) is not None,
            _model_date_from_name(root.name) or "",
        ),
        reverse=True,
    )


def _data_dir_from_models_root(root: Path) -> Path:
    root = root.resolve()
    if root.parent.name == "Pre_trained_models":
        return root.parent.parent
    return root.parent


def _resolve_models_root(
    models_path: Union[str, Path],
    *,
    model_date: Optional[str] = None,
) -> Path:
    """
    Resolve a path to an atlas root. When the path contains several dated
    releases, choose the highest YYMMDD date unless ``model_date`` is given.
    """
    p = Path(models_path).expanduser().resolve()
    requested = _normalize_model_date(model_date)

    if not p.exists():
        raise FileNotFoundError(f"models_path does not exist: {p}")

    # An explicitly supplied atlas root always wins, provided it matches the
    # requested date when one was supplied.
    if any_existing_multiclass_bundle(p) and (
        p.name == ATLAS_NAME or _model_date_from_name(p.name) is not None
    ):
        actual = _model_date_from_name(p.name)
        if requested is not None and actual != requested:
            raise FileNotFoundError(
                f"Requested model_date={requested}, but explicit models_path points to "
                f"release {actual or 'unversioned'}: {p}"
            )
        return p

    if any_existing_multiclass_bundle(p) and any(
        (p / atlas / "Release").is_dir()
        for atlas in ("Hao", "Zhang", "Triana", "Luecken")
    ):
        return p

    candidates = _models_roots_under(p)

    # Preserve support for a higher-level parent directory.
    for match in p.glob(f"**/{ATLAS_NAME}*"):
        if not match.is_dir():
            continue
        if match.name != ATLAS_NAME and _model_date_from_name(match.name) is None:
            continue
        if any_existing_multiclass_bundle(match):
            candidates.append(match.resolve())

    selected = _select_models_root(candidates, model_date=requested)
    if selected is not None:
        return selected

    expected = _atlas_dir_name(requested) if requested else f"{ATLAS_NAME}_YYMMDD"
    raise FileNotFoundError(
        "Could not resolve models_path to a usable atlas root.\n"
        f"Requested release: {requested or 'latest installed'}\n"
        f"Expected a directory such as: {expected}\n"
        f"Received: {p}"
    )


def ensure_models_available(
    *,
    local_archive: Optional[str] = None,
    force: bool = False,
    model_date: Optional[str] = None,
) -> Path:
    """
    Ensure a model release exists and return its data directory.

    When ``model_date`` is omitted, the newest installed dated release is used.
    If no dated release is installed, a legacy unversioned installation is used;
    if no models are installed at all, the newest release published on
    Hugging Face is downloaded automatically (equivalent to calling
    ``download_models(model_date="latest")``).
    """
    if model_date is None:
        model_date = os.environ.get("ESPRESSOPRO_MODEL_DATE")
    requested = _normalize_model_date(model_date)

    # Explicit model location. It may point to one root or a parent containing
    # several dated roots.
    env_models = os.environ.get("ESPRESSOPRO_MODELS")
    if env_models:
        p = Path(env_models).expanduser().resolve()
        if p.is_dir():
            try:
                root = _resolve_models_root(p, model_date=requested)
                return _data_dir_from_models_root(root)
            except FileNotFoundError:
                pass
        print(f"[ensure_models_available] ESPRESSOPRO_MODELS set but unusable: {p}")

    # Explicit data location.
    env_data = os.environ.get("ESPRESSOPRO_DATA")
    if env_data:
        d = Path(env_data).expanduser().resolve()
        if d.is_dir():
            root = _select_models_root(_models_roots_under(d), model_date=requested)
            if root is not None:
                return d
        print(f"[ensure_models_available] ESPRESSOPRO_DATA set but models not found under: {d}")

    # Common package/repository/user locations.
    roots = _candidate_models_dirs()
    selected = _select_models_root(roots, model_date=requested)
    if selected is not None and not force:
        selected_date = _model_date_from_name(selected.name) or "unversioned"
        print(f"[ensure_models_available] Using installed model release: {selected_date}")
        return _data_dir_from_models_root(selected)

    # Download the requested release. If nothing was requested AND nothing is
    # installed locally, fetch the newest release published on Hugging Face
    # rather than silently falling back to the unversioned archive.
    download_date = requested if requested is not None else _LATEST_SENTINEL
    data_dir = download_models(
        local_archive=local_archive,
        force=force,
        model_date=download_date,
    )

    # `requested` (not `download_date`) is used here on purpose: when it is
    # None, this means "pick the newest installed release", which is exactly
    # what was just downloaded above.
    selected = _select_models_root(_models_roots_under(data_dir), model_date=requested)
    if selected is not None:
        return data_dir

    raise FileNotFoundError(
        "Models directory not found after download/extraction.\n"
        f"Requested model_date: {requested or 'latest installed'}\n"
        "Set ESPRESSOPRO_MODELS or ESPRESSOPRO_DATA, pass explicit paths to "
        "generate_predictions(...), or call download_models(local_archive='…')."
    )


def get_default_models_path(*, model_date: Optional[str] = None) -> Path:
    """
    Return the exact atlas root for a requested YYMMDD release, or the newest
    installed dated release when ``model_date`` is omitted.
    """
    requested = _normalize_model_date(model_date)
    data_dir = ensure_models_available(model_date=requested)
    root = _select_models_root(_models_roots_under(data_dir), model_date=requested)

    if root is None:
        # The selected installation may live outside the package data directory
        # through ESPRESSOPRO_MODELS. Search all known roots as a fallback.
        root = _select_models_root(_candidate_models_dirs(), model_date=requested)

    if root is None:
        raise FileNotFoundError(
            f"No usable model root found for release {requested or 'latest installed'}."
        )

    selected_date = _model_date_from_name(root.name) or "unversioned"
    print(f"[get_default_models_path] Selected model release: {selected_date} ({root})")
    return root


def get_default_data_path(*, model_date: Optional[str] = None) -> Path:
    """Return the data directory containing the selected model release."""
    return ensure_models_available(model_date=model_date)


def get_package_data_path(*, model_date: Optional[str] = None) -> Path:
    """
    Resolve the package data directory using:

      1) $ESPRESSOPRO_DATA
      2) importlib.resources
      3) pkg_resources
      4) ./data next to this file
      5) ensure_models_available()

    Parameters
    ----------
    model_date
        Optional model release date in YYMMDD format, e.g. "260720".
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
    return ensure_models_available(model_date=model_date)


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

            .../TotalSeqD_Heme_Oncology_CAT399906_260720
            .../Pre_trained_models
            .../data
            .../data/Pre_trained_models/TotalSeqD_Heme_Oncology_CAT399906_260720

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