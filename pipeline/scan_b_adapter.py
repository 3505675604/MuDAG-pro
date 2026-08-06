"""Convert GEO GSE96058 SCAN-B files into MuDAG-Pro evaluation inputs."""

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Mapping, Sequence

import numpy as np
import pandas as pd

from src.pathway_scoring import compute_rank_pathway_scores


ADAPTER_VERSION = 3
EXPRESSION_FILENAME = (
    "GSE96058_gene_expression_3273_samples_and_136_replicates_transformed.csv.gz"
)
CLINICAL_FILENAME = "GSE96058_clinical_metadata.csv"
TITLE_COLUMN = "title"
TIME_COLUMN = "characteristics_ch1.21.overall survival days"
EVENT_COLUMN = "characteristics_ch1.22.overall survival event"
CACHE_FILENAMES = {
    "expr": "pathway_scores.csv",
    "clinical": "clinical.csv",
}


def _require_columns(
    frame: pd.DataFrame,
    required: Sequence[str],
    source_path: Path,
) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(
            f"{source_path} is missing required columns: {', '.join(missing)}"
        )


def build_scan_b_pathway_scores(
    expression_path: str | Path,
    genesets: Mapping[str, Sequence[str]],
    chunk_size: int = 256,
) -> pd.DataFrame:
    """Stream SCAN-B genes and build scores for the 3,273 primary samples."""
    source_path = Path(expression_path)
    header = pd.read_csv(source_path, nrows=0)
    if len(header.columns) < 2:
        raise ValueError(f"{source_path} has no expression sample columns.")

    gene_column = header.columns[0]
    sample_columns = [
        column
        for column in header.columns[1:]
        if not str(column).lower().endswith("repl")
    ]
    if not sample_columns:
        raise ValueError(f"{source_path} has no primary expression samples.")
    if len(sample_columns) != len(set(sample_columns)):
        raise ValueError("SCAN-B primary expression sample IDs must be unique.")

    pathway_names = list(genesets)
    target_genes = {
        str(value).strip()
        for pathway_genes in genesets.values()
        for value in pathway_genes
        if str(value).strip()
    }
    if not pathway_names or not target_genes:
        raise ValueError("Configured pathway gene sets are empty.")

    matched_chunks = []
    seen_genes: set[str] = set()

    chunks = pd.read_csv(
        source_path,
        usecols=[gene_column, *sample_columns],
        chunksize=chunk_size,
        low_memory=False,
    )
    for chunk in chunks:
        gene_labels = chunk.pop(gene_column).astype(str).str.strip()
        matched = gene_labels.isin(target_genes)
        if not matched.any():
            continue

        selected_genes = gene_labels.loc[matched].tolist()
        duplicate_genes = seen_genes.intersection(selected_genes)
        if duplicate_genes:
            raise ValueError(
                f"{source_path} contains duplicate gene symbol: "
                f"{sorted(duplicate_genes)[0]}"
            )
        seen_genes.update(selected_genes)
        selected_values = chunk.loc[matched, sample_columns].apply(
            pd.to_numeric,
            errors="coerce",
        ).astype(np.float32)
        if not np.isfinite(selected_values.to_numpy(dtype=np.float32)).all():
            raise ValueError(
                f"{source_path} contains non-finite values for pathway genes."
            )
        selected_values.index = selected_genes
        matched_chunks.append(selected_values)

    if not matched_chunks:
        raise ValueError(
            f"{source_path} has no genes matching the configured pathways."
        )
    gene_expression = pd.concat(matched_chunks, axis=0)
    scores = compute_rank_pathway_scores(gene_expression.T, genesets)
    scores.index = pd.Index(sample_columns, name="sample_id")
    if not np.isfinite(scores.to_numpy(dtype=float)).all():
        raise ValueError("SCAN-B pathway scores contain non-finite values.")
    return scores


def build_scan_b_clinical(clinical_path: str | Path) -> pd.DataFrame:
    """Convert SCAN-B overall-survival fields and remove technical replicates."""
    source_path = Path(clinical_path)
    clinical = pd.read_csv(source_path, low_memory=False)
    _require_columns(
        clinical,
        [TITLE_COLUMN, TIME_COLUMN, EVENT_COLUMN],
        source_path,
    )

    titles = clinical[TITLE_COLUMN].astype("string").str.strip()
    times = pd.to_numeric(clinical[TIME_COLUMN], errors="coerce")
    events = pd.to_numeric(clinical[EVENT_COLUMN], errors="coerce")
    valid = (
        titles.notna()
        & ~titles.str.lower().str.endswith("repl", na=False)
        & times.notna()
        & (times > 0)
        & events.isin([0, 1])
    )
    result = pd.DataFrame(
        {
            "time": times.loc[valid].astype(float).to_numpy(),
            "event": events.loc[valid].astype(float).to_numpy(),
        },
        index=titles.loc[valid].astype(str).to_numpy(),
    )
    result.index.name = "sample_id"
    if result.empty:
        raise ValueError(f"{source_path} has no valid primary survival records.")
    if result.index.has_duplicates:
        raise ValueError("SCAN-B primary clinical sample IDs must be unique.")
    return result


def _source_signature(path: Path) -> Dict[str, int]:
    stat = path.stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _expected_manifest(
    raw_dir: Path,
    genesets_path: Path,
) -> Dict[str, object]:
    source_paths = {
        "expression": raw_dir / EXPRESSION_FILENAME,
        "clinical": raw_dir / CLINICAL_FILENAME,
    }
    missing = [str(path) for path in source_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing SCAN-B source files: " + ", ".join(missing)
        )
    if not genesets_path.is_file():
        raise FileNotFoundError(f"Pathway gene-set file not found: {genesets_path}")

    return {
        "adapter_version": ADAPTER_VERSION,
        "normalization": "single-sample-gene-percentile-pathway-mean-v1",
        "genesets_sha256": hashlib.sha256(genesets_path.read_bytes()).hexdigest(),
        "sources": {
            name: _source_signature(path) for name, path in source_paths.items()
        },
    }


def _cache_paths(cache_dir: Path) -> Dict[str, Path]:
    return {
        name: cache_dir / filename for name, filename in CACHE_FILENAMES.items()
    }


def _cache_is_valid(
    cache_dir: Path,
    expected_manifest: Mapping[str, object],
    pathway_names: Sequence[str],
) -> bool:
    paths = _cache_paths(cache_dir)
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.is_file() or not all(path.is_file() for path in paths.values()):
        return False

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest != expected_manifest:
            return False
        score_header = pd.read_csv(paths["expr"], nrows=0).columns.tolist()
        if score_header != ["sample_id", *pathway_names]:
            return False
        clinical_header = pd.read_csv(
            paths["clinical"],
            nrows=0,
        ).columns.tolist()
        return clinical_header == ["sample_id", "time", "event"]
    except (OSError, ValueError, json.JSONDecodeError, pd.errors.ParserError):
        return False


def _cohort_result(cache_dir: Path) -> Dict[str, object]:
    paths = _cache_paths(cache_dir)
    return {
        "expr": str(paths["expr"]),
        "clinical": str(paths["clinical"]),
        "mutations": None,
        "precomputed_pathways": True,
    }


def prepare_scan_b(
    raw_dir: str | Path,
    genesets_path: str | Path,
    cache_dir: str | Path,
) -> Dict[str, object]:
    """Build or reuse validated SCAN-B evaluation caches."""
    raw_dir = Path(raw_dir)
    genesets_path = Path(genesets_path)
    cache_dir = Path(cache_dir)
    expected_manifest = _expected_manifest(raw_dir, genesets_path)
    genesets = json.loads(genesets_path.read_text(encoding="utf-8"))
    if not isinstance(genesets, dict) or not genesets:
        raise ValueError(f"Pathway gene-set file is empty or invalid: {genesets_path}")
    pathway_names = list(genesets)

    if _cache_is_valid(cache_dir, expected_manifest, pathway_names):
        return _cohort_result(cache_dir)

    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{cache_dir.name}-build-",
            dir=cache_dir.parent,
        )
    )
    backup_dir = cache_dir.with_name(f".{cache_dir.name}-previous")
    try:
        scores = build_scan_b_pathway_scores(
            raw_dir / EXPRESSION_FILENAME,
            genesets,
        )
        clinical = build_scan_b_clinical(raw_dir / CLINICAL_FILENAME)
        common_samples = scores.index.intersection(clinical.index)
        if common_samples.empty:
            raise ValueError(
                "SCAN-B expression and clinical data have no overlapping IDs."
            )
        scores = scores.loc[common_samples]
        clinical = clinical.loc[common_samples]

        temporary_paths = _cache_paths(temporary_dir)
        scores.to_csv(temporary_paths["expr"])
        clinical.to_csv(temporary_paths["clinical"])
        (temporary_dir / "manifest.json").write_text(
            json.dumps(expected_manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        if not _cache_is_valid(
            temporary_dir,
            expected_manifest,
            pathway_names,
        ):
            raise ValueError("Generated SCAN-B cache failed schema validation.")

        if backup_dir.exists():
            shutil.rmtree(backup_dir)
        if cache_dir.exists():
            os.replace(cache_dir, backup_dir)
        try:
            os.replace(temporary_dir, cache_dir)
        except Exception:
            if backup_dir.exists() and not cache_dir.exists():
                os.replace(backup_dir, cache_dir)
            raise
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
    finally:
        if temporary_dir.exists():
            shutil.rmtree(temporary_dir)

    return _cohort_result(cache_dir)
