"""Convert GEO GSE2034 files into MuDAG-Pro evaluation inputs."""

import csv
import gzip
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from src.pathway_scoring import compute_rank_pathway_scores


ADAPTER_VERSION = 2
EXPRESSION_FILENAME = "GSE2034_expression_matrix.csv"
SOFT_FILENAME = "GSE2034_family.soft.gz"
SAMPLE_COLUMN = "GEO asscession number"
TIME_COLUMN = "time to relapse or last follow-up (months)"
EVENT_COLUMN = "relapse (1=True)"
CACHE_FILENAMES = {
    "expr": "pathway_scores.csv",
    "clinical": "clinical.csv",
}


def _parse_tsv_line(line: str) -> list[str]:
    return next(csv.reader([line], delimiter="\t"))


def load_gse2034_soft_tables(
    soft_path: str | Path,
) -> Tuple[pd.DataFrame, Dict[str, list[str]]]:
    """Read the embedded RFS table and GPL96 probe annotations."""
    source_path = Path(soft_path)
    clinical_rows: list[Dict[str, str]] = []
    probe_to_genes: Dict[str, list[str]] = {}
    mode = None
    header: list[str] | None = None

    with gzip.open(source_path, "rt", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\r\n")
            if line.startswith(
                "!series_table_begin = Patient clinical parameters"
            ):
                mode = "clinical_header"
                header = None
                continue
            if line == "!platform_table_begin":
                mode = "platform_header"
                header = None
                continue
            if line in {"!series_table_end", "!platform_table_end"}:
                mode = None
                header = None
                continue
            if mode == "clinical_header":
                header = _parse_tsv_line(line)
                mode = "clinical"
                continue
            if mode == "platform_header":
                header = _parse_tsv_line(line)
                if "ID" not in header or "Gene Symbol" not in header:
                    raise ValueError(
                        f"{source_path} GPL96 table lacks ID or Gene Symbol."
                    )
                mode = "platform"
                continue
            if mode == "clinical" and header is not None:
                values = _parse_tsv_line(line)
                values.extend([""] * (len(header) - len(values)))
                clinical_rows.append(dict(zip(header, values)))
                continue
            if mode == "platform" and header is not None:
                values = _parse_tsv_line(line)
                values.extend([""] * (len(header) - len(values)))
                row = dict(zip(header, values))
                probe_id = row["ID"].strip()
                symbols = [
                    symbol.strip()
                    for symbol in row["Gene Symbol"].split(" /// ")
                    if symbol.strip() and symbol.strip() != "---"
                ]
                if probe_id:
                    if probe_id in probe_to_genes:
                        raise ValueError(
                            f"{source_path} contains duplicate probe ID: {probe_id}"
                        )
                    probe_to_genes[probe_id] = list(dict.fromkeys(symbols))

    clinical = pd.DataFrame(clinical_rows)
    required_clinical = [SAMPLE_COLUMN, TIME_COLUMN, EVENT_COLUMN]
    missing = [column for column in required_clinical if column not in clinical]
    if missing:
        raise ValueError(
            f"{source_path} lacks GSE2034 clinical columns: {', '.join(missing)}"
        )
    if not probe_to_genes:
        raise ValueError(f"{source_path} has no GPL96 probe annotations.")
    return clinical, probe_to_genes


def build_geo_gse2034_pathway_scores(
    expression_path: str | Path,
    probe_to_genes: Mapping[str, Sequence[str]],
    genesets: Mapping[str, Sequence[str]],
) -> pd.DataFrame:
    """Map probes to genes, normalize gene profiles, and score pathways."""
    source_path = Path(expression_path)
    expression = pd.read_csv(source_path, index_col=0, low_memory=False)
    expression.index = expression.index.astype(str).str.strip()
    expression.columns = expression.columns.astype(str).str.strip()
    if expression.empty or expression.shape[1] == 0:
        raise ValueError(f"{source_path} contains no expression values.")
    if expression.index.has_duplicates:
        raise ValueError("GSE2034 expression probe IDs must be unique.")
    if expression.columns.has_duplicates:
        raise ValueError("GSE2034 expression sample IDs must be unique.")

    target_genes = {
        str(gene).strip()
        for pathway_genes in genesets.values()
        for gene in pathway_genes
        if str(gene).strip()
    }
    mapped_probes: list[str] = []
    mapped_genes: list[str] = []
    available_probes = set(expression.index)
    for probe_id, symbols in probe_to_genes.items():
        if probe_id not in available_probes:
            continue
        for symbol in dict.fromkeys(str(value).strip() for value in symbols):
            if symbol in target_genes:
                mapped_probes.append(probe_id)
                mapped_genes.append(symbol)
    if not mapped_probes:
        raise ValueError(
            f"{source_path} has no probes matching the configured pathways."
        )

    probe_values = expression.loc[mapped_probes].apply(
        pd.to_numeric,
        errors="coerce",
    )
    values = probe_values.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError(
            f"{source_path} must contain finite positive probe intensities."
        )
    probe_values = np.log2(probe_values)
    probe_values.index = mapped_genes
    gene_expression = probe_values.groupby(level=0, sort=False).mean()

    scores = compute_rank_pathway_scores(gene_expression.T, genesets)
    scores.index.name = "sample_id"
    if not np.isfinite(scores.to_numpy(dtype=float)).all():
        raise ValueError("GSE2034 pathway scores contain non-finite values.")
    return scores


def build_geo_gse2034_clinical(
    clinical_table: pd.DataFrame,
) -> pd.DataFrame:
    """Convert embedded relapse-free survival fields to time/event columns."""
    required = [SAMPLE_COLUMN, TIME_COLUMN, EVENT_COLUMN]
    missing = [column for column in required if column not in clinical_table]
    if missing:
        raise ValueError(
            "GSE2034 clinical table is missing columns: " + ", ".join(missing)
        )

    sample_ids = clinical_table[SAMPLE_COLUMN].astype("string").str.strip()
    times = pd.to_numeric(clinical_table[TIME_COLUMN], errors="coerce")
    events = pd.to_numeric(clinical_table[EVENT_COLUMN], errors="coerce")
    valid = (
        sample_ids.notna()
        & times.notna()
        & (times > 0)
        & events.isin([0, 1])
    )
    result = pd.DataFrame(
        {
            "time": times.loc[valid].astype(float).to_numpy(),
            "event": events.loc[valid].astype(float).to_numpy(),
        },
        index=sample_ids.loc[valid].astype(str).to_numpy(),
    )
    result.index.name = "sample_id"
    if result.empty:
        raise ValueError("GSE2034 has no valid relapse-free survival records.")
    if result.index.has_duplicates:
        raise ValueError("GSE2034 clinical sample IDs must be unique.")
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
        "soft": raw_dir / SOFT_FILENAME,
    }
    missing = [str(path) for path in source_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing GSE2034 source files: " + ", ".join(missing)
        )
    if not genesets_path.is_file():
        raise FileNotFoundError(f"Pathway gene-set file not found: {genesets_path}")
    return {
        "adapter_version": ADAPTER_VERSION,
        "normalization": "probe-log2-gene-mean_single-sample-rank-v1",
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


def prepare_geo_gse2034(
    raw_dir: str | Path,
    genesets_path: str | Path,
    cache_dir: str | Path,
) -> Dict[str, object]:
    """Build or reuse validated GEO GSE2034 evaluation caches."""
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
        clinical_table, probe_to_genes = load_gse2034_soft_tables(
            raw_dir / SOFT_FILENAME
        )
        scores = build_geo_gse2034_pathway_scores(
            raw_dir / EXPRESSION_FILENAME,
            probe_to_genes,
            genesets,
        )
        clinical = build_geo_gse2034_clinical(clinical_table)
        common_samples = scores.index.intersection(clinical.index)
        if common_samples.empty:
            raise ValueError(
                "GSE2034 expression and clinical data have no overlapping IDs."
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
            raise ValueError("Generated GSE2034 cache failed schema validation.")

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
