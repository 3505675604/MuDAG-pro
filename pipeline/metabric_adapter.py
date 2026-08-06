"""Convert cBioPortal METABRIC files into MuDAG-Pro evaluation inputs."""

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd

from src.pathway_scoring import compute_rank_pathway_scores


ADAPTER_VERSION = 2
EXPRESSION_FILENAME = (
    "data_mrna_illumina_microarray_zscores_ref_diploid_samples.txt"
)
CLINICAL_FILENAME = "data_clinical_patient.txt"
MUTATIONS_FILENAME = "data_mutations.txt"
CACHE_FILENAMES = {
    "expr": "pathway_scores.csv",
    "clinical": "clinical.csv",
    "mutations": "mutations.json",
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


def build_metabric_pathway_scores(
    expression_path: str | Path,
    genesets: Mapping[str, Sequence[str]],
) -> pd.DataFrame:
    """Build a sample-by-pathway score table from cBioPortal expression."""
    source_path = Path(expression_path)
    expression = pd.read_csv(source_path, sep="\t", low_memory=False)
    _require_columns(
        expression,
        ["Hugo_Symbol", "Entrez_Gene_Id"],
        source_path,
    )

    sample_columns = [
        column
        for column in expression.columns
        if column not in {"Hugo_Symbol", "Entrez_Gene_Id"}
    ]
    if not sample_columns:
        raise ValueError(f"{source_path} has no expression sample columns.")

    target_genes = {
        str(gene).strip()
        for genes in genesets.values()
        for gene in genes
        if str(gene).strip()
    }
    expression["Hugo_Symbol"] = expression["Hugo_Symbol"].astype(str).str.strip()
    expression = expression.loc[
        expression["Hugo_Symbol"].isin(target_genes),
        ["Hugo_Symbol", *sample_columns],
    ].copy()
    if expression.empty:
        raise ValueError(
            f"{source_path} has no genes matching the configured pathways."
        )

    expression[sample_columns] = expression[sample_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )
    gene_expression = expression.groupby("Hugo_Symbol", sort=False)[
        sample_columns
    ].mean()

    scores = compute_rank_pathway_scores(gene_expression.T, genesets)
    scores.index.name = "sample_id"
    if scores.index.has_duplicates:
        raise ValueError("METABRIC expression sample IDs must be unique.")
    if not np.isfinite(scores.to_numpy(dtype=float)).all():
        raise ValueError("METABRIC pathway scores contain non-finite values.")
    return scores


def build_metabric_clinical(clinical_path: str | Path) -> pd.DataFrame:
    """Convert cBioPortal overall-survival fields to time/event columns."""
    source_path = Path(clinical_path)
    clinical = pd.read_csv(source_path, sep="\t", comment="#", low_memory=False)
    required = ["PATIENT_ID", "OS_MONTHS", "OS_STATUS"]
    _require_columns(clinical, required, source_path)

    months = pd.to_numeric(clinical["OS_MONTHS"], errors="coerce")
    event = pd.to_numeric(
        clinical["OS_STATUS"].astype(str).str.extract(r"^([01])", expand=False),
        errors="coerce",
    )
    valid = months.notna() & event.notna()
    result = pd.DataFrame(
        {
            "time": (months.loc[valid].astype(float) * 30.44).to_numpy(),
            "event": event.loc[valid].astype(float).to_numpy(),
        },
        index=clinical.loc[valid, "PATIENT_ID"].astype(str).to_numpy(),
    )
    result.index.name = "sample_id"
    if result.index.has_duplicates:
        raise ValueError("METABRIC clinical patient IDs must be unique.")
    return result


def build_metabric_mutations(
    mutations_path: str | Path,
) -> Dict[str, List[List[str]]]:
    """Group cBioPortal mutation records into patient mutation signatures."""
    source_path = Path(mutations_path)
    mutations = pd.read_csv(
        source_path,
        sep="\t",
        comment="#",
        low_memory=False,
    )
    required = [
        "Hugo_Symbol",
        "Tumor_Sample_Barcode",
        "HGVSp_Short",
        "Consequence",
    ]
    _require_columns(mutations, required, source_path)

    grouped: Dict[str, List[List[str]]] = {}
    for row in mutations[required].itertuples(index=False):
        gene = "" if pd.isna(row.Hugo_Symbol) else str(row.Hugo_Symbol).strip()
        sample = (
            ""
            if pd.isna(row.Tumor_Sample_Barcode)
            else str(row.Tumor_Sample_Barcode).strip()
        )
        if not gene or not sample:
            continue

        protein_change = (
            "" if pd.isna(row.HGVSp_Short) else str(row.HGVSp_Short).strip()
        )
        consequence = (
            "" if pd.isna(row.Consequence) else str(row.Consequence).strip()
        )
        grouped.setdefault(sample, []).append(
            [gene, protein_change or consequence]
        )
    return grouped


def _source_signature(path: Path) -> Dict[str, int]:
    stat = path.stat()
    return {
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _expected_manifest(
    raw_dir: Path,
    genesets_path: Path,
) -> Dict[str, object]:
    source_paths = {
        "expression": raw_dir / EXPRESSION_FILENAME,
        "clinical": raw_dir / CLINICAL_FILENAME,
        "mutations": raw_dir / MUTATIONS_FILENAME,
    }
    missing = [str(path) for path in source_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing extracted METABRIC source files: " + ", ".join(missing)
        )
    if not genesets_path.is_file():
        raise FileNotFoundError(f"Pathway gene-set file not found: {genesets_path}")

    return {
        "adapter_version": ADAPTER_VERSION,
        "normalization": "single-sample-gene-percentile-pathway-mean-v1",
        "genesets_sha256": hashlib.sha256(genesets_path.read_bytes()).hexdigest(),
        "sources": {
            name: _source_signature(path)
            for name, path in source_paths.items()
        },
    }


def _cache_paths(cache_dir: Path) -> Dict[str, Path]:
    return {
        name: cache_dir / filename
        for name, filename in CACHE_FILENAMES.items()
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
        if clinical_header != ["sample_id", "time", "event"]:
            return False

        mutations = json.loads(paths["mutations"].read_text(encoding="utf-8"))
        return isinstance(mutations, dict)
    except (OSError, ValueError, json.JSONDecodeError, pd.errors.ParserError):
        return False


def _cohort_result(cache_dir: Path) -> Dict[str, object]:
    paths = _cache_paths(cache_dir)
    return {
        "expr": str(paths["expr"]),
        "clinical": str(paths["clinical"]),
        "mutations": str(paths["mutations"]),
        "precomputed_pathways": True,
    }


def prepare_metabric(
    raw_dir: str | Path,
    genesets_path: str | Path,
    cache_dir: str | Path,
) -> Dict[str, object]:
    """Build or reuse validated METABRIC evaluation caches."""
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
        scores = build_metabric_pathway_scores(
            raw_dir / EXPRESSION_FILENAME,
            genesets,
        )
        clinical = build_metabric_clinical(raw_dir / CLINICAL_FILENAME)
        common_samples = scores.index.intersection(clinical.index)
        if common_samples.empty:
            raise ValueError(
                "METABRIC expression and clinical data have no overlapping IDs."
            )
        scores = scores.loc[common_samples]
        clinical = clinical.loc[common_samples]
        mutations = build_metabric_mutations(raw_dir / MUTATIONS_FILENAME)
        mutations = {
            sample: mutations[sample]
            for sample in common_samples
            if sample in mutations
        }

        temporary_paths = _cache_paths(temporary_dir)
        scores.to_csv(temporary_paths["expr"])
        clinical.to_csv(temporary_paths["clinical"])
        temporary_paths["mutations"].write_text(
            json.dumps(mutations, ensure_ascii=False),
            encoding="utf-8",
        )
        (temporary_dir / "manifest.json").write_text(
            json.dumps(expected_manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        if not _cache_is_valid(
            temporary_dir,
            expected_manifest,
            pathway_names,
        ):
            raise ValueError("Generated METABRIC cache failed schema validation.")

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
