"""Declarative acquisition specs for open bulk datasets.

Each spec is what a manifest needs to make a dataset *fetchable*: where it lives,
how big it is, how to verify it, and what license governs reuse. Checksums are
recorded on first verified fetch (`.downloads.json`) and pinned here once known;
until then a fetch is size-verified against the server's Content-Length.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Mapping

from ..runtime.component import (ComponentManifest, LicenseSpec, Permissions, Provider,
                                 Requirements, RuntimeSpec, Validation)
from ..providers.base import Provider as ProviderBase


@dataclass(frozen=True)
class AcquisitionSpec:
    url: str
    filename: str
    host: str
    license: str
    description: str
    domain: str
    strategy: str = "single_file"          # single_file | per_file_bucket | api_paginated
    expected_bytes: int | None = None
    checksum: str | None = None
    fmt: str = "tsv"
    source_project: str = "public-data"
    version: str = "current"
    notes: str = ""


#: Open bulk datasets with stable URLs. Sizes were read from the servers when
#: this table was written; the downloader re-reads them at fetch time.
ACQUIRABLE: tuple[AcquisitionSpec, ...] = (
    AcquisitionSpec(
        "https://public-download-files.storage.googleapis.com/hgnc/tsv/tsv/hgnc_complete_set.txt",
        "hgnc_complete_set.txt", "public-download-files.storage.googleapis.com", "CC0-1.0",
        "HGNC complete approved human gene nomenclature set (symbols, aliases, cross-references).",
        "genomics", fmt="tsv", source_project="HGNC"),
    AcquisitionSpec(
        "https://reactome.org/download/current/ReactomePathways.txt",
        "ReactomePathways.txt", "reactome.org", "CC-BY-4.0",
        "All Reactome pathways: stable id, name, species.", "pathways", fmt="tsv",
        source_project="Reactome", notes="headerless: pathway_id\\tname\\tspecies"),
    AcquisitionSpec(
        "https://reactome.org/download/current/Ensembl2Reactome.txt",
        "Ensembl2Reactome.txt", "reactome.org", "CC-BY-4.0",
        "Ensembl gene → Reactome pathway mapping (lowest-level pathways).", "pathways", fmt="tsv",
        source_project="Reactome", notes="headerless: ensembl_id\\tpathway_id\\turl\\tname\\tevidence\\tspecies"),
    AcquisitionSpec(
        "https://stringdb-downloads.org/download/protein.info.v12.0/9606.protein.info.v12.0.txt.gz",
        "9606.protein.info.v12.0.txt.gz", "stringdb-downloads.org", "CC-BY-4.0",
        "STRING v12 human protein identifiers and annotations.", "proteomics", fmt="tsv.gz",
        expected_bytes=1970090, checksum="sha256:144de4b0d98c6a7dfde6ddc2591cf88657f27b989eadff4f501450c3ed1f0f1c", source_project="STRING", version="12.0"),
    AcquisitionSpec(
        "https://stringdb-downloads.org/download/protein.links.v12.0/9606.protein.links.v12.0.txt.gz",
        "9606.protein.links.v12.0.txt.gz", "stringdb-downloads.org", "CC-BY-4.0",
        "STRING v12 human protein–protein interaction scores.", "proteomics", fmt="tsv.gz",
        expected_bytes=83164437, source_project="STRING", version="12.0"),
    AcquisitionSpec(
        "https://gcp-public-data--gnomad.storage.googleapis.com/release/4.1/constraint/gnomad.v4.1.constraint_metrics.tsv",
        "gnomad.v4.1.constraint_metrics.tsv", "gcp-public-data--gnomad.storage.googleapis.com", "CC0-1.0",
        "gnomAD v4.1 per-gene constraint metrics (pLI, LOEUF, o/e).", "genomics", fmt="tsv",
        expected_bytes=95546041, checksum="sha256:68d8abdb7fc48f570869b02dfaa74b9fecaece7fcc5f301ddca40ec1ce12da00", source_project="gnomAD", version="4.1"),
    AcquisitionSpec(
        "https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/chembl_uniprot_mapping.txt",
        "chembl_uniprot_mapping.txt", "ftp.ebi.ac.uk", "CC-BY-SA-3.0",
        "ChEMBL target → UniProt accession mapping.", "chemistry", fmt="tsv",
        expected_bytes=1254330, source_project="ChEMBL"),
    AcquisitionSpec(
        "https://rest.kegg.jp/list/pathway/hsa", "kegg_pathways_hsa.tsv", "rest.kegg.jp",
        "Academic use free; commercial requires license", "KEGG human pathway list (id, name).",
        "pathways", fmt="tsv", source_project="KEGG", notes="headerless: pathway_id\\tname"),
)


class BiomniLakeSource:
    """Biomni's data lake: per-file objects in a public S3 bucket.

    The bundled zip returns 403; Biomni's own downloader fetches per file, and so
    does this. The 76 declared filenames come from the catalogue.
    """

    BASE = "https://biomni-release.s3.amazonaws.com/data_lake/"
    HOST = "biomni-release.s3.amazonaws.com"
    LICENSE = "Apache-2.0 (code); per-dataset terms apply to contents"

    def __init__(self, filenames: Mapping[str, str] | None = None) -> None:
        self.filenames = dict(filenames or {})

    def spec(self, filename: str, description: str = "", expected_bytes: int | None = None) -> AcquisitionSpec:
        return AcquisitionSpec(self.BASE + filename, filename, self.HOST, self.LICENSE,
                               description or f"Biomni data lake file {filename}", "general",
                               strategy="per_file_bucket", expected_bytes=expected_bytes,
                               fmt=Path(filename).suffix.lstrip("."), source_project="Biomni")


def acquisition_for(manifest: ComponentManifest) -> AcquisitionSpec | None:
    """Return the acquisition spec a dataset manifest declares, if any."""
    acq = (manifest.inputs or {}).get("acquisition")
    if not acq:
        return None
    if isinstance(acq, AcquisitionSpec):
        return acq
    fields = set(AcquisitionSpec.__dataclass_fields__)
    return AcquisitionSpec(**{k: v for k, v in dict(acq).items() if k in fields})


def fetch_command(manifest: ComponentManifest) -> str:
    spec = acquisition_for(manifest)
    if spec is None:
        return ""
    return f"bioagent fetch {manifest.id}   # -> {spec.filename} from {spec.host}"


class BulkDatasetProvider(ProviderBase):
    """Yields dataset components for every ACQUIRABLE bulk source."""

    name = "bulk-datasets"

    def __init__(self, specs: tuple[AcquisitionSpec, ...] = ACQUIRABLE) -> None:
        self.specs = specs

    def discover(self) -> Iterator[ComponentManifest]:
        for s in self.specs:
            yield ComponentManifest(
                id=f"{s.source_project.lower()}.dataset.{s.filename.lower().replace('.', '_')}",
                kind="dataset", name=s.filename, version=s.version, description=s.description,
                domain=s.domain, provider=Provider(project=s.source_project, repo=s.url),
                runtime=RuntimeSpec(backend="dataset", deterministic=True),
                inputs={"acquisition": s.__dict__},
                requires=Requirements(datasets=(s.filename,)),
                permissions=Permissions(network=(s.host,)),
                license=LicenseSpec(spdx=s.license, integration_mode="native"),
                validation=Validation(smoke_test="read_head"),
                offline_capable=True,
            )
