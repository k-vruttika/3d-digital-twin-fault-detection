"""Ingest pipeline: normalize -> validate (quarantine) -> dedup -> reorder -> event bus."""

from twin_backend.ingest.pipeline import IngestPipeline

__all__ = ["IngestPipeline"]
