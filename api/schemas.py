"""Pydantic request/response models for the API."""
from typing import Literal

from pydantic import BaseModel, Field


# --------------------------------------------------
# Requests
# --------------------------------------------------

class QueryRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        description="Natural-language question to answer.",
        examples=["How does Alice meet the Mad Hatter?"],
    )
    top_k: int | None = Field(
        default=None,
        ge=1,
        le=200,
        description="Max chunks used as context. Default: vector 3, symbolic all matching chunks. "
        "Set the same value on both endpoints to compare them on an equal context budget.",
    )


class SymbolicIngestRequest(BaseModel):
    limit: int | None = Field(
        default=None,
        ge=1,
        description="Only send the first N chunks to the WSD service (same as the CLI --limit flag). "
        "Omit to process every chunk (slow: ~8s per chunk).",
    )
    offset: int = Field(
        default=0,
        ge=0,
        description="Skip the first N chunks. Combine with `limit` to ingest in batches "
        "(offset=0, 50, 100, ...); a batch past the last chunk processes nothing.",
    )
    workers: int = Field(
        default=1,
        ge=1,
        le=16,
        description="Parallel connections to the WSD service within this batch (default 1 = sequential).",
    )


# --------------------------------------------------
# Responses
# --------------------------------------------------

class IngestResponse(BaseModel):
    status: Literal["success"] = "success"
    message: str


class SymbolicIngestResponse(IngestResponse):
    chunks_total: int = Field(description="Chunks sent to the WSD entity-linking service.")
    chunks_inserted: int = Field(description="Chunk nodes written to Neo4j.")
    senses_processed: int = Field(description="Sense links written to Neo4j.")
    failed_chunks: list[str] = Field(description="IDs of chunks the WSD service failed to link.")


class VectorQueryResponse(BaseModel):
    query: str
    found: bool = Field(description="False when no chunk scored above the relevance threshold (0.7).")
    answer: str | None = None
    sources: list[str | None] = Field(default_factory=list)


class QuerySynset(BaseModel):
    surface: str = Field(description="Text in the query that was linked.")
    synset_id: str


class SymbolicQueryResponse(BaseModel):
    query: str
    found: bool = Field(description="False when no chunk in Neo4j mentions any of the query's synsets.")
    answer: str | None = None
    sources: list[str] = Field(default_factory=list, description="'<source> (<chunk_id>)', best match first.")
    synsets: list[QuerySynset] = Field(default_factory=list, description="Synsets the query's entities linked to.")


class ErrorResponse(BaseModel):
    detail: str


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
