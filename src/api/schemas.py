"""Request and response models: the API's contract."""
from typing import Literal

from pydantic import BaseModel, Field


class TopKItem(BaseModel):
    label: str
    confidence: float = Field(ge=0, le=1)


class Prediction(BaseModel):
    label: str = Field(description="Most likely defect pattern")
    confidence: float = Field(ge=0, le=1, description="Probability of the predicted pattern")
    uncertain: bool = Field(description="True when confidence is below the uncertainty threshold")
    top_k: list[TopKItem]


class Source(BaseModel):
    label: str = Field(description="Citation label used in the report, e.g. S1")
    title: str
    authors: str
    year: int | str
    url: str


class SimilarWafer(BaseModel):
    id: str
    label: str
    index: int = Field(description="Index of the wafer in the training set")
    similarity: float


class AnalysisResponse(BaseModel):
    prediction: Prediction
    patterns: list[str] = Field(description="Patterns the report explains (2 when uncertain)")
    report_markdown: str
    sources: list[Source]
    similar_wafers: list[SimilarWafer]
    generated_by: Literal["llm", "template"]
    warnings: list[str]
    timings_ms: dict[str, float]


class AnalyzeRequest(BaseModel):
    wafer_map: list[list[int]] = Field(
        description="2D grid: 0 = no die, 1 = pass, 2 = fail. Any size from 10x10 to 512x512.",
        examples=[[[0, 1, 1, 1, 0]] * 2 + [[1, 1, 2, 1, 1]] * 6 + [[0, 1, 1, 1, 0]] * 2],
    )


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    classifier: bool
    retriever: bool
    similar_index: bool
    llm: bool
    model_epoch: int | None
    version: str


class SampleList(BaseModel):
    samples: list[str]


class SampleWafer(BaseModel):
    name: str
    wafer_map: list[list[int]]


class ErrorResponse(BaseModel):
    detail: str