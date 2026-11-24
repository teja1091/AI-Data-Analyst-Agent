
from typing import Literal

from pydantic import BaseModel, Field


AnalysisExecutionStatus = Literal[
    "success",
    "failed",
    "skipped",
]

OverallAnalysisStatus = Literal[
    "success",
    "partial_failure",
    "failed",
]


class AnalysisExecution(BaseModel):
    """Execution information for one analyzer."""

    status: AnalysisExecutionStatus

    error: str | None = None

    warnings: list[str] = Field(default_factory=list)


class AnalysisResult(BaseModel):
    """Unified output produced by AnalysisOrchestrator."""

    overall_status: OverallAnalysisStatus

    dataset_summary: dict = Field(default_factory=dict)

    analyses: dict[str, dict] = Field(
        default_factory=dict
    )

    execution: dict[str, AnalysisExecution] = Field(
        default_factory=dict
    )

    warnings: list[str] = Field(
        default_factory=list
    )
