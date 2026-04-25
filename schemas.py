from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, field_validator, model_validator


class TaskType(str, Enum):
    classification = "classification"
    regression = "regression"
    clustering = "clustering"


class ColumnInfo(BaseModel):
    name: str
    dtype: str
    is_categorical: bool
    missing_count: int
    missing_pct: float      
    unique_count: int
    sample_values: list[Any]


class DatasetInfo(BaseModel):
    rows: int
    total_columns: int
    numeric_columns: list[str]
    categorical_columns: list[str]
    missing_values_total: int


class UploadResponse(BaseModel):
    job_id: str
    filename: str
    dataset_info: DatasetInfo
    columns: list[ColumnInfo]
    preview: list[dict]       


class TrainRequest(BaseModel):
    task_type: TaskType
    target_column: Optional[str] = None
    ordinal_columns: list[str] = []
    nominal_columns: list[str] = []

    @field_validator("target_column")
    @classmethod
    def target_required_for_supervised(cls, v, info):
        task = info.data.get("task_type")
        if task in (TaskType.classification, TaskType.regression) and not v:
            raise ValueError("target_column is required for classification and regression tasks.")
        return v

    @model_validator(mode="after")
    def no_overlap_between_encoding_types(self):
        overlap = set(self.ordinal_columns) & set(self.nominal_columns)
        if overlap:
            raise ValueError(f"These columns appear in both ordinal and nominal lists: {overlap}")
        return self


class TrainResponse(BaseModel):
    job_id: str
    status: str
    message: str


class JobResult(BaseModel):
    job_id: str
    status: str
    results: Optional[dict[str, Any]] = None
    error: Optional[str] = None