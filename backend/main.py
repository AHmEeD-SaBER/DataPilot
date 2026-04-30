"""
DataPilot ML – FastAPI Backend
================================
Endpoints
---------
POST /upload                  → upload CSV / XLSX, get preview + column list
POST /train/{job_id}          → kick off async training pipeline
GET  /results/{job_id}        → poll for status / results
GET  /download/{job_id}       → download the serialised model bundle (.joblib)
GET  /health                  → quick liveness check
"""

import os
import shutil
import uuid

import joblib
import numpy as np
import pandas as pd
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from preprocessing import preprocess_data
from schemas import (
    ColumnInfo,
    DatasetInfo,
    JobResult,
    TaskType,
    TrainRequest,
    TrainResponse,
    UploadResponse,
)
from training import train_and_evaluate
from utils import to_serializable

# ──────────────────────────────────────────────────────────────────────────────
# App setup
# ──────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="DataPilot ML API",
    description="Automated ML backend – upload data, train models, download results.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
MODELS_DIR = "models"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

_uploads: dict[str, str] = {}
_jobs: dict[str, dict] = {}


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

SUPPORTED_EXTENSIONS = {"csv", "xlsx"}


def _read_dataframe(path: str) -> pd.DataFrame:
    ext = path.rsplit(".", 1)[-1].lower()
    if ext == "csv":
        return pd.read_csv(path)
    return pd.read_excel(path)


def _get_job_or_404(job_id: str) -> dict:
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return _jobs[job_id]


def _run_training(job_id: str, request: TrainRequest) -> None:
    try:
        file_path = _uploads[job_id]
        df = _read_dataframe(file_path)

        # 1. Preprocess
        X, y, preprocessor = preprocess_data(
            df, request.task_type, request.target_column
        )

        # 2. Train + evaluate
        best_model, best_name, comparison, best_metrics = train_and_evaluate(
            X, y, request.task_type
        )

        # 3. Persist model bundle (model + preprocessor together)
        model_path = os.path.join(MODELS_DIR, f"{job_id}.joblib")
        joblib.dump(
            {
                "model": best_model,
                "preprocessor": preprocessor,
                "task_type": request.task_type.value,
            },
            model_path,
        )

        _jobs[job_id].update(
            {
                "status": "completed",
                "model_path": model_path,
                "results": to_serializable(
                    {
                        "best_model": best_name,
                        "metrics": best_metrics,
                        "all_models": comparison,
                        "task_type": request.task_type.value,
                        "target_column": request.target_column,
                    }
                ),
                "error": None,
            }
        )

    except Exception as exc:
        _jobs[job_id].update({"status": "failed", "error": str(exc)})


# ──────────────────────────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────────────────────────


@app.get("/health", tags=["Utility"])
def health_check():
    return {"status": "ok"}


@app.post("/upload", response_model=UploadResponse, tags=["Data"])
async def upload_file(file: UploadFile = File(...)):
    ext = (file.filename or "").rsplit(".", 1)[-1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '.{ext}'. Accepted: {SUPPORTED_EXTENSIONS}.",
        )

    job_id = str(uuid.uuid4())
    file_path = os.path.join(UPLOAD_DIR, f"{job_id}.{ext}")

    with open(file_path, "wb") as dst:
        shutil.copyfileobj(file.file, dst)

    try:
        df = _read_dataframe(file_path)
    except Exception as exc:
        os.remove(file_path)
        raise HTTPException(status_code=422, detail=f"Could not parse file: {exc}")

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()

    columns = [
        ColumnInfo(
            name=col,
            dtype=str(df[col].dtype),
            is_categorical=col in categorical_cols,
            missing_count=int(df[col].isna().sum()),
            missing_pct=round(df[col].isna().mean() * 100, 2),
            unique_count=int(df[col].nunique()),
            sample_values=df[col].dropna().unique()[:5].tolist(),
        )
        for col in df.columns
    ]

    dataset_info = DatasetInfo(
        rows=len(df),
        total_columns=len(df.columns),
        numeric_columns=numeric_cols,
        categorical_columns=categorical_cols,
        missing_values_total=int(df.isna().sum().sum()),
    )

    _uploads[job_id] = file_path
    _jobs[job_id] = {
        "status": "uploaded",
        "results": None,
        "model_path": None,
        "error": None,
    }

    return UploadResponse(
        job_id=job_id,
        filename=file.filename or "",
        dataset_info=dataset_info,
        columns=columns,
        preview=df.head(10).fillna("").to_dict(orient="records"),
    )


@app.post("/train/{job_id}", response_model=TrainResponse, tags=["Training"])
async def train(job_id: str, request: TrainRequest, background_tasks: BackgroundTasks):
    """
    Start the automated ML pipeline for a previously uploaded file.
    Training runs in the background; poll `/results/{job_id}` for progress.
    """
    job = _get_job_or_404(job_id)

    if job["status"] == "training":
        raise HTTPException(
            status_code=409, detail="Training is already in progress for this job."
        )

    if job["status"] == "completed":
        raise HTTPException(
            status_code=409,
            detail="This job has already completed. Upload a new file to start again.",
        )

    # Validate that target column exists in the file (early feedback)
    if (
        request.task_type in (TaskType.classification, TaskType.regression)
        and request.target_column
    ):
        df = _read_dataframe(_uploads[job_id])
        if request.target_column not in df.columns:
            raise HTTPException(
                status_code=422,
                detail=f"Column '{request.target_column}' not found. "
                f"Available columns: {list(df.columns)}",
            )

    _jobs[job_id]["status"] = "training"
    background_tasks.add_task(_run_training, job_id, request)

    return TrainResponse(
        job_id=job_id,
        status="training",
        message="Training started. Poll GET /results/{job_id} for updates.",
    )


@app.get("/results/{job_id}", response_model=JobResult, tags=["Training"])
def get_results(job_id: str):
    """
    Poll for training status and results.

    `status` values:
    - `uploaded`  – file is ready, training not started yet
    - `training`  – pipeline is running
    - `completed` – results are available
    - `failed`    – an error occurred (see `error` field)
    """
    job = _get_job_or_404(job_id)
    return JobResult(
        job_id=job_id,
        status=job["status"],
        results=job["results"],
        error=job.get("error"),
    )


@app.get("/download/{job_id}", tags=["Model"])
def download_model(job_id: str):
    """
    Download the serialised model bundle as a `.joblib` file.
    The bundle contains `{"model": ..., "preprocessor": ..., "task_type": ...}`.
    Load it with: `bundle = joblib.load("model.joblib")`
    """
    job = _get_job_or_404(job_id)

    if job["status"] != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Model not ready. Current status: '{job['status']}'.",
        )

    model_path = job["model_path"]
    if not model_path or not os.path.exists(model_path):
        raise HTTPException(status_code=500, detail="Model file not found on server.")

    return FileResponse(
        path=model_path,
        media_type="application/octet-stream",
        filename=f"datapilot_model_{job_id}.joblib",
    )
