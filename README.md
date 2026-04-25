# DataPilot ML – Backend

FastAPI-based backend for the IS424 Automated ML project.

## Setup

### Local Run

```bash
cd backend

# 1. Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start the server
uvicorn main:app --reload --port 8000
```

### Running with Docker

Docker is the easiest way to run the service as it packages all dependencies automatically.

**Option 1: Using Docker Compose (Recommended)**
```bash
docker-compose up -d --build
```
You can stop it later with `docker-compose down`.

**Option 2: Using standard Docker commands**

First, build the image:
```bash
docker build -t datapilot-backend .
```

Then, run the container (assuming you are in the `backend` directory), choose the command based on your operating system:

**Linux / macOS:**
```bash
docker run -d -p 8000:8000 -v $(pwd)/uploads:/app/uploads -v $(pwd)/models:/app/models datapilot-backend
```

**Windows (PowerShell):**
```powershell
docker run -d -p 8000:8000 -v ${PWD}/uploads:/app/uploads -v ${PWD}/models:/app/models datapilot-backend
```

**Windows (Command Prompt):**
```cmd
docker run -d -p 8000:8000 -v %cd%\uploads:/app/uploads -v %cd%\models:/app/models datapilot-backend
```

Interactive API docs are available at **http://localhost:8000/docs** once the server is running.

---

## Endpoints

| Method | URL | Description |
|--------|-----|-------------|
| `GET`  | `/health` | Liveness check |
| `POST` | `/upload` | Upload a CSV or XLSX file → returns `job_id` + data preview |
| `POST` | `/train/{job_id}` | Start training (async) |
| `GET`  | `/results/{job_id}` | Poll for training status / metrics |
| `GET`  | `/download/{job_id}` | Download `.joblib` model bundle |

---

## Example Flow

### 1. Upload

```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@iris.csv"
```

```json
{
  "job_id": "abc-123",
  "filename": "iris.csv",
  "rows": 150,
  "columns": ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"],
  "preview": [...],
  "dtypes": {...}
}
```

### 2. Train

```bash
curl -X POST http://localhost:8000/train/abc-123 \
  -H "Content-Type: application/json" \
  -d '{"task_type": "classification", "target_column": "species"}'
```

```json
{
  "job_id": "abc-123",
  "status": "training",
  "message": "Training started. Poll GET /results/abc-123 for updates."
}
```

### 3. Poll for results

```bash
curl http://localhost:8000/results/abc-123
```

```json
{
  "job_id": "abc-123",
  "status": "completed",
  "results": {
    "best_model": "RandomForestClassifier",
    "task_type": "classification",
    "target_column": "species",
    "metrics": {
      "accuracy": 0.9667,
      "precision": 0.967,
      "recall": 0.9667,
      "f1_score": 0.9666,
      "confusion_matrix": [[10, 0, 0], [0, 9, 1], [0, 0, 10]]
    },
    "all_models": {
      "RandomForestClassifier":    {"accuracy": 0.9667, "f1_score": 0.9666, ...},
      "LogisticRegression":        {"accuracy": 0.9333, "f1_score": 0.9329, ...},
      "GradientBoostingClassifier":{"accuracy": 0.9333, "f1_score": 0.9329, ...}
    }
  }
}
```

### 4. Download model

```bash
curl -OJ http://localhost:8000/download/abc-123
```

### 5. Use the model in Python

```python
import joblib
import pandas as pd

bundle = joblib.load("datapilot_model_abc-123.joblib")

preprocessor = bundle["preprocessor"]
model        = bundle["model"]

new_data = pd.DataFrame([{
    "sepal_length": 5.1, "sepal_width": 3.5,
    "petal_length": 1.4, "petal_width": 0.2
}])

X = preprocessor.transform(new_data)
print(model.predict(X))   # e.g. [0]
```

---

## Supported ML Tasks

| Task | Algorithms | Selection criterion |
|------|-----------|---------------------|
| Classification | RandomForest, LogisticRegression, GradientBoosting | highest weighted F1 |
| Regression | RandomForest, LinearRegression, GradientBoosting | lowest MAE |
| Clustering | KMeans, AgglomerativeClustering | highest silhouette score |

### Preprocessing pipeline (auto-applied)

| Step | Technique |
|------|-----------|
| Missing values (numeric) | Median imputation |
| Missing values (categorical) | Most-frequent imputation |
| Categorical encoding | OneHotEncoder |
| Feature scaling | StandardScaler |
| Class imbalance (classification) | SMOTE / RandomOverSampler |

---

## Project structure

```
backend/
├── main.py            ← FastAPI app & endpoints
├── preprocessing.py   ← automated data pipeline
├── training.py        ← model training & evaluation
├── schemas.py         ← Pydantic request/response models
├── utils.py           ← numpy serialisation helper
├── requirements.txt
├── uploads/           ← uploaded files (created at runtime)
└── models/            ← saved .joblib bundles (created at runtime)
```
