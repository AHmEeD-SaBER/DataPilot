# DataPilot Frontend Guide

This document explains the frontend in practical terms for teammates who are not Angular developers.

The actual Angular app lives in the DataPilot folder:

- frontend/DataPilot

## 1) What This Frontend Does

The app is a 3-step machine learning workflow UI:

1. Upload dataset file (CSV or XLSX)
2. Configure and start training
3. Watch results and download trained model

It talks to the backend API running on http://localhost:8000.

## 2) Tech Stack (Simple View)

- Angular 20 (UI framework)
- TypeScript (typed JavaScript)
- RxJS (async streams and polling)
- Angular Router (page navigation)
- Angular Reactive Forms (form validation on training page)

You do not need deep Angular knowledge to work safely in this project if you follow the file map and flow below.

## 3) Run It Locally

From frontend/DataPilot:

1. Install dependencies

   npm install

2. Start development server

   npm start

3. Open browser

   http://localhost:4200

Important: backend must also be running at http://localhost:8000.

## 4) Project Structure You Should Know

Main app source is under frontend/DataPilot/src/app.

- app.ts
  - Root shell component
  - Shows top navigation and backend health badge

- app.routes.ts
  - Defines page routes:
    - /upload
    - /train
    - /results
  - Uses guards to prevent invalid navigation order

- core/
  - Shared infrastructure logic
  - config/api.config.ts: endpoint constants and API base token
  - services/datapilot-api.service.ts: all HTTP calls to backend
  - services/job-polling.service.ts: periodic polling of results endpoint
  - services/api-error.mapper.ts: normalizes API errors for UI
  - state/job-state.service.ts: central in-memory state for current job
  - guards/workflow.guards.ts: blocks routes if prerequisites are missing

- features/
  - Actual pages users interact with
  - upload/: upload dataset and preview metadata
  - train/: configure task type and encoding settings
  - results/: show status, metrics, charts, model comparison, download model

- shared/models/datapilot.models.ts
  - Type definitions for API request/response payloads

- environments/
  - environment.ts and environment.development.ts
  - API base URL config (currently both point to localhost:8000)

## 5) Mental Model: How Data Moves

Think of JobStateService as the frontend session memory.

Flow:

1. Upload page calls upload API
2. Upload response is saved in JobStateService
3. Train page reads saved upload data and builds training payload
4. Train page starts training API call and stores request/response in state
5. Results page starts polling API every 2.5s for status updates
6. Polling stops automatically when status becomes completed or failed
7. User can download model file from results page

The guards enforce correct order:

- Cannot access Train before Upload context exists
- Cannot access Results unless a job is in progress or completed

## 6) Page-by-Page Behavior

### Upload Page

Location: features/upload

- Supports file picker and drag-and-drop
- Accepts only .csv and .xlsx
- Max file size: 20 MB
- Shows:
  - dataset summary (rows, columns, missing values)
  - per-column profile
  - preview table
- After successful upload, user can continue to training

### Train Page

Location: features/train

- Lets user select task type:
  - classification
  - regression
  - clustering
- Target column is required for classification/regression
- User marks categorical columns as ordinal or nominal
- Validation ensures a column cannot be both ordinal and nominal
- Shows payload preview before submit
- On submit, starts backend training and navigates to results

### Results Page

Location: features/results

- Polls backend until terminal status
- Handles states:
  - training
  - completed
  - failed
- Shows rich result views when completed:
  - best model
  - metrics report
  - performance bars
  - leaderboard
  - confusion matrix (if present)
  - feature importance (if present)
  - cluster distribution (if present)
  - model comparison cards
- User can retry training, start new job, or download model

## 7) API Contract Used by Frontend

All API calls are centralized in core/services/datapilot-api.service.ts.

Endpoints:

- GET /health
- POST /upload
- POST /train/:jobId
- GET /results/:jobId
- GET /download/:jobId

If backend response shape changes, update:

1. shared/models/datapilot.models.ts (types)
2. datapilot-api.service.ts (mapping logic if needed)
3. affected feature component templates

## 8) Common Changes and Where to Edit

### Change upload validation rules

- features/upload/upload-page.component.ts

### Add or change training fields

- features/train/train-page.component.ts
- shared/models/datapilot.models.ts

### Change how results are visualized

- features/results/results-page.component.ts
- features/results/results-page.component.html
- features/results/results-page.component.css

### Add a new backend endpoint call

1. Add method in core/services/datapilot-api.service.ts
2. Add request/response types in shared/models/datapilot.models.ts
3. Call method from relevant feature component

### Change routing or page access rules

- app.routes.ts
- core/guards/workflow.guards.ts

## 9) State Management Notes

This app uses Angular signals in JobStateService (not NgRx).

Key state values:

- uploadResponse
- trainRequest
- trainResponse
- latestResult
- pollingActive
- computed jobId

When starting over, Results page calls state.reset().

## 10) Error Handling Strategy

- HTTP errors are normalized in api-error.mapper.ts
- Components consume a consistent ApiError shape:
  - status
  - message
  - detail
  - timestamp
- Most pages expose error text through local error signals and render it in template

## 11) Quick Troubleshooting

If frontend shows API offline:

1. Verify backend is running on port 8000
2. Check environments API base URL
3. Verify CORS settings on backend

If you cannot open Train or Results pages:

1. This is often expected due to route guards
2. Upload must happen before Train
3. Train must start before Results

If results page never finishes:

1. Inspect backend results endpoint for that job_id
2. Check browser devtools network requests for polling failures

## 12) Suggested Onboarding Path (30 Minutes)

1. Run app and backend
2. Read app.routes.ts and workflow.guards.ts
3. Read JobStateService to understand app state
4. Walk one page at a time: Upload -> Train -> Results
5. Place console logs in DatapilotApiService methods to see request flow

After this, you can safely implement most feature-level changes.
