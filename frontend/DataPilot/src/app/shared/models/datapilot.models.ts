export type TaskType = 'classification' | 'regression' | 'clustering';

export interface ColumnInfo {
  name: string;
  dtype: string;
  is_categorical: boolean;
  missing_count: number;
  missing_pct: number;
  unique_count: number;
  sample_values: unknown[];
}

export interface DatasetInfo {
  rows: number;
  total_columns: number;
  numeric_columns: string[];
  categorical_columns: string[];
  missing_values_total: number;
}

export interface UploadResponse {
  job_id: string;
  filename: string;
  dataset_info: DatasetInfo;
  columns: ColumnInfo[];
  preview: Record<string, unknown>[];
}

export interface TrainRequest {
  task_type: TaskType;
  target_column?: string | null;
  ordinal_columns: string[];
  nominal_columns: string[];
}

export interface TrainResponse {
  job_id: string;
  status: string;
  message: string;
}

export interface TrainingResultsPayload {
  best_model?: string;
  metrics?: Record<string, unknown>;
  all_models?: Record<string, unknown>[] | Record<string, unknown>;
  task_type?: TaskType;
  target_column?: string | null;
  [key: string]: unknown;
}

export interface JobResult {
  job_id: string;
  status: 'uploaded' | 'training' | 'completed' | 'failed' | string;
  results?: TrainingResultsPayload | null;
  error?: string | null;
}

export interface ApiError {
  status: number;
  message: string;
  detail?: unknown;
  timestamp: string;
}

export interface DownloadedModel {
  blob: Blob;
  filename: string;
}
