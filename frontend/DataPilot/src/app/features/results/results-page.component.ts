import { CommonModule } from '@angular/common';
import { Component, computed, DestroyRef, inject, signal } from '@angular/core';
import { Router } from '@angular/router';
import { Subject, takeUntil } from 'rxjs';

import { DatapilotApiService } from '../../core/services/datapilot-api.service';
import { JobPollingService } from '../../core/services/job-polling.service';
import { JobStateService } from '../../core/state/job-state.service';
import {
  ApiError,
  TaskType,
  TrainRequest,
  TrainResponse,
} from '../../shared/models/datapilot.models';

interface MetricBar {
  label: string;
  value: number;
  width: number;
}

interface ModelComparisonRow {
  model: string;
  metrics: Record<string, unknown>;
}

interface MetricReportItem {
  key: string;
  label: string;
  value: number;
  direction: 'higher' | 'lower' | 'neutral';
  quality: 'good' | 'ok' | 'poor' | 'neutral';
  interpretation: string;
}

interface LeaderboardRow {
  rank: number;
  model: string;
  score: number;
  normalized: number;
  direction: 'higher' | 'lower';
  scoreKey: string;
}

interface FeatureImportanceBar {
  feature: string;
  importance: number;
  width: number;
}

interface ConfusionMatrixView {
  labels: string[];
  matrix: number[][];
  rowTotals: number[];
  columnTotals: number[];
  grandTotal: number;
  maxCellValue: number;
}

@Component({
  selector: 'app-results-page',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './results-page.component.html',
  styleUrl: './results-page.component.css',
})
export class ResultsPageComponent {
  private readonly polling = inject(JobPollingService);
  private readonly api = inject(DatapilotApiService);
  private readonly state = inject(JobStateService);
  private readonly router = inject(Router);
  private readonly destroyRef = inject(DestroyRef);

  private readonly cancelPolling$ = new Subject<void>();

  protected readonly result = this.state.latestResult;
  protected readonly trainRequest = this.state.trainRequest;
  protected readonly uploadResponse = this.state.uploadResponse;
  protected readonly pollingActive = this.state.pollingActive;
  protected readonly error = signal<string | null>(null);
  protected readonly retrying = signal(false);

  protected readonly taskType = computed<TaskType | null>(() => {
    const fromResults = this.result()?.results?.task_type;
    if (fromResults) {
      return fromResults;
    }
    return this.trainRequest()?.task_type ?? null;
  });

  protected readonly statusView = computed(() => this.result()?.status ?? 'unknown');
  protected readonly isCompleted = computed(() => this.statusView() === 'completed');
  protected readonly isFailed = computed(() => this.statusView() === 'failed');

  protected readonly bestModel = computed(() => {
    const best = this.result()?.results?.best_model;
    return typeof best === 'string' ? best : null;
  });

  protected readonly modelComparisons = computed<ModelComparisonRow[]>(() => {
    const allModels = this.result()?.results?.all_models;
    if (!allModels || typeof allModels !== 'object' || Array.isArray(allModels)) {
      return [];
    }

    return Object.entries(allModels).map(([model, metrics]) => ({
      model,
      metrics:
        typeof metrics === 'object' && metrics !== null ? (metrics as Record<string, unknown>) : {},
    }));
  });

  protected readonly scalarMetrics = computed(() => {
    const metrics = this.result()?.results?.metrics;
    if (!metrics || typeof metrics !== 'object') {
      return [];
    }

    return Object.entries(metrics)
      .filter(([, value]) => typeof value === 'number')
      .map(([label, value]) => ({ label, value: value as number }));
  });

  protected readonly metricReport = computed<MetricReportItem[]>(() => {
    const metrics = this.metricMap();
    if (!metrics) {
      return [];
    }

    const entries = Object.entries(metrics)
      .filter(([, value]) => typeof value === 'number')
      .map(([key, value]) => {
        const numericValue = value as number;
        const direction = this.metricDirection(key);
        return {
          key,
          label: this.metricLabel(key),
          value: numericValue,
          direction,
          quality: this.metricQuality(key, numericValue),
          interpretation: this.metricInterpretation(key, numericValue),
        };
      });

    const priorityOrder: Record<string, number> = {
      accuracy: 1,
      precision: 2,
      recall: 3,
      f1_score: 4,
      r2_score: 5,
      mae: 6,
      mse: 7,
      silhouette_score: 8,
      n_clusters: 9,
    };

    return entries.sort((a, b) => {
      const pa = priorityOrder[a.key] ?? 999;
      const pb = priorityOrder[b.key] ?? 999;
      return pa - pb;
    });
  });

  protected readonly chartBars = computed<MetricBar[]>(() => {
    const type = this.taskType();
    const metricsMap = this.metricMap();
    if (!type || !metricsMap) {
      return [];
    }

    const keysByTask: Record<TaskType, string[]> = {
      classification: ['accuracy', 'precision', 'recall', 'f1_score'],
      regression: ['r2_score', 'mae', 'mse'],
      clustering: ['silhouette_score'],
    };

    const keys = keysByTask[type];
    return keys
      .filter((key) => typeof metricsMap[key] === 'number')
      .map((key) => {
        const value = metricsMap[key] as number;
        return {
          label: key,
          value,
          width: this.metricToWidth(type, key, value),
        };
      });
  });

  protected readonly clusterSizes = computed(() => {
    const metricsMap = this.metricMap();
    const clusterSizes = metricsMap?.['cluster_sizes'];

    if (!clusterSizes || typeof clusterSizes !== 'object' || Array.isArray(clusterSizes)) {
      return [];
    }

    const entries = Object.entries(clusterSizes as Record<string, unknown>)
      .map(([cluster, count]) => ({
        cluster,
        count: typeof count === 'number' ? count : Number(count ?? 0),
      }))
      .filter((item) => Number.isFinite(item.count));

    const total = entries.reduce((sum, item) => sum + item.count, 0);
    return entries.map((entry) => ({
      ...entry,
      width: Math.max(8, Math.round((entry.count / total) * 100)),
    }));
  });

  protected readonly modelLeaderboard = computed<LeaderboardRow[]>(() => {
    const type = this.taskType();
    if (!type) {
      return [];
    }

    const byTask: Record<TaskType, { key: string; direction: 'higher' | 'lower' }> = {
      classification: { key: 'f1_score', direction: 'higher' },
      regression: { key: 'mae', direction: 'lower' },
      clustering: { key: 'silhouette_score', direction: 'higher' },
    };

    const config = byTask[type];
    const rows = this.modelComparisons()
      .map((row) => {
        const rawScore = row.metrics[config.key];
        const score = typeof rawScore === 'number' ? rawScore : null;
        return score === null
          ? null
          : {
              model: row.model,
              score,
            };
      })
      .filter((row): row is { model: string; score: number } => row !== null);

    if (!rows.length) {
      return [];
    }

    const values = rows.map((row) => row.score);
    const min = Math.min(...values);
    const max = Math.max(...values);
    let range = Math.max(1e-9, max - min);

    // When scores are very close, use a minimum range to avoid exaggerating tiny differences
    const relativeRange = max !== 0 ? range / Math.abs(max) : 0;
    if (relativeRange < 0.02) {
      range = Math.abs(max) * 0.02;
    }

    const withNormalized = rows.map((row) => {
      const normalizedRaw =
        config.direction === 'higher' ? (row.score - min) / range : (max - row.score) / range;

      return {
        ...row,
        normalized: max === min ? 100 : Math.round(Math.min(95, normalizedRaw * 100)),
      };
    });

    const sorted = withNormalized.sort((a, b) => b.normalized - a.normalized);
    return sorted.map((row, index) => ({
      rank: index + 1,
      model: row.model,
      score: row.score,
      normalized: Math.max(8, row.normalized),
      direction: config.direction,
      scoreKey: config.key,
    }));
  });

  protected readonly featureImportance = computed<FeatureImportanceBar[]>(() => {
    const resultsPayload = this.result()?.results;
    if (!resultsPayload || typeof resultsPayload !== 'object') {
      return [];
    }

    const candidate =
      (resultsPayload['feature_importance'] as unknown) ??
      (resultsPayload['feature_importances'] as unknown) ??
      (this.metricMap()?.['feature_importance'] as unknown) ??
      (this.metricMap()?.['feature_importances'] as unknown);

    if (!candidate) {
      return [];
    }

    let entries: Array<{ feature: string; importance: number }> = [];

    if (Array.isArray(candidate)) {
      entries = candidate
        .map((item) => {
          if (!item || typeof item !== 'object') {
            return null;
          }

          const feature = (item as Record<string, unknown>)['feature'];
          const importance = (item as Record<string, unknown>)['importance'];

          if (typeof feature !== 'string' || typeof importance !== 'number') {
            return null;
          }

          return { feature, importance };
        })
        .filter((item): item is { feature: string; importance: number } => item !== null);
    } else if (typeof candidate === 'object') {
      entries = Object.entries(candidate as Record<string, unknown>)
        .map(([feature, importance]) => ({
          feature,
          importance: typeof importance === 'number' ? importance : Number.NaN,
        }))
        .filter((item) => Number.isFinite(item.importance));
    }

    if (!entries.length) {
      return [];
    }

    const sorted = entries.sort((a, b) => b.importance - a.importance).slice(0, 12);
    const maxImportance = Math.max(...sorted.map((item) => item.importance), 1e-9);

    return sorted.map((item) => ({
      feature: item.feature,
      importance: item.importance,
      width: Math.max(8, Math.round((item.importance / maxImportance) * 100)),
    }));
  });

  protected readonly confusionMatrix = computed<ConfusionMatrixView | null>(() => {
    const matrixRaw = this.metricMap()?.['confusion_matrix'];

    if (!Array.isArray(matrixRaw) || !matrixRaw.length) {
      return null;
    }

    const matrix = matrixRaw
      .map((row) =>
        Array.isArray(row)
          ? row
              .map((cell) => (typeof cell === 'number' ? cell : Number(cell ?? 0)))
              .filter((cell) => Number.isFinite(cell))
          : [],
      )
      .filter((row) => row.length);

    if (!matrix.length) {
      return null;
    }

    const colCount = matrix[0].length;
    const normalized = matrix.map((row) => {
      if (row.length === colCount) {
        return row;
      }

      if (row.length > colCount) {
        return row.slice(0, colCount);
      }

      return [...row, ...Array(colCount - row.length).fill(0)];
    });

    const rowTotals = normalized.map((row) => row.reduce((sum, cell) => sum + cell, 0));
    const columnTotals = Array.from({ length: colCount }, (_, colIndex) =>
      normalized.reduce((sum, row) => sum + row[colIndex], 0),
    );
    const grandTotal = rowTotals.reduce((sum, total) => sum + total, 0);
    const maxCellValue = Math.max(...normalized.flat(), 1);

    const labels = Array.from({ length: normalized.length }, (_, i) => `Class ${i}`);

    return {
      labels,
      matrix: normalized,
      rowTotals,
      columnTotals,
      grandTotal,
      maxCellValue,
    };
  });

  protected readonly insightNotes = computed<string[]>(() => {
    const type = this.taskType();
    const notes: string[] = [];

    if (type === 'classification') {
      const f1 = this.getMetricValue('f1_score');
      const accuracy = this.getMetricValue('accuracy');

      if (f1 !== null && accuracy !== null) {
        notes.push(
          `F1 score is ${f1.toFixed(4)} and accuracy is ${accuracy.toFixed(4)}. Prioritize F1 when classes are imbalanced.`,
        );
      }

      if (this.confusionMatrix()) {
        notes.push('Use the confusion matrix to spot which classes are commonly misclassified.');
      }
    }

    if (type === 'regression') {
      const mae = this.getMetricValue('mae');
      const r2 = this.getMetricValue('r2_score');

      if (mae !== null) {
        notes.push(`MAE is ${mae.toFixed(4)}. Lower MAE means smaller average prediction error.`);
      }

      if (r2 !== null) {
        notes.push(`R2 score is ${r2.toFixed(4)}. Values closer to 1 indicate a stronger fit.`);
      }
    }

    if (type === 'clustering') {
      const silhouette = this.getMetricValue('silhouette_score');
      if (silhouette !== null) {
        notes.push(
          `Silhouette score is ${silhouette.toFixed(4)}. Higher values indicate clearer cluster separation.`,
        );
      }

      if (this.clusterSizes().length) {
        notes.push('Review cluster size distribution to detect strongly imbalanced segments.');
      }
    }

    return notes;
  });

  constructor() {
    this.destroyRef.onDestroy(() => {
      this.cancelPolling$.next();
      this.cancelPolling$.complete();
    });

    const jobId = this.state.jobId();
    if (!jobId) {
      void this.router.navigate(['/upload']);
      return;
    }

    this.startPolling(jobId);
  }

  protected downloadModel(): void {
    const jobId = this.state.jobId();
    if (!jobId) {
      return;
    }

    this.api.downloadModel(jobId).subscribe({
      next: ({ blob, filename }) => {
        const objectUrl = URL.createObjectURL(blob);
        const anchor = document.createElement('a');
        anchor.href = objectUrl;
        anchor.download = filename;
        anchor.click();
        URL.revokeObjectURL(objectUrl);
      },
      error: (apiError: ApiError) => this.error.set(apiError.message),
    });
  }

  protected retryTraining(): void {
    const upload = this.uploadResponse();
    const request = this.trainRequest();
    if (!upload || !request) {
      this.error.set('Cannot retry because training context is missing.');
      return;
    }

    this.retrying.set(true);
    this.error.set(null);

    this.api.startTraining(upload.job_id, request).subscribe({
      next: (response: TrainResponse) => {
        this.retrying.set(false);
        this.state.setTrainResponse(response);
        this.startPolling(upload.job_id);
      },
      error: (apiError: ApiError) => {
        this.retrying.set(false);
        this.error.set(apiError.message);
      },
    });
  }

  protected startNewJob(): void {
    this.cancelCurrentPolling();
    this.state.reset();
    void this.router.navigate(['/upload']);
  }

  protected metricEntriesForModel(row: ModelComparisonRow): Array<{ key: string; value: unknown }> {
    return Object.entries(row.metrics)
      .filter(([, value]) => typeof value === 'number' || typeof value === 'string')
      .map(([key, value]) => ({ key, value }));
  }

  protected metricDisplayValue(metric: MetricReportItem): string {
    if (this.isProbabilityMetric(metric.key)) {
      return `${(metric.value * 100).toFixed(2)}%`;
    }

    return metric.value.toFixed(4);
  }

  protected leaderboardScoreDisplay(row: LeaderboardRow): string {
    if (this.isProbabilityMetric(row.scoreKey)) {
      return `${(row.score * 100).toFixed(2)}%`;
    }

    return row.score.toFixed(4);
  }

  protected confusionCellOpacity(value: number, max: number): number {
    if (!Number.isFinite(value) || max <= 0) {
      return 0.12;
    }

    const ratio = value / max;
    return Math.max(0.14, Math.min(0.95, 0.14 + ratio * 0.81));
  }

  private startPolling(jobId: string): void {
    this.cancelCurrentPolling();

    this.polling
      .pollJob(jobId, { cancel$: this.cancelPolling$ })
      .pipe(takeUntil(this.cancelPolling$))
      .subscribe({
        error: (apiError: ApiError) => this.error.set(apiError.message),
      });
  }

  private cancelCurrentPolling(): void {
    this.cancelPolling$.next();
  }

  private metricMap(): Record<string, unknown> | null {
    const metrics = this.result()?.results?.metrics;
    if (!metrics || typeof metrics !== 'object') {
      return null;
    }

    return metrics;
  }

  private getMetricValue(key: string): number | null {
    const metric = this.metricMap()?.[key];
    return typeof metric === 'number' ? metric : null;
  }

  private metricLabel(key: string): string {
    const labels: Record<string, string> = {
      accuracy: 'Accuracy',
      precision: 'Precision',
      recall: 'Recall',
      f1_score: 'F1 Score',
      mae: 'Mean Absolute Error (MAE)',
      mse: 'Mean Squared Error (MSE)',
      r2_score: 'R2 Score',
      silhouette_score: 'Silhouette Score',
      n_clusters: 'Detected Clusters',
    };

    return labels[key] ?? key.replace(/_/g, ' ');
  }

  private metricDirection(key: string): 'higher' | 'lower' | 'neutral' {
    if (
      ['accuracy', 'precision', 'recall', 'f1_score', 'r2_score', 'silhouette_score'].includes(key)
    ) {
      return 'higher';
    }

    if (['mae', 'mse'].includes(key)) {
      return 'lower';
    }

    return 'neutral';
  }

  private metricQuality(key: string, value: number): 'good' | 'ok' | 'poor' | 'neutral' {
    if (this.metricDirection(key) === 'lower') {
      return 'neutral';
    }

    if (key === 'r2_score') {
      if (value >= 0.7) {
        return 'good';
      }
      if (value >= 0.3) {
        return 'ok';
      }
      return 'poor';
    }

    if (key === 'silhouette_score') {
      if (value >= 0.5) {
        return 'good';
      }
      if (value >= 0.25) {
        return 'ok';
      }
      return 'poor';
    }

    if (['accuracy', 'precision', 'recall', 'f1_score'].includes(key)) {
      if (value >= 0.9) {
        return 'good';
      }
      if (value >= 0.75) {
        return 'ok';
      }
      return 'poor';
    }

    return 'neutral';
  }

  private metricInterpretation(key: string, value: number): string {
    if (key === 'accuracy') {
      return 'Overall prediction correctness.';
    }
    if (key === 'precision') {
      return 'How many predicted positives were correct.';
    }
    if (key === 'recall') {
      return 'How many real positives were recovered.';
    }
    if (key === 'f1_score') {
      return 'Balance between precision and recall.';
    }
    if (key === 'mae') {
      return 'Average absolute error. Lower is better.';
    }
    if (key === 'mse') {
      return 'Penalizes larger errors strongly. Lower is better.';
    }
    if (key === 'r2_score') {
      return 'Explained variance versus baseline prediction.';
    }
    if (key === 'silhouette_score') {
      return 'Cluster separation quality from -1 to 1.';
    }
    if (key === 'n_clusters') {
      return `Model produced ${Math.round(value)} cluster groups.`;
    }

    return 'Performance metric from the trained model.';
  }

  private isProbabilityMetric(key: string): boolean {
    return ['accuracy', 'precision', 'recall', 'f1_score'].includes(key);
  }

  private metricToWidth(taskType: TaskType, key: string, value: number): number {
    if (taskType === 'classification') {
      return Math.max(6, Math.min(100, Math.round(value * 100)));
    }

    if (taskType === 'clustering') {
      return Math.max(6, Math.min(100, Math.round(((value + 1) / 2) * 100)));
    }

    if (taskType === 'regression') {
      if (key === 'r2_score') {
        return Math.max(6, Math.min(100, Math.round(((value + 1) / 2) * 100)));
      }

      const inverse = 1 / (1 + Math.max(0, value));
      return Math.max(6, Math.min(100, Math.round(inverse * 100)));
    }

    return 10;
  }
}
