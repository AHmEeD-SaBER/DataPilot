import { computed, Injectable, signal } from '@angular/core';

import {
  JobResult,
  TrainRequest,
  TrainResponse,
  UploadResponse,
} from '../../shared/models/datapilot.models';

@Injectable({ providedIn: 'root' })
export class JobStateService {
  private readonly uploadResponseState = signal<UploadResponse | null>(null);
  private readonly trainRequestState = signal<TrainRequest | null>(null);
  private readonly trainResponseState = signal<TrainResponse | null>(null);
  private readonly latestResultState = signal<JobResult | null>(null);
  private readonly pollingActiveState = signal(false);

  readonly uploadResponse = this.uploadResponseState.asReadonly();
  readonly trainRequest = this.trainRequestState.asReadonly();
  readonly trainResponse = this.trainResponseState.asReadonly();
  readonly latestResult = this.latestResultState.asReadonly();
  readonly pollingActive = this.pollingActiveState.asReadonly();

  readonly jobId = computed(() => {
    return (
      this.uploadResponseState()?.job_id ??
      this.trainResponseState()?.job_id ??
      this.latestResultState()?.job_id ??
      null
    );
  });

  readonly canConfigureTraining = computed(() => !!this.uploadResponseState()?.job_id);

  readonly canViewResults = computed(() => {
    const status = this.latestResultState()?.status;
    return (
      !!this.jobId() &&
      (!!this.trainResponseState() || status === 'completed' || status === 'training')
    );
  });

  setUploadResponse(payload: UploadResponse): void {
    this.uploadResponseState.set(payload);
    this.latestResultState.set({
      job_id: payload.job_id,
      status: 'uploaded',
      results: null,
      error: null,
    });
  }

  setTrainRequest(payload: TrainRequest): void {
    this.trainRequestState.set(payload);
  }

  setTrainResponse(payload: TrainResponse): void {
    this.trainResponseState.set(payload);
    this.latestResultState.update((current) => {
      if (!current || current.job_id !== payload.job_id) {
        return { job_id: payload.job_id, status: payload.status, results: null, error: null };
      }
      return { ...current, status: payload.status };
    });
  }

  setLatestResult(payload: JobResult): void {
    this.latestResultState.set(payload);
  }

  setPollingActive(active: boolean): void {
    this.pollingActiveState.set(active);
  }

  reset(): void {
    this.uploadResponseState.set(null);
    this.trainRequestState.set(null);
    this.trainResponseState.set(null);
    this.latestResultState.set(null);
    this.pollingActiveState.set(false);
  }
}
