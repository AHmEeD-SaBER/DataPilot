import { inject, Injectable } from '@angular/core';
import { Observable, timer } from 'rxjs';
import { finalize, shareReplay, switchMap, takeUntil, takeWhile, tap } from 'rxjs/operators';

import { JobResult } from '../../shared/models/datapilot.models';
import { DatapilotApiService } from './datapilot-api.service';
import { JobStateService } from '../state/job-state.service';

export interface PollingOptions {
  intervalMs?: number;
  cancel$?: Observable<unknown>;
}

@Injectable({ providedIn: 'root' })
export class JobPollingService {
  private readonly api = inject(DatapilotApiService);
  private readonly jobState = inject(JobStateService);

  pollJob(jobId: string, options: PollingOptions = {}): Observable<JobResult> {
    const intervalMs = options.intervalMs ?? 2500;

    this.jobState.setPollingActive(true);

    let stream = timer(0, intervalMs).pipe(
      switchMap(() => this.api.getResults(jobId)),
      tap((result) => this.jobState.setLatestResult(result)),
      takeWhile((result) => !this.isTerminal(result.status), true),
      finalize(() => this.jobState.setPollingActive(false)),
      shareReplay({ bufferSize: 1, refCount: true }),
    );

    if (options.cancel$) {
      stream = stream.pipe(takeUntil(options.cancel$));
    }

    return stream;
  }

  private isTerminal(status: string): boolean {
    return status === 'completed' || status === 'failed';
  }
}
