import { Component, DestroyRef, inject, signal } from '@angular/core';
import { RouterLink, RouterLinkActive, RouterOutlet } from '@angular/router';
import { catchError, map, of, Subject, switchMap, takeUntil, timer } from 'rxjs';

import { DatapilotApiService } from './core/services/datapilot-api.service';
import { JobStateService } from './core/state/job-state.service';

@Component({
  selector: 'app-root',
  imports: [RouterOutlet, RouterLink, RouterLinkActive],
  templateUrl: './app.html',
  styleUrl: './app.css',
})
export class App {
  private readonly api = inject(DatapilotApiService);
  private readonly jobState = inject(JobStateService);
  private readonly destroyRef = inject(DestroyRef);
  private readonly destroy$ = new Subject<void>();

  protected readonly latestResult = this.jobState.latestResult;
  protected readonly jobId = this.jobState.jobId;
  protected readonly backendHealth = signal<'checking' | 'online' | 'offline'>('checking');

  constructor() {
    this.destroyRef.onDestroy(() => {
      this.destroy$.next();
      this.destroy$.complete();
    });

    timer(0, 15000)
      .pipe(
        switchMap(() =>
          this.api.healthCheck().pipe(
            map(() => 'online' as const),
            catchError(() => of('offline' as const)),
          ),
        ),
        takeUntil(this.destroy$),
      )
      .subscribe((status) => this.backendHealth.set(status));
  }
}
