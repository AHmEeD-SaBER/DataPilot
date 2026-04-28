import { inject } from '@angular/core';
import { CanActivateFn, Router } from '@angular/router';

import { JobStateService } from '../state/job-state.service';

export const canAccessTrainingConfig: CanActivateFn = () => {
  const state = inject(JobStateService);
  const router = inject(Router);
  return state.canConfigureTraining() ? true : router.createUrlTree(['/upload']);
};

export const canAccessResults: CanActivateFn = () => {
  const state = inject(JobStateService);
  const router = inject(Router);
  return state.canViewResults() ? true : router.createUrlTree(['/upload']);
};
