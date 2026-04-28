import { HttpErrorResponse } from '@angular/common/http';

import { ApiError } from '../../shared/models/datapilot.models';

function normalizeDetail(detail: unknown): string {
  if (typeof detail === 'string') {
    return detail;
  }

  if (Array.isArray(detail)) {
    return detail
      .map((item) => (typeof item === 'string' ? item : JSON.stringify(item)))
      .join(', ');
  }

  if (detail && typeof detail === 'object') {
    return JSON.stringify(detail);
  }

  return 'Unexpected error occurred.';
}

export function mapHttpError(error: HttpErrorResponse): ApiError {
  const detail = error.error?.detail ?? error.error ?? error.message;
  const message = normalizeDetail(detail);

  return {
    status: error.status || 0,
    message,
    detail,
    timestamp: new Date().toISOString(),
  };
}
