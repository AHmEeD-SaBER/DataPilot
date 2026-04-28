import { HttpClient, HttpErrorResponse, HttpResponse } from '@angular/common/http';
import { inject, Injectable } from '@angular/core';
import { catchError, map, Observable, OperatorFunction, throwError } from 'rxjs';

import { API_BASE_URL, API_ENDPOINTS } from '../config/api.config';
import {
  ApiError,
  DownloadedModel,
  JobResult,
  TrainRequest,
  TrainResponse,
  UploadResponse,
} from '../../shared/models/datapilot.models';
import { mapHttpError } from './api-error.mapper';

@Injectable({ providedIn: 'root' })
export class DatapilotApiService {
  private readonly http = inject(HttpClient);
  private readonly apiBaseUrl = inject(API_BASE_URL);

  healthCheck(): Observable<{ status: string }> {
    return this.http
      .get<{ status: string }>(this.buildUrl(API_ENDPOINTS.health))
      .pipe(this.handleApiError());
  }

  uploadFile(file: File): Observable<UploadResponse> {
    const formData = new FormData();
    formData.append('file', file, file.name);

    return this.http
      .post<UploadResponse>(this.buildUrl(API_ENDPOINTS.upload), formData)
      .pipe(this.handleApiError());
  }

  startTraining(jobId: string, payload: TrainRequest): Observable<TrainResponse> {
    return this.http
      .post<TrainResponse>(this.buildUrl(API_ENDPOINTS.train(jobId)), payload)
      .pipe(this.handleApiError());
  }

  getResults(jobId: string): Observable<JobResult> {
    return this.http
      .get<JobResult>(this.buildUrl(API_ENDPOINTS.results(jobId)))
      .pipe(this.handleApiError());
  }

  downloadModel(jobId: string): Observable<DownloadedModel> {
    return this.http
      .get(this.buildUrl(API_ENDPOINTS.download(jobId)), {
        observe: 'response',
        responseType: 'blob',
      })
      .pipe(
        map((response: HttpResponse<Blob>) => ({
          blob: response.body ?? new Blob(),
          filename: this.extractFilename(response.headers.get('content-disposition'), jobId),
        })),
        this.handleApiError(),
      );
  }

  private buildUrl(path: string): string {
    return `${this.apiBaseUrl}${path}`;
  }

  private extractFilename(contentDisposition: string | null, jobId: string): string {
    if (!contentDisposition) {
      return `datapilot_model_${jobId}.joblib`;
    }

    const match = /filename\*?=(?:UTF-8''|\")?([^\";]+)/i.exec(contentDisposition);
    if (!match?.[1]) {
      return `datapilot_model_${jobId}.joblib`;
    }

    try {
      return decodeURIComponent(match[1].replace(/\"/g, '').trim());
    } catch {
      return match[1].replace(/\"/g, '').trim();
    }
  }

  private handleApiError<T>(): OperatorFunction<T, T> {
    return catchError((error: HttpErrorResponse) => {
      const mapped = mapHttpError(error);
      return throwError(() => mapped as ApiError);
    });
  }
}
