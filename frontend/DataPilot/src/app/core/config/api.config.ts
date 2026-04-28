import { InjectionToken } from '@angular/core';

export const API_BASE_URL = new InjectionToken<string>('API_BASE_URL');

export const API_ENDPOINTS = {
  health: '/health',
  upload: '/upload',
  train: (jobId: string) => `/train/${jobId}`,
  results: (jobId: string) => `/results/${jobId}`,
  download: (jobId: string) => `/download/${jobId}`,
};
