import { CommonModule } from '@angular/common';
import { Component, computed, inject, signal } from '@angular/core';
import { Router } from '@angular/router';

import { DatapilotApiService } from '../../core/services/datapilot-api.service';
import { JobStateService } from '../../core/state/job-state.service';
import { ApiError } from '../../shared/models/datapilot.models';

@Component({
  selector: 'app-upload-page',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './upload-page.component.html',
  styleUrl: './upload-page.component.css',
})
export class UploadPageComponent {
  private static readonly MAX_FILE_SIZE_BYTES = 20 * 1024 * 1024;

  private readonly api = inject(DatapilotApiService);
  private readonly jobState = inject(JobStateService);
  private readonly router = inject(Router);

  protected readonly uploadResponse = this.jobState.uploadResponse;
  protected readonly selectedFile = signal<File | null>(null);
  protected readonly submitting = signal(false);
  protected readonly error = signal<string | null>(null);
  protected readonly isDragActive = signal(false);

  protected readonly previewColumns = computed(() => {
    const preview = this.uploadResponse()?.preview;
    if (!preview?.length) {
      return [];
    }
    return Object.keys(preview[0]);
  });

  protected readonly prettyFileSize = computed(() => {
    const file = this.selectedFile();
    if (!file) {
      return null;
    }

    const sizeMb = file.size / (1024 * 1024);
    return `${sizeMb.toFixed(2)} MB`;
  });

  protected onFileSelected(event: Event): void {
    const input = event.target as HTMLInputElement;
    const file = input.files?.[0] ?? null;
    this.applySelectedFile(file);
  }

  protected onDragOver(event: DragEvent): void {
    event.preventDefault();
    this.isDragActive.set(true);
  }

  protected onDragLeave(event: DragEvent): void {
    event.preventDefault();
    this.isDragActive.set(false);
  }

  protected onDrop(event: DragEvent): void {
    event.preventDefault();
    this.isDragActive.set(false);

    const file = event.dataTransfer?.files?.[0] ?? null;
    this.applySelectedFile(file);
  }

  protected pickFile(input: HTMLInputElement): void {
    input.click();
  }

  protected goToTraining(): void {
    void this.router.navigate(['/train']);
  }

  protected upload(): void {
    const file = this.selectedFile();
    if (!file) {
      this.error.set('Please select a CSV or XLSX file first.');
      return;
    }

    this.submitting.set(true);
    this.error.set(null);

    this.api.uploadFile(file).subscribe({
      next: (response) => {
        this.jobState.setUploadResponse(response);
        this.submitting.set(false);
        this.selectedFile.set(null);
      },
      error: (apiError: ApiError) => {
        this.submitting.set(false);
        this.error.set(apiError.message);
      },
    });
  }

  protected formatCell(value: unknown): string {
    if (value === null || value === undefined || value === '') {
      return '-';
    }
    if (typeof value === 'object') {
      return JSON.stringify(value);
    }
    return String(value);
  }

  private applySelectedFile(file: File | null): void {
    this.error.set(null);

    if (!file) {
      this.selectedFile.set(null);
      return;
    }

    const extension = file.name.split('.').pop()?.toLowerCase() ?? '';
    if (!['csv', 'xlsx'].includes(extension)) {
      this.selectedFile.set(null);
      this.error.set("Only '.csv' and '.xlsx' files are supported.");
      return;
    }

    if (file.size > UploadPageComponent.MAX_FILE_SIZE_BYTES) {
      this.selectedFile.set(null);
      this.error.set('File is too large. Maximum allowed size is 20 MB.');
      return;
    }

    this.selectedFile.set(file);
  }
}
