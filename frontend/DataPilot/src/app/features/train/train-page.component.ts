import { CommonModule } from '@angular/common';
import { Component, computed, inject, signal } from '@angular/core';
import { toSignal } from '@angular/core/rxjs-interop';
import {
  AbstractControl,
  FormBuilder,
  ReactiveFormsModule,
  ValidationErrors,
  ValidatorFn,
  Validators,
} from '@angular/forms';
import { Router } from '@angular/router';
import { startWith } from 'rxjs';

import { DatapilotApiService } from '../../core/services/datapilot-api.service';
import { JobStateService } from '../../core/state/job-state.service';
import { ApiError, TaskType, TrainRequest } from '../../shared/models/datapilot.models';

const SUPERVISED_TASKS: TaskType[] = ['classification', 'regression'];

function targetRequiredForSupervised(): ValidatorFn {
  return (control: AbstractControl): ValidationErrors | null => {
    const taskType = control.get('task_type')?.value as TaskType | undefined;
    const target = control.get('target_column')?.value as string | null | undefined;

    if (taskType && SUPERVISED_TASKS.includes(taskType) && !target) {
      return { targetRequired: true };
    }

    return null;
  };
}

function noOverlapBetweenOrdinalAndNominal(): ValidatorFn {
  return (control: AbstractControl): ValidationErrors | null => {
    const ordinal = (control.get('ordinal_columns')?.value as string[] | undefined) ?? [];
    const nominal = (control.get('nominal_columns')?.value as string[] | undefined) ?? [];

    const overlap = ordinal.filter((column) => nominal.includes(column));
    if (overlap.length) {
      return { encodingOverlap: overlap };
    }

    return null;
  };
}

@Component({
  selector: 'app-train-page',
  standalone: true,
  imports: [CommonModule, ReactiveFormsModule],
  templateUrl: './train-page.component.html',
  styleUrl: './train-page.component.css',
})
export class TrainPageComponent {
  private readonly fb = inject(FormBuilder);
  private readonly api = inject(DatapilotApiService);
  private readonly jobState = inject(JobStateService);
  private readonly router = inject(Router);

  protected readonly submitting = signal(false);
  protected readonly error = signal<string | null>(null);
  protected readonly upload = this.jobState.uploadResponse;
  protected readonly initialRequest = this.jobState.trainRequest;

  protected readonly allColumns = computed(
    () => this.upload()?.columns.map((column) => column.name) ?? [],
  );
  protected readonly categoricalColumns = computed(
    () =>
      this.upload()
        ?.columns.filter((column) => column.is_categorical)
        .map((column) => column.name) ?? [],
  );

  protected readonly form = this.fb.group(
    {
      task_type: this.fb.nonNullable.control<TaskType>('classification', {
        validators: [Validators.required],
      }),
      target_column: this.fb.control<string | null>(null),
      ordinal_columns: this.fb.nonNullable.control<string[]>([]),
      nominal_columns: this.fb.nonNullable.control<string[]>([]),
    },
    {
      validators: [targetRequiredForSupervised(), noOverlapBetweenOrdinalAndNominal()],
    },
  );

  private readonly formValue = toSignal(
    this.form.valueChanges.pipe(startWith(this.form.getRawValue())),
    {
      initialValue: this.form.getRawValue(),
    },
  );

  protected readonly isSupervisedTask = computed(() => {
    const taskType = this.formValue().task_type ?? 'classification';
    return SUPERVISED_TASKS.includes(taskType);
  });

  protected readonly ordinalColumns = computed(() => this.formValue().ordinal_columns ?? []);
  protected readonly nominalColumns = computed(() => this.formValue().nominal_columns ?? []);

  protected readonly payloadPreview = computed<TrainRequest>(() => {
    const value = this.formValue();
    const taskType = value.task_type ?? 'classification';

    return {
      task_type: taskType,
      target_column: taskType === 'clustering' ? null : value.target_column ?? null,
      ordinal_columns: value.ordinal_columns ?? [],
      nominal_columns: value.nominal_columns ?? [],
    };
  });

  constructor() {
    const request = this.initialRequest();
    if (request) {
      this.form.patchValue({
        task_type: request.task_type,
        target_column: request.target_column ?? null,
        ordinal_columns: request.ordinal_columns,
        nominal_columns: request.nominal_columns,
      });
    }
  }

  protected toggleColumnEncoding(
    columnName: string,
    target: 'ordinal_columns' | 'nominal_columns',
  ): void {
    const ordinal = [...this.form.controls.ordinal_columns.value];
    const nominal = [...this.form.controls.nominal_columns.value];

    if (target === 'ordinal_columns') {
      if (ordinal.includes(columnName)) {
        this.form.controls.ordinal_columns.setValue(
          ordinal.filter((column) => column !== columnName),
        );
      } else {
        this.form.controls.ordinal_columns.setValue([...ordinal, columnName]);
        this.form.controls.nominal_columns.setValue(
          nominal.filter((column) => column !== columnName),
        );
      }
    } else {
      if (nominal.includes(columnName)) {
        this.form.controls.nominal_columns.setValue(
          nominal.filter((column) => column !== columnName),
        );
      } else {
        this.form.controls.nominal_columns.setValue([...nominal, columnName]);
        this.form.controls.ordinal_columns.setValue(
          ordinal.filter((column) => column !== columnName),
        );
      }
    }

    this.form.updateValueAndValidity();
  }

  protected clearEncodingSelections(): void {
    this.form.controls.ordinal_columns.setValue([]);
    this.form.controls.nominal_columns.setValue([]);
    this.form.updateValueAndValidity();
  }

  protected isEncodingSelected(
    controlName: 'ordinal_columns' | 'nominal_columns',
    columnName: string,
  ): boolean {
    return this.form.controls[controlName].value.includes(columnName);
  }

  protected submit(): void {
    const upload = this.upload();
    if (!upload) {
      this.error.set('No upload context found. Please upload a file first.');
      return;
    }

    if (this.form.invalid) {
      this.form.markAllAsTouched();
      this.error.set('Please fix form validation issues before starting training.');
      return;
    }

    const payload: TrainRequest = {
      task_type: this.form.controls.task_type.value,
      target_column:
        this.form.controls.task_type.value === 'clustering'
          ? null
          : this.form.controls.target_column.value,
      ordinal_columns: this.form.controls.ordinal_columns.value,
      nominal_columns: this.form.controls.nominal_columns.value,
    };

    this.submitting.set(true);
    this.error.set(null);
    this.jobState.setTrainRequest(payload);

    this.api.startTraining(upload.job_id, payload).subscribe({
      next: (response) => {
        this.jobState.setTrainResponse(response);
        this.submitting.set(false);
        void this.router.navigate(['/results']);
      },
      error: (apiError: ApiError) => {
        this.submitting.set(false);

        if (apiError.status === 409) {
          const message =
            typeof apiError.detail === 'string'
              ? apiError.detail.toLowerCase()
              : apiError.message.toLowerCase();

          const inferredStatus = message.includes('completed') ? 'completed' : 'training';
          this.jobState.setLatestResult({
            job_id: upload.job_id,
            status: inferredStatus,
            results: null,
            error: null,
          });

          this.error.set(
            inferredStatus === 'completed'
              ? 'This job already completed. Opening results page.'
              : 'This job is already training. Opening live results page.',
          );

          void this.router.navigate(['/results']);
          return;
        }

        this.error.set(apiError.message);
      },
    });
  }

  protected getOverlapMessage(): string | null {
    const overlap = this.form.errors?.['encodingOverlap'] as string[] | undefined;
    if (!overlap?.length) {
      return null;
    }

    return `Columns cannot be both ordinal and nominal: ${overlap.join(', ')}`;
  }
}
