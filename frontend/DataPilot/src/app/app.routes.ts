import { Routes } from '@angular/router';
import { canAccessResults, canAccessTrainingConfig } from './core/guards/workflow.guards';
import { ResultsPageComponent } from './features/results/results-page.component';
import { TrainPageComponent } from './features/train/train-page.component';
import { UploadPageComponent } from './features/upload/upload-page.component';

export const routes: Routes = [
  { path: '', pathMatch: 'full', redirectTo: 'upload' },
  { path: 'upload', component: UploadPageComponent },
  { path: 'train', component: TrainPageComponent, canActivate: [canAccessTrainingConfig] },
  { path: 'results', component: ResultsPageComponent, canActivate: [canAccessResults] },
  { path: '**', redirectTo: 'upload' },
];
