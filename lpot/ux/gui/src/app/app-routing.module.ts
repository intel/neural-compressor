import { NgModule } from '@angular/core';
import { Routes, RouterModule } from '@angular/router';
import { ErrorComponent } from './error/error.component';
import { ImportModelComponent } from './import-model/import-model.component';
import { SummaryComponent } from './summary/summary.component';
import { TuneComponent } from './tune/tune.component';
import { ModelListComponent } from './model-list/model-list.component';
import { PredefinedModelsComponent } from './predefined-models/predefined-models.component';


const routes: Routes = [
  { path: 'config-wizard', component: ImportModelComponent },
  { path: 'my-models', component: ModelListComponent },
  { path: 'model-zoo', component: PredefinedModelsComponent },
  { path: 'summary', component: SummaryComponent },
  { path: 'tune', component: TuneComponent },
  { path: '', redirectTo: '/my-models', pathMatch: 'full' },
];

@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule]
})
export class AppRoutingModule { }
