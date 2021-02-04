import { NgModule } from '@angular/core';
import { Routes, RouterModule } from '@angular/router';
import { StartPageComponent } from './start-page/start-page.component';
import { ImportModelComponent } from './import-model/import-model.component';
import { SummaryComponent } from './summary/summary.component';
import { TuneComponent } from './tune/tune.component';


const routes: Routes = [
  { path: 'config-wizard', component: ImportModelComponent },
  { path: 'my-models', component: StartPageComponent },
  { path: 'summary', component: SummaryComponent },
  { path: 'tune', component: TuneComponent },
  { path: '', redirectTo: '/my-models', pathMatch: 'full' },
];

@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule]
})
export class AppRoutingModule { }
