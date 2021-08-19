// Copyright (c) 2021 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
import { NgModule } from '@angular/core';
import { Routes, RouterModule } from '@angular/router';
import { DetailsComponent } from './details/details.component';
import { HomeComponent } from './home/home.component';
import { ImportModelComponent } from './import-model/import-model.component';
import { ModelListComponent } from './model-list/model-list.component';
import { PredefinedModelsComponent } from './predefined-models/predefined-models.component';


const routes: Routes = [
  { path: 'config-wizard', component: ImportModelComponent },
  { path: 'home', component: HomeComponent },
  { path: 'my-models', component: ModelListComponent },
  { path: 'model-zoo', component: PredefinedModelsComponent },
  { path: 'details/:id', component: DetailsComponent },
  { path: '', redirectTo: 'home', pathMatch: 'full' },
];

@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule]
})
export class AppRoutingModule { }
