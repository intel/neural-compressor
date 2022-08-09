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
import { BrowserModule } from '@angular/platform-browser';
import { NgModule } from '@angular/core';

import { AppRoutingModule } from './app-routing.module';
import { AppComponent } from './app.component';
import { MenuComponent } from './menu/menu.component';
import { ErrorComponent } from './error/error.component';
import { BrowserAnimationsModule } from '@angular/platform-browser/animations';
import { MatButtonModule } from '@angular/material/button';
import { MatCardModule } from '@angular/material/card';
import { MatDialogModule, MatDialogRef } from '@angular/material/dialog';
import { MatExpansionModule } from '@angular/material/expansion';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatInputModule } from '@angular/material/input';
import { MatMenuModule } from '@angular/material/menu';
import { MatProgressSpinnerModule } from '@angular/material/progress-spinner';
import { MatSelectModule } from '@angular/material/select';
import { MatStepperModule } from '@angular/material/stepper';
import { MatTabsModule } from '@angular/material/tabs';
import { MatToolbarModule } from '@angular/material/toolbar';
import { MatTooltipModule } from '@angular/material/tooltip';
import { FlexLayoutModule } from '@angular/flex-layout';
import { PredefinedModelsComponent } from './predefined-models/predefined-models.component';
import { ModelListPipe } from './pipes/model-list.pipe';
import { FormsModule, ReactiveFormsModule } from '@angular/forms';
import { ModelService } from './services/model.service';
import { HttpClientModule, HTTP_INTERCEPTORS } from '@angular/common/http';
import { SocketService } from './services/socket.service';
import { FileBrowserComponent } from './file-browser/file-browser.component';
import { AuthenticationInterceptor } from './authentication/authentication.interceptor';
import { MatCheckboxModule } from '@angular/material/checkbox';
import { GraphComponent } from './graph/graph.component';
import { NgxGraphModule } from '@swimlane/ngx-graph';
import { MatSidenavModule } from '@angular/material/sidenav';
import { CommonModule } from '@angular/common';
import { MatSlideToggleModule } from '@angular/material/slide-toggle';
import { NgxChartsModule } from '@swimlane/ngx-charts';
import { HomeComponent } from './home/home.component';
import { NgDatePipesModule } from 'ngx-pipes';
import { MatButtonToggleModule } from '@angular/material/button-toggle';
import { SystemInfoComponent } from './system-info/system-info.component';
import { LongNamePipe } from './pipes/long-name.pipe';
import { MatProgressBarModule } from '@angular/material/progress-bar';
import { ProjectComponent } from './project/project.component';
import { OptimizationsComponent } from './optimizations/optimizations.component';
import { BenchmarksComponent } from './benchmarks/benchmarks.component';
import { ProfilingComponent } from './profiling/profiling.component';
import { DatasetsComponent } from './datasets/datasets.component';
import { ProjectFormComponent } from './project-form/project-form.component';
import { BenchmarkFormComponent } from './benchmark-form/benchmark-form.component';
import { OptimizationFormComponent } from './optimization-form/optimization-form.component';
import { DatasetFormComponent } from './dataset-form/dataset-form.component';
import { ProfilingFormComponent } from './profiling-form/profiling-form.component';
import { MatSnackBarModule } from '@angular/material/snack-bar';
import { NotificationComponent } from './notification/notification.component';
import { JoyrideModule } from 'ngx-joyride';
import { MatRadioModule } from '@angular/material/radio';
import { PinBenchmarkComponent } from './pin-benchmark/pin-benchmark.component';
import { ProjectRemoveComponent } from './project-remove/project-remove.component';
import { ConfirmationDialogComponent } from './confirmation-dialog/confirmation-dialog.component';
import { DiagnosisComponent } from './diagnosis/diagnosis.component';
import { MatBadgeModule } from '@angular/material/badge';
import { ModelWiseComponent } from './model-wise/model-wise.component';
import * as PlotlyJS from 'plotly.js-dist-min';
import { PlotlyModule } from 'angular-plotly.js';
import { HistogramComponent } from './histogram/histogram.component';

PlotlyModule.plotlyjs = PlotlyJS;
import { ConfigPreviewComponent } from './config-preview/config-preview.component';
import { PrintJsonPipe } from './pipes/print-json.pipe';
import { GenerateConfigDialogComponent } from './generate-config-dialog/generate-config-dialog.component';
import { WarningComponent } from './warning/warning.component';

@NgModule({
  declarations: [
    AppComponent,
    MenuComponent,
    ErrorComponent,
    WarningComponent,
    PredefinedModelsComponent,
    ModelListPipe,
    LongNamePipe,
    PrintJsonPipe,
    FileBrowserComponent,
    GraphComponent,
    HomeComponent,
    SystemInfoComponent,
    ProjectComponent,
    OptimizationsComponent,
    BenchmarksComponent,
    ProfilingComponent,
    DatasetsComponent,
    ProjectFormComponent,
    BenchmarkFormComponent,
    OptimizationFormComponent,
    DatasetFormComponent,
    ProfilingFormComponent,
    NotificationComponent,
    PinBenchmarkComponent,
    ProjectRemoveComponent,
    ConfirmationDialogComponent,
    DiagnosisComponent,
    ModelWiseComponent,
    HistogramComponent,
    ConfigPreviewComponent,
    GenerateConfigDialogComponent
  ],
  imports: [
    BrowserModule,
    CommonModule,
    AppRoutingModule,
    BrowserAnimationsModule,
    FlexLayoutModule,
    FormsModule,
    HttpClientModule,
    JoyrideModule.forRoot(),
    MatExpansionModule,
    MatBadgeModule,
    MatButtonModule,
    MatButtonToggleModule,
    MatCardModule,
    MatCheckboxModule,
    MatDialogModule,
    MatFormFieldModule,
    MatInputModule,
    MatMenuModule,
    MatRadioModule,
    MatSelectModule,
    MatSidenavModule,
    MatProgressBarModule,
    MatProgressSpinnerModule,
    MatSnackBarModule,
    MatStepperModule,
    MatTabsModule,
    MatSlideToggleModule,
    MatToolbarModule,
    MatTooltipModule,
    NgDatePipesModule,
    NgxChartsModule,
    NgxGraphModule,
    PlotlyModule,
    ReactiveFormsModule,
  ],
  providers: [
    ModelService,
    SocketService,
    {
      provide: MatDialogRef,
      useValue: {}
    },
    {
      provide: HTTP_INTERCEPTORS,
      useClass: AuthenticationInterceptor,
      multi: true
    },

  ],
  entryComponents: [
    ErrorComponent,
    WarningComponent,
    FileBrowserComponent,
    ProjectRemoveComponent
  ],
  bootstrap: [AppComponent]
})
export class AppModule { }
