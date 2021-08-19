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
import { ImportModelComponent } from './import-model/import-model.component';
import { ModelListComponent } from './model-list/model-list.component';
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
import { DetailsComponent } from './details/details.component';
import { NgxChartsModule } from '@swimlane/ngx-charts';
import { HomeComponent } from './home/home.component';
import { NgDatePipesModule } from 'ngx-pipes';
import { MatButtonToggleModule } from '@angular/material/button-toggle';
import { SystemInfoComponent } from './system-info/system-info.component';
import { LongNamePipe } from './pipes/long-name.pipe';
import { MatProgressBarModule } from '@angular/material/progress-bar';

@NgModule({
  declarations: [
    AppComponent,
    MenuComponent,
    ErrorComponent,
    PredefinedModelsComponent,
    ImportModelComponent,
    ModelListComponent,
    ModelListPipe,
    LongNamePipe,
    FileBrowserComponent,
    GraphComponent,
    DetailsComponent,
    HomeComponent,
    SystemInfoComponent
  ],
  imports: [
    BrowserModule,
    CommonModule,
    AppRoutingModule,
    BrowserAnimationsModule,
    FlexLayoutModule,
    FormsModule,
    HttpClientModule,
    MatExpansionModule,
    MatButtonModule,
    MatButtonToggleModule,
    MatCardModule,
    MatCheckboxModule,
    MatDialogModule,
    MatFormFieldModule,
    MatInputModule,
    MatMenuModule,
    MatSelectModule,
    MatSidenavModule,
    MatProgressBarModule,
    MatProgressSpinnerModule,
    MatStepperModule,
    MatTabsModule,
    MatSlideToggleModule,
    MatToolbarModule,
    MatTooltipModule,
    NgDatePipesModule,
    NgxChartsModule,
    NgxGraphModule,
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
    FileBrowserComponent
  ],
  bootstrap: [AppComponent]
})
export class AppModule { }
