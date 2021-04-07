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
import { DialogComponent } from './dialog/dialog.component';
import { FileComponent } from './file/file.component';
import { FileBrowserComponent } from './file-browser/file-browser.component';
import { AuthenticationInterceptor } from './authentication/authentication.interceptor';
import { MatCheckboxModule } from '@angular/material/checkbox';
import { GraphComponent } from './graph/graph.component';
import { NgxGraphModule } from '@swimlane/ngx-graph';
import { MatSidenavModule } from '@angular/material/sidenav';


@NgModule({
  declarations: [
    AppComponent,
    MenuComponent,
    ErrorComponent,
    PredefinedModelsComponent,
    ImportModelComponent,
    ModelListComponent,
    ModelListPipe,
    DialogComponent,
    FileComponent,
    FileBrowserComponent,
    GraphComponent
  ],
  imports: [
    BrowserModule,
    AppRoutingModule,
    BrowserAnimationsModule,
    FlexLayoutModule,
    FormsModule,
    HttpClientModule,
    MatExpansionModule,
    MatButtonModule,
    MatCardModule,
    MatCheckboxModule,
    MatDialogModule,
    MatFormFieldModule,
    MatInputModule,
    MatMenuModule,
    MatSelectModule,
    MatSidenavModule,
    MatProgressSpinnerModule,
    MatStepperModule,
    MatTabsModule,
    MatToolbarModule,
    MatTooltipModule,
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
    DialogComponent,
    ErrorComponent,
    FileBrowserComponent
  ],
  bootstrap: [AppComponent]
})
export class AppModule { }
