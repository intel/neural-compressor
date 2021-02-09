import { BrowserModule } from '@angular/platform-browser';
import { NgModule } from '@angular/core';

import { AppRoutingModule } from './app-routing.module';
import { AppComponent } from './app.component';
import { MenuComponent } from './menu/menu.component';
import { ErrorComponent } from './error/error.component';
import { BrowserAnimationsModule } from '@angular/platform-browser/animations';
import { MatButtonModule, MatCardModule, MatDialogModule, MatDialogRef, MatExpansionModule, MatFormFieldModule, MatInputModule, MatMenuModule, MatProgressSpinnerModule, MatSelectModule, MatStepperModule, MatTabsModule, MatToolbarModule, MatTooltipModule } from '@angular/material';
import { FlexLayoutModule } from '@angular/flex-layout';
import { PredefinedModelsComponent } from './predefined-models/predefined-models.component';
import { ImportModelComponent } from './import-model/import-model.component';
import { TuneComponent } from './tune/tune.component';
import { SummaryComponent } from './summary/summary.component';
import { SystemInfoComponent } from './system-info/system-info.component';
import { ModelListComponent } from './model-list/model-list.component';
import { ModelListPipe } from './pipes/model-list.pipe';
import { FormsModule, ReactiveFormsModule } from '@angular/forms';
import { ModelService } from './services/model.service';
import { HttpClientModule } from '@angular/common/http';
import { SocketService } from './services/socket.service';
import { DialogComponent } from './dialog/dialog.component';
import { FileComponent } from './file/file.component';
import { FileBrowserComponent } from './file-browser/file-browser.component';

@NgModule({
  declarations: [
    AppComponent,
    MenuComponent,
    ErrorComponent,
    PredefinedModelsComponent,
    ImportModelComponent,
    TuneComponent,
    SummaryComponent,
    SystemInfoComponent,
    ModelListComponent,
    ModelListPipe,
    DialogComponent,
    FileComponent,
    FileBrowserComponent
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
    MatDialogModule,
    MatFormFieldModule,
    MatInputModule,
    MatMenuModule,
    MatSelectModule,
    MatProgressSpinnerModule,
    MatStepperModule,
    MatTabsModule,
    MatToolbarModule,
    MatTooltipModule,
    ReactiveFormsModule,
  ],
  providers: [
    ModelService,
    SocketService,
    {
      provide: MatDialogRef,
      useValue: {}
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
