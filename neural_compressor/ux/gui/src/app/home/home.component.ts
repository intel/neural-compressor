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
import { Component, OnInit } from '@angular/core';
import { MatDialog } from '@angular/material/dialog';
import { ProjectFormComponent } from '../project-form/project-form.component';
import { ModelService } from '../services/model.service';

@Component({
  selector: 'app-home',
  templateUrl: './home.component.html',
  styleUrls: ['./home.component.scss', './../error/error.component.scss']
})
export class HomeComponent implements OnInit {

  clicked: 'systemInfo' | 'details';
  chosenRow = {};
  showSpinner = false;

  constructor(
    private modelService: ModelService,
    public dialog: MatDialog
  ) { }

  ngOnInit(): void {
  }

  createNewProject() {
    const dialogRef = this.dialog.open(ProjectFormComponent, {
      width: '60%',
    });
    dialogRef.afterClosed().subscribe(response => {
      if (response !== undefined) {
        this.showSpinner = true;
      }
    });
  }

  systemInfo() {
    return this.modelService.systemInfo;
  }

}
