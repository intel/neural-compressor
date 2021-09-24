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
import { ErrorComponent } from './error/error.component';
import { ModelService } from './services/model.service';
import { SystemInfoComponent } from './system-info/system-info.component';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.scss']
})
export class AppComponent implements OnInit {
  tokenIsSet = false;
  workspacePath: string;

  constructor(
    private modelService: ModelService,
    public dialog: MatDialog
  ) { }

  ngOnInit() {
    this.modelService.setToken(window.location.search.replace('?token=', ''));
    this.tokenIsSet = true;
    this.getWorkspace();
    this.modelService.getSystemInfo();
  }

  getWorkspace() {
    this.modelService.getDefaultPath('workspace')
      .subscribe(repoPath => {
        this.workspacePath = repoPath['path'];
        this.modelService.workspacePath = repoPath['path'];
      },
        error => {
          this.openErrorDialog(error);
        }
      );
  }

  openErrorDialog(error) {
    const dialogRef = this.dialog.open(ErrorComponent, {
      data: error
    });
  }

  showSystemInfo() {
    const dialogRef = this.dialog.open(SystemInfoComponent, {
      data: this.modelService.systemInfo
    });
  }
}