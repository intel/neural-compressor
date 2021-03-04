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
import { ErrorComponent } from '../error/error.component';
import { FileBrowserComponent } from '../file-browser/file-browser.component';
import { ModelService } from '../services/model.service';


@Component({
  selector: 'app-menu',
  templateUrl: './menu.component.html',
  styleUrls: ['./menu.component.scss', './../error/error.component.scss']
})
export class MenuComponent implements OnInit {

  workspacePath: string;

  constructor(
    private modelService: ModelService,
    public dialog: MatDialog
  ) { }

  ngOnInit() {
    this.getWorkspace();
  }

  getWorkspace() {
    this.modelService.getDefaultPath('workspace')
      .subscribe(repoPath => {
        this.workspacePath = repoPath['path'];
        this.modelService.workspacePath = repoPath['path'];
      },
        error => {
          if (error.error === 'Access denied') {
            this.modelService.getToken()
              .subscribe(response => {
                this.modelService.setToken(response['token']);
              });
          }
        }
      );
  }

  openUrl(url: string) {
    window.open(url);
  }

  openDialog() {
    const dialogRef = this.dialog.open(FileBrowserComponent, {
      width: '60%',
      height: '60%',
      data: {
        path: this.modelService.workspacePath,
        filter: 'directories'
      }
    });

    dialogRef.afterClosed().subscribe(chosenFile => {
      if (chosenFile) {
        this.workspacePath = chosenFile;
        this.modelService.setWorkspacePath(chosenFile)
          .subscribe(
            response => {
              this.modelService.workspacePathChange.next(true);
            },
            error => {
              this.openErrorDialog(error);
            }
          );
      }
    });;
  }

  openErrorDialog(error) {
    const dialogRef = this.dialog.open(ErrorComponent, {
      data: error
    });
  }

}
