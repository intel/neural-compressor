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
import { Component, Inject, OnInit } from '@angular/core';
import { MatDialog, MatDialogRef, MAT_DIALOG_DATA } from '@angular/material/dialog';
import { FileBrowserFilter, ModelService } from '../services/model.service';

@Component({
  selector: 'app-file-browser',
  templateUrl: './file-browser.component.html',
  styleUrls: ['./file-browser.component.scss', './../error/error.component.scss']
})
export class FileBrowserComponent implements OnInit {

  contents = [];
  foundFiles = [];
  currentPath: string;
  chosenFile: string;
  filter: FileBrowserFilter;
  showSpinner = false;
  isWindows = false;
  slash = '/';

  constructor(
    private modelService: ModelService,
    public dialogRef: MatDialogRef<FileBrowserComponent>,
    @Inject(MAT_DIALOG_DATA) public data,
    public dialog: MatDialog
  ) { }

  ngOnInit() {
    this.filter = this.data.filter;
    this.getFileSystem(this.data.path);
  }

  getFileSystem(path: string) {
    this.showSpinner = true;
    this.isWindows = this.modelService.systemInfo.systeminfo.system.toLowerCase().includes('windows');
    this.slash = this.isWindows ? '\\' : '/';
    if (path[0] !== '/' && !this.isWindows) {
      path = '/' + path;
    }
    this.modelService.getFileSystem(path, this.filter)
      .subscribe(
        (resp: { contents: any; path: string }) => {
          this.showSpinner = false;
          this.contents = resp.contents;
          this.currentPath = resp.path;
        },
        error => {
          this.showSpinner = false;
          this.modelService.openErrorDialog(error);
        }
      );
  }

  chooseFile(name: string, close: boolean) {
    this.chosenFile = name;
    if (close) {
      this.checkForFiles();
    }
  }

  checkForFiles() {
    this.modelService.getFileSystem(this.currentPath, 'all')
      .subscribe(
        (resp: { contents: any; path: string }) => {
          this.contents = resp.contents;
          this.currentPath = resp.path;
          if (this.data.filesToFind) {
            this.data.filesToFind.forEach(file => {
              if (this.contents.find(x => x.name.includes(file))) {
                this.foundFiles.push(this.contents.find(x => x.name.includes(file)));
              }
            });
          }
          this.dialogRef.close({
            chosenFile: this.chosenFile,
            foundFiles: this.foundFiles
          });
        },
        error => {
          this.modelService.openErrorDialog(error);
        }
      );
  }

  currentPathChange(event) {
    this.getFileSystem(event.srcElement.value);
  }

  goToParentDirectory() {
    const pathArray = this.currentPath.split(this.slash);
    pathArray.pop();
    this.getFileSystem(pathArray.join(this.slash));
  }

}
