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
import { ErrorComponent } from '../error/error.component';
import { FileBrowserFilter, ModelService } from '../services/model.service';

@Component({
  selector: 'app-file-browser',
  templateUrl: './file-browser.component.html',
  styleUrls: ['./file-browser.component.scss', './../error/error.component.scss']
})
export class FileBrowserComponent implements OnInit {

  contents = [];
  currentPath: string;
  chosenFile: string;
  filter: FileBrowserFilter;

  constructor(
    private modelService: ModelService,
    public dialogRef: MatDialogRef<FileBrowserComponent>,
    @Inject(MAT_DIALOG_DATA) public data,
    public dialog: MatDialog
  ) { }

  ngOnInit() {
    this.filter = this.data.filter
    this.getFileSystem(this.data.path)
  }

  getFileSystem(path: string) {
    this.modelService.getFileSystem(path, this.filter)
      .subscribe(
        resp => {
          this.contents = resp['contents'];
          this.currentPath = resp['path'];
        },
        error => {
          this.openErrorDialog(error);
        }
      )
  }

  openErrorDialog(error) {
    const dialogRef = this.dialog.open(ErrorComponent, {
      data: error
    });
  }

  chooseFile(name: string, close: boolean) {
    this.chosenFile = name;
    if (close) {
      this.dialogRef.close(this.chosenFile);
    }
  }

  currentPathChange(event) {
    this.getFileSystem(event.srcElement.value);
  }

  goToParentDirectory() {
    var pathArray = this.currentPath.split('/');
    pathArray.pop();
    this.getFileSystem(pathArray.join('/'));
  }

}
