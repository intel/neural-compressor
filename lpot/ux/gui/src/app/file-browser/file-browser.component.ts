import { Component, Inject, OnInit } from '@angular/core';
import { MatDialog, MatDialogRef, MAT_DIALOG_DATA } from '@angular/material';
import { DialogComponent } from '../dialog/dialog.component';
import { ErrorComponent } from '../error/error.component';
import { ModelService } from '../services/model.service';

@Component({
  selector: 'app-file-browser',
  templateUrl: './file-browser.component.html',
  styleUrls: ['./file-browser.component.scss', './../error/error.component.scss']
})
export class FileBrowserComponent implements OnInit {

  contents = [];
  currentPath: string;
  chosenFile: string;
  files: boolean;
  modelsOnly: boolean;

  constructor(
    private modelService: ModelService,
    public dialogRef: MatDialogRef<DialogComponent>,
    @Inject(MAT_DIALOG_DATA) public data,
    public dialog: MatDialog
  ) { }

  ngOnInit() {
    this.files = this.data.files;
    this.modelsOnly = this.data.modelsOnly;
    this.getFileSystem(this.data.path)
  }

  getFileSystem(path: string) {
    this.modelService.getFileSystem(path, this.files, this.modelsOnly)
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
