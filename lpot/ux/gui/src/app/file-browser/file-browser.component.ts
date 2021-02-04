import { Component, Inject, OnInit } from '@angular/core';
import { MatDialogRef, MAT_DIALOG_DATA } from '@angular/material';
import { DialogComponent } from '../dialog/dialog.component';
import { ModelService } from '../services/model.service';

@Component({
  selector: 'app-file-browser',
  templateUrl: './file-browser.component.html',
  styleUrls: ['./file-browser.component.scss', './../start-page/start-page.component.scss']
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
    @Inject(MAT_DIALOG_DATA) public data
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
        }
      )
  }

  chooseFile(name: string) {
    this.chosenFile = name;
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
