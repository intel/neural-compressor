import { Component, Inject, OnInit } from '@angular/core';
import { MatDialogRef, MAT_DIALOG_DATA } from '@angular/material';

@Component({
  selector: 'app-dialog',
  templateUrl: './dialog.component.html',
  styleUrls: ['./dialog.component.scss']
})
export class DialogComponent implements OnInit {

  refresh = false;

  constructor(
    public dialogRef: MatDialogRef<DialogComponent>,
    @Inject(MAT_DIALOG_DATA) public data
  ) { }

  ngOnInit() {
  }

  getFileName(): string {
    if (this.data.path) {
      return this.data.path.replace(/^.*[\\\/]/, '');
    }
    return 'No file path'
  }

  refreshOutput() {
    this.refresh = !this.refresh
  }

}
