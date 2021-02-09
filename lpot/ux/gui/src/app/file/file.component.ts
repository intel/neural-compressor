import { Component, Input, OnDestroy, OnInit } from '@angular/core';
import { MatDialog } from '@angular/material';
import { interval, Subscription } from 'rxjs';
import { ErrorComponent } from '../error/error.component';
import { ModelService } from '../services/model.service';

@Component({
  selector: 'app-file',
  templateUrl: './file.component.html',
  styleUrls: ['./file.component.scss']
})
export class FileComponent implements OnInit, OnDestroy {

  @Input() path: string;
  @Input() fileType: string;
  @Input() set refresh(value: boolean) {
    this.getFile();
  }

  fileText = '';
  outputSubscription: Subscription;

  constructor(
    private modelService: ModelService,
    public dialog: MatDialog
  ) { }

  ngOnInit() {
    if (this.fileType === 'output') {
      this.outputSubscription = interval(3000).subscribe(x => {
        this.getFile();
      });
    }
  }

  ngOnDestroy() {
    if (this.fileType === 'output') {
      this.outputSubscription.unsubscribe();
    }
  }

  getFile() {
    this.modelService.getFile(this.path)
      .subscribe(
        data => {
          this.fileText = String(data);
        },
        error => {
          this.openErrorDialog(error);
        });
  }

  openErrorDialog(error) {
    const dialogRef = this.dialog.open(ErrorComponent, {
      data: error
    });
  }

}
