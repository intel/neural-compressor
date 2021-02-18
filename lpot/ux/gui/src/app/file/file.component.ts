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
import { Component, Input, OnDestroy, OnInit } from '@angular/core';
import { MatDialog } from '@angular/material/dialog';
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
