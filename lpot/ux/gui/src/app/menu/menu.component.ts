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
import { ModelService } from '../services/model.service';
import { SocketService } from '../services/socket.service';

@Component({
  selector: 'app-menu',
  templateUrl: './menu.component.html',
  styleUrls: ['./menu.component.scss', './../error/error.component.scss'],
})
export class MenuComponent implements OnInit {

  showSpinner = true;
  modelList = [];

  constructor(
    private modelService: ModelService,
    private socketService: SocketService,
    public dialog: MatDialog
  ) { }

  ngOnInit() {
    this.modelService.getSystemInfo();

    this.modelService.configurationSaved
      .subscribe(result => {
        this.getAllModels();
      });
    this.socketService.optimizationStart$
      .subscribe(result => {
        this.getAllModels();
      });
    this.socketService.optimizationFinish$
      .subscribe(result => {
        this.getAllModels();
      });
    this.socketService.benchmarkStart$
      .subscribe(result => {
        this.getAllModels();
      });
    this.socketService.benchmarkFinish$
      .subscribe(result => {
        if (result['data'] && result['data']['current_step'] === result['data']['number_of_steps']) {
          this.getAllModels();
        }
      });
    this.getAllModels();
  }

  getAllModels() {
    this.modelService.getDefaultPath('workspace')
      .subscribe(
        response => {
          this.modelService.getAllModels()
            .subscribe(
              list => {
                this.showSpinner = false;
                this.modelList = list['workloads_list'];
              },
              error => {
                this.showSpinner = false;
                this.openErrorDialog(error);
              });
        },
        error => {
          this.showSpinner = false;
          this.openErrorDialog(error);
        });
  }

  openErrorDialog(error) {
    const dialogRef = this.dialog.open(ErrorComponent, {
      data: error
    });
  }

  getFileName(path: string): string {
    return path.replace(/^.*[\\\/]/, '');
  }

  getDate(date: string) {
    return new Date(date);
  }

}
