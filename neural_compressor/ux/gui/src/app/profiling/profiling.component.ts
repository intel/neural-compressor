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
import { AfterViewInit, Component, OnInit } from '@angular/core';
import { MatDialog } from '@angular/material/dialog';
import { ActivatedRoute } from '@angular/router';
import * as saveAs from 'file-saver';
import { ShortcutInput } from 'ng-keyboard-shortcuts';
import { environment } from 'src/environments/environment';
import { ConfirmationDialogComponent } from '../confirmation-dialog/confirmation-dialog.component';
import { ProfilingFormComponent } from '../profiling-form/profiling-form.component';
import { ModelService } from '../services/model.service';
import { SocketService } from '../services/socket.service';

declare let require: any;
const shajs = require('sha.js');

@Component({
  selector: 'app-profiling',
  templateUrl: './profiling.component.html',
  styleUrls: ['./profiling.component.scss',
    './../error/error.component.scss',
    './../home/home.component.scss',
    './../datasets/datasets.component.scss',
    './../optimizations/optimizations.component.scss']
})
export class ProfilingComponent implements OnInit, AfterViewInit {

  shortcuts: ShortcutInput[] = [];

  apiBaseUrl = environment.baseUrl;
  token = '';

  profiling = {
    name: 'Quantized model profiling',
    project_id: 1,
    model_id: 2,
    dataset_id: 1,
    batch_size: 1,
    num_threads: 7
  };
  activeProfilingId = -1;
  profilingList = [];

  profilingData = [];
  profilingDataHeaders = [];
  profilingChartData = [{
    name: 'node',
    style: {
      text: 'red'
    },
    series: []
  }];
  showChart = false;
  showInChart = {
    0: true,
    1: true,
    2: true,
    3: true,
    4: true
  };

  xAxis = true;
  yAxis = true;
  showYAxisLabel = true;
  showXAxisLabel = true;
  viewLine: any[] = [900, 300];
  fontColor = localStorage.getItem('darkMode') === 'darkMode' ? '#fff' : '#000';

  customColor = {
    domain: [
      '#005B85',
      '#0095CA',
      '#00C7FD',
      '#047271',
      '#07b3b0',
      '#9E8A87',
      '#333471',
      '#5153B0',
      '#ED6A5E ',
      '#9D79BC',
      '#A14DA0',
    ]
  };

  constructor(
    private modelService: ModelService,
    private socketService: SocketService,
    public activatedRoute: ActivatedRoute,
    public dialog: MatDialog
  ) { }

  ngOnInit(): void {
    this.token = this.modelService.getToken();
    this.getProfilingList();
    this.modelService.projectChanged$
      .subscribe((response: { id: number }) => {
        this.getProfilingList(response.id);
        this.activeProfilingId = -1;
        this.profilingData = [];
      });
    this.socketService.profilingStart$
      .subscribe(resp => {
        this.getProfilingList();
      });
    this.socketService.profilingFinish$
      .subscribe(resp => {
        this.getProfilingList();
      });
    this.modelService.colorMode$
      .subscribe(resp => {
        this.fontColor = resp === '' ? '#000' : '#fff';
      });
  }

  ngAfterViewInit(): void {
    this.shortcuts.push(
      {
        key: 'shift + alt + p',
        preventDefault: true,
        command: e => this.openProfilingForm()
      }
    );
  }

  openProfilingForm() {
    const dialogRef = this.dialog.open(ProfilingFormComponent, {
      width: '60%',
      data:
      {
        projectId: this.activatedRoute.snapshot.params.id,
        index: this.profilingList.length
      }
    });

    dialogRef.afterClosed().subscribe(response => this.getProfilingList());
  }

  executeProfiling(profilingId: number) {
    const dateTime = Date.now();
    const requestId = shajs('sha384').update(String(dateTime)).digest('hex');

    this.profilingList.find(profiling => profiling.id === profilingId).status = 'wip';
    this.modelService.executeProfiling(profilingId, requestId)
      .subscribe(
        response => { },
        error => {
          this.modelService.openErrorDialog(error);
        }
      );
  }

  getProfilingList(id?: number) {
    this.modelService.getProfilingList(id ?? this.activatedRoute.snapshot.params.id)
      .subscribe(
        (response: { profilings: any }) => {
          this.profilingList = response.profilings;
        },
        error => {
          this.modelService.openErrorDialog(error);
        });
  }

  getProfilingDetails(id) {
    this.activeProfilingId = id;
    this.profilingData = [];
    this.modelService.getProfilingDetails(id)
      .subscribe(
        (response: { results: any }) => {
          this.profilingData = response.results?.sort((a, b) => a.total_execution_time < b.total_execution_time ? 1 : -1);
          if (this.profilingData) {
            this.profilingDataHeaders = Object.keys(this.profilingData[0]).filter(key => !key.includes('id'));
            this.showProfilingChart();
          }
        },
        error => {
          this.modelService.openErrorDialog(error);
        });
  }

  downloadProfilingData(id: number) {
    this.modelService.downloadProfiling(id)
      .subscribe(
        data =>
          saveAs(data, 'profiling' + id + '.csv'),
        error =>
          this.modelService.openErrorDialog(error)
      );
  }

  showProfilingChart() {
    this.profilingChartData[0].series = [];
    Object.keys(this.showInChart).forEach(index => {
      if (this.showInChart[index]) {
        this.profilingChartData[0].series.push({
          name: this.profilingData[index].node_name,
          value: this.profilingData[index].total_execution_time
        });
      }
    });
    this.profilingChartData = [...this.profilingChartData];
    this.showChart = true;
  }

  deleteProfiling(id: number, name: string) {
    const dialogRef = this.dialog.open(ConfirmationDialogComponent, {
      data: {
        what: 'profiling',
        id,
        name
      }
    });

    dialogRef.afterClosed().subscribe(
      response => {
        if (response.confirm) {
          this.modelService.delete('profiling', id, name)
            .subscribe(
              deleted =>
                this.getProfilingList(),
              error =>
                this.modelService.openErrorDialog(error)
            );
        }
      });
  }

  editProfiling(id: number) {
    const dialogRef = this.dialog.open(ProfilingFormComponent, {
      width: '60%',
      data: {
        projectId: this.activatedRoute.snapshot.params.id,
        profilingId: id,
        editing: true,
        index: this.profilingList.length,
      }
    });

    dialogRef.afterClosed().subscribe(
      response => {
        this.getProfilingList();
      });
  }

  openLogs(id: number) {
    const autoRefreshTime = this.getAutoRefreshTime(id);
    window.open(`${this.apiBaseUrl}api/profiling/output.log?id=${id}&autorefresh=${autoRefreshTime}&token=${this.token}`, '_blank');
  }

  getAutoRefreshTime(id: number): number {
    if (this.profilingList.find(profiling => profiling.id === id).status === 'wip') {
      return 3;
    }
    return 0;
  }

  typeOf(obj): string {
    return typeof obj;
  }

}
