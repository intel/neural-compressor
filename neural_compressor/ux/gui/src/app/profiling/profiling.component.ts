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
import { ActivatedRoute } from '@angular/router';
import { environment } from 'src/environments/environment';
import { ProfilingFormComponent } from '../profiling-form/profiling-form.component';
import { ModelService } from '../services/model.service';
import { SocketService } from '../services/socket.service';

declare var require: any;
var shajs = require('sha.js');

@Component({
  selector: 'app-profiling',
  templateUrl: './profiling.component.html',
  styleUrls: ['./profiling.component.scss',
    './../error/error.component.scss',
    './../home/home.component.scss',
    './../datasets/datasets.component.scss',
    './../optimizations/optimizations.component.scss']
})
export class ProfilingComponent implements OnInit {

  apiBaseUrl = environment.baseUrl;
  token = '';

  profiling = {
    "name": "Quantized model profiling",
    "project_id": 1,
    "model_id": 2,
    "dataset_id": 1,
    "batch_size": 1,
    "num_threads": 7
  };
  activeProfilingId = -1;
  profilingList = [];

  profilingData = [];
  profilingDataHeaders = [];
  profilingChartData = [{
    name: 'Profiling',
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

  xAxis: boolean = true;
  yAxis: boolean = true;
  showYAxisLabel: boolean = true;
  showXAxisLabel: boolean = true;
  viewLine: any[] = [900, 300];

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
    this.socketService.profilingStart$
      .subscribe(resp => {
        this.getProfilingList();
      });
    this.socketService.profilingFinish$
      .subscribe(resp => {
        this.getProfilingList();
      });
    this.modelService.projectChanged$
      .subscribe(resp => {
        this.getProfilingList();
        this.activeProfilingId = -1;
        this.profilingData = [];
      });
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
    let requestId = shajs('sha384').update(String(dateTime)).digest('hex');

    this.profilingList.find(profiling => profiling.id === profilingId).status = 'wip';
    this.modelService.executeProfiling(profilingId, requestId)
      .subscribe();
  }

  getProfilingList() {
    this.modelService.getProfilingList(this.activatedRoute.snapshot.params.id)
      .subscribe(response => {
        this.profilingList = response['profilings'];
      });
  }

  getProfilingDetails(id) {
    this.activeProfilingId = id;
    this.profilingData = [];
    this.modelService.getProfilingDetails(id)
      .subscribe(
        response => {
          this.profilingData = response['results']?.sort((a, b) => a.total_execution_time < b.total_execution_time ? 1 : -1);
          if (this.profilingData) {
            this.profilingDataHeaders = Object.keys(this.profilingData[0]).filter(key => !key.includes('id'));
            this.showProfilingChart();
          }
        });
  }

  showProfilingChart() {
    this.profilingChartData[0].series = [];
    Object.keys(this.showInChart).forEach(index => {
      if (this.showInChart[index]) {
        this.profilingChartData[0].series.push({
          name: this.profilingData[index]['node_name'],
          value: this.profilingData[index]['total_execution_time']
        });
      }
    });
    this.profilingChartData = [...this.profilingChartData];
    this.showChart = true;
  }

  typeOf(obj): string {
    return typeof obj;
  }

}
