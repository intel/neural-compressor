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
import { AfterViewInit, Component, Input, OnInit } from '@angular/core';
import { MatDialog } from '@angular/material/dialog';
import { ActivatedRoute } from '@angular/router';
import { Color, ColorHelper, ScaleType } from '@swimlane/ngx-charts';
import { environment } from 'src/environments/environment';
import { ConfirmationDialogComponent } from '../confirmation-dialog/confirmation-dialog.component';
import { BenchmarkFormComponent } from '../benchmark-form/benchmark-form.component';
import { DatasetFormComponent } from '../dataset-form/dataset-form.component';
import { ModelService } from '../services/model.service';
import { SocketService } from '../services/socket.service';
import { ShortcutInput } from 'ng-keyboard-shortcuts';

declare let require: any;
const shajs = require('sha.js');

@Component({
  selector: 'app-benchmarks',
  templateUrl: './benchmarks.component.html',
  styleUrls: ['./benchmarks.component.scss',
    './../error/error.component.scss',
    './../home/home.component.scss',
    './../datasets/datasets.component.scss',
    './../optimizations/optimizations.component.scss']
})
export class BenchmarksComponent implements OnInit, AfterViewInit {
  @Input() framework;
  @Input() domain;
  @Input() domainFlavour;

  apiBaseUrl = environment.baseUrl;
  token = '';

  shortcuts: ShortcutInput[] = [];
  benchmarks = [];
  activeBenchmarkId = -1;
  benchmarkDetails: any;
  comparison = {};
  showComparison = false;
  hiddenFields = {
    tensorflow: ['id', 'supports'],
    pytorch: ['id', 'supports', 'nodes', 'domain']
  };

  accuracyData;
  throughputData;
  view: any[] = [307, 300];

  // options

  throughputLegend = [];
  accuracyLegend = [];

  showLabels = true;
  animations = true;
  xAxis = true;
  yAxis = true;
  yScaleMin: number;
  yScaleMax: number;
  showYAxisLabel = true;
  showXAxisLabel = true;
  timeline = true;
  fontColor = localStorage.getItem('darkMode') === 'darkMode' ? '#fff' : '#000';

  customColor = [
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
  ];

  colorDomain = { domain: this.customColor };
  colorScheme: Color = { domain: this.customColor, group: ScaleType.Ordinal, selectable: true, name: 'Customer Usage' };
  colorHelper = new ColorHelper(this.colorScheme, ScaleType.Ordinal, this.customColor);

  fields = ['created_at', 'last_run_at', 'config_path', 'execution_command', 'log_path'];

  constructor(
    private modelService: ModelService,
    private socketService: SocketService,
    public activatedRoute: ActivatedRoute,
    public dialog: MatDialog,
  ) { }

  ngOnInit(): void {
    this.token = this.modelService.getToken();
    this.getBenchmarksList();
    this.modelService.benchmarkCreated$
      .subscribe(response => this.getBenchmarksList());
    this.socketService.benchmarkFinish$
      .subscribe((response: { data: any }) => {
        if (String(this.activatedRoute.snapshot.params.id) === String(response.data.project_id)) {
          this.getBenchmarksList();
          if (this.activeBenchmarkId > 0) {
            this.getBenchmarkDetails(this.activeBenchmarkId);
          }
        }
      });
    this.modelService.projectChanged$
      .subscribe((response: { id: number }) => {
        this.getBenchmarksList(response.id);
        this.comparison = {};
        this.showComparison = false;
        this.activeBenchmarkId = -1;
        this.benchmarkDetails = null;
      });
    this.modelService.colorMode$
      .subscribe(resp => {
        this.fontColor = resp === '' ? '#000' : '#fff';
      });
  }

  ngAfterViewInit(): void {
    this.shortcuts.push(
      {
        key: 'shift + alt + b',
        preventDefault: true,
        command: e => this.addBenchmark()
      },
    );
  }

  getBenchmarksList(id?: number) {
    this.modelService.getBenchmarksList(id ?? this.activatedRoute.snapshot.params.id)
      .subscribe(
        (response: { benchmarks: [] }) => {
          this.benchmarks = response.benchmarks;
          if (this.activeBenchmarkId > 0) {
            this.getBenchmarkDetails(this.activeBenchmarkId);
          }
        },
        error => {
          this.modelService.openErrorDialog(error);
        });
  }

  addBenchmark() {
    const dialogRef = this.dialog.open(BenchmarkFormComponent, {
      maxWidth: '90vw',
      maxHeight: '90vh',
      data:
      {
        projectId: this.activatedRoute.snapshot.params.id,
        index: this.benchmarks.length,
        framework: this.framework.toLowerCase()
      }
    });

    this.modelService.openDatasetDialog$.subscribe(
      response => this.addDataset()
    );
  }

  addDataset() {
    const dialogRef = this.dialog.open(DatasetFormComponent, {
      maxWidth: '90vw',
      maxHeight: '90vh',
      restoreFocus: true,
      data: {
        projectId: 1,
        index: 2,
        framework: 'TensorFlow',
        domain: 'Image Recognition',
        domainFlavour: ''
      }
    });
  }

  executeBenchmark(benchmarkId: number) {
    const dateTime = Date.now();
    const requestId = shajs('sha384').update(String(dateTime)).digest('hex');

    this.benchmarks.find(benchmark => benchmark.id === benchmarkId).status = 'wip';
    this.benchmarks.find(benchmark => benchmark.id === benchmarkId).requestId = requestId;
    this.modelService.executeBenchmark(benchmarkId, requestId)
      .subscribe(
        response => { },
        error => {
          this.modelService.openErrorDialog(error);
        }
      );
  }

  getBenchmarkDetails(id) {
    this.activeBenchmarkId = id;
    this.modelService.getBenchmarkDetails(id)
      .subscribe(
        response => {
          this.benchmarkDetails = response;
        },
        error => {
          this.modelService.openErrorDialog(error);
        });
  }

  editBenchmark(id: number) {
    const dialogRef = this.dialog.open(BenchmarkFormComponent, {
      maxWidth: '90vw',
      maxHeight: '90vh',
      data:
      {
        projectId: this.activatedRoute.snapshot.params.id,
        benchmarkId: id,
        editing: true,
        index: this.benchmarks.length,
        framework: this.framework.toLowerCase()
      }
    });
  }

  compare() {
    const accuracySeries = [];
    const throughputSeries = [];
    this.throughputLegend = [];
    this.accuracyLegend = [];
    let record: any;

    Object.keys(this.comparison).forEach(benchmarkId => {
      if (this.comparison[benchmarkId]) {
        record = this.benchmarks.find(benchmark => String(benchmark.id) === String(benchmarkId));
        if (record.result?.accuracy || record.result?.accuracy === 0) {
          this.accuracyLegend.push(record.name);
          accuracySeries.push({
            value: record.result.accuracy,
            name: record.name
          });
        }
        if (record.result?.performance) {
          this.throughputLegend.push(record.name);
          throughputSeries.push({
            value: record.result.performance,
            name: record.name
          });
        }
      }
    });

    if (accuracySeries.length) {
      this.accuracyData = [{
        name: 'Accuracy',
        series: accuracySeries
      }];
    }

    if (throughputSeries.length) {
      this.throughputData = [{
        name: 'Throughput',
        series: throughputSeries
      }];
    }

    this.showComparison = true;
  }

  deleteBenchmark(id: number, name: string) {
    const dialogRef = this.dialog.open(ConfirmationDialogComponent, {
      data: {
        what: 'benchmark',
        id,
        name
      }
    });

    dialogRef.afterClosed().subscribe(
      response => {
        if (response.confirm) {
          this.modelService.delete('benchmark', id, name)
            .subscribe(
              deleted =>
                this.modelService.projectChanged$.next({ id: this.activatedRoute.snapshot.params.id, tab: 'benchmarks' }),
              error =>
                this.modelService.openErrorDialog(error)
            );
        }
      });
  }

  isParameterVisible(parameter: string): boolean {
    let isVisible = true;
    this.hiddenFields[this.framework.toLowerCase()]
      .forEach(field => {
        if (parameter.includes(field)) {
          isVisible = false;
          return;
        }
      });
    return isVisible;
  }

  openLogs(id: number) {
    const autoRefreshTime = this.getAutoRefreshTime(id);
    window.open(`${this.apiBaseUrl}api/benchmark/output.log?id=${id}&autorefresh=${autoRefreshTime}&token=${this.token}`, '_blank');
  }

  getAutoRefreshTime(id: number): number {
    if (this.benchmarks.find(benchmark => benchmark.id === id).status === 'wip') {
      return 3;
    }
    return 0;
  }

  typeOf(object) {
    return typeof object;
  }

  copyToClipboard(text: string) {
    const textArea = document.createElement('textarea');
    textArea.value = text;
    document.body.appendChild(textArea);
    textArea.select();
    try {
      document.execCommand('copy');
    } catch (err) {
      console.error('Unable to copy', err);
    }
    document.body.removeChild(textArea);
  }

}
