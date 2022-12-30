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
import { AfterViewInit, Component, ElementRef, Input, OnInit, ViewChild } from '@angular/core';
import { FormControl, FormGroup } from '@angular/forms';
import { MatDialog } from '@angular/material/dialog';
import { ActivatedRoute, Router } from '@angular/router';
import * as saveAs from 'file-saver';
import { ShortcutInput } from 'ng-keyboard-shortcuts';
import { environment } from 'src/environments/environment';
import { ConfirmationDialogComponent } from '../confirmation-dialog/confirmation-dialog.component';
import { DatasetFormComponent } from '../dataset-form/dataset-form.component';
import { OptimizationFormComponent } from '../optimization-form/optimization-form.component';
import { PinBenchmarkComponent } from '../pin-benchmark/pin-benchmark.component';
import { PruningComponent } from '../pruning/pruning.component';
import { ModelService } from '../services/model.service';
import { SocketService } from '../services/socket.service';
declare let require: any;
const shajs = require('sha.js');

@Component({
  selector: 'app-optimizations',
  templateUrl: './optimizations.component.html',
  styleUrls: ['./optimizations.component.scss', './../error/error.component.scss', './../home/home.component.scss',
    './../datasets/datasets.component.scss']
})
export class OptimizationsComponent implements OnInit, AfterViewInit {

  @ViewChild('accChart', { read: ElementRef, static: false }) accChartRef: ElementRef;
  @ViewChild('perfChart', { read: ElementRef, static: false }) perfChartRef: ElementRef;
  @ViewChild(PruningComponent) pruningComponent: PruningComponent;

  @Input() framework: string;
  @Input() domain: string;
  @Input() domainFlavour: string;
  @Input() supportsPruning: boolean;

  shortcuts: ShortcutInput[] = [];
  apiBaseUrl = environment.baseUrl;
  token = '';

  model = {};
  historyData = {
    accuracy: null,
    performance: null
  };
  optimizations = [];
  activeOptimizationId = 0;
  requestId = '';
  optimizationDetails: any;
  parsedOptimizationDetails: { batch_size: number; sampling_size: number };
  pinnedAccuracyBenchmarks = {};
  pinnedPerformanceBenchmarks = {};
  allBenchmarks = [];
  availableAccuracyBenchmarks = {};
  availablePerformanceBenchmarks = {};
  showAccuracyDropdown = {};
  showPerformanceDropdown = {};
  labels = ['Input', 'Optimized'];
  tuningDetailsEditable = false;

  detailsFormGroup: FormGroup;

  hiddenFields = {
    onnxrt: ['id', 'supports', 'nodes', 'domain'],
    pytorch: ['id', 'supports', 'nodes', 'domain'],
    tensorflow: ['id', 'supports'],
  };

  xAxis = true;
  yAxis = true;
  showYAxisLabel = true;
  showXAxisLabel = true;
  viewLine: any[] = [600, 300];
  referenceLines = {
    accuracy: {},
    performance: {}
  };
  chartsReady = false;
  fontColor = localStorage.getItem('darkMode') === 'darkMode' ? '#fff' : '#000';

  colorScheme = {
    domain: [
      '#0095CA',
      '#004A86',
      '#EDB200',
      '#B24501',
      '#41728A',
      '#525252',
      '#653171',
      '#708541',
      '#000F8A',
      '#C81326',
      '#005B85',
      '#183544',
      '#515A3D',
      '#C98F00',]
  };

  constructor(
    private modelService: ModelService,
    private socketService: SocketService,
    public activatedRoute: ActivatedRoute,
    public dialog: MatDialog,
    private router: Router) {
  }

  ngOnInit() {
    this.initializeOptimizations();
    this.token = this.modelService.getToken();
    this.modelService.projectChanged$
      .subscribe((response: { id: number }) => {
        this.getOptimizations(response.id);
        this.optimizationDetails = null;
        this.activeOptimizationId = -1;
      });
    this.modelService.colorMode$
      .subscribe(resp => {
        this.fontColor = resp === '' ? '#000' : '#fff';
      });
  }

  ngAfterViewInit(): void {
    this.shortcuts.push(
      {
        key: 'shift + alt + o',
        preventDefault: true,
        command: e => this.addOptimization()
      },
    );
  }

  initializeOptimizations(): void {
    this.getOptimizations();
    this.modelService.optimizationCreated$
      .subscribe(response => this.getOptimizations());
    this.socketService.benchmarkFinish$
      .subscribe(response => this.getOptimizations());
    this.socketService.optimizationFinish$
      .subscribe((response: { data: any }) => {
        if (String(this.activatedRoute.snapshot.params.id) === String(response.data.project_id)) {
          this.getOptimizations();
          if (this.activeOptimizationId > 0) {
            this.getOptimizationDetails(this.activeOptimizationId);
          }
        }
      });
    this.socketService.tuningHistory$
      .subscribe((response: { data: any; status: string }) => {
        if (response.status === 'success' && this.activeOptimizationId === response.data.optimization_id) {
          this.getHistoryData(response.data);
        }
      });
  }

  getOptimizations(id?: number) {
    this.modelService.getOptimizationList(id ?? this.activatedRoute.snapshot.params.id)
      .subscribe(
        (response: { optimizations: any }) => {
          this.optimizations = response.optimizations;
          this.getBenchmarksList();
        },
        error => {
          this.modelService.openErrorDialog(error);
        });
  }

  getBenchmarksList(id?: number) {
    this.modelService.getBenchmarksList(id ?? this.activatedRoute.snapshot.params.id)
      .subscribe(
        (response: { benchmarks: any }) => {
          this.allBenchmarks = response.benchmarks;

          this.optimizations.forEach(optimization => {
            this.availableAccuracyBenchmarks[optimization.id] = this.allBenchmarks.filter(x =>
              x.model.name.toLowerCase().replace(' ', '_') === optimization.name.toLowerCase().replace(' ', '_')
              && x.mode === 'accuracy');

            this.availablePerformanceBenchmarks[optimization.id] = this.allBenchmarks.filter(x =>
              x.model.name.toLowerCase().replace(' ', '_') === optimization.name.toLowerCase().replace(' ', '_')
              && x.mode === 'performance');

            this.pinnedAccuracyBenchmarks[optimization.id] = this.allBenchmarks
              .find(x => x.id === optimization.accuracy_benchmark_id);
            this.pinnedPerformanceBenchmarks[optimization.id] = this.allBenchmarks
              .find(x => x.id === optimization.performance_benchmark_id);
          });
        },
        error => {
          this.modelService.openErrorDialog(error);
        });
  }

  openPinDialog(mode: string, optimizationId: number) {
    let benchmarks = [];
    if (mode === 'accuracy' && this.availableAccuracyBenchmarks[optimizationId]) {
      benchmarks = this.availableAccuracyBenchmarks[optimizationId];
    } else if (mode === 'performance' && this.availablePerformanceBenchmarks[optimizationId]) {
      benchmarks = this.availablePerformanceBenchmarks[optimizationId];
    }

    if ((mode === 'accuracy' && this.availableAccuracyBenchmarks[optimizationId].length)
      || (mode === 'performance' && this.availablePerformanceBenchmarks[optimizationId].length)) {
      const dialogRef = this.dialog.open(PinBenchmarkComponent, {
        data: {
          mode,
          optimizationId,
          benchmarks
        }
      });

      dialogRef.afterClosed()
        .subscribe(response => {
          if (response) {
            if (mode === 'accuracy') {
              this.pinnedAccuracyBenchmarks[optimizationId] = this.allBenchmarks.find(x => x.id === response.chosenBenchmarkId);
            } else if (mode === 'performance') {
              this.pinnedPerformanceBenchmarks[optimizationId] = this.allBenchmarks.find(x => x.id === response.chosenBenchmarkId);
            }
            this.getOptimizations();
          }
        });
    }
  }

  getOptimizationDetails(id) {
    this.activeOptimizationId = id;
    this.modelService.getOptimizationDetails(id)
      .subscribe(
        (response: { tuning_details: any }) => {
          this.tuningDetailsEditable = false;
          this.optimizationDetails = response;
          this.parseOptimizationDetails(response);
          if (response.tuning_details && response.tuning_details.tuning_history) {
            this.getHistoryData(response.tuning_details.tuning_history);
          } else {
            this.historyData = {
              accuracy: null,
              performance: null
            };;
            this.chartsReady = false;
          }
        },
        error => {
          this.modelService.openErrorDialog(error);
        });
  }

  parseOptimizationDetails(response) {
    this.parsedOptimizationDetails = {
      batch_size: response.batch_size,
      sampling_size: response.sampling_size
    };
    if (response.tuning_details) {
      Object.keys(response.tuning_details).forEach(key => {
        if (!['id', 'tuning_history'].includes(key)) {
          if (typeof response.tuning_details[key] === 'string' || typeof response.tuning_details[key] === 'number') {
            this.parsedOptimizationDetails[key] = response.tuning_details[key];
          } else if (typeof response.tuning_details[key] === 'object' && response.tuning_details[key]) {
            Object.keys(response.tuning_details[key]).forEach(objectKey => {
              this.parsedOptimizationDetails[objectKey] = response.tuning_details[key][objectKey];
            });
          }
        }
      });
    }
    const formGroupDetails = {};
    Object.keys(this.parsedOptimizationDetails).forEach(key => {
      formGroupDetails[key] = new FormControl(this.parsedOptimizationDetails[key]);
    });
    this.detailsFormGroup = new FormGroup(formGroupDetails);
  }

  addOptimization() {
    const dialogRef = this.dialog.open(OptimizationFormComponent, {
      width: '60%',
      data:
      {
        projectId: this.activatedRoute.snapshot.params.id,
        index: this.optimizations.length,
        framework: this.framework.toLowerCase(),
        supportsPruning: this.supportsPruning
      }
    });

    this.modelService.openDatasetDialog$.subscribe(
      response => this.addDataset()
    );
  }

  addDataset() {
    const dialogRef = this.dialog.open(DatasetFormComponent, {
      width: '60%',
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

  executeOptimization(optimizationId) {
    const dateTime = Date.now();
    this.requestId = shajs('sha384').update(String(dateTime)).digest('hex');

    this.optimizations.find(optimization => optimization.id === optimizationId).status = 'wip';
    this.optimizations.find(optimization => optimization.id === optimizationId).requestId = this.requestId;
    this.modelService.executeOptimization(optimizationId, this.requestId)
      .subscribe(
        response => { },
        error => {
          this.modelService.openErrorDialog(error);
        }
      );
  }

  saveTuningDetails() {
    const parsedTuningDetails = this.detailsFormGroup.value;
    parsedTuningDetails.exit_policy = {};
    parsedTuningDetails.exit_policy.timeout = this.detailsFormGroup.value.timeout;
    parsedTuningDetails.exit_policy.max_trials = this.detailsFormGroup.value.max_trials;
    delete parsedTuningDetails.timeout;
    delete parsedTuningDetails.max_trials;
    this.modelService.editOptimization({
      id: this.activeOptimizationId,
      tuning_details: parsedTuningDetails
    })
      .subscribe(
        response => {
          this.tuningDetailsEditable = false;
          this.getOptimizationDetails(this.activeOptimizationId);
        },
        error => this.modelService.openErrorDialog(error)
      );
  }

  getHistoryData(result) {
    ['accuracy', 'performance'].forEach(type => {
      this.historyData[type] = [{
        name: type,
        series: []
      }];

      result.history.forEach((record, index) => {
        if (result['baseline_' + type]) {
          this.historyData[type][0].series.push({
            name: index + 1,
            value: record[type][0]
          });
        }
      });

      this.referenceLines[type] = [{
        name: 'baseline ' + type,
        value: result['baseline_' + type]
      }];
    });

    if (this.historyData.accuracy[0].series.length || this.historyData.performance[0].series) {
      this.chartsReady = true;
    }

    setTimeout(() => { this.fixChart(); }, 1000);
  }

  fixChart() {
    if (this.accChartRef) {
      this.accChartRef.nativeElement.querySelectorAll('g.line-series path').forEach((el) => {
        el.setAttribute('stroke-width', '10');
        el.setAttribute('stroke-linecap', 'round');
      });
      this.perfChartRef.nativeElement.querySelectorAll('g.line-series path').forEach((el) => {
        el.setAttribute('stroke-width', '10');
        el.setAttribute('stroke-linecap', 'round');
      });
    }
  }

  downloadModel(filePath: string, modelId: number) {
    this.modelService.downloadModel(modelId)
      .subscribe(
        data =>
          saveAs(data, this.getFileName(filePath)),
        error =>
          this.modelService.openErrorDialog(error)
      );
  }

  editOptimization(id: number) {
    const dialogRef = this.dialog.open(OptimizationFormComponent, {
      width: '60%',
      data:
      {
        projectId: this.activatedRoute.snapshot.params.id,
        optimizationId: id,
        editing: true,
        index: this.optimizations.length,
        framework: this.framework.toLowerCase()
      }
    });
  }

  deleteOptimization(id: number, name: string) {
    const dialogRef = this.dialog.open(ConfirmationDialogComponent, {
      data: {
        what: 'optimization',
        id,
        name
      }
    });

    dialogRef.afterClosed().subscribe(
      response => {
        if (response.confirm) {
          this.modelService.delete('optimization', id, name)
            .subscribe(
              deleted =>
                this.modelService.projectChanged$.next({ id: this.activatedRoute.snapshot.params.id, tab: 'optimizations' }),
              error =>
                this.modelService.openErrorDialog(error)
            );
        }
      });
  }

  axisFormat(val) {
    if (val % 1 === 0) {
      return val.toLocaleString();
    } else {
      return '';
    }
  }

  goToBenchmarks() {
    this.router.navigate(['project', this.activatedRoute.snapshot.params.id, 'benchmarks'], { queryParamsHandling: 'merge' });
    this.modelService.projectChanged$.next({ id: this.activatedRoute.snapshot.params.id, tab: 'benchmarks' });
  }

  getFileName(path: string): string {
    return path.replace(/^.*[\\\/]/, '');
  }

  openLogs(id: number) {
    const autoRefreshTime = this.getAutoRefreshTime(id);
    window.open(`${this.apiBaseUrl}api/optimization/output.log?id=${id}&autorefresh=${autoRefreshTime}&token=${this.token}`, '_blank');
  }

  getAutoRefreshTime(id: number): number {
    if (this.optimizations.find(optimization => optimization.id === id).status === 'wip') {
      return 3;
    }
    return 0;
  }

  isParameterVisible(parameter: string): boolean {
    let isVisible = true;
    let hiddenFields = ['id', 'supports'];
    if (this.hiddenFields[this.framework.toLowerCase()]) {
      hiddenFields = this.hiddenFields[this.framework.toLowerCase()];
    }
    hiddenFields.forEach(field => {
      if (parameter.includes(field)) {
        isVisible = false;
        return;
      }
    });
    return isVisible;
  }

  typeOf(object) {
    return typeof object;
  }

  getType(value) {
    if (typeof value === 'number') { return 'number'; }
    return 'text';
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
