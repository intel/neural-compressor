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
import { Component, Input, OnInit } from '@angular/core';
import { MatDialog } from '@angular/material/dialog';
import { ActivatedRoute, NavigationEnd, Router } from '@angular/router';
import { environment } from 'src/environments/environment';
import { OptimizationFormComponent } from '../optimization-form/optimization-form.component';
import { ModelService } from '../services/model.service';
import { SocketService } from '../services/socket.service';
declare var require: any;
var shajs = require('sha.js');

@Component({
  selector: 'app-optimizations',
  templateUrl: './optimizations.component.html',
  styleUrls: ['./optimizations.component.scss', './../error/error.component.scss', './../home/home.component.scss', './../datasets/datasets.component.scss']
})
export class OptimizationsComponent implements OnInit {

  apiBaseUrl = environment.baseUrl;
  token = '';

  @Input() framework: string;
  model = {};
  historyData = {};
  optimizations = [];
  activeOptimizationId = 0;
  requestId = '';
  optimizationDetails: any;
  labels = ['Input', 'Optimized'];

  xAxis: boolean = true;
  yAxis: boolean = true;
  yScaleMin: number;
  yScaleMax: number;
  showYAxisLabel: boolean = true;
  showXAxisLabel: boolean = true;
  viewLine: any[] = [700, 300];
  accuracyReferenceLines = {};

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
      .subscribe(response => {
        this.getOptimizations(response['id']);
        this.optimizationDetails = null;
        this.activeOptimizationId = -1;
      });
  }

  initializeOptimizations(): void {
    this.getOptimizations();
    this.modelService.optimizationCreated$
      .subscribe(response => this.getOptimizations());
    this.socketService.optimizationFinish$
      .subscribe(response => {
        if (String(this.activatedRoute.snapshot.params.id) === String(response['data']['project_id'])) {
          this.getOptimizations();
          if (this.activeOptimizationId > 0) {
            this.getOptimizationDetails(this.activeOptimizationId);
          }
        }
      });
  }

  getOptimizations(id?: number) {
    this.modelService.getOptimizationList(id ?? this.activatedRoute.snapshot.params.id)
      .subscribe(
        response => {
          this.optimizations = response['optimizations'];
        },
        error => {
          this.modelService.openErrorDialog(error);
        });
  }

  getOptimizationDetails(id) {
    this.activeOptimizationId = id;
    this.modelService.getOptimizationDetails(id)
      .subscribe(
        response => {
          this.optimizationDetails = response;
        },
        error => {
          this.modelService.openErrorDialog(error);
        });
  }

  addOptimization() {
    const dialogRef = this.dialog.open(OptimizationFormComponent, {
      width: '60%',
      data:
      {
        projectId: this.activatedRoute.snapshot.params.id,
        index: this.optimizations.length,
        framework: this.framework
      }
    });
  }

  executeOptimization(optimizationId) {
    const dateTime = Date.now();
    this.requestId = shajs('sha384').update(String(dateTime)).digest('hex');

    this.optimizations.find(optimization => optimization.id === optimizationId)['status'] = 'wip';
    this.optimizations.find(optimization => optimization.id === optimizationId)['requestId'] = this.requestId;
    this.modelService.executeOptimization(optimizationId, this.requestId)
      .subscribe(
        response => { },
        error => {
          this.modelService.openErrorDialog(error);
        }
      );
  }

  getHistoryData(result) {
    this.historyData['accuracy'] = [{
      "name": "Accuracy",
      "series": []
    }];
    this.historyData['performance'] = [{
      "name": "Performance",
      "series": []
    }];
    this.yScaleMin = result['data']['history'][0]['accuracy'];
    this.yScaleMax = result['data']['history'][0]['accuracy'];

    result['data']['history'].forEach((record, index) => {
      if (this.historyData['data']['baseline_performance']) {
        this.historyData['performance'][0]['series'].push({
          name: index + 1,
          value: record['performance']
        });
      }
      this.historyData['accuracy'][0]['series'].push({
        name: index + 1,
        value: record['accuracy']
      });

      this.findMinMaxAccuracy(record['accuracy']);
    });
    this.findMinMaxAccuracy(result['data']['baseline_accuracy']);
    this.findMinMaxAccuracy(result['data']['baseline_accuracy']);
    this.setAccuracyScale();

    this.accuracyReferenceLines = [{
      name: 'baseline accuracy',
      value: result['data']['baseline_accuracy']
    },
    {
      name: 'minimal accepted accuracy',
      value: result['data']['minimal_accuracy']
    }];
  }

  findMinMaxAccuracy(accuracyValue: number) {
    accuracyValue < this.yScaleMin ? this.yScaleMin = accuracyValue : null;
    accuracyValue > this.yScaleMax ? this.yScaleMax = accuracyValue : null;
  }

  setAccuracyScale() {
    this.yScaleMin = 0.98 * this.yScaleMin;
    this.yScaleMax = 1.02 * this.yScaleMax;
  }

  axisFormat(val) {
    if (val % 1 === 0) {
      return val.toLocaleString();
    } else {
      return '';
    }
  }

  onResize() {
    if (window.innerWidth < 1500) {
      this.viewLine = [800, 200];
    } else if (window.innerWidth < 2000) {
      this.viewLine = [900, 250];
    } else {
      this.viewLine = [1000, 300];
    }
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
