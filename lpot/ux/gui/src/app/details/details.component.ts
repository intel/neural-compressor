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
import { OnInit } from '@angular/core';
import { Component, Input, OnChanges } from '@angular/core';
import { MatDialog } from '@angular/material/dialog';
import { ActivatedRoute, Router } from '@angular/router';
import { ColorHelper, ScaleType } from '@swimlane/ngx-charts';
import { environment } from 'src/environments/environment';
import { ErrorComponent } from '../error/error.component';
import { GraphComponent } from '../graph/graph.component';
import { ModelService } from '../services/model.service';
import { SocketService } from '../services/socket.service';

@Component({
  selector: 'app-details',
  templateUrl: './details.component.html',
  styleUrls: ['./details.component.scss', './../error/error.component.scss', './../model-list/model-list.component.scss']
})
export class DetailsComponent implements OnInit, OnChanges {

  @Input() model;

  token = '';
  apiBaseUrl = environment.baseUrl;
  showSpinner = true;
  graph = {};
  showGraphButton = false;

  accuracyData;
  performanceData;
  sizeData;
  executionDetails = {};
  view: any[] = window.innerWidth < 1500 ? [273, 200] : [307, 300];

  // options
  labels = ['Input', 'Optimized'];
  colors;
  showLabels: boolean = true;
  animations: boolean = true;
  xAxis: boolean = true;
  yAxis: boolean = true;
  showYAxisLabel: boolean = true;
  showXAxisLabel: boolean = true;
  timeline: boolean = true;

  colorScheme = {
    domain: ['#004A86',
      '#0095CA',
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
    private activatedRoute: ActivatedRoute,
    private modelService: ModelService,
    private socketService: SocketService,
    public dialog: MatDialog,
    private router: Router
  ) {
    this.router.routeReuseStrategy.shouldReuseRoute = () => false;
  }

  onResize(event) {
    this.view = window.innerWidth < 1500 ? [273, 200] : [307, 300];
  }

  ngOnInit() {
    this.startDetails();
    this.token = this.modelService.getToken();
    this.socketService.optimizationFinish$
      .subscribe(result => {
        if (result['data']) {
          if (result['status'] === 'success') {
            this.updateResult(result);
            this.executionDetails['optimization'] = result['data']['execution_details']['optimization'];
          } else {
            this.model['status'] = result['status'];
            this.openErrorDialog(result['data']['message']);
          }
        }
      });
    this.socketService.benchmarkFinish$
      .subscribe(result => {
        if (result['data'] && this.model) {
          if (result['status'] === 'success') {
            this.model['perf_throughput_input_model'] = result['data']['perf_throughput_input_model'];
            this.model['perf_throughput_optimized_model'] = result['data']['perf_throughput_optimized_model'];
            this.executionDetails['input_model_benchmark'] = result['data']['execution_details']['input_model_benchmark'];
            this.executionDetails['optimized_model_benchmark'] = result['data']['execution_details']['optimized_model_benchmark'];
            if (result['data']['acc_input_model']) {
              this.model['acc_input_model'] = result['data']['acc_input_model'];
              this.model['acc_optimized_model'] = result['data']['acc_optimized_model'];
            }
            if (result['data']['current_step'] === result['data']['number_of_steps']) {
              this.model['status'] = result['status'];
              this.getDataForChart();
            }
          } else {
            this.openErrorDialog(result['data']['message']);
          }
        }
      });
  }

  ngOnChanges() {
    this.startDetails();
  }

  startDetails(): void {
    if ((this.model && this.model.id === this.activatedRoute.snapshot.params.id) || (this.model && !this.activatedRoute.snapshot.params.id)) {
      this.getDataForChart();
      this.showSpinner = false;
    } else {
      this.modelService.getDefaultPath('workspace')
        .subscribe(
          response => {
            this.modelService.getAllModels()
              .subscribe(
                list => {
                  this.model = list['workloads_list'].filter(x => x.id === this.activatedRoute.snapshot.params.id)[0];
                  this.showSpinner = false;

                  if (this.model) {
                    this.getDataForChart();

                    this.modelService.getModelGraph(this.model['model_path'])
                      .subscribe(
                        graph => {
                          this.graph = graph;
                          this.showGraphButton = true;
                        },
                        error => {
                          this.showGraphButton = false;
                        }
                      );
                  }
                },
                error => {
                  this.openErrorDialog(error);
                });
          },
          error => {
            this.openErrorDialog(error);
          });
    }
  }

  openErrorDialog(error) {
    const dialogRef = this.dialog.open(ErrorComponent, {
      data: error
    });
  }

  getDataForChart() {
    this.labels = ['Original ' + this.model['input_precision'], 'Optimized ' + this.model['output_precision']];
    this.colors = new ColorHelper(this.colorScheme, ScaleType.Ordinal, this.labels, this.colorScheme);

    this.accuracyData = [{
      "name": "Accuracy [%]",
      "series": [
        {
          "value": this.model['acc_input_model'] ? this.model['acc_input_model'] * 100 : 0,
          "name": this.labels[0]
        },
        {
          "value": this.model['acc_optimized_model'] ? this.model['acc_optimized_model'] * 100 : 0,
          "name": this.labels[1]
        },
      ]
    }];

    this.performanceData = [{
      "name": "Throughput [sample/sec]",
      "series": [
        {
          "value": this.model['perf_throughput_input_model'] ? this.model['perf_throughput_input_model'] : 0,
          "name": this.labels[0]
        },
        {
          "value": this.model['perf_throughput_optimized_model'] ? this.model['perf_throughput_optimized_model'] : 0,
          "name": this.labels[1]
        },
      ]
    }];

    this.sizeData = [{
      "name": "Size [MB]",
      "series": [
        {
          "value": this.model['size_input_model'] ? this.model['size_input_model'] : 0,
          "name": this.labels[0]
        },
        {
          "value": this.model['size_optimized_model'] ? this.model['size_optimized_model'] : 0,
          "name": this.labels[1]
        },
      ]
    }];
  }

  getTooltip(execution_details): string | null {
    if (execution_details) {
      let tooltip = '';
      if (execution_details.input_model_benchmark) {
        tooltip += 'INPUT MODEL BENCHMARK\n\n';
        if (execution_details.input_model_benchmark.accuracy) {
          tooltip += 'ACCURACY\n' +
            'cores per instance: ' + execution_details.input_model_benchmark.accuracy.cores_per_instance + '\n' +
            'instances: ' + execution_details.input_model_benchmark.accuracy.instances + '\n\n';
        }
        if (execution_details.input_model_benchmark.performance) {
          tooltip += 'PERFORMANCE\n' +
            'cores per instance: ' + execution_details.input_model_benchmark.performance.cores_per_instance + '\n' +
            'instances: ' + execution_details.input_model_benchmark.performance.instances + '\n\n\n';
        }
      }
      if (execution_details.optimized_model_benchmark) {
        tooltip += 'OUTPUT MODEL BENCHMARK\n\n';
        if (execution_details.optimized_model_benchmark.accuracy) {
          tooltip += 'ACCURACY\n' +
            'cores per instance: ' + execution_details.optimized_model_benchmark.accuracy.cores_per_instance + '\n' +
            'instances: ' + execution_details.optimized_model_benchmark.accuracy.instances + '\n\n';
        }
        if (execution_details.optimized_model_benchmark.performance) {
          tooltip += 'PERFORMANCE\n' +
            'cores per instance: ' + execution_details.optimized_model_benchmark.performance.cores_per_instance + '\n' +
            'instances: ' + execution_details.optimized_model_benchmark.performance.instances + '\n\n\n';
        }
      }
      if (execution_details.optimization) {
        tooltip += 'OPTIMIZATION\n' +
          'cores per instance: ' + execution_details.optimization.cores_per_instance + '\n' +
          'instances: ' + execution_details.optimization.instances + '\n\n';
      }
      return tooltip;
    }
    return null;
  }

  optimize() {
    this.clearModel();
    this.model['status'] = 'wip';
    this.modelService.optimize(this.model)
      .subscribe(
        response => { },
        error => {
          this.openErrorDialog(error);
        }
      );
  }

  updateResult(result: {}) {
    if (this.model) {
      this.model['message'] = result['data']['message'];
      this.model['acc_input_model'] = result['data']['acc_input_model'];
      this.model['acc_optimized_model'] = result['data']['acc_optimized_model'];
      this.model['optimization_time'] = result['data']['optimization_time'];
      this.model['size_optimized_model'] = result['data']['size_optimized_model'];
      this.model['model_output_path'] = result['data']['model_output_path'];
      this.model['input_precision'] = result['data']['execution_details']['optimization']['input_precision'];
      this.model['output_precision'] = result['data']['execution_details']['optimization']['output_precision'];
      this.model['perf_throughput_fp32'] = null;
      this.model['perf_throughput_int8'] = null;
    }
  }

  showGraph(model) {
    this.showSpinner = true;
    let height = window.innerHeight < 1000 ? '99%' : '95%';
    this.showSpinner = false;
    this.dialog.open(GraphComponent, {
      width: '90%',
      height: height,
      data: {
        graph: this.graph,
        modelPath: model['model_path'],
        viewSize: [window.innerWidth * 0.9, window.innerHeight * 0.9]
      }
    });
  }

  clearModel() {
    ["acc_input_model", "acc_optimized_model", "optimization_time", "perf_throughput_input_model", "perf_throughput_optimized_model", "status", "execution_details", "size_optimized_model"].forEach(property => {
      this.model[property] = null;
    });
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

  getFileName(path: string): string {
    return path.replace(/^.*[\\\/]/, '');
  }

}
