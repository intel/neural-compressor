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
import { environment } from 'src/environments/environment';
import { Component, OnInit, ViewChild } from '@angular/core';
import { MatDialog } from '@angular/material/dialog';
import { ErrorComponent } from '../error/error.component';
import { ModelService, NewModel } from '../services/model.service';
import { SocketService } from '../services/socket.service';

@Component({
  selector: 'app-model-list',
  templateUrl: './model-list.component.html',
  styleUrls: ['./model-list.component.scss', './../error/error.component.scss']
})
export class ModelListComponent implements OnInit {

  modelList = [];
  visibleColumns = ['model_name', 'framework', 'config', 'console_output', 'acc_input_model', 'acc_optimized_model'];
  showSpinner = true;
  token = "";
  apiBaseUrl = environment.baseUrl;
  detailedView = true;
  chosenRow = {};

  @ViewChild('drawer') drawer;

  constructor(
    private modelService: ModelService,
    private socketService: SocketService,
    public dialog: MatDialog
  ) { }

  ngOnInit() {
    this.socketService.optimizationStart$
      .subscribe(result => {
        if (result['data']) {
          if (result['status'] === 'success') {
            this.updateResult(result, 'start');
          } else {
            this.openErrorDialog(result['data']['message']);
          }
        }
      });
    this.socketService.optimizationFinish$
      .subscribe(result => {
        if (result['data']) {
          if (result['status'] === 'success') {
            this.updateResult(result, 'finish');
          } else {
            const index = this.modelList.indexOf(this.modelList.find(model => model.id === result['data']['id']));
            if (index !== -1) {
              this.modelList[index]['status'] = result['status'];
              this.openErrorDialog(result['data']['message']);
            }
          }
        }
      });
    this.socketService.benchmarkStart$
      .subscribe(result => {
        if (result['data']) {
          if (result['status'] === 'success') {
            const index = this.modelList.indexOf(this.modelList.find(model => model.id === result['data']['id']));
          } else {
            this.openErrorDialog(result['data']['message']);
          }
        }
      });
    this.socketService.benchmarkFinish$
      .subscribe(result => {
        if (result['data']) {
          const index = this.modelList.indexOf(this.modelList.find(model => model.id === result['data']['id']));
          if (index !== -1) {
            this.modelList[index]['status'] = result['status'];
            if (result['status'] === 'success') {
              this.modelList[index]['perf_throughput_input_model'] = result['data']['perf_throughput_input_model'];
              this.modelList[index]['perf_throughput_optimized_model'] = result['data']['perf_throughput_optimized_model'];
              if (result['data']['acc_input_model']) {
                this.modelList[index]['acc_input_model'] = result['data']['acc_input_model'];
                this.modelList[index]['acc_optimized_model'] = result['data']['acc_optimized_model'];
              }
            } else {
              this.openErrorDialog(result['data']['message']);
            }
          }
        }
      });
    this.getAllModels();
    this.token = this.modelService.getToken();
  }

  systemInfo() {
    return this.modelService.systemInfo;
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

  objectKeys(obj): string[] {
    return Object.keys(obj);
  }

  updateResult(result: {}, param: string) {
    const index = this.modelList.indexOf(this.modelList.find(model => model.id === result['data']['id']));
    if (this.modelList[index]) {
      if (param === 'finish') {
        this.modelList[index]['message'] = result['data']['message'];
        this.modelList[index]['acc_input_model'] = result['data']['acc_input_model'];
        this.modelList[index]['acc_optimized_model'] = result['data']['acc_optimized_model'];
        this.modelList[index]['optimization_time'] = result['data']['optimization_time'];
        this.modelList[index]['size_optimized_model'] = result['data']['size_optimized_model'];
        this.modelList[index]['model_output_path'] = result['data']['model_output_path'];
        this.modelList[index]['input_precision'] = result['data']['execution_details']['optimization']['input_precision'];
        this.modelList[index]['output_precision'] = result['data']['execution_details']['optimization']['output_precision'];
        this.modelList[index]['perf_throughput_fp32'] = null;
        this.modelList[index]['perf_throughput_int8'] = null;
      } else if (param === 'start') {
        this.modelList[index]['size_input_model'] = result['data']['size_input_model'];
      }
    }
  }

  optimize(model: NewModel) {
    this.clearModel(model);
    model['status'] = 'wip';
    this.modelService.optimize(model)
      .subscribe(
        response => { },
        error => {
          this.openErrorDialog(error);
        }
      );
  }

  clearModel(model: NewModel) {
    ["acc_input_model", "acc_optimized_model", "optimization_time", "perf_throughput_input_model", "perf_throughput_optimized_model", "status"].forEach(property => {
      model[property] = null;
    })
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

  showDrawer(model?) {
    this.chosenRow = model;
    this.drawer.toggle();
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
