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
import { DialogComponent } from '../dialog/dialog.component';
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
  visibleColumns = ['model_name', 'framework', 'config', 'console_output', 'acc_fp32', 'acc_int8'];
  showSpinner = true;

  constructor(
    private modelService: ModelService,
    private socketService: SocketService,
    public dialog: MatDialog
  ) { }

  ngOnInit() {
    this.socketService.tuningStart$
      .subscribe(result => {
        if (result['data']) {
          if (result['status'] === 'success') {
            this.updateResult(result, 'start');
          } else {
            this.openErrorDialog(result['data']['message']);
          }
        }
      });
    this.socketService.tuningFinish$
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
              this.modelList[index]['perf_throughput_fp32'] = result['data']['perf_throughput_fp32'];
              this.modelList[index]['perf_throughput_int8'] = result['data']['perf_throughput_int8'];
            } else {
              this.openErrorDialog(result['data']['message']);
            }
          }
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

  objectKeys(obj): string[] {
    return Object.keys(obj);
  }

  updateResult(result: {}, param: string) {
    const index = this.modelList.indexOf(this.modelList.find(model => model.id === result['data']['id']));
    if (this.modelList[index]) {
      if (param === 'finish') {
        this.modelList[index]['message'] = result['data']['message'];
        this.modelList[index]['acc_fp32'] = result['data']['acc_fp32'];
        this.modelList[index]['acc_int8'] = result['data']['acc_int8'];
        this.modelList[index]['tuning_time'] = result['data']['tuning_time'];
        this.modelList[index]['model_int8'] = result['data']['model_int8'];
        this.modelList[index]['perf_latency_int8'] = result['data']['perf_latency_int8'];
        this.modelList[index]['perf_latency_fp32'] = result['data']['perf_latency_fp32'];
        this.modelList[index]['size_int8'] = result['data']['size_int8'];
        this.modelList[index]['model_output_path'] = result['data']['model_output_path'];

        this.modelList[index]['perf_throughput_fp32'] = null;
        this.modelList[index]['perf_throughput_int8'] = null;
      } else if (param === 'start') {
        this.modelList[index]['size_fp32'] = result['data']['size_fp32'];
      }
    }
  }

  openDialog(path: string, fileType: string): void {
    const dialogRef = this.dialog.open(DialogComponent, {
      width: '90%',
      data: {
        path: path,
        fileType: fileType
      }
    });
  }

  tune(model: NewModel) {
    model['status'] = 'wip';
    this.modelService.tune(model)
      .subscribe(
        response => { },
        error => {
          this.openErrorDialog(error);
        }
      );
  }

  getTooltip(execution_details): string | null {
    if (execution_details) {
      let tooltip = '';
      if (execution_details.fp32_benchmark) {
        tooltip += 'FP32 BENCHMARK\n' +
          'cores per instance: ' + execution_details.fp32_benchmark.cores_per_instance + '\n' +
          'instances: ' + execution_details.fp32_benchmark.instances + '\n\n'
      }
      if (execution_details.int8_benchmark) {
        tooltip += 'INT8 BENCHMARK\n' +
          'cores per instance: ' + execution_details.int8_benchmark.cores_per_instance + '\n' +
          'instances: ' + execution_details.int8_benchmark.instances + '\n\n';
      }
      if (execution_details.tuning) {
        tooltip += 'TUNING\n' +
          'cores per instance: ' + execution_details.tuning.cores_per_instance + '\n' +
          'instances: ' + execution_details.tuning.instances + '\n\n';
      }
      return tooltip;
    }
    return null;
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
