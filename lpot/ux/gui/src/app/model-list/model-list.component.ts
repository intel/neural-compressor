import { Component, OnInit } from '@angular/core';
import { MatDialog } from '@angular/material';
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
  benchmarkSpinner = [];
  showSpinner = true;

  constructor(
    private modelService: ModelService,
    private socketService: SocketService,
    public dialog: MatDialog
  ) { }

  ngOnInit() {
    this.socketService.tuningStart$
      .subscribe(result => {
        if (Object.keys(result).length > 0) {
          this.updateResult(result, 'start');
        }
      });
    this.socketService.tuningFinish$
      .subscribe(result => {
        if (Object.keys(result).length > 0) {
          this.updateResult(result, 'finish');
        }
      });
    this.socketService.benchmarkFinish$
      .subscribe(result => {
        if (result['data']) {
          const index = this.modelList.indexOf(this.modelList.find(model => model.id === result['data']['id']));
          this.benchmarkSpinner[index] = false;
          this.modelList[index]['perf_throughput_fp32'] = result['data']['perf_throughput_fp32'];
          this.modelList[index]['perf_throughput_int8'] = result['data']['perf_throughput_int8'];
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
        this.modelList[index]['status'] = result['status'];
        this.modelList[index]['message'] = result['data']['message'];
        this.modelList[index]['acc_fp32'] = result['data']['acc_fp32'];
        this.modelList[index]['acc_int8'] = result['data']['acc_int8'];
        this.modelList[index]['tuning_time'] = result['data']['tuning_time'];
        this.modelList[index]['model_int8'] = result['data']['model_int8'];
        this.modelList[index]['perf_latency_int8'] = result['data']['perf_latency_int8'];
        this.modelList[index]['perf_latency_fp32'] = result['data']['perf_latency_fp32'];
        this.modelList[index]['size_int8'] = result['data']['size_int8'];
        this.modelList[index]['model_output_path'] = result['data']['model_output_path'];

        this.modelService.benchmark(
          result['data']['id'],
          this.modelList[index]['model_path'],
          result['data']['model_output_path']
        )
          .subscribe(
            response => {
              this.benchmarkSpinner[index] = true;
              this.modelList[index]['perf_throughput_fp32'] = null;
              this.modelList[index]['perf_throughput_int8'] = null;
            },
            error => {
              this.openErrorDialog(error);
            }
          );
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
