import { Component, OnChanges, OnInit } from '@angular/core';
import { MatDialog } from '@angular/material';
import { DialogComponent } from '../dialog/dialog.component';
import { ModelService, NewModel } from '../services/model.service';
import { SocketService } from '../services/socket.service';

@Component({
  selector: 'app-model-list',
  templateUrl: './model-list.component.html',
  styleUrls: ['./model-list.component.scss', './../start-page/start-page.component.scss']
})
export class ModelListComponent implements OnInit, OnChanges {

  modelList = [];
  tuningInProgress = false;
  visibleColumns = ['model_name', 'framework', 'config', 'console_output', 'acc_fp32', 'acc_int8'];

  constructor(
    private modelService: ModelService,
    private socketService: SocketService,
    public dialog: MatDialog
  ) { }


  ngOnInit() {
    this.socketService.tuningStart$
      .subscribe(result => {
        this.tuningInProgress = true;
        if (Object.keys(result).length > 0) {
          this.updateResult(result, 'start');
        }
      });
    this.socketService.tuningFinish$
      .subscribe(result => {
        this.tuningInProgress = false;
        if (Object.keys(result).length > 0) {
          this.updateResult(result, 'finish');
        }
      }
      );
    this.modelList = this.modelService.getAllModels();
  }

  ngOnChanges() {
    this.modelList = this.modelService.getAllModels();
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
        this.modelList[index]['perf_throughput_int8'] = result['data']['perf_throughput_int8'];
        this.modelList[index]['perf_latency_int8'] = result['data']['perf_latency_int8'];
        this.modelList[index]['perf_throughput_fp32'] = result['data']['perf_throughput_fp32'];
        this.modelList[index]['perf_latency_fp32'] = result['data']['perf_latency_fp32'];
        this.modelList[index]['size_int8'] = result['data']['size_int8'];
        this.modelList[index]['model_output_path'] = result['data']['model_output_path'];
      } else if (param === 'start') {
        this.modelList[index]['size_fp32'] = result['data']['size_fp32'];
      }
      localStorage.setItem('myModels', JSON.stringify(this.modelList));
    }
  }

  openDialog(fileName: string, fileType: string): void {
    const dialogRef = this.dialog.open(DialogComponent, {
      width: '90%',
      data: {
        fileName: fileName,
        fileType: fileType
      }
    });
  }

  tune(model: NewModel) {
    model['status'] = 'wip';
    this.modelService.tune(model)
      .subscribe();
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
