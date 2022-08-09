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
// See the License for the specifU77777777777
// ic language governing permissions and
// limitations under the License.
import { Component, Inject, OnInit } from '@angular/core';
import { FormGroup, FormBuilder } from '@angular/forms';
import { MAT_DIALOG_DATA } from '@angular/material/dialog';
import { ModelService } from '../services/model.service';

@Component({
  selector: 'app-benchmark-form',
  templateUrl: './benchmark-form.component.html',
  styleUrls: ['./benchmark-form.component.scss', './../error/error.component.scss', './../home/home.component.scss', './../datasets/datasets.component.scss']
})
export class BenchmarkFormComponent implements OnInit {

  name: string;
  modelId;
  models = [];
  datasetId;
  datasets = [];
  mode = 'performance';
  modes = ['accuracy', 'performance'];
  benchmarkFormGroup: FormGroup;
  allSamples = true;

  constructor(
    @Inject(MAT_DIALOG_DATA) public data,
    public modelService: ModelService,
    private _formBuilder: FormBuilder,
  ) { }

  ngOnInit(): void {
    this.name = 'Benchmark' + String(this.data.index + 1);
    if (this.data.framework === "pytorch"){
      this.modes = ['performance']
          }
    this.getDatasetList()
    this.modelService.getModelList(this.data.projectId)
      .subscribe(
        response => {
          this.models = response['models'];
          if (this.models.length > 0) {
            this.modelId = this.models[0].id;
          }
        },
        error => {
          this.modelService.openErrorDialog(error);
        });

    this.benchmarkFormGroup = this._formBuilder.group({
      batchSize: [1],
      warmup: [5],
      iterations: [10],
      numOfInstance: [this.modelService.systemInfo['cores_per_socket'] * this.modelService.systemInfo['sockets'] / 4],
      coresPerInstance: [4],
      commandLine:['']
    });

    this.modelService.datasetCreated$.subscribe(response => this.getDatasetList());
  }

  getDatasetList() {
    this.modelService.getDatasetList(this.data.projectId)
      .subscribe(
        response => {
          this.datasets = response['datasets'];
          if (this.datasets.length > 0) {
            this.datasetId = this.datasets[0].id;
          }
        },
        error => {
          this.modelService.openErrorDialog(error);
        });
  }

  coresValidated(): boolean {
    return this.benchmarkFormGroup.get('coresPerInstance').value * this.benchmarkFormGroup.get('numOfInstance').value <= this.modelService.systemInfo['cores_per_socket'] * this.modelService.systemInfo['sockets'];
  }

  openDatasetDialog() {
    this.modelService.openDatasetDialog$.next(true);
  }

  addBenchmark() {
    this.modelService.addBenchmark({
      project_id: this.data.projectId,
      name: this.name,
      mode: this.mode,
      dataset_id: this.datasetId,
      model_id: this.modelId,
      batch_size: this.benchmarkFormGroup.get('batchSize').value,
      iterations: this.allSamples ? -1 : this.benchmarkFormGroup.get('iterations').value,
      number_of_instance: this.benchmarkFormGroup.get('numOfInstance').value,
      cores_per_instance: this.benchmarkFormGroup.get('coresPerInstance').value,
      warmup_iterations: this.benchmarkFormGroup.get('warmup').value,
      command_line:this.benchmarkFormGroup.get('commandLine').value
    })
      .subscribe(
        response => {
          this.modelService.benchmarkCreated$.next(true);
        },
        error => {
          this.modelService.openErrorDialog(error);
        });
  }

}
