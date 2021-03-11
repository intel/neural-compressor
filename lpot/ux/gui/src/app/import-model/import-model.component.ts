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
import { FormBuilder, FormGroup, Validators } from '@angular/forms';
import { ModelService, NewModel } from '../services/model.service';
import { Md5 } from 'ts-md5/dist/md5';
import { FileBrowserComponent } from '../file-browser/file-browser.component';
import { MatDialog } from '@angular/material/dialog';
import { SocketService } from '../services/socket.service';
import { debounceTime, filter, map, pairwise } from 'rxjs/operators';
import { ErrorComponent } from '../error/error.component';

@Component({
  selector: 'app-import-model',
  templateUrl: './import-model.component.html',
  styleUrls: ['./import-model.component.scss', './../error/error.component.scss']
})
export class ImportModelComponent implements OnInit {

  frameworks = [];
  domains = [];
  metrics = [];
  metricParam: string | boolean;
  metricParams = [];
  objectives = [];
  approaches = [];
  dataLoaders = [];
  dataLoaderParams = {
    evaluation: [],
    quantization: []
  };
  showDatasetLocation = {
    evaluation: true,
    quantization: true
  };
  transformations = [];
  transformationParams = [];
  tunings = [];
  inputs = [];
  outputs = [];
  order = {
    input: [],
    output: []
  };
  frameworkVersion: string;
  frameworkWarning: string;

  firstFormGroup: FormGroup;
  secondFormGroup: FormGroup;

  saved = false;
  id: string;
  showSpinner = false;
  useCalibrationData = true;

  constructor(
    private _formBuilder: FormBuilder,
    private modelService: ModelService,
    private socketService: SocketService,
    public dialog: MatDialog
  ) { }

  ngOnInit() {
    const id = new Md5();
    const dateTime = Date.now();
    this.id = String(id.appendStr(String(dateTime)).end());

    this.setDefaultValues();

    this.firstFormGroup.valueChanges
      .pipe(
        pairwise(),
        debounceTime(1000),
        map(([oldState, newState]) => {
          let changes = 0;
          ['modelLocation', 'modelDomain'].forEach(field => {
            if (oldState[field] !== newState[field]) {
              changes++;
            };
          });
          return changes;
        }),
        filter(changes => changes > 0))
      .subscribe(response => {
        if (this.firstFormGroup.get('modelLocation').value && this.firstFormGroup.get('modelDomain').value) {
          this.getConfig();
        }
      }
      );

    this.firstFormGroup.get('modelLocation').valueChanges
      .subscribe(response => {
        if (this.firstFormGroup.get('modelLocation').value) {
          this.showSpinner = true;
          this.frameworkVersion = null;
          this.frameworkWarning = null;
          ['input', 'output'].forEach(type => {
            this.firstFormGroup.get(type).setValue([]);
            this.order[type] = [];
          });
          this.socketService.getBoundaryNodes(this.getNewModel()).subscribe();
        }
      });

    this.socketService.boundaryNodesFinish$
      .subscribe(result => {
        this.showSpinner = false;
        if (result['data'] && this.firstFormGroup.get('modelLocation').value) {
          if (result['status'] === 'error') {
            this.frameworkWarning = result['data']['message'];
          } else {
            this.firstFormGroup.get('framework').setValue(result['data']['framework']);
            if (result['data']['framework'] === 'tensorflow') {
              this.firstFormGroup.controls['input'].setValidators([Validators.required]);
              this.firstFormGroup.controls['output'].setValidators([Validators.required]);
              this.firstFormGroup.controls['input'].updateValueAndValidity();
              this.firstFormGroup.controls['output'].updateValueAndValidity();
            } else {
              this.firstFormGroup.controls['input'].clearValidators();
              this.firstFormGroup.controls['output'].clearValidators();
              this.firstFormGroup.controls['input'].updateValueAndValidity();
              this.firstFormGroup.controls['output'].updateValueAndValidity();
            }
            this.getPossibleValues();
            this.frameworkVersion = result['data']['framework_version'];
            ['inputs', 'outputs'].forEach(param => {
              this[param] = result['data'][param];
              if (result['data'][param]) {
                const nonCustomParams = result['data'][param].filter(param => param !== 'custom');
                if (nonCustomParams.length === 1) {
                  this.firstFormGroup.get(param.slice(0, -1)).setValue(nonCustomParams);
                }
              }
            });
          }
        }
      });
  }

  boundaryNodesChanged(value, type: 'input' | 'output') {
    if (value === 'custom') {
      if (!this.order[type].includes(value)) {
        this.firstFormGroup.get(type).setValue([value]);
        this.order[type] = [value];
      } else {
        this.firstFormGroup.get(type).setValue([]);
        this.order[type] = [];
      }
    } else {
      if (!this.order[type].includes(value)) {
        this.order[type].push(value);
      } else {
        this.order[type].splice(this.order[type].indexOf(value), 1);
      }
    }
  }

  setDefaultValues() {
    this.firstFormGroup = this._formBuilder.group({
      framework: ['', Validators.required],
      modelLocation: ['', Validators.required],
      modelDomain: ['', Validators.required],
      input: [''],
      inputOther: [''],
      output: [''],
      outputOther: [''],
    });
    this.secondFormGroup = this._formBuilder.group({
      accuracyGoal: [0.01],
      dataLoaderEvaluation: [''],
      dataLoaderQuantization: [''],
      datasetLocationEvaluation: [''],
      datasetLocationQuantization: [''],
      datasetParams0: [''],
      datasetParams1: [''],
      transform: [''],
      transformParams: [''],
      samplingSize: [100],
      op: [''],
      strategy: [''],
      batchSize: [1],
      cores_per_instance: [''],
      num_of_instance: [''],
      inter_num_of_threads: [''],
      intra_num_of_threads: [''],
      kmp_blocktime: [''],
      warmup: [],
      iteration: [-1],
      metric: [''],
      objective: [''],
      timeout: [0],
      maxTrials: [100],
      randomSeed: [],
      approach: [],
    });
  }

  getPossibleValues() {
    this.modelService.getPossibleValues('framework', {})
      .subscribe(
        resp => this.frameworks = resp['framework'],
        error => this.openErrorDialog(error));
    this.modelService.getPossibleValues('metric', { framework: this.firstFormGroup.get('framework').value })
      .subscribe(
        resp => this.metrics = resp['metric'],
        error => this.openErrorDialog(error));
    this.modelService.getPossibleValues('domain', { framework: this.firstFormGroup.get('framework').value })
      .subscribe(
        resp => this.domains = resp['domain'],
        error => this.openErrorDialog(error));
    this.modelService.getPossibleValues('objective', { framework: this.firstFormGroup.get('framework').value })
      .subscribe(
        resp => this.objectives = resp['objective'],
        error => this.openErrorDialog(error));
    this.modelService.getPossibleValues('quantization_approach', { framework: this.firstFormGroup.get('framework').value })
      .subscribe(
        resp => this.approaches = resp['quantization_approach'],
        error => this.openErrorDialog(error));
    this.modelService.getPossibleValues('dataloader', { framework: this.firstFormGroup.get('framework').value })
      .subscribe(
        resp => this.dataLoaders = resp['dataloader'],
        error => this.openErrorDialog(error));
    this.modelService.getPossibleValues('transform', { framework: this.firstFormGroup.get('framework').value })
      .subscribe(
        resp => this.transformations = resp['transform'],
        error => this.openErrorDialog(error));
    this.modelService.getPossibleValues('strategy', { framework: this.firstFormGroup.get('framework').value })
      .subscribe(
        resp => this.tunings = resp['strategy'],
        error => this.openErrorDialog(error));
  }

  setDefaultMetricParam(event) {
    this.metricParams = this.metrics.find(x => x.name === event.value).params;
    if (this.metricParams) {
      this.metricParam = this.metricParams[0].value;
      if (Array.isArray(this.metricParams[0].value)) {
        this.metricParam = this.metricParams[0].value[0];
      }
    } else {
      this.metricParam = null;
    }
  }

  setDefaultDataLoaderParam(event, section: 'quantization' | 'evaluation') {
    const parameters = this.dataLoaders.find(x => x.name === event.value).params;
    this.dataLoaderParams[section] = [];
    if (Array.isArray(parameters)) {
      parameters.forEach((param, index) => {
        this.dataLoaderParams[section][index] = {};
        Object.keys(param).forEach(paramValue => {
          this.dataLoaderParams[section][index][paramValue] = param[paramValue];
        });
      })
    }
    this.showDatasetLocation[section] = this.dataLoaders.find(x => x.name === event.value).show_dataset_location;
    if (section === 'quantization' && this.useCalibrationData) {
      this.secondFormGroup.controls['datasetLocationEvaluation'].clearValidators();
      this.secondFormGroup.controls['datasetLocationEvaluation'].updateValueAndValidity();
    }
    const controlName = 'datasetLocation' + section.charAt(0).toUpperCase() + section.substr(1).toLowerCase();
    if (this.showDatasetLocation[section]) {
      this.secondFormGroup.controls[controlName].setValidators([Validators.required]);
      this.secondFormGroup.controls[controlName].updateValueAndValidity();
    } else {
      this.secondFormGroup.controls[controlName].clearValidators();
      this.secondFormGroup.controls[controlName].updateValueAndValidity();
    }
  }

  setDefaultTransformationParam(event, index: number) {
    this.transformationParams[index]['params'] = this.transformations.find(x => x.name === event.value).params;
  }

  addNewTransformation(name?: string) {
    this.transformationParams.push({ name: name ? name : '', params: {} });
  }

  removeTransformation(index: number) {
    this.transformationParams.splice(index, 1);;
  }

  getConfig() {
    const newModel = this.getNewModel();
    this.modelService.getConfiguration(newModel)
      .subscribe(resp => {
        if (resp['config']) {
          this.firstFormGroup.get('framework').setValue(resp['framework']);
          this.firstFormGroup.get('modelDomain').setValue(resp['domain']);

          if (resp['config']['quantization']) {
            this.secondFormGroup.get('samplingSize').setValue(resp['config']['quantization'].calibration.sampling_size);
            this.secondFormGroup.get('approach').setValue(resp['config']['quantization'].approach);
            this.transformationParams = [];
            let transform = {};
            if (resp['config']['quantization'].calibration.dataloader.transform) {
              transform = resp['config']['quantization'].calibration.dataloader.transform;
            } else if (resp['config']['evaluation'].dataloader && resp['config']['evaluation'].dataloader.transform) {
              transform = resp['config']['evaluation'].dataloader.transform;
            } else if (resp['config']['evaluation'].accuracy && resp['config']['evaluation'].accuracy.postprocess.transform) {
              transform = resp['config']['evaluation'].accuracy.postprocess.transform;
            }
            this.secondFormGroup.get('transform').setValue(transform);
            const transformNames = Object.keys(transform);
            transformNames.forEach((name, index) => {
              this.addNewTransformation(name);
              this.transformationParams[index]['params'] = this.transformations.find(x => x.name === name).params;
              if (Array.isArray(this.transformationParams[index]['params'])) {
                this.transformationParams[index]['params'].forEach(param => {
                  param.value = transform[name][param.name];
                });
              }
            });

            const dataLoader = Object.keys(resp['config']['quantization'].calibration.dataloader.dataset)[0];
            this.secondFormGroup.get('dataLoaderQuantization').setValue(dataLoader);
            this.secondFormGroup.get('dataLoaderEvaluation').setValue(dataLoader);
            this.setDefaultDataLoaderParam(this.secondFormGroup.get('dataLoaderQuantization'), 'quantization');
            this.setDefaultDataLoaderParam(this.secondFormGroup.get('dataLoaderEvaluation'), 'evaluation');
          }

          if (resp['config']['evaluation']) {
            this.secondFormGroup.get('metric').setValue(Object.keys(resp['config']['evaluation'].accuracy.metric)[0]);
            this.setDefaultMetricParam(this.secondFormGroup.get('metric'));
            this.secondFormGroup.get('kmp_blocktime').setValue(resp['config']['evaluation'].performance.configs.kmp_blocktime);
            this.secondFormGroup.get('warmup').setValue(resp['config']['evaluation'].performance.warmup);
            this.secondFormGroup.get('iteration').setValue(resp['config']['evaluation'].performance.iteration);
            this.secondFormGroup.get('batchSize').setValue(resp['config']['evaluation'].performance.dataloader.batch_size);
          }

          if (resp['config']['tuning']) {
            this.secondFormGroup.get('strategy').setValue(resp['config']['tuning'].strategy.name);
            this.secondFormGroup.get('timeout').setValue(resp['config']['tuning'].exit_policy.timeout);
            this.secondFormGroup.get('randomSeed').setValue(resp['config']['tuning'].random_seed);
          }
        }
      },
        error => {
          this.openErrorDialog(error);
        });
  }

  calibrationDataChange(copied: boolean) {
    this.useCalibrationData = copied;
    if (copied) {
      this.secondFormGroup.get('dataLoaderEvaluation').setValue(this.secondFormGroup.get('dataLoaderQuantization').value);
      this.secondFormGroup.get('datasetLocationEvaluation').setValue(this.secondFormGroup.get('datasetLocationQuantization').value);
      this.setDefaultDataLoaderParam(this.secondFormGroup.get('dataLoaderEvaluation'), 'evaluation');
    } else {
      this.secondFormGroup.get('dataLoaderEvaluation').reset();
      this.secondFormGroup.get('datasetLocationEvaluation').setValue('');
    }
  }

  useForEvaluation() {
    if (this.useCalibrationData) {
      this.secondFormGroup.get('datasetLocationEvaluation').setValue(this.secondFormGroup.get('datasetLocationQuantization').value);
    }
  }

  addModel() {
    this.modelService.saveWorkload(this.getFullModel())
      .subscribe(
        response => { },
        error => {
          this.openErrorDialog(error);
        }
      );
  }

  getNewModel(): NewModel {
    let model: NewModel;
    model = {
      domain: this.firstFormGroup.get('modelDomain').value,
      framework: this.firstFormGroup.get('framework').value,
      id: this.id,
      model_path: this.firstFormGroup.get('modelLocation').value,
    };
    return model;
  }

  getFullModel(): FullModel {
    let model: FullModel;
    ['evaluation', 'quantization'].forEach(section => {
      if (this.dataLoaderParams[section] && Object.keys(this.dataLoaderParams[section]).includes('root')) {
        this.dataLoaderParams[section]['root'] = this.secondFormGroup.get('datasetLocationQuantization').value;
      }
    });
    model = {
      domain: this.firstFormGroup.get('modelDomain').value,
      framework: this.firstFormGroup.get('framework').value,
      id: this.id,
      model_path: this.firstFormGroup.get('modelLocation').value,
      inputs: this.getBoundaryNodes('input'),
      outputs: this.getBoundaryNodes('output'),
      transform: this.getTransformParams(this.transformationParams),
      quantization: {
        dataset_path: this.secondFormGroup.get('datasetLocationQuantization').value.length ? this.secondFormGroup.get('datasetLocationQuantization').value : 'no_dataset_location',
        dataloader: {
          name: this.secondFormGroup.get('dataLoaderQuantization').value,
          params: this.dataLoaderParams['quantization'] ? this.getParams(this.dataLoaderParams['quantization']) : null,
        },
        accuracy_goal: this.secondFormGroup.get('accuracyGoal').value,
        sampling_size: this.secondFormGroup.get('samplingSize').value,
        strategy: this.secondFormGroup.get('strategy').value,
        approach: this.secondFormGroup.get('approach').value,
        objective: this.secondFormGroup.get('objective').value,
        timeout: this.secondFormGroup.get('timeout').value,
        max_trials: this.secondFormGroup.get('maxTrials').value,
        random_seed: this.secondFormGroup.get('randomSeed').value
      },
      evaluation: {
        dataset_path: this.getEvaluationDatasetPath(),
        dataloader: this.getEvaluationDataloader(),
        metric: this.secondFormGroup.get('metric').value,
        metric_param: this.metricParam,
        batch_size: this.secondFormGroup.get('batchSize').value,
        cores_per_instance: this.secondFormGroup.get('cores_per_instance').value,
        instances: this.secondFormGroup.get('num_of_instance').value,
        inter_nr_of_threads: this.secondFormGroup.get('inter_num_of_threads').value,
        intra_nr_of_threads: this.secondFormGroup.get('intra_num_of_threads').value,
        iterations: this.secondFormGroup.get('iteration').value,
        warmup: this.secondFormGroup.get('warmup').value,
        kmp_blocktime: this.secondFormGroup.get('kmp_blocktime').value,
      }
    }
    return model;
  }

  getEvaluationDatasetPath() {
    if (this.useCalibrationData) {
      return this.secondFormGroup.get('datasetLocationQuantization').value.length ? this.secondFormGroup.get('datasetLocationQuantization').value : 'no_dataset_location'
    }
    return this.secondFormGroup.get('datasetLocationEvaluation').value.length ? this.secondFormGroup.get('datasetLocationEvaluation').value : 'no_dataset_location'
  }

  getEvaluationDataloader() {
    if (this.useCalibrationData) {
      return {
        name: this.secondFormGroup.get('dataLoaderQuantization').value,
        params: this.dataLoaderParams['quantization'] ? this.getParams(this.dataLoaderParams['quantization']) : null
      };
    }
    return {
      name: this.secondFormGroup.get('dataLoaderEvaluation').value,
      params: this.dataLoaderParams['evaluation'] ? this.getParams(this.dataLoaderParams['evaluation']) : null
    };
  }

  getBoundaryNodes(type: 'input' | 'output') {
    if (this.firstFormGroup.get(type + 'Other').value) {
      return [this.firstFormGroup.get(type + 'Other').value];
    }
    if (this.order[type]) {
      return this.order[type];
    }
    if (typeof this.firstFormGroup.get(type).value === 'string') {
      return [this.firstFormGroup.get(type).value];
    }
    return this.firstFormGroup.get(type).value;
  }

  getParams(obj: any[]): {} {
    let newObj = {};
    obj.forEach(item => {
      newObj[item['name']] = item['value'];
    });
    return newObj;
  }

  getTransformParams(obj: any[]): { name: string; params: {}; }[] {
    let newObj = [];
    obj.forEach(item => {
      newObj.push({
        name: item.name,
        params: item.params ? this.getParams(item.params) : null
      })
    });
    return newObj;
  }

  openDialog(fieldName: string, filter: 'models' | 'datasets' | 'directories') {
    let form = 'firstFormGroup';
    if (filter === 'datasets') {
      form = 'secondFormGroup';
    }
    const dialogRef = this.dialog.open(FileBrowserComponent, {
      width: '60%',
      height: '60%',
      data: {
        path: this[form].get(fieldName).value ? this[form].get(fieldName).value.split("/").slice(0, -1).join("/") : this.modelService.workspacePath,
        filter: filter
      }
    });

    dialogRef.afterClosed().subscribe(chosenFile => {
      if (chosenFile) {
        this[form].get(fieldName).setValue(chosenFile);
        if (fieldName === 'datasetLocationQuantization' && this.useCalibrationData) {
          this.secondFormGroup.get('datasetLocationEvaluation').setValue(chosenFile);
        }
      }
    });;
  }

  openErrorDialog(error) {
    const dialogRef = this.dialog.open(ErrorComponent, {
      data: error
    });
  }

  objectKeys(obj: {}): string[] {
    return Object.keys(obj);
  }

  typeOf(obj: any): string {
    return typeof obj;
  }

  isArray(obj: any): boolean {
    if (Array.isArray(obj)) {
      return true;
    }
    return false;
  }
}

export interface FullModel {
  domain: string;
  framework: string;
  id: string;
  model_path: string;
  inputs: string[];
  outputs: string[];
  transform: {
    name: string;
    params: {};
  }[]
  quantization: {
    accuracy_goal: number;
    sampling_size: number;
    op?: string[];
    strategy: string;
    approach: string;
    objective: string;
    timeout: number;
    max_trials: number;
    random_seed: number;
    dataset_path: string;
    dataloader: {
      name: string;
      params: {};
    }
  }
  evaluation: {
    metric: string;
    metric_param: string | boolean;
    batch_size: number;
    cores_per_instance: number;
    instances: number;
    inter_nr_of_threads: number;
    intra_nr_of_threads: number;
    iterations: number;
    warmup: number;
    kmp_blocktime: string;
    dataset_path: string;
    dataloader: {
      name: string;
      params: {};
    }
  }
}
