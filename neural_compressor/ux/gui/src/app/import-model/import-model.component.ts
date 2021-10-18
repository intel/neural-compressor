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
import { FileBrowserFilter, ModelService, NewModel } from '../services/model.service';
import { FileBrowserComponent } from '../file-browser/file-browser.component';
import { MatDialog } from '@angular/material/dialog';
import { SocketService } from '../services/socket.service';
import { debounceTime, filter, map, pairwise } from 'rxjs/operators';
import { ErrorComponent } from '../error/error.component';
import { GraphComponent } from '../graph/graph.component';
import { Router } from '@angular/router';
import { Subject, combineLatest } from 'rxjs';
declare var require: any;
var shajs = require('sha.js')

@Component({
  selector: 'app-import-model',
  templateUrl: './import-model.component.html',
  styleUrls: ['./import-model.component.scss', './../error/error.component.scss', './../home/home.component.scss']
})
export class ImportModelComponent implements OnInit {

  showAdvancedParams = false;
  frameworks = [];
  domains = [];
  metrics = [];
  precisions = [];
  precisionIndex;
  metricParam: string | boolean;
  metricParams = [];
  objectives = [];
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
  tuningEnabled = false;
  dummy = true;
  tunings = [];
  inputs = [];
  outputs = [];
  order = {
    input: [],
    output: []
  };
  graph = {};
  sliderOptions;
  resizeValues;
  resizeCustom;

  boundaryNodes: {
    inputs: 'none' | 'custom' | 'select',
    outputs: 'none' | 'custom' | 'select',
  };

  frameworkWarning: string;
  trustedShape: boolean;
  firstFormGroup: FormGroup;
  secondFormGroup: FormGroup;

  saved = false;
  finishDisabled = false;
  id: string;
  showSpinner = false;
  showBigSpinner = false;
  showGraphSpinner = false;
  showGraphButton = false;
  showDomainSpinner = false;
  useEvaluationData = true;
  fileBrowserParams = ['label_file', 'vocab_file', 'anno_path'];

  metricList$: Subject<boolean> = new Subject<boolean>();
  metricValue$: Subject<boolean> = new Subject<boolean>();
  dataLoaderList$: Subject<boolean> = new Subject<boolean>();
  dataLoaderValue$: Subject<boolean> = new Subject<boolean>();

  constructor(
    private _formBuilder: FormBuilder,
    public modelService: ModelService,
    private socketService: SocketService,
    public dialog: MatDialog,
    private router: Router
  ) { }

  ngOnInit() {
    const dateTime = Date.now();
    this.id = shajs('sha384').update(String(dateTime)).digest('hex');

    this.boundaryNodes = {
      inputs: 'none',
      outputs: 'none'
    };

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
              this.showDomainSpinner = true;
            };
          });
          return changes;
        }),
        filter(changes => changes > 0))
      .subscribe(response => {
        if (this.firstFormGroup.get('modelLocation').value && this.firstFormGroup.get('modelDomain').value) {
          this.getConfig();
        } else {
          this.showDomainSpinner = false;
        }
      }
      );

    this.firstFormGroup.get('modelLocation').valueChanges
      .pipe(
        debounceTime(1000))
      .subscribe(response => {
        if (this.firstFormGroup.get('modelLocation').value) {
          this.showSpinner = true;
          this.showGraphButton = false;
          this.frameworkWarning = null;
          this.socketService.getBoundaryNodes(this.getNewModel()).subscribe();
          this.modelService.getModelGraph(this.firstFormGroup.get('modelLocation').value)
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
      });

    this.socketService.boundaryNodesFinish$
      .subscribe(result => {
        this.showSpinner = false;
        if (result['data'] && this.firstFormGroup.get('modelLocation').value && result['data'].id === this.id) {
          if (result['status'] === 'error') {
            this.frameworkWarning = result['data']['message'];
          } else {
            this.firstFormGroup.get('domainFlavour').setValue(result['data']['domain_flavour']);
            this.secondFormGroup.get('shape').setValue(result['data']['shape']);
            this.trustedShape = result['data']['trusted'];
            if (result['data']['domain'].length) {
              this.firstFormGroup.get('modelDomain').setValue(result['data']['domain']);
              this.getConfig();
              this.showDomainSpinner = true;
            } else {
              this.firstFormGroup.get('modelDomain').reset();
            }
            this.firstFormGroup.get('framework').setValue(result['data']['framework']);
            if (this.firstFormGroup.get('framework').value !== 'tensorflow') {
              this.tuningEnabled = true;
              this.isFieldRequired('secondFormGroup', 'datasetLocationEvaluation', true);
              this.isFieldRequired('secondFormGroup', 'shape', false);
            }
            this.onTuningEnabledChange();
            this.getPossibleValues();
            ['inputs', 'outputs'].forEach(param => {
              this[param] = result['data'][param];
              if (Array.isArray(result['data'][param])) {
                this.isFieldRequired('firstFormGroup', 'input', true);
                this.isFieldRequired('firstFormGroup', 'output', true);
                if (result['data'][param].length === 0) {
                  this.boundaryNodes[param] = 'custom';
                } else if (result['data'][param].length === 1) {
                  this.boundaryNodes[param] = 'custom';
                  this.firstFormGroup.get(param.slice(0, -1)).setValue(result['data'][param]);
                } else {
                  this.boundaryNodes[param] = 'select';
                  if (result['data']['domain'] === 'object_detection' && result['data']['domain_flavour'] === 'ssd') {
                    if (["detection_bboxes", "detection_scores", "detection_classes"].every((val) => result['data']['outputs'].includes(val))) {
                      this.firstFormGroup.get('output').setValue(["detection_bboxes", "detection_scores", "detection_classes"]);
                    }
                    this.resizeValues = [[1200, 1200], [300, 300], 'custom'];
                  } else {
                    const nonCustomParams = result['data'][param].filter(param => param !== 'custom');
                    if (nonCustomParams.length === 1) {
                      this.firstFormGroup.get(param.slice(0, -1)).setValue(nonCustomParams);
                    } else if (nonCustomParams.includes('softmax_tensor')) {
                      this.firstFormGroup.get(param.slice(0, -1)).setValue(['softmax_tensor']);
                    }
                  }
                }
              } else {
                this.boundaryNodes[param] = 'none';
                this.isFieldRequired('firstFormGroup', 'input', false);
                this.isFieldRequired('firstFormGroup', 'output', false);
              }
            });
          }
        }
      });

    combineLatest([
      this.metricList$,
      this.metricValue$
    ]).subscribe(([metricList, metricValue]) => {
      if (metricList === true && metricValue === true) {
        this.setDefaultMetricParam(this.secondFormGroup.get('metric'));
      }
    });

    combineLatest([
      this.dataLoaderList$,
      this.dataLoaderValue$
    ]).subscribe(([dataLoaderList, dataLoaderValue]) => {
      if (dataLoaderList === true && dataLoaderValue === true) {
        this.setDefaultDataLoaderParam(this.secondFormGroup.get('dataLoaderQuantization'), 'quantization');
        this.setDefaultDataLoaderParam(this.secondFormGroup.get('dataLoaderEvaluation'), 'evaluation');
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
      name: [''],
      framework: ['', Validators.required],
      modelLocation: ['', Validators.required],
      modelDomain: ['', Validators.required],
      domainFlavour: [''],
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
      calibrationBatchSize: [100],
      op: [''],
      strategy: [''],
      batchSize: [1],
      cores_per_instance: [],
      num_of_instance: [],
      inter_num_of_threads: [''],
      intra_num_of_threads: [''],
      kmp_blocktime: [''],
      warmup: [],
      iteration: [-1],
      metric: [''],
      objective: ['performance'],
      timeout: [0],
      maxTrials: [100],
      randomSeed: [],
      shape: ['', Validators.required],
    });
  }

  getPossibleValues() {
    this.modelService.getPossibleValues('framework', {})
      .subscribe(
        resp => this.frameworks = resp['framework'],
        error => this.openErrorDialog(error));
    this.modelService.getPossibleValues('precision', { framework: this.firstFormGroup.get('framework').value })
      .subscribe(
        resp => {
          this.precisions = resp['precision'];
          this.precisionIndex = this.precisions.length - 1;
          this.sliderOptions = {
            floor: 0,
            ceil: this.precisions.length ? this.precisions.length - 1 : 2,
            step: 1,
            showTicks: true,
            showTicksValues: false,
            showTicksTooltips: true,
            hideLimitLabels: true,
            hidePointerLabels: true,
            showSelectionBar: true,
            getLegend: (value: number): string => {
              return this.precisions[value].label ?? this.precisions[value].name;
            },
            ticksTooltip: (value: number): string => {
              return this.precisions[value].help;
            }
          };
        },
        error => this.openErrorDialog(error));
    this.modelService.getPossibleValues('metric', { framework: this.firstFormGroup.get('framework').value })
      .subscribe(
        resp => {
          this.metrics = resp['metric'];
          this.metricList$.next(true);
        },
        error => this.openErrorDialog(error));
    this.modelService.getPossibleValues('domain', { framework: this.firstFormGroup.get('framework').value })
      .subscribe(
        resp => this.domains = resp['domain'],
        error => this.openErrorDialog(error));
    this.modelService.getPossibleValues('objective', { framework: this.firstFormGroup.get('framework').value })
      .subscribe(
        resp => this.objectives = resp['objective'],
        error => this.openErrorDialog(error));
    this.modelService.getPossibleValues('dataloader', { framework: this.firstFormGroup.get('framework').value })
      .subscribe(
        resp => {
          this.dataLoaders = resp['dataloader'];
          this.dataLoaderList$.next(true);
        },
        error => this.openErrorDialog(error));
    this.modelService.getPossibleValues('strategy', { framework: this.firstFormGroup.get('framework').value })
      .subscribe(
        resp => this.tunings = resp['strategy'],
        error => this.openErrorDialog(error));
  }

  setDefaultMetricParam(event) {
    if (this.metrics.length) {
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
  }

  setDefaultDataLoaderParam(event, section: 'quantization' | 'evaluation') {
    if (this.dataLoaders.length) {
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
      if (section === 'evaluation' && this.useEvaluationData) {
        this.isFieldRequired('secondFormGroup', 'datasetLocationQuantization', false);
      }
      const controlName = 'datasetLocation' + section.charAt(0).toUpperCase() + section.substr(1).toLowerCase();
      if (this.showDatasetLocation[section] && !this.dummy) {
        this.isFieldRequired('secondFormGroup', controlName, true);
      } else {
        this.isFieldRequired('secondFormGroup', controlName, false);
      }
    }
  }

  setDefaultTransformationParam(event, index: number) {
    this.transformationParams[index]['params'] = this.transformations.find(x => x.name === event.value).params;
  }

  addNewTransformation(name?: string) {
    this.transformationParams.push({ name: name ?? '', params: {} });
  }

  removeTransformation(index: number) {
    this.transformationParams.splice(index, 1);;
  }

  onTuningEnabledChange() {
    if (!this.tuningEnabled) {
      this.secondFormGroup.get('datasetLocationQuantization').setValue('');
      this.isFieldRequired('secondFormGroup', 'datasetLocationQuantization', false);
      if (this.firstFormGroup.get('framework').value === 'tensorflow') {
        this.dummy = true;
        this.isFieldRequired('secondFormGroup', 'datasetLocationEvaluation', false);
        this.isFieldRequired('secondFormGroup', 'shape', true);
      }
    } else {
      this.dummy = false;
      this.isFieldRequired('secondFormGroup', 'datasetLocationEvaluation', true);
      this.isFieldRequired('secondFormGroup', 'shape', false);
    }

  }

  isFieldRequired(form: string, field: string, required: boolean) {
    if (required) {
      this[form].controls[field].setValidators([Validators.required]);
    } else {
      this[form].controls[field].clearValidators();
    }
    this[form].controls[field].updateValueAndValidity();
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
            this.transformationParams = [];
            let transform = {};
            if (resp['config']['quantization'].calibration.dataloader && resp['config']['quantization'].calibration.dataloader.transform) {
              transform = resp['config']['quantization'].calibration.dataloader.transform;
            } else if (resp['config']['evaluation'] && resp['config']['evaluation'].dataloader && resp['config']['evaluation'].dataloader.transform) {
              transform = resp['config']['evaluation'].dataloader.transform;
            } else if (resp['config']['evaluation'] && resp['config']['evaluation'].accuracy && resp['config']['evaluation'].accuracy.postprocess && resp['config']['evaluation'].accuracy.postprocess.transform) {
              transform = resp['config']['evaluation'].accuracy.postprocess.transform;
            }
            this.secondFormGroup.get('transform').setValue(transform);
            const transformNames = Object.keys(transform);
            this.modelService.getPossibleValues('transform', { framework: this.firstFormGroup.get('framework').value, domain: this.firstFormGroup.get('modelDomain').value, domain_flavour: this.firstFormGroup.get('domainFlavour').value })
              .subscribe(
                resp => {
                  this.showDomainSpinner = false;
                  this.transformations = resp['transform'];
                  transformNames.forEach((name, index) => {
                    this.addNewTransformation(name);
                    if (this.transformations.find(x => x.name === name)) {
                      this.transformationParams[index]['params'] = this.transformations.find(x => x.name === name).params;
                      if (Array.isArray(this.transformationParams[index]['params'])) {
                        this.transformationParams[index]['params'].forEach(param => {
                          param.value = transform[name][param.name];
                          if (this.resizeValues && param.name === 'size') {
                            param.value = this.resizeValues[0];
                          }
                        });
                      }
                    }
                  });
                },
                error => this.openErrorDialog(error));

            if (resp['config']['quantization'].calibration.dataloader) {
              const dataLoader = Object.keys(resp['config']['quantization'].calibration.dataloader.dataset)[0];
              this.secondFormGroup.get('dataLoaderQuantization').setValue(dataLoader);
              this.secondFormGroup.get('dataLoaderEvaluation').setValue(dataLoader);
              this.dataLoaderValue$.next(true);
            }
          }

          if (resp['config']['evaluation']) {
            this.secondFormGroup.get('metric').setValue(Object.keys(resp['config']['evaluation'].accuracy.metric)[0]);
            this.metricValue$.next(true);
            this.secondFormGroup.get('kmp_blocktime').setValue(resp['config']['evaluation'].performance.configs.kmp_blocktime);
            this.secondFormGroup.get('cores_per_instance').setValue(resp['config']['evaluation'].performance.configs.cores_per_instance);
            this.secondFormGroup.get('num_of_instance').setValue(resp['config']['evaluation'].performance.configs.num_of_instance);
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
    this.useEvaluationData = copied;
    if (copied) {
      this.secondFormGroup.get('dataLoaderQuantization').setValue(this.secondFormGroup.get('dataLoaderEvaluation').value);
      this.secondFormGroup.get('datasetLocationQuantization').setValue(this.secondFormGroup.get('datasetLocationEvaluation').value);
      this.setDefaultDataLoaderParam(this.secondFormGroup.get('dataLoaderQuantization'), 'evaluation');
    } else {
      this.secondFormGroup.get('dataLoaderQuantization').reset();
      this.secondFormGroup.get('datasetLocationQuantization').setValue('');
    }
  }

  useForQuantization() {
    if (this.useEvaluationData) {
      this.secondFormGroup.get('datasetLocationQuantization').setValue(this.secondFormGroup.get('datasetLocationEvaluation').value);
    }
  }

  addModel() {
    this.finishDisabled = true;
    this.showBigSpinner = true;
    this.modelService.saveWorkload(this.getFullModel())
      .subscribe(
        response => {
          this.router.navigate(['/details', this.id], { queryParamsHandling: "merge" });
          this.modelService.configurationSaved.next(true);
        },
        error => {
          this.openErrorDialog(error);
          this.finishDisabled = false;
          this.showBigSpinner = false;
        }
      );
  }

  getNewModel(): NewModel {
    let model: NewModel;
    model = {
      domain: this.firstFormGroup.get('modelDomain').value,
      domain_flavour: this.firstFormGroup.get('domainFlavour').value,
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
      project_name: this.getFileName(this.firstFormGroup.get('modelLocation').value),
      domain: this.firstFormGroup.get('modelDomain').value,
      domain_flavour: this.firstFormGroup.get('domainFlavour').value,
      framework: this.firstFormGroup.get('framework').value,
      id: this.id,
      model_path: this.firstFormGroup.get('modelLocation').value,
      inputs: this.getBoundaryNodes('input'),
      outputs: this.getBoundaryNodes('output'),
      precision: this.precisions[this.precisionIndex].name,
      transform: this.getTransformParams(this.transformationParams),
      tuning: this.tuningEnabled,
      shape: this.secondFormGroup.get('shape').value,
      quantization: {
        dataset_path: this.getQuantizationDatasetPath(),
        dataloader: this.getQuantizationDataloader(),
        accuracy_goal: this.secondFormGroup.get('accuracyGoal').value,
        sampling_size: this.secondFormGroup.get('samplingSize').value,
        batch_size: this.secondFormGroup.get('calibrationBatchSize').value,
        strategy: this.secondFormGroup.get('strategy').value,
        objective: this.secondFormGroup.get('objective').value,
        timeout: this.secondFormGroup.get('timeout').value,
        max_trials: this.secondFormGroup.get('maxTrials').value,
        random_seed: this.secondFormGroup.get('randomSeed').value
      },
      evaluation: {
        dataset_path: this.secondFormGroup.get('datasetLocationEvaluation').value.length ? this.secondFormGroup.get('datasetLocationEvaluation').value : 'no_dataset_location',
        dataloader: {
          name: this.secondFormGroup.get('dataLoaderEvaluation').value,
          params: this.dataLoaderParams['evaluation'] ? this.getParams(this.dataLoaderParams['evaluation']) : null,
        },
        metric: this.secondFormGroup.get('metric').value,
        metric_param: this.metricParam,
        batch_size: this.secondFormGroup.get('batchSize').value,
        cores_per_instance: this.secondFormGroup.get('cores_per_instance').value,
        instances: this.secondFormGroup.get('num_of_instance').value ?? Math.floor(Number(this.modelService.systemInfo['cores_per_socket']) / 4),
        inter_nr_of_threads: this.secondFormGroup.get('inter_num_of_threads').value,
        intra_nr_of_threads: this.secondFormGroup.get('intra_num_of_threads').value,
        iterations: this.secondFormGroup.get('iteration').value,
        warmup: this.secondFormGroup.get('warmup').value,
        kmp_blocktime: this.secondFormGroup.get('kmp_blocktime').value,
      }
    }
    return model;
  }

  getQuantizationDatasetPath() {
    if (this.useEvaluationData) {
      return this.secondFormGroup.get('datasetLocationEvaluation').value.length ? this.secondFormGroup.get('datasetLocationEvaluation').value : 'no_dataset_location'
    }
    return this.secondFormGroup.get('datasetLocationQuantization').value.length ? this.secondFormGroup.get('datasetLocationQuantization').value : 'no_dataset_location'
  }

  getQuantizationDataloader() {
    if (this.useEvaluationData) {
      return {
        name: this.secondFormGroup.get('dataLoaderEvaluation').value,
        params: this.dataLoaderParams['evaluation'] ? this.getParams(this.dataLoaderParams['evaluation']) : null
      };
    }
    return {
      name: this.secondFormGroup.get('dataLoaderQuantization').value,
      params: this.dataLoaderParams['quantization'] ? this.getParams(this.dataLoaderParams['quantization']) : null
    };
  }

  getBoundaryNodes(type: 'input' | 'output') {
    if (this.firstFormGroup.get(type + 'Other').value.length) {
      return [this.firstFormGroup.get(type + 'Other').value];
    }
    if (this.order[type].length) {
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
      if (item['name'] === 'size' && this.resizeCustom && item['value'] === 'custom') {
        newObj[item['name']] = this.resizeCustom;
      } else {
        newObj[item['name']] = item['value'];
      }
    });
    return newObj;
  }

  getTransformParams(obj: any[]): { name: string; params: {}; }[] {
    let newObj = [];
    obj.forEach(item => {
      newObj.push({
        name: item.name,
        params: item.params && Array.isArray(item.params) ? this.getParams(item.params) : null
      })
    });
    return newObj;
  }

  showGraph() {
    this.showGraphSpinner = true;
    let height = window.innerHeight < 1000 ? '99%' : '95%';
    this.showGraphSpinner = false;
    this.dialog.open(GraphComponent, {
      width: '90%',
      height: height,
      data: {
        graph: this.graph,
        modelPath: this.firstFormGroup.get('modelLocation').value,
        viewSize: [window.innerWidth * 0.9, window.innerHeight * 0.9]
      }
    });
  }

  boundaryNodesVisible(): boolean {
    return (this.boundaryNodes.inputs !== 'none' || this.boundaryNodes.outputs !== 'none') && this.firstFormGroup.get('modelLocation').value && !this.showSpinner;
  }

  openDialog(fieldName: string, filter: FileBrowserFilter, paramFile?) {
    let form = 'firstFormGroup';
    const fileCategories = { 'dev-v1.1.json': 'label_file', 'vocab.txt': 'vocab_file' };
    if (filter === 'datasets') {
      form = 'secondFormGroup';
    }

    const dialogRef = this.dialog.open(FileBrowserComponent, {
      width: '60%',
      height: '60%',
      data: {
        path: this[form].get(fieldName) && this[form].get(fieldName).value ? this[form].get(fieldName).value.split("/").slice(0, -1).join("/") : this.modelService.workspacePath,
        filter: filter,
        filesToFind: paramFile ? Object.keys(fileCategories) : null
      }
    });

    dialogRef.afterClosed().subscribe(response => {
      if (response.chosenFile) {
        if (paramFile && paramFile !== 'datasetLocation') {
          if (paramFile === 'evaluation' || paramFile === 'quantization') {
            this.dataLoaderParams[paramFile].find(x => x.name === fieldName).value = response.chosenFile;
          } else if (paramFile === 'metric') {
            this.metricParam = response.chosenFile;
          } else {
            paramFile.find(x => x.name === fieldName).value = response.chosenFile;
          }
        } else {
          this[form].get(fieldName).setValue(response.chosenFile);
          if (fieldName === 'datasetLocationEvaluation' && this.useEvaluationData) {
            this.secondFormGroup.get('datasetLocationQuantization').setValue(response.chosenFile);
          }
        }

        if (response.foundFiles.length) {
          Object.keys(fileCategories).forEach((fileCategory, categoryIndex) => {
            const fileName = Object.keys(fileCategories)[categoryIndex];
            const fieldName = fileCategories[fileCategory];
            if (fieldName === 'label_file') {
              this.dataLoaderParams['evaluation'].find(x => x.name === fieldName).value = response.foundFiles.find(x => x.name.includes(fileName)).name;
              this.dataLoaderParams['quantization'].find(x => x.name === fieldName).value = response.foundFiles.find(x => x.name.includes(fileName)).name;
            }
            this.transformationParams
              .find(transformation => transformation.params.find(param => param.name === fieldName)).params
              .find(param => param.name === fieldName).value = response.foundFiles.find(x => x.name.includes(fileName)).name;
          });
        }
      }
    });;
  }

  updateCalibrationBatchSize() {
    const samplingSize = this.secondFormGroup.get('samplingSize').value;
    if (samplingSize.includes(',')) {
      this.secondFormGroup.get('calibrationBatchSize').setValue(samplingSize.substring(0, samplingSize.indexOf(',')));
    } else {
      this.secondFormGroup.get('calibrationBatchSize').setValue(samplingSize);
    }
  }

  openErrorDialog(error) {
    const dialogRef = this.dialog.open(ErrorComponent, {
      data: error
    });
  }

  coresValidated(): boolean {
    return this.secondFormGroup.get('cores_per_instance').value * this.secondFormGroup.get('num_of_instance').value <= this.modelService.systemInfo['cores_per_socket'] * this.modelService.systemInfo['sockets'];
  }

  getFileName(path: string): string {
    return path.replace(/^.*[\\\/]/, '');
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
  project_name: string;
  domain: string;
  domain_flavour: string;
  framework: string;
  id: string;
  model_path: string;
  inputs: string[];
  outputs: string[];
  precision: string;
  shape?: string;
  tuning: boolean,
  transform: {
    name: string;
    params: {};
  }[]
  quantization: {
    accuracy_goal: number;
    sampling_size: number;
    batch_size: number;
    op?: string[];
    strategy: string;
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
