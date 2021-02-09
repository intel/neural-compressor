import { Component, OnInit } from '@angular/core';
import { FormBuilder, FormGroup, Validators } from '@angular/forms';
import { ModelService, NewModel } from '../services/model.service';
import { Md5 } from 'ts-md5/dist/md5';
import { FileBrowserComponent } from '../file-browser/file-browser.component';
import { MatDialog } from '@angular/material';
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
  dataLoaderParams = [];
  transformations = [];
  transformationParams = [];
  tunings = [];
  inputs = [];
  outputs = [];
  frameworkVersion: string;
  frameworkWarning: string;

  firstFormGroup: FormGroup;
  secondFormGroup: FormGroup;

  saved = false;
  id: string;
  showSpinner = false;

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
          this.socketService.getBoundaryNodes(this.getNewModel()).subscribe()
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
            this.getPossibleValues();
            this.frameworkVersion = result['data']['framework_version'];
            ['inputs', 'outputs'].forEach(param => {
              if (result['data'][param]) {
                if (result['data'][param].length > 1) {
                  this[param] = result['data'][param];
                } else {
                  this[param] = result['data'][param];
                  this.firstFormGroup.get(param.slice(0, -1)).setValue(result['data'][param]);
                }
              }
            });
          }
        }
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
    }
  }

  setDefaultDataLoaderParam(event) {
    this.dataLoaderParams = this.dataLoaders.find(x => x.name === event.value).params;
  }

  setDefaultTransformationParam(event, index) {
    this.transformationParams[index]['params'] = this.transformations.find(x => x.name === event.value).params;
  }

  addNewTransformation(name?: string) {
    this.transformationParams.push({ name: name ? name : '', params: {} });
  }

  removeTransformation(index: number) {
    this.transformationParams.splice(index, 1);;
  }

  setDefaultValues() {
    this.firstFormGroup = this._formBuilder.group({
      framework: ['', Validators.required],
      datasetLocation: ['', Validators.required],
      modelLocation: ['', Validators.required],
      modelDomain: ['', Validators.required],
      input: [''],
      output: [''],
    });
    this.secondFormGroup = this._formBuilder.group({
      accuracyGoal: [0.01],
      dataLoader: [''],
      datasetParams0: [''],
      datasetParams1: [''],
      transform: [''],
      transformParams: [''],
      samplingSize: [''],
      op: [''],
      strategy: [''],
      batchSize: [1],
      cores_per_instance: [''],
      num_of_instance: [''],
      inter_num_of_threads: [''],
      intra_num_of_threads: [''],
      kmp_blocktime: [''],
      warmup: [],
      iteration: [],
      metric: [''],
      objective: [''],
      timeout: [0],
      maxTrials: [100],
      randomSeed: [],
      approach: [],
    });
  }

  getConfig() {
    const newModel = this.getNewModel();
    this.modelService.getConfiguration(newModel)
      .subscribe(resp => {
        if (resp['config']) {
          this.firstFormGroup.get('framework').setValue(resp['framework']);
          this.firstFormGroup.get('modelDomain').setValue(resp['domain']);

          if (resp['config']['quantization']) {
            this.secondFormGroup.get('dataLoader').setValue(resp['config']['quantization'].calibration.dataloader.dataset);
            this.secondFormGroup.get('transform').setValue(resp['config']['quantization'].calibration.dataloader.transform);
            this.secondFormGroup.get('samplingSize').setValue(resp['config']['quantization'].calibration.sampling_size);
            this.secondFormGroup.get('approach').setValue(resp['config']['quantization'].approach);
            const dataLoader = Object.keys(resp['config']['quantization'].calibration.dataloader.dataset)[0];
            const transformNames = Object.keys(resp['config']['quantization'].calibration.dataloader.transform);
            this.secondFormGroup.get('dataLoader').setValue(dataLoader);
            this.transformationParams = [];
            transformNames.forEach((name, index) => {
              this.addNewTransformation(name);
              this.transformationParams[index]['params'] = this.transformations.find(x => x.name === name).params;
              if (Array.isArray(this.transformationParams[index]['params'])) {
                this.transformationParams[index]['params'].forEach(param => {
                  param.value = resp['config']['quantization'].calibration.dataloader.transform[name][param.name];
                });
              }
            });
            this.dataLoaderParams = this.dataLoaders.find(x => x.name === dataLoader).params;
          }

          if (resp['config']['evaluation']) {
            this.secondFormGroup.get('metric').setValue(Object.keys(resp['config']['evaluation'].accuracy.metric)[0]);
            this.metricParams = this.metrics.find(x => x.name === this.secondFormGroup.get('metric').value).params;
            this.metricParam = this.metricParams[0].value;
            if (Array.isArray(this.metricParams[0].value)) {
              this.metricParam = this.metricParams[0].value[0];
            }
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

          if (typeof resp['config']['model'].outputs === 'string') {
            this.outputs = [resp['config']['model'].outputs];
            this.firstFormGroup.get('output').setValue(resp['config']['model'].outputs);
          } else if (Array.isArray(resp['config']['model'].outputs)) {
            this.outputs = resp['config']['model'].outputs;
          }
        }
      },
        error => {
          this.openErrorDialog(error);
        });
  }

  addModel() {
    const newModel = this.getFullModel();
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
      dataset_path: this.firstFormGroup.get('datasetLocation').value,
      domain: this.firstFormGroup.get('modelDomain').value,
      framework: this.firstFormGroup.get('framework').value,
      id: this.id,
      model_path: this.firstFormGroup.get('modelLocation').value,
    };
    return model;
  }

  getFullModel(): FullModel {
    let model: FullModel;
    if (this.dataLoaderParams && Object.keys(this.dataLoaderParams).includes('root')) {
      this.dataLoaderParams['root'] = this.firstFormGroup.get('datasetLocation').value;
    }
    model = {
      dataset_path: this.firstFormGroup.get('datasetLocation').value,
      domain: this.firstFormGroup.get('modelDomain').value,
      framework: this.firstFormGroup.get('framework').value,
      id: this.id,
      model_path: this.firstFormGroup.get('modelLocation').value,
      inputs: this.firstFormGroup.get('input').value,
      outputs: this.firstFormGroup.get('output').value,
      dataloader: {
        name: this.secondFormGroup.get('dataLoader').value,
        params: this.dataLoaderParams ? this.getParams(this.dataLoaderParams) : null,
      },
      transform: this.getTransformParams(this.transformationParams),
      quantization: {
        accuracy_goal: this.secondFormGroup.get('accuracyGoal').value,
        sampling_size: this.secondFormGroup.get('samplingSize').value,
        // op: this.secondFormGroup.get('op').value,
        strategy: this.secondFormGroup.get('strategy').value,
        approach: this.secondFormGroup.get('approach').value,
        objective: this.secondFormGroup.get('objective').value,
        timeout: this.secondFormGroup.get('timeout').value,
        max_trials: this.secondFormGroup.get('maxTrials').value,
        random_seed: this.secondFormGroup.get('randomSeed').value
      },
      evaluation: {
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

  openDialog(fieldName: string, files: boolean, modelsOnly: boolean) {
    const dialogRef = this.dialog.open(FileBrowserComponent, {
      width: '60%',
      height: '60%',
      data: {
        path: this.firstFormGroup.get(fieldName).value ? this.firstFormGroup.get(fieldName).value.split("/").slice(0, -1).join("/") : this.modelService.workspacePath,
        files: files,
        modelsOnly: modelsOnly
      }
    });

    dialogRef.afterClosed().subscribe(chosenFile => {
      if (chosenFile) {
        this.firstFormGroup.get(fieldName).setValue(chosenFile);
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
  dataset_path: string;
  domain: string;
  framework: string;
  id: string;
  model_path: string;
  inputs: string[];
  outputs: string[];
  dataloader: {
    name: string;
    params: {};
  }
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
  }
}
