import { Component, OnInit } from '@angular/core';
import { FormBuilder, FormGroup, Validators } from '@angular/forms';
import { ModelService, NewModel } from '../services/model.service';
import { Md5 } from 'ts-md5/dist/md5';
import { debounceTime, filter, map, pairwise } from 'rxjs/operators';
import { FileBrowserComponent } from '../file-browser/file-browser.component';
import { MatDialog } from '@angular/material';
import { SocketService } from '../services/socket.service';

@Component({
  selector: 'app-import-model',
  templateUrl: './import-model.component.html',
  styleUrls: ['./import-model.component.scss', './../start-page/start-page.component.scss']
})
export class ImportModelComponent implements OnInit {

  frameworks = ['tensorflow', 'pytorch'];
  datasets = {
    tensorflow: ['dummy', 'ImageNet', 'TFRecordDataset', 'COCORecord', 'style_transfer'],
    pytorch: ['dummy', 'ImageNet', 'ImageFolder', 'DatasetFolder', 'Bert'],
  };
  domains = ['image_recognition', 'NLP', 'object_detection', 'recommendation'];
  ops = [];
  transforms = {
    tensorflow: ['Resize', 'CenterCrop', 'RandomResizedCrop', 'Normalize', 'RandomCrop', 'Compose', 'CropAndResize',
      'RandomHorizontalFlip', 'RandomVerticalFlip', 'DecodeImage', 'EncodeJped', 'Transpose', 'CropToBoundingBox', 'ConvertImageDtype'],
    pytorch: ['Resize', 'CenterCrop', 'RandomResizedCrop', 'Normalize', 'RandomCrop', 'Compose', 'RandomHorizontalFlip',
      'RandomVerticalFlip', 'ToTensor', 'ToPILImage', 'Pad', 'ColorJitter'],
  };
  metrics = {};
  dataLoaders = {};
  dataLoaderParams = {};
  transformations = {};
  transformationParams = {};
  tunings = [];
  outputs = [];

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
          this.socketService.getBoundaryNodes(this.getNewModel()).subscribe()
        }
      }
      );

    this.socketService.boundaryNodesStart$
      .subscribe(result => {
        this.showSpinner = true;
      });

    this.socketService.boundaryNodesFinish$
      .subscribe(result => {
        this.showSpinner = false;
        if (result['data'] && result['data']['inputs'] && result['data']['outputs']) {
          this.firstFormGroup.get('input').setValue(result['data']['inputs']);
          this.outputs = [result['data']['outputs']];
          this.firstFormGroup.get('output').setValue(result['data']['outputs']);
        }
      });
  }

  getPossibleValues() {
    this.modelService.getPossibleValues('metric', { framework: this.firstFormGroup.get('framework').value })
      .subscribe(resp =>
        this.metrics = resp['metric']);
    this.modelService.getPossibleValues('dataloader', { framework: this.firstFormGroup.get('framework').value })
      .subscribe(resp =>
        this.dataLoaders = resp['dataloader']);
    this.modelService.getPossibleValues('transform', { framework: this.firstFormGroup.get('framework').value })
      .subscribe(resp =>
        this.transformations = resp['transform']);
    this.modelService.getPossibleValues('strategy', { framework: this.firstFormGroup.get('framework').value })
      .subscribe(resp =>
        this.tunings = resp['strategy']);
  }

  setDefaultMetricParam(event) {
    if (this.metrics[event.value]) {
      if (Array.isArray(this.metrics[event.value][Object.keys(this.metrics[event.value])[0]])) {
        this.secondFormGroup.get('metricParam').setValue(this.metrics[event.value][Object.keys(this.metrics[event.value])[0]][0]);
      } else {
        this.secondFormGroup.get('metricParam').setValue(this.metrics[event.value][Object.keys(this.metrics[event.value])[0]]);
      }
    }
  }

  setDefaultDataLoaderParam(event) {
    if (this.dataLoaders[event.value]) {
      this.dataLoaderParams = this.dataLoaders[event.value];
    }
  }

  setDefaultTransformationParam(event) {
    if (this.transformations[event.value]) {
      this.transformationParams = this.transformations[event.value];
    }
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
      strategy: ['basic'],
      batchSize: [1],
      cores_per_instance: [''],
      num_of_instance: [''],
      inter_num_of_threads: [''],
      intra_num_of_threads: [''],
      kmp_blocktime: [''],
      warmup: [''],
      iteration: [''],
      metric: [''],
      metricParam: [''],
    });
  }

  getConfig() {
    const newModel = this.getNewModel();
    this.modelService.getConfiguration(newModel)
      .subscribe(resp => {
        if (resp['config']) {
          this.firstFormGroup.get('framework').setValue(resp['framework']);
          this.getPossibleValues();

          this.firstFormGroup.get('modelDomain').setValue(resp['domain']);
          this.secondFormGroup.get('dataLoader').setValue(resp['config']['quantization'].calibration.dataloader.dataset);
          this.secondFormGroup.get('transform').setValue(resp['config']['quantization'].calibration.dataloader.transform);
          this.secondFormGroup.get('metric').setValue(Object.keys(resp['config']['evaluation'].accuracy.metric)[0]);
          this.secondFormGroup.get('samplingSize').setValue(resp['config']['quantization'].calibration.sampling_size);
          this.secondFormGroup.get('strategy').setValue(resp['config']['tuning'].strategy.name);
          this.secondFormGroup.get('kmp_blocktime').setValue(resp['config']['evaluation'].performance.configs.kmp_blocktime);
          this.secondFormGroup.get('warmup').setValue(resp['config']['evaluation'].performance.warmup);
          this.secondFormGroup.get('iteration').setValue(resp['config']['evaluation'].performance.iteration);
          this.secondFormGroup.get('batchSize').setValue(resp['config']['evaluation'].performance.dataloader.batch_size);

          const dataLoader = Object.keys(resp['config']['quantization'].calibration.dataloader.dataset)[0];
          const transform = Object.keys(resp['config']['quantization'].calibration.dataloader.transform)[0];
          this.secondFormGroup.get('dataLoader').setValue(dataLoader);
          this.secondFormGroup.get('transform').setValue(transform);
          this.dataLoaderParams = resp['config']['quantization'].calibration.dataloader.dataset[dataLoader];
          this.transformationParams = resp['config']['quantization'].calibration.dataloader.transform[transform];

          if (typeof resp['config']['model'].outputs === 'string') {
            this.outputs = [resp['config']['model'].outputs];
            this.firstFormGroup.get('output').setValue(resp['config']['model'].outputs);
          } else if (Array.isArray(resp['config']['model'].outputs)) {
            this.outputs = resp['config']['model'].outputs;
          }
        }
      });
  }

  addModel() {
    const newModel = this.getNewModel();
    this.modelService.addModel(newModel);
    this.modelService.saveWorkload(this.getFullModel())
      .subscribe();
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
    model = {
      dataset_path: this.firstFormGroup.get('datasetLocation').value,
      domain: this.firstFormGroup.get('modelDomain').value,
      framework: this.firstFormGroup.get('framework').value,
      id: this.id,
      model_path: this.firstFormGroup.get('modelLocation').value,
      dataloader: {
        name: this.secondFormGroup.get('dataLoader').value,
        params: this.dataLoaderParams,
      },
      transform: {
        name: this.secondFormGroup.get('transform').value,
        params: this.transformationParams,
      },
      quantization: {
        accuracy_goal: this.secondFormGroup.get('accuracyGoal').value,
        sampling_size: this.secondFormGroup.get('samplingSize').value,
        // op: this.secondFormGroup.get('op').value,
        strategy: this.secondFormGroup.get('strategy').value,
      },
      evaluation: {
        metric: this.secondFormGroup.get('metric').value,
        metric_param: this.secondFormGroup.get('metricParam').value,
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

  objectKeys(obj: {}): string[] {
    return Object.keys(obj);
  }

  typeOf(obj: any): string {
    return typeof obj;
  }
}

export interface FullModel {
  dataset_path: string;
  domain: string;
  framework: string;
  id: string;
  model_path: string;
  dataloader: {
    name: string;
    params: {};
  }
  transform: {
    name: string;
    params: {};
  }
  quantization: {
    accuracy_goal: number;
    sampling_size: number;
    op?: string[];
    strategy: string;
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
