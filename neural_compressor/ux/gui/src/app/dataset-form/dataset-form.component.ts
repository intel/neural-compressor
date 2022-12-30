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
import { Component, Inject, OnInit } from '@angular/core';
import { FormControl, FormGroup, Validators } from '@angular/forms';
import { MatDialog, MAT_DIALOG_DATA } from '@angular/material/dialog';
import { combineLatest, Subject } from 'rxjs';
import { FileBrowserComponent } from '../file-browser/file-browser.component';
import { FileBrowserFilter, ModelService } from '../services/model.service';

@Component({
  selector: 'app-dataset-form',
  templateUrl: './dataset-form.component.html',
  styleUrls: ['./dataset-form.component.scss', './../error/error.component.scss', './../project-form/project-form.component.scss']
})
export class DatasetFormComponent implements OnInit {
  datasetFormGroup: FormGroup;

  metrics = [];
  metricParams = [];
  metricParamsCopy = [];

  dataLoaders = [];
  dataLoaderParams = [];

  showDatasetLocation = true;
  datasetLocationPlaceholder = '';

  transformations = [];
  transformationParams = [];

  useEvaluationData = true;
  fileBrowserParams = ['label_file', 'vocab_file', 'anno_path', 'annotation_path'];

  resizeValues;
  resizeCustom;

  metricList$: Subject<boolean> = new Subject<boolean>();
  metricValue$: Subject<boolean> = new Subject<boolean>();

  constructor(
    public dialog: MatDialog,
    public modelService: ModelService,
    @Inject(MAT_DIALOG_DATA) public data
  ) { }

  ngOnInit(): void {
    this.setFormValues();

    combineLatest([
      this.metricList$,
      this.metricValue$
    ]).subscribe(
      ([metricList, metricValue]) => {
        if (metricList === true && metricValue === true) {
          this.setDefaultMetricParam(this.datasetFormGroup.get('metric'));
        }
      },
      error => {
        this.modelService.openErrorDialog(error);
      }
    );

    this.modelService.getDictionaryWithParam('dataloaders', 'framework', { framework: this.data.framework.toLowerCase() })
      .subscribe(
        (response: { dataloaders: any }) => {
          this.dataLoaders = response.dataloaders;
        },
        error => {
          this.modelService.openErrorDialog(error);
        });
    this.modelService.getDictionaryWithParam('transforms', 'framework', { framework: this.data.framework.toLowerCase() })
      .subscribe(
        (response: { transforms: any }) => {
          this.transformations = response.transforms;
        },
        error => {
          this.modelService.openErrorDialog(error);
        }
      );

    this.getPossibleValues();
  }

  getPossibleValues() {
    this.modelService.getPossibleValues('metric', { framework: this.data.framework.toLowerCase() })
      .subscribe(
        (response: { metric: any }) => {
          this.metrics = response.metric;
          this.metricList$.next(true);
          this.getPredefinedDatasets();
        },
        error => this.modelService.openErrorDialog(error));
  }

  getPredefinedDatasets() {
    this.modelService.getPredefinedDatasets(this.data.framework, this.data.domain, this.data.domainFlavour)
      .subscribe((response: { transform: [{ name: string }]; dataloader: any; metric: any }) => {
        for (let i = 0; i < response.transform.length; i++) {
          this.addNewTransformation(response.transform[i].name);
          this.setDefaultTransformationParam({ value: response.transform[i] }, i);
        }

        this.datasetFormGroup.get('dataLoader').setValue(response.dataloader.name);
        this.setDefaultDataLoaderParam({ value: response.dataloader.name });
        if (response.dataloader.params?.[0].name === 'root') {
          this.datasetLocationPlaceholder = response.dataloader.params[0].value;
          this.isFieldRequired('datasetLocation', true);
        } else {
          this.isFieldRequired('datasetLocation', false);
        }

        this.datasetFormGroup.get('metric').setValue(response.metric);
        this.setDefaultMetricParam({ value: response.metric });
      });
  }

  setFormValues() {
    this.datasetFormGroup = new FormGroup({
      name: new FormControl('Dataset' + String(this.data.index + 1), Validators.required),
      dataLoader: new FormControl('', Validators.required),
      datasetLocation: new FormControl(''),
      datasetParams0: new FormControl(''),
      datasetParams1: new FormControl(''),
      transform: new FormControl(''),
      transformParams: new FormControl(''),
      metric: new FormControl('', Validators.required),
      metricParam: new FormControl(''),
    });
  }

  setDefaultDataLoaderParam(event) {
    if (this.dataLoaders.length) {
      const parameters = this.dataLoaders.find(x => x.name === event.value)?.params;
      this.dataLoaderParams = [];
      if (Array.isArray(parameters)) {
        parameters.forEach((param, index) => {
          this.dataLoaderParams[index] = {};
          Object.keys(param).forEach(paramValue => {
            this.dataLoaderParams[index][paramValue] = param[paramValue];
          });
        });
      }
      this.showDatasetLocation = this.dataLoaders.find(x => x.name === event.value)?.show_dataset_location;
      const hasRootEntry = this.dataLoaderParams.find(x => x.name === 'root') !== undefined;
      if (hasRootEntry) {
        this.isFieldRequired('datasetLocation', true);
      } else {
        this.isFieldRequired('datasetLocation', false);
      }
    }
  }

  isFieldRequired(field: string, required: boolean) {
    if (required) {
      this.datasetFormGroup.controls[field].setValidators([Validators.required]);
    } else {
      this.datasetFormGroup.controls[field].clearValidators();
    }
    this.datasetFormGroup.controls[field].updateValueAndValidity();
  }


  setDefaultTransformationParam(event, index: number) {
    if (!event.value.hasOwnProperty('params')) {
      // Case when event source is MatSelect
      this.transformationParams[index].params = this.transformations.find(x => x.name === event.value).params;
      return;
    }

    const transformParameters: { name: string; value: any }[] = this.transformations.find(x => x.name === event.value.name).params;
    event.value.params.forEach(item => {
      transformParameters.find(x => x.name === item.name).value = item.value;
    });
    this.transformationParams[index].params = transformParameters;
  }

  addNewTransformation(name?: string) {
    this.transformationParams.push({ name: name ?? '', params: {} });
  }

  removeTransformation(index: number) {
    this.transformationParams.splice(index, 1);;
  }

  setDefaultMetricParam(event) {
    if (this.metrics.length) {
      this.metricParams = this.metrics.find(x => x.name === event.value)?.params;
      this.metricParamsCopy = this.metricParams ? (JSON.parse(JSON.stringify(this.metricParams))) : undefined;
      if (this.metricParams && this.metricParams.length === 1) {
        this.metricParamsCopy[0].value = this.metricParams[0].value;
        if (Array.isArray(this.metricParams[0].value)) {
          this.metricParamsCopy[0].value = this.metricParams[0].value[0];
        }
      }
    }
  }

  addDataset() {
    const readyDataset = {
      project_id: this.data.projectId,
      name: this.datasetFormGroup.get('name').value,
      dataset_path: this.datasetFormGroup.get('datasetLocation').value,
      transform: this.getTransformParams(this.transformationParams),
      dataloader: {
        name: this.datasetFormGroup.get('dataLoader').value,
        params: this.dataLoaderParams ? this.getParams(this.dataLoaderParams) : null,
      },
      metric: this.datasetFormGroup.get('metric').value,
      metric_param: this.getMetricParams(),
    };

    this.modelService.addDataset(readyDataset)
      .subscribe(
        response => { this.modelService.datasetCreated$.next(true); },
        error => { this.modelService.openErrorDialog(error); }
      );
  }

  getMetricParams(): any {
    const parsedMetricParameter = {};
    if (Array.isArray(this.metricParamsCopy)) {
      this.metricParamsCopy.forEach(param => {
        if (param.params) {
          parsedMetricParameter[param.name] = {};
          param.params.forEach(nestedParam => {
            parsedMetricParameter[param.name][nestedParam.name] = nestedParam.value;
          });
        } else {
          parsedMetricParameter[param.name] = param.value;
        }
      });
      if (this.metricParamsCopy[0].name === 'k') {
        return this.metricParamsCopy[0].value;
      }
    }
    return parsedMetricParameter;
  }

  getParams(obj: any[]): any {
    const newObj = {};
    obj.forEach(item => {
      if (item.name === 'size' && this.resizeCustom && item.value === 'custom') {
        newObj[item.name] = this.resizeCustom;
      } else {
        newObj[item.name] = item.value;
      }
    });
    return newObj;
  }

  getTransformParams(obj: any[]): { name: string; params: any }[] {
    if (this.datasetFormGroup.get('dataLoader').value === 'custom') {
      return [];
    }
    const newObj = [];
    obj.forEach(item => {
      newObj.push({
        name: item.name,
        params: item.params && Array.isArray(item.params) ? this.getParams(item.params) : null
      });
    });
    return newObj;
  }

  openDialog(inputFieldName: string, filter: FileBrowserFilter, paramFile?) {
    const fileCategories = { 'dev-v1.1.json': 'label_file', 'vocab.txt': 'vocab_file' };

    const dialogRef = this.dialog.open(FileBrowserComponent, {
      width: '60%',
      height: '60%',
      data: {
        path: this.getPath(inputFieldName),
        filter,
        filesToFind: paramFile ? Object.keys(fileCategories) : null
      }
    });

    dialogRef.afterClosed().subscribe(
      response => {
        if (response.chosenFile) {
          if (paramFile && paramFile !== 'datasetLocation') {
            if (paramFile === 'dataset') {
              this.dataLoaderParams.find(x => x.name === inputFieldName).value = response.chosenFile;
            } else if (paramFile === 'metric') {
              this.metricParamsCopy.find(x => x.name === inputFieldName).value = response.chosenFile;
            } else {
              paramFile.find(x => x.name === inputFieldName).value = response.chosenFile;
            }
          } else {
            this.datasetFormGroup.get(inputFieldName).setValue(response.chosenFile);
          }

          if (response.foundFiles.length) {
            Object.keys(fileCategories).forEach((fileCategory, categoryIndex) => {
              const fileName = Object.keys(fileCategories)[categoryIndex];
              const fieldName = fileCategories[fileCategory];
              if (fieldName === 'label_file') {
                this.dataLoaderParams.find(x => x.name === fieldName).value = response.foundFiles.find(x => x.name.includes(fileName)).name;
              }
              this.transformationParams
                .find(transformation => transformation.params.find(param => param.name === fieldName)).params
                .find(param => param.name === fieldName).value = response.foundFiles.find(x => x.name.includes(fileName)).name;
            });
          }
        }
      },
      error => {
        this.modelService.openErrorDialog(error);
      });;
  }

  getPath(fieldName: string) {
    return this.datasetFormGroup.get(fieldName) && this.datasetFormGroup.get(fieldName).value
      && !this.datasetFormGroup.get(fieldName).value.includes('/path/to/')
      ? this.datasetFormGroup.get(fieldName).value.split('/').slice(0, -1).join('/')
      : this.modelService.workspacePath;
  }

  isArray(obj: any): boolean {
    return Array.isArray(obj);
  }

  typeOf(obj: any): string {
    return typeof obj;
  }
}
