
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
import { FormBuilder, FormGroup, Validators } from '@angular/forms';
import { MatDialog, MAT_DIALOG_DATA } from '@angular/material/dialog';
import { combineLatest, Subject } from 'rxjs';
import { ErrorComponent } from '../error/error.component';
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
  metricParam: string | boolean;
  metricParams = [];

  dataLoaders = [];
  dataLoaderParams = [];

  showDatasetLocation = true;

  transformations = [];
  transformationParams = [];

  useEvaluationData = true;
  fileBrowserParams = ['label_file', 'vocab_file', 'anno_path'];

  resizeValues;
  resizeCustom;

  metricList$: Subject<boolean> = new Subject<boolean>();
  metricValue$: Subject<boolean> = new Subject<boolean>();

  constructor(
    private _formBuilder: FormBuilder,
    public dialog: MatDialog,
    public modelService: ModelService,
    @Inject(MAT_DIALOG_DATA) public data
  ) { }

  ngOnInit(): void {
    this.setFormValues();

    combineLatest([
      this.metricList$,
      this.metricValue$
    ]).subscribe(([metricList, metricValue]) => {
      if (metricList === true && metricValue === true) {
        this.setDefaultMetricParam(this.datasetFormGroup.get('metric'));
      }
    });

    this.modelService.getDictionaryWithParam('dataloaders', 'framework', { framework: this.data.framework.toLowerCase() })
      .subscribe(response => this.dataLoaders = response['dataloaders']);
    this.modelService.getDictionaryWithParam('transforms', 'framework', { framework: this.data.framework.toLowerCase() })
      .subscribe(response => this.transformations = response['transforms']);

    this.getPossibleValues();
  }

  getPossibleValues() {
    this.modelService.getPossibleValues('metric', { framework: 'tensorflow' })
      .subscribe(
        resp => {
          this.metrics = resp['metric'];
          this.metricList$.next(true);
        },
        error => this.openErrorDialog(error));
  }


  setFormValues() {
    this.datasetFormGroup = this._formBuilder.group({
      name: ['Dataset' + String(this.data.index + 1), Validators.required],
      dataLoaderEvaluation: [''],
      datasetLocationEvaluation: [''],
      datasetParams0: [''],
      datasetParams1: [''],
      transform: [''],
      transformParams: [''],
      metric: ['', Validators.required],
      metricParam: [''],
    });
  }

  setDefaultDataLoaderParam(event) {
    if (this.dataLoaders.length) {
      const parameters = this.dataLoaders.find(x => x.name === event.value).params;
      this.dataLoaderParams = [];
      if (Array.isArray(parameters)) {
        parameters.forEach((param, index) => {
          this.dataLoaderParams[index] = {};
          Object.keys(param).forEach(paramValue => {
            this.dataLoaderParams[index][paramValue] = param[paramValue];
          });
        })
      }
      this.showDatasetLocation = this.dataLoaders.find(x => x.name === event.value).show_dataset_location;
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


  setDefaultTransformationParam(event, index: number) {
    this.transformationParams[index]['params'] = this.transformations.find(x => x.name === event.value).params;
  }

  addNewTransformation(name?: string) {
    this.transformationParams.push({ name: name ?? '', params: {} });
  }

  removeTransformation(index: number) {
    this.transformationParams.splice(index, 1);;
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

  addDataset() {
    const readyDataset = {
      "project_id": this.data.projectId,
      "name": this.datasetFormGroup.get('name').value,
      "dataset_path": this.datasetFormGroup.get('datasetLocationEvaluation').value,
      "transform": this.getTransformParams(this.transformationParams),
      "dataloader": {
        "name": this.datasetFormGroup.get('dataLoaderEvaluation').value,
        "params": this.dataLoaderParams ? this.getParams(this.dataLoaderParams) : null,
      },
      "metric": this.datasetFormGroup.get('metric').value,
      "metric_param": this.metricParam
    }

    this.modelService.addDataset(readyDataset)
      .subscribe(response => this.modelService.datasetCreated$.next(true));
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

  openDialog(fieldName: string, filter: FileBrowserFilter, paramFile?) {
    let form = 'datasetFormGroup';
    const fileCategories = { 'dev-v1.1.json': 'label_file', 'vocab.txt': 'vocab_file' };
    if (filter === 'datasets') {
      form = 'datasetFormGroup';
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
          if (paramFile === 'evaluation') {
            this.dataLoaderParams.find(x => x.name === fieldName).value = response.chosenFile;
          } else if (paramFile === 'metric') {
            this.metricParam = response.chosenFile;
          } else {
            paramFile.find(x => x.name === fieldName).value = response.chosenFile;
          }
        } else {
          this[form].get(fieldName).setValue(response.chosenFile);
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
    });;
  }

  isArray(obj: any): boolean {
    return Array.isArray(obj);
  }

  typeOf(obj: any): string {
    return typeof obj;
  }

  openErrorDialog(error) {
    const dialogRef = this.dialog.open(ErrorComponent, {
      data: error
    });
  }

}
