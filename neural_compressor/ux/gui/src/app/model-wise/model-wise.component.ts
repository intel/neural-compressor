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
import { MatDialogRef, MAT_DIALOG_DATA } from '@angular/material/dialog';
import { ModelService } from '../services/model.service';

@Component({
  selector: 'app-model-wise',
  templateUrl: './model-wise.component.html',
  styleUrls: ['./model-wise.component.scss', './../error/error.component.scss', './../datasets/datasets.component.scss']
})
export class ModelWiseComponent implements OnInit {

  modelWiseParams;
  selectorParams = {
    weight: [],
    activation: [],
  };
  visibleParams = {
    weight: [],
    activation: [],
  };
  chosenParams = {
    weight: {},
    activation: {}
  };
  params = {
    weight: [],
    activation: [],
  };
  paramIndex = {
    weight: [],
    activation: [],
  };

  constructor(
    private modelService: ModelService,
    @Inject(MAT_DIALOG_DATA) public data,
    public dialogRef: MatDialogRef<ModelWiseComponent>,
  ) { }

  ngOnInit(): void {
    if (Object.keys(this.data.modelWise).length) {
      ['weight', 'activation'].forEach(type => {
        Object.keys(this.data.modelWise[type]).forEach((paramName, index) => {
          this.addNewParam(type);
          this.selectParameter(type, paramName);
          this.params[type][index] = paramName;
          this.chosenParams[type][paramName] = this.data.modelWise[type][paramName];
        })
      });
    }

    this.modelService.getModelWiseParams(this.data.optimizationId)
      .subscribe(
        response => {
          this.modelWiseParams = response['model_wise'];
          this.selectorParams.weight = Object.keys(response['model_wise']['weight']);
          this.selectorParams.activation = Object.keys(response['model_wise']['activation']);
        },
        error => {
          this.modelService.openErrorDialog(error);
        });
  }

  selectParameter(type: string, parameter: string) {
    this.visibleParams[type].push(parameter);
  }

  addNewParam(type: string) {
    this.paramIndex[type].push(1);
  }

  removeParameter(type: 'weight' | 'activation', index: number, paramName: string) {
    delete this.params[type][index];
    delete this.chosenParams[type][paramName];
    delete this.paramIndex[type][index];
  }

  saveModelWise() {
    this.dialogRef.close({
      modelWise: this.chosenParams,
    });
  }


  objectKeys(obj: any): string[] {
    return Object.keys(obj);
  }
}
