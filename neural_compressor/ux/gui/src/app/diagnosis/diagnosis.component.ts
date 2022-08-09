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

import { Component, Input, OnInit } from '@angular/core';
import { ActivatedRoute, Router } from '@angular/router';
import { ModelService } from '../services/model.service';
import { ModelWiseComponent } from '../model-wise/model-wise.component';
import { MatDialog } from '@angular/material/dialog';
import { ConfigPreviewComponent } from '../config-preview/config-preview.component';
import { GenerateConfigDialogComponent } from '../generate-config-dialog/generate-config-dialog.component';

@Component({
  selector: 'app-diagnosis',
  templateUrl: './diagnosis.component.html',
  styleUrls: ['./diagnosis.component.scss', './../error/error.component.scss', './../datasets/datasets.component.scss']
})
export class DiagnosisComponent implements OnInit {

  @Input() modelPath: string;
  @Input() inputModelId: number;
  models = [];
  modelId: number;
  optimizationId?: number;
  opList = {};
  opDetails = {};
  activeOp = '';
  activeType: 'weights' | 'activation';
  chartOptions;
  updatedValues = {};
  modelWise = {};
  precision: 'int8' | 'float32';
  granularity: 'per_channel' | 'per_tensor';
  showOps = true;
  showSpinner = false;

  constructor(
    private modelService: ModelService,
    public activatedRoute: ActivatedRoute,
    public dialog: MatDialog,
    private router: Router
  ) { }

  ngOnInit(): void {
    this.getModels();
    this.modelId = this.inputModelId;
    this.optimizationId = null;
    this.getOpList(this.inputModelId);
    this.modelService.getNodeDetails$
      .subscribe(
        response => this.getOpDetails(this.modelId, response)
      )
  }

  getModels(id?: number) {
    this.modelService.getModelList(id ?? this.activatedRoute.snapshot.params.id)
      .subscribe(
        response => {
          this.models = response['models'];
        },
        error => {
          this.modelService.openErrorDialog(error);
        });
  }

  getOpList(modelId: number) {
    this.showSpinner = true;
    this.modelId = modelId;

    // Update optimization ID
    this.modelService.getOptimizationList(this.activatedRoute.snapshot.params.id)
      .subscribe(
        response => {
          this.optimizationId = response['optimizations'].find(x => x.optimized_model?.id == this.modelId)?.id
        },
        error => {
          if (error.error === 'No row was found when one was required') {
          } else {
            this.modelService.openErrorDialog(error);
          }
        });

    this.modelService.getOpList(this.activatedRoute.snapshot.params.id, modelId)
      .subscribe(
        response => {
          this.showSpinner = false;
          this.opList = response;
          this.showOps = true;
        },
        error => {
          this.showSpinner = false;
          if (error.error === 'No row was found when one was required') {
            this.showOps = false;
          } else if (error.error.match("No such file or directory: '.*/dequan_min_max.pkl'")) {
            this.modelService.openWarningDialog(
              "Diagnosis is supported only for real data quantization."
            )
            this.showOps = false;
          } else {
            this.modelService.openErrorDialog(error);
          }
        });
  }
  

  getOpDetails(modelId: number, opName: string) {
    this.activeOp = opName;
    this.modelService.getOpDetails(this.activatedRoute.snapshot.params.id, modelId, opName)
      .subscribe(
        response => {
          if (Object.keys(response).length) {
            this.showOps = true;
            this.opDetails = response;
            this.precision = response['Pattern']['precision'];
            this.granularity = response['Weights']['granularity'];
          }
        },
        error => {
          if (error.error === 'No row was found when one was required') {
            this.showOps = false;
          } else {
            this.modelService.openErrorDialog(error);
          }
        });
  }

  updateValue(valueName: string, value: string, opName: string) {
    if (!this.updatedValues[opName]) {
      this.updatedValues[opName] = {};
    }

    this.updatedValues[opName][valueName] = value;
  }

  applyChanges() {
    const dialogRef = this.dialog.open(GenerateConfigDialogComponent);
    let updatedValuesToSave = {
      op_wise: {},
      model_wise: {},
      project_id: this.activatedRoute.snapshot.params.id,
      model_id: this.modelId,
      optimization_id: this.optimizationId,
    };

    Object.keys(this.updatedValues).forEach(opName => {
      updatedValuesToSave.op_wise[opName] = {};
      if (this.updatedValues[opName].precision) {
        updatedValuesToSave.op_wise[opName]['pattern'] = {
          precision: this.updatedValues[opName].precision
        };
      }
      if (this.updatedValues[opName].granularity) {
        updatedValuesToSave.op_wise[opName]['weight'] = {
          granularity: this.updatedValues[opName].granularity
        };
      }
    });

    updatedValuesToSave.model_wise = this.modelWise;

    dialogRef.afterClosed().subscribe(response => {
      if (response) {
        updatedValuesToSave['optimization_name'] = response['optimizationName'];

        this.modelService.generateOptimization(updatedValuesToSave)
          .subscribe(
            response => {
              this.router.navigate(['project', this.activatedRoute.snapshot.params.id, 'optimizations'], { queryParamsHandling: "merge" });
              this.modelService.projectChanged$.next({ id: this.activatedRoute.snapshot.params.id, tab: 'optimizations' });
            },
            error => {
              this.modelService.openErrorDialog(error);
            });
      }
    });
  }

  viewConfiguration() {
    let updatedValuesToShow = {
      op_wise: {},
      model_wise: {},
    };
    updatedValuesToShow['model_wise'] = this.modelWise;
    updatedValuesToShow['op_wise'] = this.updatedValues;
    const dialogRef = this.dialog.open(ConfigPreviewComponent, {
      data: {
        updatedValues: updatedValuesToShow
      }
    });
  }

  resetChanges(opName: string) {
    delete this.updatedValues[opName];
    this.precision = this.opDetails['Pattern']['precision'];
    this.granularity = this.opDetails['Weights']['granularity'];
  }

  openModelWiseDialog() {
    const dialogRef = this.dialog.open(ModelWiseComponent, {
      data: {
        optimizationId: this.optimizationId,
        modelWise: this.modelWise
      }
    });

    dialogRef.afterClosed().subscribe(response => {
      if (response) {
        this.modelWise = response['modelWise'];
      }
    });
  }

  returnZero() {
    return 0;
  }

  typeOf(obj): string {
    return typeof obj;
  }

  objectKeys(obj: any): string[] {
    return Object.keys(obj);
  }
}
