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
import { MatDialog, MAT_DIALOG_DATA } from '@angular/material/dialog';
import { ModelService } from '../services/model.service';

@Component({
  selector: 'app-optimization-form',
  templateUrl: './optimization-form.component.html',
  styleUrls: ['./optimization-form.component.scss', './../error/error.component.scss', './../project-form/project-form.component.scss',]
})
export class OptimizationFormComponent implements OnInit {

  precisions = [];
  precisionId: number;

  optimizationTypes = [];
  optimizationTypeId: number;

  datasets = [];
  datasetId: number;

  name: string;

  constructor(
    private dialog: MatDialog,
    public modelService: ModelService,
    @Inject(MAT_DIALOG_DATA) public data
  ) { }

  ngOnInit(): void {
    this.name = "Optimization" + String(this.data.index + 1);
    this.getPrecisions();
    this.getDatasets();
  }

  getPrecisions() {
    this.modelService.getDictionary('precisions')
      .subscribe(
        response => {
          this.precisions = response['precisions'];
          this.precisionId = this.precisions.find(x => x.name === 'int8').id;
          this.getOptimizationTypes();
        },
        error => {
          this.modelService.openErrorDialog(error);
        });
  }

  getOptimizationTypes() {
    this.modelService.getDictionaryWithParam('optimization_types', 'precision', { precision: this.precisions.find(x => x.id === this.precisionId).name })
      .subscribe(
        response => {
          this.optimizationTypes = response['optimization_types'];
          for (let type of this.optimizationTypes) {
            if (type.is_supported) {
              this.optimizationTypeId = type.id;
              break;
            }
          };
        },
        error => {
          this.modelService.openErrorDialog(error);
        });
  }

  getDatasets() {
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

  addOptimization() {
    this.modelService.addOptimization({
      project_id: this.data.projectId,
      name: this.name,
      precision_id: this.precisionId,
      optimization_type_id: this.optimizationTypeId,
      dataset_id: this.datasetId
    })
      .subscribe(
        response => { this.modelService.optimizationCreated$.next(true) },
        error => {
          this.modelService.openErrorDialog(error);
        });
  }

}
