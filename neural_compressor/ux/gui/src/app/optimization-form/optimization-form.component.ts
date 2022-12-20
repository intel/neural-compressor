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
import { AfterViewInit, Component, Inject, OnInit } from '@angular/core';
import { FormControl, FormGroup, Validators } from '@angular/forms';
import { MAT_DIALOG_DATA } from '@angular/material/dialog';
import { ShortcutInput } from 'ng-keyboard-shortcuts';
import { ModelService } from '../services/model.service';

@Component({
  selector: 'app-optimization-form',
  templateUrl: './optimization-form.component.html',
  styleUrls: ['./optimization-form.component.scss', './../error/error.component.scss', './../project-form/project-form.component.scss',]
})
export class OptimizationFormComponent implements OnInit, AfterViewInit {

  shortcuts: ShortcutInput[] = [];

  precisions = [];
  precisionsPyTorch = [];
  precisionsOther = [];

  optimizationTypes = [];
  supportedTypes = [];

  datasets = [];

  optimizationFormGroup: FormGroup;

  constructor(
    public modelService: ModelService,
    @Inject(MAT_DIALOG_DATA) public data
  ) { }

  ngOnInit(): void {
    this.getPrecisions();
    this.getDatasets();
    this.setFormValues();
    this.modelService.datasetCreated$.subscribe(response => this.getDatasets());
  }

  ngAfterViewInit(): void {
    this.shortcuts.push(
      {
        key: 'ctrl + right',
        preventDefault: true,
        command: e => {
          document.getElementsByName('next')[0].click();
        }
      },
    );
  }

  setFormValues() {
    this.optimizationFormGroup = new FormGroup({
      name: new FormControl('Optimization' + String(this.data.index + 1), Validators.required),
      precisionId: new FormControl('', Validators.required),
      datasetId: new FormControl('', Validators.required),
      optimizationTypeId: new FormControl('', Validators.required)
    });
  }

  getPrecisions() {
    this.modelService.getDictionary('precisions')
      .subscribe(
        (response: { precisions: any }) => {
          this.precisions = response.precisions;
          this.optimizationFormGroup.get('precisionId').setValue(this.precisions.find(x => x.name === 'int8').id);
          this.precisions.forEach((element) => {
            if (element.name === 'int8 dynamic quantization' || element.name === 'int8 static quantization') {
              const insert = Object.assign({}, element);
              this.precisionsPyTorch.push(insert);
            } else {
              const tmp = Object.assign({}, element);
              this.precisionsOther.push(tmp);
            }
          });
          if (this.data.framework === 'pytorch') {
            this.precisions = this.precisionsPyTorch;
          } else {
            this.precisions = this.precisionsOther;
          }
          this.getOptimizationTypes();
        },
        error => {
          this.modelService.openErrorDialog(error);
        });
  }

  getOptimizationTypes() {
    this.supportedTypes = [];
    this.modelService.getDictionaryWithParam('optimization_types', 'precision',
      { precision: this.precisions.find(x => x.id === this.optimizationFormGroup.get('precisionId').value).name })
      .subscribe(
        (response: { optimization_types: any }) => {
          this.optimizationTypes = response.optimization_types;
          this.supportedTypes = this.optimizationTypes.filter(x => x.is_supported === true);
          if (this.optimizationFormGroup.get('precisionId').value === 2) {
            this.optimizationFormGroup.get('optimizationTypeId').setValue(
              this.optimizationTypes.find(x => x.name === 'Mixed precision').id
            );
          } else {
            this.optimizationFormGroup.get('optimizationTypeId').setValue(
              this.supportedTypes[0].id
            );
          }
        },
        error => {
          this.modelService.openErrorDialog(error);
        });
  }

  getDatasets() {
    this.modelService.getDatasetList(this.data.projectId)
      .subscribe(
        (response: { datasets: any }) => {
          this.datasets = response.datasets;
          if (this.datasets.length > 0) {
            this.optimizationFormGroup.get('datasetId').setValue(this.datasets[0].id);
          }
        },
        error => {
          this.modelService.openErrorDialog(error);
        });
  }

  openDatasetDialog() {
    this.modelService.openDatasetDialog$.next(true);
  }

  addOptimization() {
    if (!this.data.editing) {
      this.modelService.addOptimization({
        project_id: this.data.projectId,
        name: this.optimizationFormGroup.get('name').value,
        precision_id: this.optimizationFormGroup.get('precisionId').value,
        optimization_type_id: this.optimizationFormGroup.get('optimizationTypeId').value,
        dataset_id: this.optimizationFormGroup.get('datasetId').value
      })
        .subscribe(
          response => { this.modelService.optimizationCreated$.next(true); },
          error => {
            this.modelService.openErrorDialog(error);
          });
    } else {
      this.modelService.editOptimization({
        id: this.data.optimizationId,
        precision_id: this.optimizationFormGroup.get('precisionId').value,
        optimization_type_id: this.optimizationFormGroup.get('optimizationTypeId').value,
        dataset_id: this.optimizationFormGroup.get('datasetId').value
      })
        .subscribe(
          response => { this.modelService.optimizationCreated$.next(true); },
          error => {
            this.modelService.openErrorDialog(error);
          });
    }
  }

}
