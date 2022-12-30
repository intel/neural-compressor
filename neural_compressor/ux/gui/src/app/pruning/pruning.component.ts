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

import { Component, Input, OnChanges, ViewChild } from '@angular/core';
import { NestedTreeControl } from '@angular/cdk/tree';
import { MatTreeNestedDataSource } from '@angular/material/tree';
import { FormGroup } from '@angular/forms';
import { FieldBase } from '../dynamic-form-helper/field-base';
import { TextboxField } from '../dynamic-form-helper/field-textbox';
import { FieldControlService } from '../dynamic-form-helper/field-control.service';
import { NodeBase } from '../dynamic-form-helper/node-base';
import { ModelService } from '../services/model.service';

@Component({
  selector: 'app-pruning',
  templateUrl: './pruning.component.html',
  styleUrls: ['./pruning.component.scss', './../error/error.component.scss', './../datasets/datasets.component.scss'],
  providers: [FieldControlService]
})
export class PruningComponent implements OnChanges {

  @Input() optimizationId: number;
  @Input() editable: boolean;
  @ViewChild('pruningTree') pruningTree;

  treeControl = new NestedTreeControl<NodeBase>(node => node.children);
  dataSource = new MatTreeNestedDataSource<any>();

  pruningFormGroup!: FormGroup;

  fieldBase = {};
  paramsToSave = {};
  displayDataReady = false;
  editableParams = ['epoch', 'target_sparsity'];
  pruningParamEditable = {
    epoch: false,
    target_sparsity: false
  };

  constructor(
    private fieldControlService: FieldControlService,
    private modelService: ModelService
  ) { }

  hasChild = (_: number, node: any) => !!node.children && node.children.length > 0;

  ngOnChanges(): void {
    this.getDisplayData();
  }

  expandAll() {
    this.pruningTree.treeControl.expandAll();
  }

  getDisplayData() {
    this.displayDataReady = false;
    this.modelService.getPruningDetails(this.optimizationId)
      .subscribe(
        (response: { pruning_details }) => {
          this.dataSource.data = response.pruning_details.config_tree;
          this.treeControl.dataNodes = response.pruning_details.config_tree;
          this.displayDataReady = true;

          const initialValueEpoch = response.pruning_details.config_tree[0].children
            .find(x => x.name === 'epoch').value;
          const initialValueTargetSparsity = response.pruning_details.config_tree[1].children
            .find(x => x.name === 'weight_compression').children
            .find(x => x.name === 'target_sparsity').value;

          this.getFormFields(initialValueEpoch, initialValueTargetSparsity);
        },
        error => this.modelService.openErrorDialog(error));
  }

  editDisplayData(data: { epoch: number; target_sparsity: number }) {
    this.dataSource.data[0].children
      .find(x => x.name === 'epoch').value = data.epoch;
    this.dataSource.data[1].children
      .find(x => x.name === 'weight_compression').children
      .find(x => x.name === 'target_sparsity').value = data.target_sparsity;
  }

  getFormFields(initialValueEpoch: number, initialValueTargetSparsity: number) {
    let field: FieldBase<string | number>;
    field =
      new TextboxField({
        value: initialValueEpoch,
        paramName: 'epoch',
        name: 'Epoch',
        required: true,
        controlType: 'numberInput',
        type: 'int',
      });
    this.pruningFormGroup = this.fieldControlService.addFieldToFormGroup(field as FieldBase<number>);
    this.fieldBase[field.paramName] = field;
    field =
      new TextboxField({
        value: initialValueTargetSparsity,
        paramName: 'target_sparsity',
        name: 'Target sparsity',
        required: true,
        controlType: 'numberInput',
        type: 'float',
      });
    this.pruningFormGroup = this.fieldControlService.addFieldToFormGroup(field as FieldBase<number>);
    this.fieldBase[field.paramName] = field;
  }

  savePruningParam() {
    this.modelService.editOptimization({
      id: this.optimizationId,
      pruning_details: this.pruningFormGroup.value
    })
      .subscribe(
        response => this.editDisplayData(this.pruningFormGroup.value),
        error => this.modelService.openErrorDialog(error));
  }
}

