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

import { Component, Input } from '@angular/core';
import { FormGroup } from '@angular/forms';
import { MatDialog } from '@angular/material/dialog';
import { FieldBase } from '../dynamic-form-helper/field-base';
import { FileBrowserComponent } from '../file-browser/file-browser.component';
import { FileBrowserFilter, ModelService } from '../services/model.service';

@Component({
  selector: 'app-field',
  templateUrl: './dynamic-form-field.component.html',
  styleUrls: ['./../error/error.component.scss', './dynamic-form-field.component.scss', './../datasets/datasets.component.scss']
})
export class DynamicFormFieldComponent {
  @Input() field!: FieldBase<string | number>;
  @Input() form!: FormGroup;

  constructor(
    private dialog: MatDialog,
    public modelService: ModelService
  ) { }

  get isValid() { return this.form.controls[this.field.paramName].valid; }

  integerOnly(event: KeyboardEvent, type: string): boolean {
    let result = true;
    if (type === 'Int') {
      const integers = /^([0-9])$/;
      result = integers.test(event.key);
    }
    return result;
  }

  openDialog(fieldName: string, filter: FileBrowserFilter) {
    const dialogRef = this.dialog.open(FileBrowserComponent, {
      width: '60%',
      height: '60%',
      data: {
        path: this.form.get(fieldName) && this.form.get(fieldName).value
          ? this.form.get(fieldName).value.split('/').slice(0, -1).join('/')
          : this.modelService.workspacePath,
        filter
      }
    });

    dialogRef.afterClosed().subscribe(response => {
      if (response && response.chosenFile) {
        this.form.get(fieldName).setValue(response.chosenFile);
      }
    });
  }
}
