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

import { Injectable } from '@angular/core';
import { FormControl, FormGroup, Validators } from '@angular/forms';

import { FieldBase } from './field-base';

@Injectable()
export class FieldControlService {

  group: any = {};
  constructor() { }

  addFieldToFormGroup(field: FieldBase<string | number>) {
    this.group[field.fieldPath] = field.required ? new FormControl(field.value || null, Validators.required)
      : new FormControl(field.value || null);
    return new FormGroup(this.group);
  }
}
