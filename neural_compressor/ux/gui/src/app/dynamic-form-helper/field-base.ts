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

export class FieldBase<T> {
  value: T | undefined;
  paramName: string;
  name: string;
  required: boolean;
  controlType: string;
  type: string;
  choices: { paramName: string; value: string }[];
  range: { min: number; max: number };
  fieldPath: string;

  constructor(options: {
    value?: T;
    paramName?: string;
    name?: string;
    required?: boolean;
    controlType?: string;
    type?: string;
    choices?: { paramName: string; value: string }[];
    range?: { min: number; max: number };
    fieldPath?: string;
  } = {}) {
    this.value = options.value;
    this.paramName = options.paramName || '';
    this.name = options.name || '';
    this.required = !!options.required;
    this.controlType = options.controlType || '';
    this.type = options.type || '';
    this.choices = options.choices || [];
    this.range = options.range || { min: null, max: null };
    this.fieldPath = options.fieldPath || options.paramName;
  }
}
