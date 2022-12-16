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
import { Pipe, PipeTransform } from '@angular/core';

@Pipe({ name: 'capitalLetter' })
export class CapitalLetterPipe implements PipeTransform {
  transform(value: string): string {
    if (typeof value === 'string') {
      value = value[0].toUpperCase() + value.substr(1).toLowerCase();
    }
    return value;
  }
}
