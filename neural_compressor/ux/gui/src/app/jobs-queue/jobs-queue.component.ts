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

import { KeyValue } from '@angular/common';
import { Component, OnInit } from '@angular/core';
import { ModelService } from '../services/model.service';

@Component({
  selector: 'app-jobs-queue',
  templateUrl: './jobs-queue.component.html',
  styleUrls: ['./jobs-queue.component.scss', './../error/error.component.scss']
})
export class JobsQueueComponent implements OnInit {

  jobsQueue = {};

  constructor(
    private modelService: ModelService
  ) { }

  ngOnInit(): void {
    this.modelService.getJobsQueue()
      .subscribe(
        response => this.jobsQueue = response,
        error => this.modelService.openErrorDialog(error)
      );
  }

  valueAscOrder = (a: KeyValue<string, string>, b: KeyValue<string, string>): number => a.value.localeCompare(b.value);

  objectKeys(obj: any): string[] {
    return Object.keys(obj);
  }
}
