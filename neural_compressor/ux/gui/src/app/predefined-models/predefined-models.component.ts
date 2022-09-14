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
import { Router } from '@angular/router';
import { ModelService } from '../services/model.service';
import { SocketService } from '../services/socket.service';
declare let require: any;
const shajs = require('sha.js');

@Component({
  selector: 'app-predefined-models',
  templateUrl: './predefined-models.component.html',
  styleUrls: ['./predefined-models.component.scss', './../error/error.component.scss']
})
export class PredefinedModelsComponent implements OnInit {

  @Input() name;

  showSpinner = true;
  showProgressBar = false;
  progressBarValue = 0;

  modelList = [];
  frameworks = [];
  domains = [];
  models = [];

  model: PredefinedModel = {
    request_id: '',
    framework: 'TensorFlow',
    model: '',
    domain: '',
    name: '',
    progress_steps: 10
  };

  downloadMessage = 'Downloading model and config. It may take a few minutes...';

  constructor(
    private modelService: ModelService,
    private socketService: SocketService,
    private router: Router
  ) { }

  ngOnInit() {
    this.getExamplesList();

    this.socketService.modelDownloadFinish$
      .subscribe((response: { status: string; data: any }) => {
        if (response.status && response.status !== 'success') {
          this.modelService.openErrorDialog({
            error: response.data.message,
          });
        }
      });

    this.socketService.modelDownloadProgress$
      .subscribe((response: { status: string; data: any }) => {
        if (response.status) {
          this.progressBarValue = response.data.progress;
        }
      });

    this.socketService.exampleFinish$
      .subscribe((response: { status: string; data: any }) => {
        if (response.status) {
          if (response.status === 'success') {
            if (response.data.project_id) {
              this.router.navigate(['project', response.data.project_id, 'optimizations'], { queryParamsHandling: 'merge' });
              this.modelService.projectCreated$.next(true);
              this.modelService.projectChanged$.next({ id: response.data.project_id, tab: 'optimization' });
            }
          } else {
            this.modelService.openErrorDialog({
              error: response.data.message,
            });
          }
        }
      });

    this.socketService.exampleProgress$
      .subscribe((response: { data: any }) => {
        this.downloadMessage = response.data.message;
      });
  }

  getExamplesList() {
    this.modelService.getExamplesList()
      .subscribe(
        (resp: []) => {
          this.showSpinner = false;
          this.modelList = resp;
          this.getValuesForForm();
        },
        error => {
          this.showSpinner = false;
          this.modelService.openErrorDialog(error);
        });
  }

  getValuesForForm() {
    this.modelList.forEach(row => {
      ['framework', 'domain', 'model'].forEach(value => {
        if (!this[value + 's'].includes(row[value])) {
          this[value + 's'].push(row[value]);
          if (value === 'domain') {
            this.model.domain = this.domains[0];
          } else if (value === 'model') {
            this.model.model = this.models[0];
          }
        }
      });
    });
  }

  onModelDomainChange(availableDomain) {
    this.model.domain = availableDomain;
    this.model.model = this.modelList.find(x => x.domain === availableDomain).model;
  }

  configExists(framework: string, domain: string, model?: string) {
    let found;
    if (model) {
      found = this.modelList.filter(x => x.domain === domain && x.framework === framework && x.model === model);
    } else {
      found = this.modelList.filter(x => x.domain === domain && x.framework === framework);
    }
    return found.length;
  }

  addExample() {
    this.showProgressBar = true;
    const dateTime = Date.now();
    this.model.request_id = shajs('sha384').update(String(dateTime)).digest('hex');
    this.model.name = this.name;
    this.modelService.addExample(this.model)
      .subscribe(
        response => { },
        error => {
          this.modelService.openErrorDialog(error);
        }
      );
  }
}

type PredefinedModel = {
  request_id: string;
  framework: string;
  model: string;
  domain: string;
  name: string;
  progress_steps: number;
};
