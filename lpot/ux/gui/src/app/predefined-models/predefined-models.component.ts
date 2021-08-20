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
import { Component, OnInit } from '@angular/core';
import { MatDialog } from '@angular/material/dialog';
import { Router } from '@angular/router';
import { ErrorComponent } from '../error/error.component';
import { FileBrowserComponent } from '../file-browser/file-browser.component';
import { FileBrowserFilter, ModelService } from '../services/model.service';
import { SocketService } from '../services/socket.service';
declare var require: any;
var shajs = require('sha.js')

@Component({
  selector: 'app-predefined-models',
  templateUrl: './predefined-models.component.html',
  styleUrls: ['./predefined-models.component.scss', './../error/error.component.scss']
})
export class PredefinedModelsComponent implements OnInit {

  showSpinner = true;
  showProgressBar = false;
  progressBarValue = 0;

  modelList = [];
  frameworks = [];
  domains = [];
  models = [];

  model: PredefinedModel = {
    id: '',
    framework: 'tensorflow',
    model: '',
    domain: '',
    model_path: '',
    yaml: '',
    project_name: '',
    dataset_path: '',
  };

  constructor(
    private modelService: ModelService,
    private socketService: SocketService,
    public dialog: MatDialog,
    private router: Router
  ) { }

  ngOnInit() {
    this.listModelZoo();

    this.modelService.workspacePathChange.subscribe(
      response => {
        this.listModelZoo();
      }
    );

    this.socketService.modelDownloadFinish$
      .subscribe(response => {
        if (response['status']) {
          if (response['status'] === 'success') {
            if (this.model.id) {
              this.router.navigate(['/details', this.model['id']], { queryParamsHandling: "merge" });
              this.modelService.configurationSaved.next(true);
            }
          } else {
            this.openErrorDialog({
              error: response['data']['message'],
            });
          }
        }
      });

    this.socketService.modelDownloadProgress$
      .subscribe(response => {
        if (response['status']) {
          this.progressBarValue = response['data']['progress'];
        }
      });
  }

  systemInfo() {
    return this.modelService.systemInfo;
  }

  listModelZoo() {
    this.modelService.listModelZoo()
      .subscribe(
        (resp: []) => {
          this.showSpinner = false;
          this.modelList = resp;
          this.getValuesForForm();
        },
        error => {
          this.showSpinner = false;
          this.openErrorDialog(error);
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

  configExists(framework: string, domain: string, model?: string) {
    let found;
    if (model) {
      found = this.modelList.filter(x => x['domain'] === domain && x['framework'] === framework && x['model'] === model);
    } else {
      found = this.modelList.filter(x => x['domain'] === domain && x['framework'] === framework);
    }
    return found.length;
  }

  objectKeys(obj): string[] {
    return Object.keys(obj);
  }

  openDialog(filter: FileBrowserFilter, index: number) {
    const dialogRef = this.dialog.open(FileBrowserComponent, {
      width: '60%',
      height: '60%',
      data: {
        path: this.modelService.workspacePath,
        filter: filter
      }
    });

    dialogRef.afterClosed().subscribe(chosenFile => {
      if (chosenFile) {
        this.model.dataset_path = chosenFile;
      }
    });;
  }

  openErrorDialog(error) {
    const dialogRef = this.dialog.open(ErrorComponent, {
      data: error,
    });
  }

  saveWorkload() {
    const dateTime = Date.now();
    const index = this.getModelIndex();
    this.model.id = shajs('sha384').update(String(dateTime)).digest('hex');
    this.model.project_name = this.model['model'];
    this.model['progress_steps'] = 20;
    this.saveReadyWorkload();
  }

  saveReadyWorkload() {
    this.showProgressBar = true;
    this.modelService.saveExampleWorkload(this.model)
      .subscribe(
        response => { },
        error => {
          this.openErrorDialog(error);
        }
      );
  }

  getModelIndex(): number {
    return this.modelList.indexOf(x => x.model === this.model.model);
  }

  getFileNameFromPath(path: string): string {
    return path.replace(/^.*[\\\/]/, '');
  }

}

type PredefinedModel = {
  id: string;
  framework: string;
  model: string;
  domain: string;
  model_path?: string;
  yaml?: string;
  project_name: string;
  dataset_path: string;
};
