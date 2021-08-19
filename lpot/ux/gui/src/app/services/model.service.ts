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
import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { Subject } from 'rxjs';
import { environment } from 'src/environments/environment';
import { FullModel } from '../import-model/import-model.component';

@Injectable({
  providedIn: 'root'
})
export class ModelService {

  baseUrl = environment.baseUrl;
  myModels = [];
  workspacePath: string;
  workspacePathChange: Subject<boolean> = new Subject<boolean>();
  systemInfoChange: Subject<boolean> = new Subject<boolean>();
  configurationSaved: Subject<boolean> = new Subject<boolean>();

  token;
  systemInfo = {};

  constructor(
    private http: HttpClient
  ) { }

  getToken(): string {
    return this.token;
  }

  setToken(token: string) {
    this.token = token;
  }

  setWorkspacePath(path: string) {
    this.workspacePath = path;
    return this.http.post(
      this.baseUrl + 'api/set_workspace',
      { path: path }
    );
  }

  getSystemInfo() {
    this.http.get(
      this.baseUrl + 'api/system_info'
    ).subscribe(resp => {
      this.systemInfo = resp;
      this.systemInfoChange.next(true);
    }
    );
  }

  getDefaultPath(name: string) {
    return this.http.post(
      this.baseUrl + 'api/get_default_path',
      { name: name }
    );
  }

  getModelGraph(path: string, groups?: string[]) {
    let groupsParam = '';
    if (groups) {
      groups.forEach(group => {
        groupsParam += '&group=' + group;
      });
    }
    return this.http.get(
      this.baseUrl + 'api/model_graph' + '?path=' + path + groupsParam
    );
  }

  getPossibleValues(param: string, config?: {}) {
    return this.http.post(
      this.baseUrl + 'api/get_possible_values',
      {
        param: param,
        config: config
      }
    );
  }

  getAllModels() {
    return this.http.post(
      this.baseUrl + 'api/get_workloads_list',
      { workspace_path: this.workspacePath }
    );
  }

  getConfiguration(newModel: NewModel) {
    return this.http.post(
      this.baseUrl + 'api/configuration',
      {
        id: newModel.id,
        model_path: newModel.model_path,
        domain: newModel.domain
      });
  }

  optimize(newModel: NewModel) {
    return this.http.post(
      this.baseUrl + 'api/optimize',
      {
        workspace_path: this.workspacePath,
        id: newModel.id
      }
    );
  }

  saveWorkload(fullModel: FullModel | {}) {
    fullModel['workspace_path'] = this.workspacePath;
    return this.http.post(
      this.baseUrl + 'api/save_workload',
      fullModel
    );
  }

  saveExampleWorkload(fullModel: FullModel | {}) {
    fullModel['workspace_path'] = this.workspacePath;
    return this.http.post(
      this.baseUrl + 'api/save_example_workload',
      fullModel
    );
  }

  getFileSystem(path: string, filter: FileBrowserFilter) {
    if (filter === 'all') {
      return this.http.get(this.baseUrl + 'api/filesystem', {
        params: {
          path: path + '/'
        }
      });
    }
    return this.http.get(this.baseUrl + 'api/filesystem', {
      params: {
        path: path + '/',
        filter: filter
      }
    });
  }

  listModelZoo() {
    return this.http.get(this.baseUrl + 'api/list_model_zoo');
  }

}

export interface NewModel {
  domain: string;
  framework: string;
  id: string;
  input?: string;
  model_path: string;
  output?: string;
}

export type FileBrowserFilter = 'models' | 'datasets' | 'directories' | 'all';
