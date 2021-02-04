import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { environment } from 'src/environments/environment';
import { FullModel } from '../import-model/import-model.component';

@Injectable({
  providedIn: 'root'
})
export class ModelService {

  baseUrl = environment.baseUrl;
  myModels = [];
  workspacePath: string;

  constructor(
    private http: HttpClient
  ) { }

  setWorkspacePath(path: string) {
    this.workspacePath = path;
    return this.http.post(
      this.baseUrl + 'api/set_workspace',
      { path: path }
    );
  }

  getDefaultPath(name: string) {
    return this.http.post(
      this.baseUrl + 'api/get_default_path',
      { name: name }
    );
  }

  getPossibleValues(param: string, config: {}) {
    return this.http.post(
      this.baseUrl + 'api/get_possible_values',
      {
        "param": param,
        "config": config
      }
    );
  }

  addModel(newModel: NewModel) {
    this.myModels.push(newModel);
    localStorage.setItem('myModels', JSON.stringify(this.myModels));
  }

  getAllModels() {
    if (JSON.parse(localStorage.getItem('myModels'))) {
      this.myModels = JSON.parse(localStorage.getItem('myModels'));
    }
    return this.myModels;
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

  tune(newModel: NewModel) {
    return this.http.post(
      this.baseUrl + 'api/tune',
      {
        workspace_path: this.workspacePath,
        id: newModel.id
      }
    );
  }

  saveWorkload(fullModel: FullModel) {
    fullModel['workspace_path'] = this.workspacePath;
    return this.http.post(
      this.baseUrl + 'api/save_workload',
      fullModel
    );
  }

  getFile(id: string, fileType: string) {
    this.getAllModels();
    const model = this.myModels.find(model => model['id'] === id);
    let url = '';
    if (fileType === 'output') {
      url = 'file' + this.workspacePath + '/' + model['id'] + '.txt';
    } else if (fileType === 'config') {
      url = 'file' + this.workspacePath + '/config.' + model['id'] + '.yaml';
    }
    return this.http.get(this.baseUrl + url, { responseType: 'text' as 'json' });
  }

  getFileSystem(path: string, files: boolean, modelsOnly: boolean) {
    return this.http.get(this.baseUrl + 'api/filesystem', {
      params: {
        path: path,
        files: String(files),
        models_only: String(modelsOnly)
      }
    });
  }
}

export interface NewModel {
  dataset_path: string;
  domain: string;
  framework: string;
  id: string;
  input?: string;
  model_path: string;
  output?: string;
}
