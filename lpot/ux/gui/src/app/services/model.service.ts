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

  constructor(
    private http: HttpClient
  ) { }

  getRepoPath() {
    return this.http.get(
      this.baseUrl + 'api/lpot_repository_path'
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
        model_path: newModel.model_path
      });
  }

  tune(newModel: NewModel) {
    return this.http.post(
      this.baseUrl + 'api/tune',
      {
        workspace_path: newModel.workspace_path,
        id: newModel.id
      }
    );
  }

  saveWorkload(fullModel: FullModel) {
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
      url = 'file' + model['workspace_path'] + '/' + model['id'] + '.txt';
    } else if (fileType === 'config') {
      url = 'file' + model['workspace_path'] + '/config.' + model['id'] + '.yaml';
    }
    return this.http.get(this.baseUrl + url, { responseType: 'text' as 'json' });
  }

  getFileSystem(path: string, files: boolean) {
    return this.http.get(this.baseUrl + 'api/filesystem', { params: { path: path, files: String(files) } });
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
  workspace_path: string;
}
