import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { Subject } from 'rxjs';
import { environment } from 'src/environments/environment';

@Injectable({
  providedIn: 'root'
})
export class ModelService {

  baseUrl = environment.baseUrl;
  myModels = [];
  workspacePath: string;
  workspacePathChange: Subject<boolean> = new Subject<boolean>();

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
      this.baseUrl + 'api/get_possible_values_v2',
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

  tune(newModel: NewModel) {
    return this.http.post(
      this.baseUrl + 'api/tune',
      {
        workspace_path: this.workspacePath,
        id: newModel.id
      }
    );
  }

  saveWorkload(fullModel) {
    fullModel['workspace_path'] = this.workspacePath;
    return this.http.post(
      this.baseUrl + 'api/save_workload',
      fullModel
    );
  }

  getFile(path: string) {
    return this.http.get(this.baseUrl + 'file' + path, { responseType: 'text' as 'json' });
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

  listModelZoo() {
    return this.http.get(this.baseUrl + 'api/list_model_zoo');
  }

  benchmark(id: string, modelPath: string, outputModelPath: string) {
    return this.http.post(
      this.baseUrl + 'api/benchmark',
      {
        id: id,
        workspace_path: this.workspacePath,
        models: [
          {
            precision: 'fp32',
            path: modelPath
          },
          {
            precision: 'int8',
            path: outputModelPath
          }
        ]

      });
  }

  downloadModel(model, index: number) {
    return this.http.post(
      this.baseUrl + 'api/download_model',
      {
        id: index,
        workspace_path: this.workspacePath,
        framework: model.framework,
        domain: model.domain,
        model: model.model,
        progress_steps: 20,
      }
    );
  }

  downloadConfig(model, index: number) {
    return this.http.post(
      this.baseUrl + 'api/download_config',
      {
        id: index,
        workspace_path: this.workspacePath,
        framework: model.framework,
        domain: model.domain,
        model: model.model,
        progress_steps: 20,
      }
    );
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
