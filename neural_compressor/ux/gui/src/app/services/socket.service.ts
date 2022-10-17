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
import { io } from 'socket.io-client';
import { environment } from 'src/environments/environment';
import { NewModel } from './model.service';

@Injectable({
  providedIn: 'root'
})
export class SocketService {

  baseUrl = environment.baseUrl;
  webSocketNamespace = environment.webSocketNamespace;
  socket;
  public optimizationStart$ = new Subject();
  public optimizationFinish$ = new Subject();
  public boundaryNodesStart$ = new Subject();
  public boundaryNodesFinish$ = new Subject();
  public modelDownloadFinish$ = new Subject();
  public modelDownloadProgress$ = new Subject();
  public benchmarkStart$ = new Subject();
  public benchmarkFinish$ = new Subject();
  public profilingStart$ = new Subject();
  public profilingFinish$ = new Subject();
  public tuningHistory$ = new Subject();
  public exampleStart$ = new Subject();
  public exampleProgress$ = new Subject();
  public exampleFinish$ = new Subject();
  public showSnackBar$ = new Subject<{ tab: string; id: number }>();

  constructor(
    private http: HttpClient
  ) {
    this.socket = io(this.webSocketNamespace);
    this.setupOptimizationConnection();
    this.setupBoundaryNodesConnection();
    this.setupModelDownload();
    this.setupBenchmark();
    this.setupTuningHistory();
    this.setupProfiling();
    this.setupExample();
  }

  setupOptimizationConnection() {
    this.socket.on('optimization_start', (data) => {
      this.optimizationStart$.next(data);
    });
    this.socket.on('optimization_finish', (data) => {
      this.optimizationFinish$.next(data);
      this.showSnackBar$.next({ tab: 'optimizations', id: data.data.project_id });
    });
  }

  setupBoundaryNodesConnection() {
    this.socket.on('boundary_nodes_start', (data) => {
      this.boundaryNodesStart$.next(data);
    });
    this.socket.on('boundary_nodes_finish', (data) => {
      this.boundaryNodesFinish$.next(data);
    });
  }

  setupModelDownload() {
    this.socket.on('download_finish', (data) => {
      this.modelDownloadFinish$.next(data);
    });
    this.socket.on('download_progress', (data) => {
      this.modelDownloadProgress$.next(data);
    });
  }

  setupBenchmark() {
    this.socket.on('benchmark_start', (data) => {
      this.benchmarkStart$.next(data);
    });
    this.socket.on('benchmark_finish', (data) => {
      this.benchmarkFinish$.next(data);
      this.showSnackBar$.next({ tab: 'benchmarks', id: data.data.project_id });
    });
  }

  setupProfiling() {
    this.socket.on('profiling_start', (data) => {
      this.profilingStart$.next(data);
    });
    this.socket.on('profiling_finish', (data) => {
      this.profilingFinish$.next(data);
      this.showSnackBar$.next({ tab: 'profiling', id: data.data.project_id });
    });
  }

  setupTuningHistory() {
    this.socket.on('tuning_history', (data) => {
      this.tuningHistory$.next(data);
    });
  }

  setupExample() {
    this.socket.on('create_example_project_start', (data) => {
      this.exampleStart$.next(data);
    });
    this.socket.on('create_example_project_progress', (data) => {
      this.exampleProgress$.next(data);
    });
    this.socket.on('create_example_project_finish', (data) => {
      this.exampleFinish$.next(data);
    });
  }

  getBoundaryNodes(newModel: NewModel) {
    return this.http.post(
      this.baseUrl + 'api/model/boundary_nodes',
      {
        request_id: newModel.id,
        model_path: newModel.model_path
      });
  }

  getTuningHistory(id: string) {
    return this.http.get(this.baseUrl + 'api/workload/tuning_history', {
      params: {
        workload_id: id,
      }
    });
  }
}
