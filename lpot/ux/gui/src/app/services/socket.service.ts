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
import { BehaviorSubject } from 'rxjs';
import { io } from 'socket.io-client';
import { environment } from 'src/environments/environment';
import { NewModel } from './model.service';

@Injectable({
  providedIn: 'root'
})
export class SocketService {

  baseUrl = environment.baseUrl;
  socket;
  public optimizationStart$ = new BehaviorSubject({});
  public optimizationFinish$ = new BehaviorSubject({});
  public boundaryNodesStart$ = new BehaviorSubject({});
  public boundaryNodesFinish$ = new BehaviorSubject({});
  public modelDownloadFinish$ = new BehaviorSubject({});
  public modelDownloadProgress$ = new BehaviorSubject({});
  public benchmarkStart$ = new BehaviorSubject({});
  public benchmarkFinish$ = new BehaviorSubject({});

  constructor(
    private http: HttpClient
  ) {
    this.socket = io(this.baseUrl);
    this.setupOptimizationConnection();
    this.setupBoundaryNodesConnection();
    this.setupModelDownload();
    this.setupBenchmark();
  }

  setupOptimizationConnection() {
    this.socket.on('optimization_start', (data) => {
      this.optimizationStart$.next(data);
    });
    this.socket.on('optimization_finish', (data) => {
      this.optimizationFinish$.next(data);
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
    });
  }

  getBoundaryNodes(newModel: NewModel) {
    return this.http.post(
      this.baseUrl + 'api/get_boundary_nodes',
      {
        id: newModel.id,
        model_path: newModel.model_path
      });
  }
}
