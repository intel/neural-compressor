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
  public tuningStart$ = new BehaviorSubject({});
  public tuningFinish$ = new BehaviorSubject({});
  public boundaryNodesStart$ = new BehaviorSubject({});
  public boundaryNodesFinish$ = new BehaviorSubject({});
  public modelDownloadFinish$ = new BehaviorSubject({});
  public benchmarkFinish$ = new BehaviorSubject({});

  constructor(
    private http: HttpClient
  ) {
    this.socket = io(this.baseUrl);
    this.setupTuningConnection();
    this.setupBoundaryNodesConnection();
    this.setupModelDownload();
    this.setupBenchmark();
  }

  setupTuningConnection() {
    this.socket.on('tuning_start', (data) => {
      this.tuningStart$.next(data);
    });
    this.socket.on('tuning_finish', (data) => {
      this.tuningFinish$.next(data);
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
  }

  setupBenchmark() {
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
