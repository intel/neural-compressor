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
import { MatDialog } from '@angular/material/dialog';
import { ActivatedRoute } from '@angular/router';
import { DatasetFormComponent } from '../dataset-form/dataset-form.component';
import { ModelService } from '../services/model.service';

@Component({
  selector: 'app-datasets',
  templateUrl: './datasets.component.html',
  styleUrls: ['./datasets.component.scss', './../error/error.component.scss', './../home/home.component.scss']
})
export class DatasetsComponent implements OnInit {
  @Input() framework;
  datasets = [];
  activeDatasetId = 0;
  datasetDetails = {};

  constructor(
    private modelService: ModelService,
    public activatedRoute: ActivatedRoute,
    public dialog: MatDialog
  ) { }

  ngOnInit(): void {
    this.getDatasetList();
    this.modelService.datasetCreated$.subscribe(response => this.getDatasetList());
    this.modelService.projectChanged$.subscribe(response => {
      this.getDatasetList();
      this.activeDatasetId = -1;
      this.datasetDetails = {};
    });
  }

  getDatasetList() {
    this.modelService.getDatasetList(this.activatedRoute.snapshot.params.id)
      .subscribe(response => this.datasets = response['datasets']);
  }

  addDataset() {
    const dialogRef = this.dialog.open(DatasetFormComponent, {
      width: '60%',
      data: {
        projectId: this.activatedRoute.snapshot.params.id,
        index: this.datasets.length,
        framework: this.framework
      }
    });
  }

  getDatasetDetails(id) {
    this.activeDatasetId = id;
    this.modelService.getDatasetDetails(id)
      .subscribe(response => this.datasetDetails = response);
  }

  objectKeys(obj: any): string[] {
    return Object.keys(obj);
  }

}
