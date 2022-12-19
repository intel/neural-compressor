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
import { AfterViewInit, Component, Input, OnInit } from '@angular/core';
import { MatDialog } from '@angular/material/dialog';
import { ActivatedRoute } from '@angular/router';
import { ShortcutInput } from 'ng-keyboard-shortcuts';
import { ConfirmationDialogComponent } from '../confirmation-dialog/confirmation-dialog.component';
import { DatasetFormComponent } from '../dataset-form/dataset-form.component';
import { ModelService } from '../services/model.service';

@Component({
  selector: 'app-datasets',
  templateUrl: './datasets.component.html',
  styleUrls: ['./datasets.component.scss', './../error/error.component.scss', './../home/home.component.scss']
})
export class DatasetsComponent implements OnInit, AfterViewInit {
  @Input() framework;
  @Input() domain;
  @Input() domainFlavour;

  shortcuts: ShortcutInput[] = [];

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
    this.modelService.projectChanged$
      .subscribe((response: { id: number }) => {
        this.getDatasetList(response.id);
        this.activeDatasetId = -1;
        this.datasetDetails = {};
      });
  }

  ngAfterViewInit(): void {
    this.shortcuts.push(
      {
        key: 'shift + alt + d',
        preventDefault: true,
        command: e => this.addDataset()
      },
    );
  }

  getDatasetList(id?: number) {
    this.modelService.getDatasetList(id ?? this.activatedRoute.snapshot.params.id)
      .subscribe(
        (response: { datasets: any }) => { this.datasets = response.datasets; },
        error => {
          this.modelService.openErrorDialog(error);
        });
  }

  addDataset() {
    const dialogRef = this.dialog.open(DatasetFormComponent, {
      maxWidth: '90vw',
      maxHeight: '90vh',
      data: {
        projectId: this.activatedRoute.snapshot.params.id,
        index: this.datasets.length,
        framework: this.framework,
        domain: this.domain,
        domainFlavour: this.domainFlavour
      }
    });
  }

  getDatasetDetails(id) {
    this.activeDatasetId = id;
    this.modelService.getDatasetDetails(id)
      .subscribe(
        response => { this.datasetDetails = response; },
        error => {
          this.modelService.openErrorDialog(error);
        });
  }

  deleteDataset(id: number, name: string) {
    const dialogRef = this.dialog.open(ConfirmationDialogComponent, {
      data: {
        what: 'dataset',
        id,
        name
      }
    });

    dialogRef.afterClosed().subscribe(
      response => {
        if (response.confirm) {
          this.modelService.delete('dataset', id, name)
            .subscribe(
              deleted =>
                this.modelService.projectChanged$.next({ id: this.activatedRoute.snapshot.params.id, tab: 'datasets' }),
              error =>
                this.modelService.openErrorDialog(error)
            );
        }
      });
  }

  typeOf(object) {
    return typeof object;
  }

  objectKeys(obj: any): string[] {
    return Object.keys(obj);
  }

  copyToClipboard(text: string) {
    const textArea = document.createElement('textarea');
    textArea.value = text;
    document.body.appendChild(textArea);
    textArea.select();

    try {
      document.execCommand('copy');
    } catch (err) {
      console.error('Unable to copy', err);
    }

    document.body.removeChild(textArea);
  }

}
