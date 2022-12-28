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
import { AfterViewInit, Component, OnInit } from '@angular/core';
import { FormControl, FormGroup, Validators } from '@angular/forms';
import { MatDialog } from '@angular/material/dialog';
import { Router } from '@angular/router';
import { debounceTime } from 'rxjs/operators';
import { FileBrowserComponent } from '../file-browser/file-browser.component';
import { GraphComponent } from '../graph/graph.component';
import { FileBrowserFilter, ModelService, NewModel } from '../services/model.service';
import { SocketService } from '../services/socket.service';
import { CdkDragDrop, moveItemInArray, transferArrayItem } from '@angular/cdk/drag-drop';
import { ShortcutInput } from 'ng-keyboard-shortcuts';
declare let require: any;
const shajs = require('sha.js');

@Component({
  selector: 'app-project-form',
  templateUrl: './project-form.component.html',
  styleUrls: ['./project-form.component.scss', './../error/error.component.scss',]
})
export class ProjectFormComponent implements OnInit, AfterViewInit {

  shortcuts: ShortcutInput[] = [];
  showGraphSpinner = false;
  showSpinner = false;
  showGraphButton = false;
  showDomain = false;
  showExamples = false;
  disableFinish = false;
  showShapeWarning: boolean;
  showPanelHeader = true;

  projectFormGroup: FormGroup;
  domains = [];
  graph = {};
  id: string;

  modelList = [];
  frameworks = [];
  models = [];

  model = {
    id: '',
    framework: '',
    model: '',
    domain: '',
    model_path: '',
    yaml: '',
    project_name: '',
    dataset_path: '',
  };

  boundaryNodes: {
    inputs: 'none' | 'custom' | 'select';
    outputs: 'none' | 'custom' | 'select';
  };

  inputs = [];
  outputs = [];
  order = {
    inputs: [],
    outputs: []
  };
  customBoundaryNodes = {
    inputs: false,
    outputs: false
  };

  constructor(
    private dialog: MatDialog,
    public modelService: ModelService,
    public socketService: SocketService,
    private router: Router
  ) { }

  ngOnInit(): void {
    const dateTime = Date.now();
    this.id = shajs('sha384').update(String(dateTime)).digest('hex');

    this.boundaryNodes = {
      inputs: 'none',
      outputs: 'none'
    };

    this.getDomains();
    this.setFormValues();

    this.projectFormGroup.get('modelLocation').valueChanges
      .pipe(
        debounceTime(1000))
      .subscribe(response => {
        if (this.projectFormGroup.get('modelLocation').value) {
          this.showSpinner = true;
          this.showGraphButton = false;
          this.order = {
            inputs: [],
            outputs: []
          };
          this.socketService.getBoundaryNodes(this.getNewModel()).subscribe(
            boundaryNodes => { },
            error => {
              this.modelService.openErrorDialog(error);
            }
          );
          this.modelService.getModelGraph(this.projectFormGroup.get('modelLocation').value)
            .subscribe(
              graph => {
                this.graph = graph;
                this.showGraphButton = true;
              },
              error => {
                this.showGraphButton = false;
              }
            );
        }
      });

    this.socketService.boundaryNodesFinish$
      .subscribe((result: { status: string; id: number; data: any }) => {
        this.showSpinner = false;
        if (result.status === 'success') {
          if (result.data && this.projectFormGroup.get('modelLocation').value && result.data.request_id === this.id) {
            this.projectFormGroup.get('domainFlavour').setValue(result.data.domain_flavour);
            if (result.data.domain?.length) {
              this.projectFormGroup.get('modelDomain').setValue(result.data.domain);
              this.showDomain = false;
            } else {
              this.projectFormGroup.get('modelDomain').reset();
              this.showDomain = true;
            }
            this.projectFormGroup.get('shape').setValue(result.data.shape);
            this.showShapeWarning = result.data.shape ? true : false;
            this.projectFormGroup.get('framework').setValue(result.data.framework);
            ['inputs', 'outputs'].forEach(param => {
              this[param] = result.data[param];
              if (Array.isArray(result.data[param])) {
                if (this[param].indexOf('custom') !== -1) {
                  this[param].splice(this[param].indexOf('custom'), 1);
                }
                this.isFieldRequired(param, true);
                if (result.data[param].length === 0) {
                  this.boundaryNodes[param] = 'custom';
                  this.customBoundaryNodes[param] = true;
                } else if (result.data[param].length === 1) {
                  this.boundaryNodes[param] = 'custom';
                  this.customBoundaryNodes[param] = true;
                  this.projectFormGroup.get(param).setValue(result.data[param]);
                } else {
                  this.boundaryNodes[param] = 'select';
                  this.isFieldRequired(param, false);
                  this.disableFinish = true;
                  if (result.data.domain === 'object_detection' && result.data.domain_flavour === 'ssd') {
                    if (['detection_bboxes', 'detection_scores', 'detection_classes'].every((val) => result.data.outputs.includes(val))) {
                      this.setDefaultBoundaryNode('outputs', ['detection_bboxes', 'detection_scores', 'detection_classes']);
                    }
                  } else {
                    const nonCustomParams = result.data[param].filter(x => x !== 'custom');
                    if (nonCustomParams.length === 1) {
                      this.projectFormGroup.get(param).setValue(nonCustomParams);
                    } else if (nonCustomParams.includes('softmax_tensor')) {
                      this.setDefaultBoundaryNode('outputs', ['softmax_tensor']);
                    }
                  }
                }
              } else {
                this.boundaryNodes[param] = 'none';
                this.isFieldRequired(param, false);
              }
            });
          }
        } else {
          this.modelService.openErrorDialog(result.data.message);
        }
      });
  }

  ngAfterViewInit(): void {
    this.shortcuts.push(
      {
        key: 'ctrl + right',
        preventDefault: true,
        command: e => {
          document.getElementsByName('next')[0].click();
        }
      },
    );
  }

  setDefaultBoundaryNode(nodeType: 'inputs' | 'outputs', values: string[]) {
    this.projectFormGroup.get(nodeType).setValue(values);
    this.order[nodeType] = values;
    this[nodeType] = this[nodeType].filter(x => !values.includes(x));
    this.disableFinish = false;
  }

  getNewModel(): NewModel {
    const model = {
      domain: this.projectFormGroup.get('modelDomain').value,
      domain_flavour: this.projectFormGroup.get('domainFlavour').value,
      framework: this.projectFormGroup.get('framework').value,
      id: this.id,
      model_path: this.projectFormGroup.get('modelLocation').value,
    };
    return model;
  }

  getDomains() {
    this.modelService.getDictionary('domains')
      .subscribe(
        (resp: { domains: any }) => this.domains = resp.domains,
        error => this.modelService.openErrorDialog(error));
  }

  getExamples(event?) {
    if (!event || event.selectedIndex === 2) {
      this.showExamples = this.projectFormGroup.get('predefined').value;
    }
  }

  setFormValues() {
    this.projectFormGroup = new FormGroup({
      name: new FormControl('Project' + String(this.modelService.projectCount + 1), Validators.required),
      framework: new FormControl('', Validators.required),
      modelLocation: new FormControl('', Validators.required),
      modelDomain: new FormControl(''),
      domainFlavour: new FormControl(''),
      inputs: new FormControl(''),
      outputs: new FormControl(''),
      shape: new FormControl(''),
      predefined: new FormControl(true),
    });
  }

  createProject() {
    this.showSpinner = true;
    const newProject = {
      name: this.projectFormGroup.get('name').value,
      model: {
        path: this.projectFormGroup.get('modelLocation').value,
        domain: this.projectFormGroup.get('modelDomain').value,
        input_nodes: this.getBoundaryNodes('inputs') === '' ? null : this.getBoundaryNodes('inputs'),
        output_nodes: this.getBoundaryNodes('outputs') === '' ? null : this.getBoundaryNodes('outputs'),
        shape: this.projectFormGroup.get('shape').value,
      }
    };

    this.modelService.createProject(newProject)
      .subscribe(
        (response: { project_id: number }) => {
          this.showSpinner = false;
          this.modelService.projectCreated$.next(true);
          this.router.navigate(['/project', response.project_id], { queryParamsHandling: 'merge' });
        },
        error => {
          this.showSpinner = false;
          this.modelService.openErrorDialog(error);
        });
  }

  drop(event: CdkDragDrop<string[]>) {
    if (event.previousContainer === event.container) {
      moveItemInArray(event.container.data, event.previousIndex, event.currentIndex);
    } else {
      transferArrayItem(
        event.previousContainer.data,
        event.container.data,
        event.previousIndex,
        event.currentIndex,
      );
    }
    if ((this.boundaryNodes.inputs === 'select' && !this.order.inputs.length)
      || (this.boundaryNodes.outputs === 'select' && !this.order.outputs.length)) {
      this.disableFinish = true;
    } else {
      this.disableFinish = false;
    }
  }

  showGraph() {
    this.showGraphSpinner = true;
    this.showGraphSpinner = false;
    this.dialog.open(GraphComponent, {
      width: '90%',
      height: '90%',
      data: {
        modelPath: this.projectFormGroup.get('modelLocation').value,
        viewSize: [window.innerWidth * 0.9, window.innerHeight * 0.9]
      }
    });
  }

  openDialog(fieldName: string, filter: FileBrowserFilter, paramFile?) {
    const dialogRef = this.dialog.open(FileBrowserComponent, {
      width: '60%',
      height: '60%',
      data: {
        path: this.projectFormGroup.get(fieldName) && this.projectFormGroup.get(fieldName).value
          ? this.projectFormGroup.get(fieldName).value.split('/').slice(0, -1).join('/')
          : this.modelService.workspacePath,
        filter,
        filesToFind: paramFile
      }
    });

    dialogRef.afterClosed().subscribe(response => {
      if (response && response.chosenFile) {
        this.projectFormGroup.get(fieldName).setValue(response.chosenFile);
      }
    });
  }

  changeCustomBoundaryNodes(nodeType: 'inputs' | 'outputs') {
    this.customBoundaryNodes[nodeType] = !this.customBoundaryNodes[nodeType];
    if (this.customBoundaryNodes[nodeType]) {
      this.boundaryNodes[nodeType] = 'custom';
      this.isFieldRequired(nodeType, true);
      this.disableFinish = false;
    } else if (this[nodeType].length) {
      this.boundaryNodes[nodeType] = 'select';
      this.isFieldRequired(nodeType, false);
      this.disableFinish = true;
    } else {
      this.boundaryNodes[nodeType] = 'none';
      this.isFieldRequired(nodeType, false);
      this.disableFinish = false;
    }
  }

  boundaryNodesVisible(): boolean {
    return (this.boundaryNodes.inputs !== 'none' || this.boundaryNodes.outputs !== 'none')
      && this.projectFormGroup.get('modelLocation').value && !this.showSpinner;
  }

  boundaryNodesChanged(value: string, type: 'inputs' | 'outputs') {
    if (value === 'custom') {
      if (!this.order[type].includes(value)) {
        this.projectFormGroup.get(type).setValue([value]);
        this.order[type] = [value];
      } else {
        this.projectFormGroup.get(type).setValue([]);
        this.order[type] = [];
      }
    } else {
      if (!this.order[type].includes(value)) {
        this.order[type].push(value);
      } else {
        this.order[type].splice(this.order[type].indexOf(value), 1);
      }
    }
  }

  getBoundaryNodes(type: 'inputs' | 'outputs'): any {
    if (this.order[type].length) {
      return this.order[type];
    }
    if (typeof this.projectFormGroup.get(type).value === 'string') {
      if (typeof this.projectFormGroup.get(type).value.includes(',')) {
        return this.projectFormGroup.get(type).value;
      }
      return [this.projectFormGroup.get(type).value];
    }
    return this.projectFormGroup.get(type).value;
  }

  isFieldRequired(field: string, required: boolean) {
    if (required) {
      this.projectFormGroup.controls[field].setValidators([Validators.required]);
    } else {
      this.projectFormGroup.controls[field].clearValidators();
    }
    this.projectFormGroup.controls[field].updateValueAndValidity();
  }

  togglePanel(show: boolean) {
    this.showPanelHeader = show;
  }
}
