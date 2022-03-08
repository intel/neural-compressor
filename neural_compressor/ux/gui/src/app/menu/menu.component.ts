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
import { Component, OnInit } from '@angular/core';
import { MatDialog } from '@angular/material/dialog';
import { ActivatedRoute, Router, RoutesRecognized } from '@angular/router';
import { ErrorComponent } from '../error/error.component';
import { ProjectFormComponent } from '../project-form/project-form.component';
import { ModelService } from '../services/model.service';

@Component({
  selector: 'app-menu',
  templateUrl: './menu.component.html',
  styleUrls: ['./menu.component.scss', './../error/error.component.scss'],
})
export class MenuComponent implements OnInit {

  showSpinner = true;
  projectList = [];
  activeTab = 'optimizations';

  constructor(
    private modelService: ModelService,
    public dialog: MatDialog,
    private activatedRoute: ActivatedRoute,
    private router: Router
  ) { }

  ngOnInit() {
    this.router.events.subscribe(val => {
      if (val instanceof RoutesRecognized) {
        this.activeTab = val.state.root.firstChild.params.tab;
      }
    });
    this.modelService.getSystemInfo();
    this.getAllProjects();
    this.modelService.projectCreated$
      .subscribe(response => this.getAllProjects());
  }

  projectChange(id) {
    this.modelService.projectChanged$.next({ id: id, tab: this.activeTab });
  }

  getAllProjects() {
    this.modelService.getProjectList()
      .subscribe(
        response => {
          this.showSpinner = false;
          this.projectList = response['projects'];
          this.modelService.projectCount = this.projectList.length;
        },
        error => {
          this.showSpinner = false;
          this.modelService.openErrorDialog(error);
        });
  }

  getDate(date: string) {
    return new Date(date);
  }

  createNewProject() {
    const dialogRef = this.dialog.open(ProjectFormComponent, {
      width: '60%',
    });
    dialogRef.afterClosed().subscribe(response => {
      if (response !== undefined) {
        this.showSpinner = true;
      }
    });
  }

}
