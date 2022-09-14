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
import { Component, Inject } from '@angular/core';
import { MAT_DIALOG_DATA } from '@angular/material/dialog';
import { Router } from '@angular/router';
import { ModelService } from '../services/model.service';

@Component({
  selector: 'app-project-remove',
  templateUrl: './project-remove.component.html',
  styleUrls: ['./project-remove.component.scss', './../error/error.component.scss']
})
export class ProjectRemoveComponent {
  projectName = '';

  constructor(
    private modelService: ModelService,
    private router: Router,
    @Inject(MAT_DIALOG_DATA) public data
  ) { }

  removeProject() {
    this.modelService.removeProject(this.data.projectId, this.projectName)
      .subscribe(response => {
        this.modelService.projectCreated$.next(true);
        if (this.router.url.includes(this.data.projectId)) {
          this.router.navigate(['home'], { queryParamsHandling: 'merge' });
        }
      },
        error => this.modelService.openErrorDialog(error)
      );
  }

}
