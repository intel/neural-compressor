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
import { Component } from '@angular/core';
import { ActivatedRoute, NavigationEnd, Router } from '@angular/router';
import { ModelService } from '../services/model.service';

@Component({
  selector: 'app-project',
  templateUrl: './project.component.html',
  styleUrls: ['./project.component.scss', './../error/error.component.scss']
})
export class ProjectComponent {

  project = {};

  constructor(
    private modelService: ModelService,
    public activatedRoute: ActivatedRoute,
    private router: Router) {
    router.events.subscribe((val) => {
      if (val instanceof NavigationEnd) {
        this.getProject();
        this.modelService.projectChanged$.next(true);
      }
    });
  }

  getProject() {
    this.modelService.getProjectDetails(this.activatedRoute.snapshot.params.id)
      .subscribe(
        response => {
          this.project = response;
        }
      )
  }

  addNotes() {
    this.modelService.addNotes(this.activatedRoute.snapshot.params.id, this.project['notes'])
      .subscribe();
  }

  getFileName(path: string): string {
    return path.replace(/^.*[\\\/]/, '');
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

