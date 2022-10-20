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
import { ActivatedRoute, Router } from '@angular/router';
import { ModelService } from '../services/model.service';
import { JoyrideService } from 'ngx-joyride';

@Component({
  selector: 'app-project',
  templateUrl: './project.component.html',
  styleUrls: ['./project.component.scss', './../error/error.component.scss']
})
export class ProjectComponent implements OnInit {

  project = {
    notes: '',
    name: ''
  };
  themeCatalog = '';
  is_pytorch = false;
  projectId;
  selectedTab = 0;
  tabs = ['optimizations', 'benchmarks', 'profiling', 'datasets', 'diagnosis', 'info'];

  constructor(
    public modelService: ModelService,
    public activatedRoute: ActivatedRoute,
    private router: Router,
    private readonly joyrideService: JoyrideService) {
  }

  ngOnInit() {
    this.setColorTheme();
    this.projectId = this.activatedRoute.snapshot.params.id;
    this.getProject(this.projectId);
    this.modelService.projectChanged$
      .subscribe((response: { id: number; tab: string }) => {
        this.getProject(response.id, response.tab);
      });
  }

  setColorTheme() {
    if (localStorage.getItem('darkMode') === 'darkMode') {
      this.themeCatalog = 'dark/';
    }
    this.modelService.colorMode$
      .subscribe(resp => {
        this.themeCatalog = resp;
      });
  }

  getProject(id: number, tab?: string) {
    this.selectedTab = this.tabs.indexOf(tab ?? this.activatedRoute.snapshot.params.tab);
    this.modelService.getProjectDetails(id)
      .subscribe(
        (response: { input_model: any; notes: string; name: string }) => {
          this.project = response;
          this.is_pytorch = response.input_model.framework.name === 'PyTorch';
        },
        error => {
          if (error.error === 'list index out of range') {
            this.router.navigate(['home'], { queryParamsHandling: 'merge' });
          } else {
            this.modelService.openErrorDialog(error);
          }
        }
      );
  }

  onTabChanged(event) {
    this.selectedTab = event.index;
    this.router.navigate(
      ['project', this.activatedRoute.snapshot.params.id, this.tabs[this.selectedTab]],
      { queryParamsHandling: 'merge' }
    );
  }

  onClick() {
    this.joyrideService.startTour(
      {
        steps: ['intro', 'addOptimizationTour', 'datasetTour', 'benchmarkTour', 'profilingTour', 'graphTour'],
        themeColor: '#005B85',
      }
    );
  }

  addNotes() {
    this.modelService.addNotes(this.activatedRoute.snapshot.params.id, this.project.notes)
      .subscribe(
        response => { },
        error => {
          this.modelService.openErrorDialog(error);
        }
      );
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

