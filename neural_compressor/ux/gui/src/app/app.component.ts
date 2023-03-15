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
import { OverlayContainer } from '@angular/cdk/overlay';
import { AfterViewInit, Component, HostBinding, OnInit } from '@angular/core';
import { FormControl } from '@angular/forms';
import { MatDialog } from '@angular/material/dialog';
import { MatSnackBar } from '@angular/material/snack-bar';
import { ActivatedRoute, Router } from '@angular/router';
import { ShortcutInput } from 'ng-keyboard-shortcuts';
import { HomeComponent } from './home/home.component';
import { JobsQueueComponent } from './jobs-queue/jobs-queue.component';
import { NotificationComponent } from './notification/notification.component';
import { ModelService } from './services/model.service';
import { SocketService } from './services/socket.service';
import { SystemInfoComponent } from './system-info/system-info.component';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.scss']
})
export class AppComponent implements OnInit, AfterViewInit {
  @HostBinding('class') className = '';
  shortcuts: ShortcutInput[] = [];
  tokenIsSet = false;
  workspacePath: string;
  toggleControl = new FormControl(false);

  constructor(
    private overlay: OverlayContainer,
    private modelService: ModelService,
    private socketService: SocketService,
    public dialog: MatDialog,
    private snackBar: MatSnackBar,
    private homeComponent: HomeComponent,
    private router: Router,
    public activatedRoute: ActivatedRoute
  ) { }

  ngOnInit() {
    this.setColorTheme();
    this.modelService.setToken(window.location.search.replace('?token=', ''));
    this.tokenIsSet = true;
    this.getWorkspace();
    this.modelService.getSystemInfo();
    this.socketService.showSnackBar$
      .subscribe(response => {
        this.openSnackBar(response.tab, response.id);
      });
  }

  ngAfterViewInit(): void {
    this.shortcuts.push(
      {
        key: 'shift + alt + c',
        preventDefault: true,
        command: e => this.homeComponent.createNewProject()
      }
    );
  }

  setColorTheme() {
    if (localStorage.getItem('darkMode') === 'darkMode') {
      this.setDarkMode();
      this.toggleControl.setValue(true);
    }
    this.toggleControl.valueChanges.subscribe((darkMode) => {
      if (darkMode) {
        this.setDarkMode();
      } else {
        this.setLightMode();
      }
    });
  }

  setDarkMode() {
    this.modelService.colorMode$.next('dark/');
    this.className = 'darkMode';
    this.overlay.getContainerElement().classList.add('darkMode');
    localStorage.setItem('darkMode', 'darkMode');
  }

  setLightMode() {
    this.modelService.colorMode$.next('');
    this.className = '';
    this.overlay.getContainerElement().classList.remove('darkMode');
    localStorage.setItem('darkMode', '');
  }

  getWorkspace() {
    this.modelService.getDefaultPath('workspace')
      .subscribe(
        (repoPath: { path: string }) => {
          this.workspacePath = repoPath.path;
          this.modelService.workspacePath = repoPath.path;
        },
        error => {
          this.modelService.openErrorDialog(error);
        }
      );
  }

  showJobsQueue() {
    const dialogRef = this.dialog.open(JobsQueueComponent, {
      width: '50vw',
      height: '50vh',
    });
  }

  showSystemInfo() {
    const dialogRef = this.dialog.open(SystemInfoComponent, {
      maxWidth: '90vw',
      maxHeight: '90vh',
      data: {
        'system info': this.modelService.systemInfo.systeminfo,
        'framework info': this.modelService.systemInfo.frameworks
      }
    });
  }

  openSnackBar(tab: string, id: number) {
    this.snackBar.openFromComponent(NotificationComponent, {
      duration: 5 * 1000,
      data: {
        tab,
        projectId: id
      }
    });
  }
}
