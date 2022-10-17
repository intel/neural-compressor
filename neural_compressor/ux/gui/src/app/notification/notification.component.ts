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
import { MAT_SNACK_BAR_DATA } from '@angular/material/snack-bar';
import { ActivatedRoute, Router } from '@angular/router';
import { ModelService } from '../services/model.service';

@Component({
  selector: 'app-notification',
  templateUrl: './notification.component.html',
  styleUrls: ['./notification.component.scss']
})
export class NotificationComponent {

  constructor(
    private router: Router,
    public activatedRoute: ActivatedRoute,
    private modelService: ModelService,
    @Inject(MAT_SNACK_BAR_DATA) public data,
  ) { }

  navigateTo() {
    this.router.navigate(['project', this.data.projectId, this.data.tab], { queryParamsHandling: 'merge' });
    this.modelService.projectChanged$.next({ id: this.data.projectId, tab: this.data.tab });
  }

}
