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
import { Component, Inject, OnInit } from '@angular/core';
import { MAT_DIALOG_DATA, MatDialogRef } from '@angular/material/dialog';

@Component({
  selector: 'app-generate-config-dialog',
  templateUrl: './generate-config-dialog.component.html',
  styleUrls: ['./generate-config-dialog.component.scss', './../error/error.component.scss', './../datasets/datasets.component.scss']
})
export class GenerateConfigDialogComponent implements OnInit {

  optimizationName = 'Generated optimization';

  constructor(
    @Inject(MAT_DIALOG_DATA) public data,
    public dialogRef: MatDialogRef<GenerateConfigDialogComponent>,
  ) { }

  ngOnInit(): void {
  }

  confirmGenerating() {
    this.dialogRef.close({
      optimizationName: this.optimizationName,
    });
  }

}
