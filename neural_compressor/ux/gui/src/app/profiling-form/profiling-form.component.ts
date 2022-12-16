import { AfterViewInit, Component, Inject, OnInit } from '@angular/core';
import { FormControl, FormGroup } from '@angular/forms';
import { MAT_DIALOG_DATA } from '@angular/material/dialog';
import { ShortcutInput } from 'ng-keyboard-shortcuts';
import { ModelService } from '../services/model.service';

@Component({
  selector: 'app-profiling-form',
  templateUrl: './profiling-form.component.html',
  styleUrls: ['./profiling-form.component.scss', './../error/error.component.scss', './../home/home.component.scss',
    './../datasets/datasets.component.scss']
})
export class ProfilingFormComponent implements OnInit, AfterViewInit {

  shortcuts: ShortcutInput[] = [];

  profilingFormGroup: FormGroup;
  models = [];
  datasets = [];

  constructor(
    @Inject(MAT_DIALOG_DATA) public data,
    public modelService: ModelService,
  ) { }

  ngOnInit(): void {
    this.modelService.getModelList(this.data.projectId)
      .subscribe(
        (response: { models: any }) => {
          this.models = response.models;
          if (this.models.length > 0) {
            this.profilingFormGroup.get('model_id').setValue(this.models[0].id);
          }
        },
        error => {
          this.modelService.openErrorDialog(error);
        });

    this.modelService.getDatasetList(this.data.projectId)
      .subscribe(
        (response: { datasets: any }) => {
          this.datasets = response.datasets;
          if (this.datasets.length > 0) {
            this.profilingFormGroup.get('dataset_id').setValue(this.datasets[0].id);
          }
        },
        error => {
          this.modelService.openErrorDialog(error);
        });

    this.profilingFormGroup = new FormGroup({
      name: new FormControl('Profiling' + String(this.data.index + 1)),
      project_id: new FormControl(this.data.projectId),
      model_id: new FormControl(),
      dataset_id: new FormControl(),
      num_threads: new FormControl(1),
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

  addProfiling() {
    if (!this.data.editing) {
      this.modelService.addProfiling(this.profilingFormGroup.value)
        .subscribe(
          response => { },
          error => {
            this.modelService.openErrorDialog(error);
          }
        );
    } else {
      this.modelService.editProfiling({
        id: this.data.profilingId,
        dataset_id: this.profilingFormGroup.get('dataset_id').value,
        num_threads: this.profilingFormGroup.get('num_threads').value
      })
        .subscribe(
          response => { },
          error => {
            this.modelService.openErrorDialog(error);
          }
        );
    }
  }

}
