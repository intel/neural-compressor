import { Component, Inject, OnInit } from '@angular/core';
import { FormBuilder, FormGroup } from '@angular/forms';
import { MAT_DIALOG_DATA } from '@angular/material/dialog';
import { ModelService } from '../services/model.service';

@Component({
  selector: 'app-profiling-form',
  templateUrl: './profiling-form.component.html',
  styleUrls: ['./profiling-form.component.scss', './../error/error.component.scss', './../home/home.component.scss', './../datasets/datasets.component.scss']
})
export class ProfilingFormComponent implements OnInit {

  profilingFormGroup: FormGroup;
  models = [];
  datasets = [];

  constructor(
    @Inject(MAT_DIALOG_DATA) public data,
    public modelService: ModelService,
    private _formBuilder: FormBuilder
  ) { }

  ngOnInit(): void {
    this.modelService.getModelList(this.data.projectId)
      .subscribe(
        response => {
          this.models = response['models'];
          if (this.models.length > 0) {
            this.profilingFormGroup.get('model_id').setValue(this.models[0]['id']);
          }
        },
        error => {
          this.modelService.openErrorDialog(error);
        });

    this.modelService.getDatasetList(this.data.projectId)
      .subscribe(
        response => {
          this.datasets = response['datasets'];
          if (this.datasets.length > 0) {
            this.profilingFormGroup.get('dataset_id').setValue(this.datasets[0].id);
          }
        },
        error => {
          this.modelService.openErrorDialog(error);
        });

    this.profilingFormGroup = this._formBuilder.group({
      name: ['Profiling' + String(this.data.index + 1)],
      project_id: this.data.projectId,
      model_id: [],
      dataset_id: [],
      num_threads: 1
    });
  }

  addProfiling() {
    this.modelService.addProfiling(this.profilingFormGroup.value)
      .subscribe(
        response => { },
        error => {
          this.modelService.openErrorDialog(error);
        }
      );
  }

}
