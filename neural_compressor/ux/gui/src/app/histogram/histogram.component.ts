import { Component, Input, OnChanges } from '@angular/core';
import { ActivatedRoute } from '@angular/router';
import { ModelService } from '../services/model.service';

@Component({
  selector: 'app-histogram',
  templateUrl: './histogram.component.html',
  styleUrls: ['./histogram.component.scss', './../error/error.component.scss']
})
export class HistogramComponent implements OnChanges {

  @Input() modelId;
  @Input() opName;
  @Input() type;

  data = [];
  layout;

  constructor(
    private modelService: ModelService,
    public activatedRoute: ActivatedRoute,
  ) { }

  ngOnChanges(): void {
    this.data = [];
    this.getHistogramData();
  }

  getHistogramData() {
    this.modelService.getHistogram(this.activatedRoute.snapshot.params.id, this.modelId, this.opName, this.type)
      .subscribe(
        response => {
          response[0]['histograms'].forEach((series, index) => {
            this.data.push(
              {
                x: series['data'],
                type: "histogram",
                opacity: 0.6,
                name: 'channel ' + index
              }
            )
          })

          this.layout = {
            bargap: 0.05,
            bargroupgap: 0.2,
            barmode: "overlay",
            xaxis: { title: "Value" },
            yaxis: { title: "Count" }
          };

          document.getElementById('histograms').scrollIntoView({ behavior: 'smooth' });

        },
        error => this.modelService.openErrorDialog(error));
  };
}
