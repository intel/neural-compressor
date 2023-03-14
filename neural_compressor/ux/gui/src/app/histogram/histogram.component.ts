import { Component, Input, OnChanges, OnInit } from '@angular/core';
import { ActivatedRoute } from '@angular/router';
import { ModelService } from '../services/model.service';

@Component({
  selector: 'app-histogram',
  templateUrl: './histogram.component.html',
  styleUrls: ['./histogram.component.scss', './../error/error.component.scss']
})
export class HistogramComponent implements OnInit, OnChanges {

  @Input() modelId;
  @Input() opName;
  @Input() type;

  data = [];
  layout;
  showSpinner = false;

  constructor(
    private modelService: ModelService,
    public activatedRoute: ActivatedRoute,
  ) { }

  ngOnInit() {
    this.modelService.colorMode$.subscribe(change => {
      const isDark = change === 'dark/';
      this.setLayout(isDark);
    });
  }

  ngOnChanges(): void {
    this.showSpinner = true;
    this.data = [];
    this.getHistogramData();
  }

  getHistogramData() {
    this.modelService.getHistogram(this.activatedRoute.snapshot.params.id, this.modelId, this.opName, this.type)
      .subscribe(
        response => {
          this.data = [];
          const colorPalette = this.generateColor(response[0].histograms.length);
          response[0].histograms.forEach((series, index) => {
            this.data.push(
              {
                x: series.data,
                type: 'violin',
                orientation: 'h',
                side: 'negative',
                y0: 'channel ' + index,
                name: 'channel ' + index,
                width: 100,
                opacity: 0.8,
                fillcolor: colorPalette[index],
                hoverinfo: 'none',
                line: {
                  width: 1,
                  color: series.data.length === 1 ? colorPalette[index] : '#fff',
                },
                points: false
              }
            );
          }
          );

          this.setLayout(localStorage.getItem('darkMode') === 'darkMode');
          this.showSpinner = false;
        },
        error => this.modelService.openErrorDialog(error));
  };

  setLayout(isDark: boolean) {
    this.layout = {
      height: 550,
      width: 800,
      yaxis: {
        autorange: 'reversed',
        showgrid: true,
      },
      legend: {
        tracegroupgap: 0,
      },
      violinmode: 'overlay',
      opacity: 1,
      margin: {
        l: 150,
        r: 50,
        b: 100,
        t: 50,
        pad: 40
      }
    };

    if (isDark) {
      this.layout.xaxis = { color: '#fff' };
      this.layout.yaxis.color = '#fff';
      this.layout.legend.font = { color: '#fff' };
      this.layout.plot_bgcolor = '#424242';
      this.layout.paper_bgcolor = '#424242';
    }
  }

  generateColor(num: number) {
    const colorPalette = [];
    const step = 100 / num;
    for (let i = num; i > 0; --i) {
      colorPalette.push(`rgb(${20 + (step * i)}, ${100 - (step * i * 0.1)}, ${200 - (step * i * 0.1)})`);
    }
    return colorPalette;
  }

}
