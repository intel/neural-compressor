// -* - coding: utf - 8 -* -
// Copyright(c) 2023 Intel Corporation
//
// Licensed under the Apache License, Version 2.0(the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
import React, { useEffect, useState } from 'react';
import './Histogram.scss';
import Plot from 'react-plotly.js';
import { api } from './../../App';
import Spinner from 'react-bootstrap/Spinner';

function Histogram({ selectedWorkload, selectedOp, histogramType, setWarningText }) {
  const [histogramData, setHistogramData] = useState(null);

  useEffect(() => {
    if (selectedOp.length && histogramType.length) {
      setHistogramData(null);
      api.post('api/diagnosis/histogram?token=' + localStorage.getItem('token'), { workload_id: selectedWorkload.uuid, op_name: selectedOp, type: histogramType })
        .then(
          response => {
            setHistogramData(response.data);
          })
        .catch(error => {
          setWarningText(error.message);
        });
    }
  }, [histogramType, selectedOp]);

  return (
    <div className="Histogram">
      <h3>{histogramType.charAt(0).toUpperCase() + histogramType.slice(1)} histogram</h3>
      {!histogramData && <Spinner className="spinner" animation="border" />}

      {histogramData?.length === 0 && <p>No histogram data for this OP.</p>}

      {histogramData?.length > 0 &&
        <div>
          <div>
            When you hover over the chart a menu will appear in the top right corner.<br />
            You can zoom the chart, save it as .png file or hide channels by clicking them in the legend.
          </div>

          <div id='myDiv'>
            <Plot
              data={getHistogramData(histogramData)}
              layout={layout}
              useResizeHandler={true}
              style={{ width: '60vw' }}>
            </Plot>
          </div>
        </div>
      }
    </div>
  )
};

const getHistogramData = (histogramData) => {
  const data = [];
  if (histogramData.length) {
    const colorPalette = generateColor(histogramData[0].histograms.length);
    histogramData[0].histograms.forEach((series, index) => {
      data.push(
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
          points: false,
          spanmode: 'hard'
        }
      );
    });
  }
  return data;
}

const layout = {
  height: 450,
  responsive: true,
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
    b: 20,
    t: 30,
    pad: 0
  }
};

const generateColor = (num) => {
  const colorPalette = [];
  const step = 100 / num;
  for (let i = num; i > 0; --i) {
    colorPalette.push(`rgb(${20 + (step * i)}, ${100 - (step * i * 0.1)}, ${200 - (step * i * 0.1)})`);
  }
  return colorPalette;
}

Histogram.propTypes = {};

Histogram.defaultProps = {};

export default Histogram;
