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
import React, { useEffect, useState, useMemo } from 'react';
import './Profiling.scss';
import { api } from './../../App';
import Plot from 'react-plotly.js';
import { getLabel } from './../Diagnosis/Diagnosis';
import Table from 'react-bootstrap/Table';

export default function Profiling({ selectedWorkload, setWarningText }) {
  const [profilingTable, setProfilingTable] = useState([]);
  const [profilingChartData, setProfilingChartData] = useState([]);
  return (
    <div>
      <ProfilingChart profilingChartData={profilingChartData} />
      <ProfilingTable selectedWorkload={selectedWorkload} profilingTable={profilingTable} setProfilingTable={setProfilingTable} setProfilingChartData={setProfilingChartData} setWarningText={setWarningText} />
    </div>
  )
}

function ProfilingTable({ selectedWorkload, profilingTable, setProfilingTable, setProfilingChartData, setWarningText }) {

  const [checked, setChecked] = useState({});
  const [sorting, setSorting] = useState({ field: 'node_name', direction: 1 });

  useEffect(() => {
    if (selectedWorkload) {
      api.post('api/profiling?token=' + localStorage.getItem('token'), { workload_id: selectedWorkload.uuid })
        .then(
          response => {
            setProfilingTable(response.data);
            setSorting({ field: 'total_execution_time', direction: 1 });
            const showOnChart = {};
            const chartData = [];
            response.data.forEach((node, index) => {
              if (index < 10) {
                showOnChart[node.node_name] = true;
                chartData.push(node);
              } else {
                showOnChart[node.node_name] = false;
              }
            });
            setChecked(showOnChart);
            setProfilingChartData(chartData);
          })
        .catch(error => {
          setWarningText(error.message);
        });
    }
  }, [selectedWorkload]);

  let sortedProfiling = useMemo(() => {
    let sortedTable = [...profilingTable];
    if (sorting !== null) {
      sortedTable.sort((a, b) => {
        if (a[sorting.field] < b[sorting.field]) {
          return sorting.direction;
        }
        if (a[sorting.field] > b[sorting.field]) {
          return -sorting.direction;
        }
        return 0;
      });
    }
    return sortedTable;
  }, [sorting]);

  const requestSorting = field => {
    let direction = -sorting.direction;
    setSorting({ field, direction });
  };

  const getSortingClass = (name) => {
    let classes = 'header clickable';
    if (sorting.field === name) {
      return classes + (sorting.direction === 1 ? ' ascending' : ' descending');
    }
    return 'header clickable';
  };

  const requestChartCheck = (nodeName, value) => {
    let chartCheck = checked;
    chartCheck[nodeName] = value;
    setChecked(chartCheck);
    const newProfilingChartData = profilingTable.filter(node => checked[node.node_name] === true);
    setProfilingChartData(newProfilingChartData);
  };

  const tableContent = sortedProfiling?.map(profiling => {
    return (
      <tr key={profiling.node_name}>
        <td className='cell'>{profiling.node_name}</td>
        <td className="cell center">{profiling.accelerator_execution_time}</td>
        <td className="cell center">{profiling.cpu_execution_time}</td>
        <td className="cell center">{profiling.op_defined}</td>
        <td className="cell center">{profiling.op_run}</td>
        <td className="cell center">{profiling.total_execution_time}</td>
        <td className="cell center">
          <input
            type="checkbox"
            defaultChecked={checked[profiling.node_name]}
            value={checked[profiling.node_name]}
            onClick={(e) => {
              requestChartCheck(profiling.node_name, e.target.checked);
            }}
          />
        </td>
      </tr >
    );
  });

  return (
    <div className="Profiling">
      <Table className="rounded data-panel">
        <tbody>
          <tr>
            <td className={getSortingClass('node_name')} onClick={() => requestSorting('node_name')}>Name</td>
            <td className={getSortingClass('accelerator_execution_time')} onClick={() => requestSorting('accelerator_execution_time')}>Accelerator execution time [μs]</td>
            <td className={getSortingClass('cpu_execution_time')} onClick={() => requestSorting('cpu_execution_time')}>CPU execution time [μs]</td>
            <td className={getSortingClass('op_defined')} onClick={() => requestSorting('op_defined')}>Op defined</td>
            <td className={getSortingClass('op_run')} onClick={() => requestSorting('op_run')}>Op run</td>
            <td className={getSortingClass('total_execution_time')} onClick={() => requestSorting('total_execution_time')}>Total execution time [μs]</td>
            <td className="header">Show on chart</td>
          </tr>
          {tableContent}
        </tbody>
      </Table>
    </div>
  );
}

function ProfilingChart({ profilingChartData }) {
  return (<div className='data-panel'>
    <Plot
      data={getChartData(profilingChartData)}
      layout={layout}
      useResizeHandler={true}
      style={{ 'width': '100%' }}
    ></Plot>
  </div >)
};

const getChartData = (profilingData) => {
  let data = [];
  if (Object.keys(profilingData).length) {
    const colorPalette = generateColor(profilingData.length);
    profilingData.forEach((node, index) => {
      data.push({
        name: getLabel(node.node_name),
        x: [node.node_name],
        y: [node.total_execution_time],
        type: 'bar',
        marker: {
          color: colorPalette[index]
        }
      });
    });
  }
  return data;
}

const layout = {
  responsive: true,
  xaxis: {
    title: 'Total execution time [μs]',
    showticklabels: false
  },
  yaxis: {
    showgrid: true,
  },
  legend: {
    tracegroupgap: 0,
  },
  opacity: 1,
};

const generateColor = (num) => {
  const colorPalette = [];
  const step = 100 / num;
  for (let i = num; i > 0; --i) {
    colorPalette.push(`rgb(${20 + (step * i)}, ${100 - (step * i * 0.1)}, ${200 - (step * i * 0.1)})`);
  }
  return colorPalette;
}

