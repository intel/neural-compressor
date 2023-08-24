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
import React, { useState } from 'react';
import './Diagnosis.scss';
import Graph from './../Graph/Graph';
import OpDetails from './../OpDetails/OpDetails';
import OpList from './../OpList/OpList';
import Histogram from './../Histogram/Histogram';
import Workloads from './../Workloads/Workloads';
import Profiling from './../Profiling/Profiling';
import Warning from './../Warning/Warning';
import Form from 'react-bootstrap/Form';
import InputGroup from 'react-bootstrap/InputGroup';
import Button from 'react-bootstrap/esm/Button';
import Spinner from 'react-bootstrap/Spinner';

function Diagnosis() {
  const [selectedNode, setSelectedNode] = useState(null);
  const [selectedWorkload, setSelectedWorkload] = useState(null);
  const [selectedOp, setSelectedOp] = useState(null);
  const [selectedPattern, setSelectedPattern] = useState([]);
  const [histogramType, setHistogramType] = useState(null);
  const [warningText, setWarningText] = useState('');

  return (
    <div className="Diagnosis">
      <Warning className="alert" warningText={warningText} setWarningText={setWarningText} />
      <div className="flexbox">
        <div className="flexbox-inside">
          <Workloads setSelectedWorkload={setSelectedWorkload} selectedWorkload={selectedWorkload} setWarningText={setWarningText} setSelectedOp={setSelectedOp} />
          {/* {selectedWorkload?.mode === 'quantization' &&
              <NodeSearch />
            } */}
          {selectedWorkload?.mode === 'quantization' &&
            <NodeProperties selectedNode={selectedNode} />
          }
        </div>
        {selectedWorkload?.mode === 'benchmark' &&
          <div className="flex-item">
            <Profiling selectedWorkload={selectedWorkload} setWarningText={setWarningText} />
          </div>
        }
        {selectedWorkload?.mode === 'quantization' &&
          <div className="flex-item">
            <Graph setSelectedNode={setSelectedNode} selectedWorkload={selectedWorkload} selectedOp={selectedOp} selectedPattern={selectedPattern} setWarningText={setWarningText} />
          </div>
        }
        {selectedWorkload?.mode === 'quantization' &&
          <div className="flex-item">
            <AccuracyResults selectedWorkload={selectedWorkload} />
            <OpList selectedWorkload={selectedWorkload} setSelectedOp={setSelectedOp} selectedOp={selectedOp} setWarningText={setWarningText} />
          </div>
        }
      </div>
      {selectedWorkload?.mode === 'quantization' && selectedOp &&
        <div className="flexbox">
          <div className="flex-item">
            <OpDetails selectedWorkload={selectedWorkload} selectedOp={selectedOp} setHistogramType={setHistogramType} setSelectedPattern={setSelectedPattern} setWarningText={setWarningText} />
          </div>
          <div className="flex-item">
            {histogramType && <Histogram selectedWorkload={selectedWorkload} selectedOp={selectedOp} histogramType={histogramType} setWarningText={setWarningText} />}
          </div>
        </div>
      }
    </div>
  )
};

function NodeProperties({ selectedNode }) {
  if (selectedNode) {
    const propertyList = Object.entries(selectedNode.properties).map(([key, value]) => {
      return (
        <tr key={key}>
          <td className="table-key">{key}</td>
          <td colSpan={2} className="table-value">{getLabel(value)}</td>
        </tr>
      )
    });

    const attributeList = selectedNode.attributes?.map(attribute => {
      return (
        <tr key={attribute.name}>
          <td className="table-key">{attribute.name}</td>
          <td className="table-value">{attribute.attribute_type}</td>
          {attribute.attribute_type !== "float32" &&
            <td className="table-value">{attribute.value.toString()}</td>
          }
          {attribute.attribute_type === "float32" &&
            <td className="table-value">{attribute.value.toExponential(2)}</td>
          }
        </tr>
      )
    });

    return (
      <div className='data-panel'>
        <h3>Node details</h3>
        <table className="property-table">
          <tbody>
            <tr>
              <td className="table-title" colSpan={2}>Properties</td>
            </tr>
            {propertyList}
            <tr>
              {attributeList && <td className="table-title" colSpan={2}>Attributes</td>}
            </tr>
            {attributeList}
          </tbody>
        </table>
      </div>
    );
  } else {
    return;
  }
}

class NodeSearch extends React.Component {
  render() {
    return (
      <div className='data-panel'>
        <h3>Node search</h3 >
        <InputGroup className="mb-3">
          <Form.Control
            placeholder="Node name"
            aria-label="Node name"
            aria-describedby="basic-addon2"
          />
          <Button variant="primary">Search</Button>
        </InputGroup>
      </div>
    )
  }
}

function AccuracyResults({ selectedWorkload }) {
  return (
    <div className='data-panel'>
      {selectedWorkload.status === 'wip' &&
        <p> Quantization is in progress.
          <div className="spinner-container">
            <Spinner className="spinner" animation="border" />
          </div>
        </p>
      }
      {selectedWorkload.status !== 'wip' &&
        !selectedWorkload.accuracy_data.ratio &&
        <table className='accuracy-table'>
          <tbody>
            <tr>
              <td className="accuracy-title">Accuracy <br /> results</td>
              <td>
                <div className="accuracy-number">N/A</div>
                <div className="accuracy-subtitle">FP32</div>
              </td>
              <td>
                <div className="accuracy-number">N/A</div>
                <div className="accuracy-subtitle">INT8</div>
              </td>
              <td>
                <div className="accuracy-number">N/A</div>
                <div className="accuracy-subtitle">Ratio</div>
              </td>
            </tr>
          </tbody>
        </table>
      }
      {selectedWorkload.status !== 'wip' &&
        selectedWorkload.accuracy_data.ratio &&
        <table className='accuracy-table'>
          <tbody>
            <tr>
              <td className="accuracy-title">Accuracy <br /> results</td>
              <td>
                <div className="accuracy-number">{(selectedWorkload.accuracy_data.baseline_accuracy * 100).toPrecision(3)}%</div>
                <div className="accuracy-subtitle">FP32</div>
              </td>
              <td>
                <div className="accuracy-number">{(selectedWorkload.accuracy_data.optimized_accuracy * 100).toPrecision(3)}%</div>
                <div className="accuracy-subtitle">INT8</div>
              </td>
              <td>
                <div className="accuracy-number">{(selectedWorkload.accuracy_data.ratio * 100).toPrecision(2)}%</div>
                <div className="accuracy-subtitle">Ratio</div>
              </td>
            </tr>
          </tbody>
        </table>
      }
    </div>
  )
}

export const getLabel = (label) => {
  if (label.includes('/')) {
    return label.replace(/^.*[\\\/]/, '');
  } else {
    return label;
  }
}

export const customColor = [
  '#5B69FF',
  '#FF848A',
  '#EDB200',
  '#1E2EB8',
  '#FF5662',
  '#C98F00',
  '#000F8A',
  '#C81326',
  '#000864',
  '#9D79BC',
  '#A14DA0',
];

export default Diagnosis;
