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
import './OpDetails.scss';
import Button from 'react-bootstrap/esm/Button';
import { api } from './../../App';

export default function OpDetails({ selectedWorkload, selectedOp, setHistogramType, setSelectedPattern, setWarningText }) {
  const [opDetails, setOpDetails] = useState({
    "OP name": "",
    "Pattern": {
      "sequence": [],
      "precision": ""
    },
    "Weights": {
      "dtype": "",
      "granularity": ""
    },
    "Activation": {
      "dtype": ""
    }
  });

  useEffect(() => {
    if (selectedOp?.length) {
      api.post('api/diagnosis/op_details?token=' + localStorage.getItem('token'), { workload_id: selectedWorkload.uuid, op_name: selectedOp })
        .then(
          response => {
            setOpDetails(response.data);
            setSelectedPattern(response.data.Pattern.sequence);
            setHistogramType(null);
          })
        .catch(error => {
          setWarningText(error.message);
        });
    }
  }, [selectedOp]);

  return (
    <div>
      {selectedOp &&
        <div id='opDetails' className='data-panel'>
          <h3>OP details</h3>
          <br />
          {selectedWorkload.framework !== 'PyTorch' &&
            <table className="property-table">
              <tbody>
                <tr>
                  <td className="table-key">OP name</td>
                  <td className="table-value" colSpan={2}>{opDetails['OP name']}</td>
                </tr>
                <tr>
                  <td className="table-title" colSpan={3}>Pattern</td>
                </tr>
                <tr>
                  <td className="table-key">Sequence</td>
                  <td className="table-value" colSpan={2}>
                    {opDetails.Pattern.sequence.map(
                      sequence => {
                        return <span key={sequence}>{sequence} </span>
                      }
                    )}
                  </td>
                </tr>
                <tr>
                  <td className="table-key">Precision</td>
                  <td className="table-value" colSpan={2}>{opDetails.Pattern.precision}</td>
                </tr>
                <tr>
                  <td className="table-title" colSpan={3}>
                    Weights
                  </td>
                </tr>
                <tr>
                  <td className="table-key">Dtype</td>
                  <td className="table-value">{opDetails.Weights.dtype}</td>
                  <td className="table-value">
                    <Button variant="primary" className="histogram-btn" onClick={() => setHistogramType('weights')}>Show weights histogram</Button>
                  </td>
                </tr>
                <tr>
                  <td className="table-key">Granularity</td>
                  <td className="table-value" colSpan={2}>{opDetails.Weights.granularity}</td>
                </tr>
                <tr>
                  <td className="table-title" colSpan={3}>
                    Activation
                  </td>
                </tr>
                <tr>
                  <td className="table-key">Dtype</td>
                  <td className="table-value">{opDetails.Activation.dtype}</td>
                  <td className="table-value">
                    <Button variant="primary" className="histogram-btn" onClick={() => setHistogramType('activation')}>Show activation histogram</Button>
                  </td>
                </tr>
              </tbody>
            </table>
          }
          {selectedWorkload.framework === 'PyTorch' &&
            <>
              <table className="property-table">
                <tbody>
                  <tr>
                    <td className="table-key">OP name</td>
                    <td className="table-value" colSpan={2}>{opDetails['OP name']}</td>
                  </tr>
                </tbody>
              </table>
              <br />
              <Button variant="primary" className="histogram-btn" onClick={() => setHistogramType('weights')}>Show weights histogram</Button>
              <br />
              <br />
              <Button variant="primary" className="histogram-btn" onClick={() => setHistogramType('activation')}>Show activation histogram</Button>
            </>
          }
        </div>
      }
    </div>
  );
}
