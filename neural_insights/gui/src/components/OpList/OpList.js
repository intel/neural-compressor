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
import { api } from '../../App';
import Table from 'react-bootstrap/Table';
import Spinner from 'react-bootstrap/Spinner';

export default function OpList({ selectedWorkload, setSelectedOp, selectedOp, setWarningText }) {
  const [opList, setOpList] = useState([]);

  useEffect(() => {
    if (selectedWorkload) {
      setOpList([]);
      api.post('api/diagnosis/op_list?token=' + localStorage.getItem('token'), { workload_id: selectedWorkload.uuid })
        .then(
          response => {
            setOpList(response.data.sort((a, b) => b.MSE - a.MSE));
          })
        .catch(error => {
          if (selectedWorkload?.status !== 'wip') {
            setWarningText(error.message + ': ' + error?.response?.data);
          }
        });
    }
  }, [selectedWorkload, selectedOp]);

  const tableContent =
    opList.map(opData => {
      return (
        <tr key={opData['OP name']}
          className={opData['OP name'] === selectedOp ? 'clickable active' : 'clickable'}
          onClick={() => {
            setSelectedOp(opData['OP name']);
            setTimeout(() => {
              document.getElementById('opDetails').scrollIntoView({ behavior: 'smooth' });
            }, 500)
          }}>
          <td className="cell">{opData['OP name']}</td>
          <td className="cell right nowrap">{opData['MSE'].toExponential(3)}</td>
          {typeof opList[0]['Activation Min'] !== 'undefined' &&
            <td className="cell right">{opData['Activation Min'].toFixed(2)}</td>
          }
          {typeof opList[0]['Activation Max'] !== 'undefined' &&
            <td className="cell right">{opData['Activation Max'].toFixed(2)}</td>
          }
        </tr>
      )
    });

  return (
    <div className="overflow-table">
      {opList.length === 0 &&
        <div className="spinner-container">
          <Spinner className="spinner" animation="border" />
        </div>
      }
      {opList.length > 0 &&
        <Table className="rounded" hover>
          <thead>
            <tr>
              <th className="header center">OP Name</th>
              <th className="header center">MSE</th>
              {typeof opList[0]['Activation Min'] !== 'undefined' &&
                <th className="header center">Activation Min</th>
              }
              {typeof opList[0]['Activation Max'] !== 'undefined' &&
                <th className="header center">Activation Max</th>
              }
            </tr>
          </thead>
          <tbody>
            {tableContent}
          </tbody>
        </Table>
      }
    </div>
  );
}