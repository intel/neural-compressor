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
import Button from 'react-bootstrap/esm/Button';
import Spinner from 'react-bootstrap/Spinner';
import Image from 'react-bootstrap/Image';
import Tooltip from 'react-bootstrap/Tooltip';
import OverlayTrigger from 'react-bootstrap/OverlayTrigger';
import './Workloads.scss';
import moment from 'moment';
import { api } from './../../App';
import { getLabel } from './../Diagnosis/Diagnosis';
import { io } from 'socket.io-client';

export default function Workloads({ setSelectedWorkload, selectedWorkload, setWarningText, setSelectedOp }) {
  const [workloads, setWorkloads] = useState([]);
  const [spinner, setSpinner] = useState(true);

  let socket = io('/');
  socket.on('Config update', data => {
    getWorkloads(false);
  });

  useEffect(() => {
    getWorkloads(true);
  }, []);

  let getWorkloads = (changeSelectedWorkload) => {
    api.get('api/workloads?token=' + localStorage.getItem('token'))
      .then(
        response => {
          if (changeSelectedWorkload) {
            setSelectedWorkload(response.data.workloads[0]);
          }
          setWorkloads(response.data.workloads);
          setSpinner(false);
        }
      )
      .catch(error => {
        setWarningText(error.message);
        setSpinner(false);
      });
  }

  let deleteWorkload = (selectedWorkload) => {
    api.post('api/workloads/delete?token=' + localStorage.getItem('token'), { workload_id: selectedWorkload.uuid })
      .then(
        response => {
          getWorkloads(true);
        }
      )
      .catch(error => {
        setWarningText(error.message);
        setSpinner(false);
      });
  }

  let workloadsList = workloads.map(workload => {
    return (
      <div key={workload.uuid} onClick={e => { setSelectedWorkload(workload); setSelectedOp(null); }}>
        <Button variant="secondary" className={workload.uuid === selectedWorkload.uuid ? 'active' : ''}>
          {workload.mode}
          <div className='date'>{moment(moment.unix(workload.creation_time)).fromNow()}</div>
        </Button>
      </div >
    );
  });

  const tooltipDelete = (
    <Tooltip id="tooltipDelete">
      Delete this workload
    </Tooltip>
  );

  const tooltipCopy = (
    <Tooltip id="tooltipCopy">
      Copy full model path
    </Tooltip>
  );

  const tooltipFullPath = (
    <Tooltip id="tooltipFullPath">
      {selectedWorkload?.model_path}
    </Tooltip>
  );

  return (
    <div>
      {spinner && <Spinner className="spinner" animation="border" />}
      {workloadsList.length > 0 &&
        <div className="data-panel workloads-list">
          <h3>Workloads</h3>
          {workloadsList}
        </div>
      }
      {workloadsList.length === 0 &&
        <div className="data-panel no-data workloads-list">
          <h3>Neural Insights</h3>
          <p>Run diagnosis or profiling process to see workloads on this page.</p>
        </div>
      }
      {selectedWorkload &&
        <div className="data-panel workloads-list">
          <h3>Details
            <OverlayTrigger placement="right" overlay={tooltipDelete}>
              <div className="delete-button" role="button" onClick={e => { deleteWorkload(selectedWorkload); setSelectedOp(null); }}>
                <Image src="icons/057a-trash-solid-red.svg"
                  onMouseOver={e => (e.currentTarget.src = "icons/057a-trash-solid.svg")}
                  onMouseOut={e => (e.currentTarget.src = "icons/057a-trash-solid-red.svg")}
                />
              </div>
            </OverlayTrigger>
          </h3>
          <table class="details-table">
            <tbody>
              <tr>
                <td>Framework:</td>
                <td>{selectedWorkload?.framework}</td>
              </tr>
              <tr>
                <td>
                  Model path:
                </td>
                <td>
                  <OverlayTrigger placement="bottom" overlay={tooltipFullPath}>
                    <div>{getLabel(selectedWorkload?.model_path)}</div>
                  </OverlayTrigger>
                </td>
                <td>
                  {selectedWorkload?.framework === 'TensorFlow' &&
                    <OverlayTrigger placement="right" overlay={tooltipCopy}>
                      <div className="delete-button" role="button" onClick={() => { navigator.clipboard.writeText(selectedWorkload.model_path) }}>
                        <Image src="icons/146b-copy-outlined.svg"
                          onMouseOver={e => (e.currentTarget.src = "icons/146b-copy-outlined-gray.svg")}
                          onMouseOut={e => (e.currentTarget.src = "icons/146b-copy-outlined.svg")}
                        />
                      </div>
                    </OverlayTrigger>
                  }
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      }
    </div>
  )

}