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
import './Workloads.scss';
import moment from 'moment';
import { api } from './../../App';
import { getLabel } from './../Diagnosis/Diagnosis';
import { io } from 'socket.io-client';

export default function Workloads({ setSelectedWorkload, selectedWorkload, setWarningText }) {
  const [workloads, setWorkloads] = useState([]);

  let socket = io('https://10.91.48.210:4567/');
  socket.on('Config update', data => {
    console.log(data);
    getWorkloads();
  });

  useEffect(() => {
    getWorkloads();
  }, []);

  let getWorkloads = () => {
    api.get('api/workloads?token=asd')
      .then(
        response => {
          setSelectedWorkload(response.data.workloads[0]);
          setWorkloads(response.data.workloads)
        }
      )
      .catch(error => {
        setWarningText(error.message);
      });
  }

  let workloadsList = workloads.map(workload => {
    return (
      <div key={workload.uuid} onClick={e => { setSelectedWorkload(workload) }}>
        <Button variant="secondary" className={workload.uuid === selectedWorkload.uuid ? 'active' : ''}>
          {workload.mode}
          <div className='date'>{moment(moment.unix(workload.creation_time)).fromNow()}</div>
        </Button>
      </div >
    );
  });

  return (
    <div>
      {workloadsList.length > 0 &&
        <div className="data-panel workloads-list">
          <h3>Workloads</h3>
          {workloadsList}
        </div>
      }
      {workloadsList.length === 0 &&
        <div className="data-panel">
          <h3>Intel Neural Insights</h3>
          <p>Run diagnosis or profiling process to see workloads on this page.</p>
        </div>
      }
      {selectedWorkload &&
        <div className="data-panel">
          <h3>Details</h3>
          <p>Framework: {selectedWorkload?.framework}</p>
          <p>Model path: {getLabel(selectedWorkload?.model_path)}</p>
        </div>
      }
    </div>
  )
}
