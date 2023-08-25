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
import './Workloads.scss';
import moment from 'moment';
import { api } from './../../App';
import { io } from 'socket.io-client';

export default function Workloads({ setSelectedWorkload, selectedWorkload, setWarningText, setSelectedOp, setSelectedNode }) {
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

  let workloadsList = workloads.map(workload => {
    return (
      <div key={workload.uuid} onClick={e => { setSelectedWorkload(workload); setSelectedOp(null); setSelectedNode(null) }}>
        <Button variant="secondary" className={workload.uuid === selectedWorkload.uuid ? 'active' : ''}>
          {workload.workload_name}
          <div className='date'>{workload.mode} [{workload.framework}]</div>
          <div className='date'>{moment(moment.unix(workload.creation_time)).fromNow()}</div>
        </Button>
      </div >
    );
  });

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
    </div>
  )

}