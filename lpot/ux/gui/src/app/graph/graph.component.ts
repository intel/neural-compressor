// Copyright (c) 2021 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
import { Component, Inject, OnInit, ViewChild } from '@angular/core';
import { MAT_DIALOG_DATA } from '@angular/material/dialog';
import { MatSidenav } from '@angular/material/sidenav';
import { Subject } from 'rxjs';

@Component({
  selector: 'app-graph',
  templateUrl: './graph.component.html',
  styleUrls: ['./graph.component.scss']
})
export class GraphComponent implements OnInit {

  edges = [];
  nodes: Node[] = [];
  nodeDetails: Node;

  layoutSettings = { orientation: 'TB' };
  panToNodeObservable: Subject<string> = new Subject<string>();

  @ViewChild('sidenav') sidenav: MatSidenav;

  customColor = {
    'Add': '#004A86',
    'Relu': '#0095CA',
    'Conv2D': '#525252',
    'FusedBatchNorm': '#41728A',
    'Identity': '#653171',
    'Pad': '#708541',
    'BiasAdd': '#B24501',
    'MatMul': '#000F8A',
    'Mean': '#C81326',
    'Softmax': '#EDB200',
    'ArgMax': '#005B85',
    'MaxPool': '#183544',
    'Placeholder': '#515A3D',
    'Squeeze': '#C98F00',
  };

  constructor(
    @Inject(MAT_DIALOG_DATA) public data,
  ) { }

  ngOnInit(): void {
    let nodes = [];
    let edges = [];
    this.data.graph.nodes.forEach(node => {
      nodes.push({
        id: node.id.replaceAll('/', '_').replaceAll(' ', '_'),
        label: node.label,
        attributes: node.attributes,
        properties: node.properties,
      });
    });
    this.data.graph.edges.forEach(edge => {
      edges.push({
        source: edge.source.replaceAll('/', '_').replaceAll(' ', '_'),
        target: edge.target.replaceAll('/', '_').replaceAll(' ', '_'),
      });
    });
    this.nodes = nodes;
    this.edges = edges;
  }

  getDetails(node: Node) {
    this.panToNodeObservable.next(node.id);
    this.nodeDetails = node;
  }

  close() {
    this.sidenav.close();
  }
}

interface Node {
  id: string;
  label: string;
  attributes: any[];
  properties: {
    name: string;
    type: string;
  },
  data?: {
    color: string
  }
}
