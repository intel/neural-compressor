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
import { ModelService } from '../services/model.service';

@Component({
  selector: 'app-graph',
  templateUrl: './graph.component.html',
  styleUrls: ['./graph.component.scss']
})
export class GraphComponent implements OnInit {

  edges: Edge[] = [];
  nodes: Node[] = [];
  viewSize = [1000, 1000];
  nodeDetails: Node;
  expandedNodesArray = [];
  showSpinner = false;

  layoutSettings = { orientation: 'TB' };
  panToNodeObservable: Subject<string> = new Subject<string>();
  updateObservable: Subject<string> = new Subject<string>();
  center$: Subject<boolean> = new Subject();
  zoomToFit$: Subject<boolean> = new Subject();

  @ViewChild('sidenav') sidenav: MatSidenav;

  customColor = [
    '#004A86',
    '#0095CA',
    '#525252',
    '#41728A',
    '#653171',
    '#708541',
    '#B24501',
    '#000F8A',
    '#C81326',
    '#EDB200',
    '#005B85',
    '#183544',
    '#515A3D',
    '#C98F00',
  ];

  constructor(
    @Inject(MAT_DIALOG_DATA) public data,
    private modelService: ModelService
  ) { }

  ngOnInit(): void {
    this.viewSize = this.data.viewSize;
    this.updateGraph(this.data['graph']);
  }

  center() {
    this.center$.next(true);
  }

  zoomToFit() {
    this.zoomToFit$.next(true);
  }

  updateGraph(graph: any) {
    let nodes = [];
    let edges = [];
    graph.nodes.forEach(node => {
      nodes.push({
        id: node.id,
        label: node.label,
        attributes: node.attributes,
        properties: node.properties,
        node_type: node.node_type,
        color: this.customColor[node.label.length % 14]
      });
    });
    graph.edges.forEach(edge => {
      edges.push({
        source: edge.source,
        target: edge.target,
      });
    });
    this.nodes = nodes;
    this.edges = edges;
    this.showSpinner = false;
  }

  getDetails(node: Node) {
    this.nodeDetails = node;
  }

  close() {
    this.sidenav.close();
  }

  expand(id: string) {
    this.showSpinner = true;
    if (this.expandedNodesArray.includes(id)) {
      this.collapse(id);
    } else {
      this.expandedNodesArray.push(id);
    }
    this.modelService.getModelGraph(this.data.modelPath, this.expandedNodesArray)
      .subscribe(graph => this.updateGraph(graph));
  }

  collapse(id: string) {
    this.expandedNodesArray = this.expandedNodesArray.filter(x => x !== id);
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
  },
  node_type: string,
  color: string,
}

interface Edge {
  target: string,
  source: string,
}
