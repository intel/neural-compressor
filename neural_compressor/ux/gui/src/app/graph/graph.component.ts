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
import { Component, Inject, Input, OnChanges, OnInit, Optional, ViewChild } from '@angular/core';
import { MAT_DIALOG_DATA } from '@angular/material/dialog';
import { MatSidenav } from '@angular/material/sidenav';
import { Subject } from 'rxjs';
import { ModelService } from '../services/model.service';

@Component({
  selector: 'app-graph',
  templateUrl: './graph.component.html',
  styleUrls: ['./graph.component.scss']
})
export class GraphComponent implements OnChanges, OnInit {

  @Input() modelPath: string;
  @Input() diagnosisTab?: boolean;
  @Input() diagnosisTabParams?: {};

  edges: Edge[] = [];
  nodes: Node[] = [];
  nodeDetails: Node;
  expandedNodesArray = [];
  showSpinner = false;
  miniMapMaxHeight = window.innerHeight;

  layoutSettings = { orientation: 'TB' };
  panToNodeObservable: Subject<string> = new Subject<string>();
  updateObservable: Subject<string> = new Subject<string>();
  center$: Subject<boolean> = new Subject();
  zoomToFit$: Subject<boolean> = new Subject();

  @ViewChild('sidenav') sidenav: MatSidenav;

  customColor = [
    '#005B85',
    '#0095CA',
    '#00C7FD',
    '#047271',
    '#07b3b0',
    '#9E8A87',
    '#333471',
    '#5153B0',
    '#ED6A5E ',
    '#9D79BC',
    '#A14DA0',
  ];

  constructor(
    @Optional() @Inject(MAT_DIALOG_DATA) public data,
    private modelService: ModelService
  ) { }

  ngOnInit() {
    this.showSpinner = true;
    this.getGraph();
  }

  ngOnChanges(): void {
    this.showSpinner = true;
    if (this.diagnosisTabParams['Pattern']) {
      this.highlightPatternInGraph();
    } else {
      this.getGraph();
    }
  }

  highlightPatternInGraph() {
    this.modelService.highlightPatternInGraph(
      this.modelPath,
      this.diagnosisTabParams['OP name'],
      this.diagnosisTabParams['Pattern']['sequence']
    )
      .subscribe(
        response => {
          this.updateGraph(response['graph']);
          this.expandedNodesArray = response['groups'];
          this.panToNodeObservable.next(this.diagnosisTabParams['OP name']);
        },
        error => {
          this.modelService.openErrorDialog(error);
        });
  }

  getGraph() {
    this.edges = [];
    this.nodes = [];
    this.modelPath = this.modelPath ?? this.data.modelPath;
    this.modelService.getModelGraph(this.modelPath)
      .subscribe(
        response => {
          this.updateGraph(response);
        },
        error => {
          this.modelService.openErrorDialog(error);
        });
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
        highlight: node.highlight,
        color: node.label.includes('Relu') ? this.customColor[1] : this.customColor[0]
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
    if (this.diagnosisTab) {
      this.modelService.getNodeDetails$.next(node.id);
    } else {
      this.nodeDetails = node;
      this.sidenav.open()
    }
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
    this.modelService.getModelGraph(this.modelPath, this.expandedNodesArray)
      .subscribe(
        graph => { this.updateGraph(graph) },
        error => {
          this.modelService.openErrorDialog(error);
        });
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
