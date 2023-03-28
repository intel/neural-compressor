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

/* eslint no-underscore-dangle: 0 */
import { Component, EventEmitter, Inject, Input, OnChanges, OnDestroy, OnInit, Optional, Output, ViewEncapsulation } from '@angular/core';
import { MAT_DIALOG_DATA } from '@angular/material/dialog';
import { ModelService } from '../services/model.service';

const cytoscape = require('cytoscape');
const dagre = require('cytoscape-dagre');
cytoscape.use(dagre);
const nodeHtmlLabel = require('cytoscape-node-html-label');
nodeHtmlLabel(cytoscape);

@Component({
  selector: 'app-graph',
  templateUrl: './graph.component.html',
  styleUrls: ['./graph.component.scss'],
  encapsulation: ViewEncapsulation.None
})
export class GraphComponent implements OnChanges, OnInit, OnDestroy {

  @Input() modelPath: string;
  @Input() showOps?: boolean;
  @Input() diagnosisTabParams?: { Pattern: { sequence: string[] } };
  @Output() showNodeDetails = new EventEmitter<Node>();

  cy;
  nav;

  edges: Edge[] = [];
  nodes: Node[] = [];
  expandedNodesArray = [];
  showSpinner = false;

  //bright
  customColor = [
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

  constructor(
    @Optional() @Inject(MAT_DIALOG_DATA) public data,
    private modelService: ModelService
  ) { }

  ngOnInit() {
    this.showSpinner = true;
    this.getGraph();
    this.modelService.projectChanged$
      .subscribe((response: { id: number; tab: string }) => {
        this.showSpinner = true;
        setTimeout(() => {
          this.getGraph();
        }, 500);
      });
  }

  cytoGraph(elements) {
    this.cy = cytoscape({

      container: document.getElementById('cy'),
      elements,
      style: [
        {
          selector: 'node',
          style: {
            'background-color': 'data(color)',
            'border-color': 'data(border_color)',
            'border-width': '3px',
            color: '#fff',
            label: 'data(label)',
            shape: 'round-rectangle',
            'text-valign': 'center',
            'text-halign': 'center',
            width: (node: any) => node.data('label').length * 10,
          }
        },
        {
          selector: 'edge',
          style: {
            'font-size': '10px',
            'source-text-offset': '10px',
            'target-text-offset': '10px',
            width: 3,
            'line-color': '#ccc',
            'target-arrow-color': '#ccc',
            'target-arrow-shape': 'triangle',
            'curve-style': 'taxi',
          }
        },
        {
          selector: 'node',
          css: {
            'font-family': 'IntelClearRg',
          }
        },
        {
          selector: 'node.selected',
          css: {
            'border-color': '#00c7fd'
          }
        },
        {
          selector: 'node.hover',
          css: {
            'border-color': '#B1BABF',
            'border-style': 'dashed',
          }
        },
        {
          selector: 'node[node_type = \'group_node\']',
          css: {
            color: 'black'
          }
        },
        {
          selector: 'node[highlight = \'true\']',
          css: {
            'border-color': '#FEC91B'
          }
        }
      ],
      layout: {
        name: 'dagre',
        padding: 24,
        spacingFactor: 1.5,
        animate: true
      },
    });

    this.cy.nodeHtmlLabel([
      {
        query: 'node[node_type = "group_node"]',
        halign: 'right',
        valign: 'bottom',
        cssClass: 'plus-sign',
        tpl: (data) => '<div>&#65291;</div>'
      }
    ]);

    this.cy.on('click', (event) => {
      if (event.target._private.data.node_type === 'group_node') {
        this.expand(event.target._private.data.id);
      }
    });

    this.cy.on('mouseover', 'node', e => {
      e.target.addClass('hover');
    });

    this.cy.on('mouseout', 'node', e => {
      e.target.removeClass('hover');
    });

    this.cy.on('tap', 'node', e => {
      this.cy.elements('node:selected').removeClass('selected');
      if (e.target._private.data.node_type === 'node') {
        e.target.addClass('selected');
        this.showNodeDetails.next(e.target._private.data);
      }
    });

    setTimeout(() => {
      if (this.cy.elements('node[highlight = \'true\']').length) {
        this.cy.reset();
        this.cy.center(this.cy.elements('node[highlight = \'true\']')[0]);
      } else {
        this.cy.zoom({
          level: 2.0
        });
        this.cy.center();
      }
    }, 1000);

    this.showSpinner = false;
  }

  ngOnDestroy(): void {
    this.cy.destroy();
  }

  ngOnChanges(): void {
    if (this.showOps) {
      setTimeout(() => {
        if (this.cy) {
          this.cy.fit();
          this.cy.zoom();
        }
      }, 1000);
      if (this.diagnosisTabParams?.Pattern) {
        this.highlightPatternInGraph();
      }
    }
  }

  highlightPatternInGraph() {
    this.modelService.highlightPatternInGraph(
      this.modelPath,
      this.diagnosisTabParams['OP name'],
      this.diagnosisTabParams.Pattern.sequence
    )
      .subscribe(
        (response: { graph: any; groups: any }) => {
          this.updateGraph(response.graph);
          this.expandedNodesArray = response.groups;
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
    this.cy.center();
  }

  zoomToFit() {
    this.cy.fit();
  }

  reset() {
    this.cy.reset();
  }

  updateGraph(graph: any) {
    const elements = [];
    const nodes = [];
    const edges = [];
    graph.nodes.forEach(node => {
      nodes.push({
        data: {
          id: node.id,
          label: this.getLabel(node.label),
          parent: node.parent,
          attributes: node.attributes,
          properties: node.properties,
          node_type: node.node_type,
          highlight: String(node.highlight),
          border_color: node.node_type === 'group_node' ? '#5B69FF' : this.customColor[node.label.length % this.customColor.length],
          color: node.node_type === 'group_node' ? '#fff' : this.customColor[node.label.length % this.customColor.length],
        },
        grabbable: false,
      });
    });
    graph.edges.forEach(edge => {
      edges.push({
        data: {
          source: edge.source,
          target: edge.target,
          // source_label: edge.source_label,
          // target_label: edge.target_label,
        }
      });
    });
    this.nodes = nodes;
    this.edges = edges;
    elements.push(... this.nodes);
    elements.push(... this.edges);
    this.cytoGraph(elements);
  }

  getLabel(label: string) {
    if (label.includes('/')) {
      return label.replace(/^.*[\\\/]/, '');
    } else {
      return label;
    }
  }

  expand(nodeId: string) {
    this.showSpinner = true;
    if (this.expandedNodesArray.includes(nodeId)) {
      this.collapse(nodeId);
    } else {
      this.expandedNodesArray.push(nodeId);
    }
    this.modelService.getModelGraph(this.modelPath, this.expandedNodesArray)
      .subscribe(
        graph => { this.updateGraph(graph); },
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
  };
  data?: {
    color: string;
  };
  node_type: string;
  color: string;
}

interface Edge {
  target: string;
  source: string;
}
