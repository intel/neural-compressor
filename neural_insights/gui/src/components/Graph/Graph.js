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
import './Graph.scss';
import { getLabel, customColor } from './../Diagnosis/Diagnosis';
import Button from 'react-bootstrap/Button';
import { api } from './../../App';

const cytoscape = require('cytoscape');
const nodeHtmlLabel = require('cytoscape-node-html-label');
cytoscape.use(nodeHtmlLabel);
const elk = require('cytoscape-elk');
cytoscape.use(elk);

export default function Graph({ setSelectedNode, selectedWorkload, selectedOp, selectedPattern, setWarningText }) {
  const [graph, setGraph] = useState(null);
  const [groupNode, setGroupNode] = useState([]);
  const groupNodeOpList = [];

  useEffect(() => {
    if (selectedWorkload) {
      const payload = {
        workload_id: selectedWorkload.uuid,
        path: [selectedWorkload.model_path],
        ...((groupNode.length || groupNodeOpList.length) && { group: [...groupNode, ...groupNodeOpList] })
      };
      api.post('api/model/graph?token=' + localStorage.getItem('token'), payload)
        .then(
          response => {
            setGraph(response.data);
          })
        .catch(error => {
          setWarningText(error.message);
        });
    }
  }, [selectedWorkload, groupNode]);

  useEffect(() => {
    if (selectedOp) {
      api.post('api/model/graph/highlight_pattern?token=' + localStorage.getItem('token'), {
        workload_id: selectedWorkload.uuid,
        path: [selectedWorkload.model_path],
        op_name: selectedOp,
        pattern: selectedPattern,
        ...((groupNode.length || groupNodeOpList.length) && { group: [...groupNode, ...groupNodeOpList] })
      })
        .then(
          response => {
            setGraph(response.data.graph);
            groupNodeOpList.push(...response.data.groups);
          })
        .catch(error => {
          if (error.response.status !== 400) {
            setWarningText(error.message);
          }
        });
    }
  }, [selectedPattern]);

  return (
    <div className="Graph">
      <CytoGraph setSelectedNode={setSelectedNode} setGroupNode={setGroupNode} groupNode={groupNode} graph={graph} />
    </div>
  )
};

class CytoGraph extends React.Component {
  constructor(props) {
    super(props);
    this.renderCytoscapeElement = this.renderCytoscapeElement.bind(this);
  }

  handleCallback = (childData) => {
    switch (childData) {
      case 'fit':
        this.cy.fit();
        break;
      case 'center':
        this.cy.center();
        break;
      case 'reset':
        this.cy.reset();
        break;
      default:
        break;
    }
  }

  renderCytoscapeElement() {
    const elements = getElements(this.props.graph);
    this.cy =
      cytoscape({
        container: document.getElementById('cy'),
        elements,
        style: [{
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
            width: (node) => node.data('label').length * 12,
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
          name: 'elk',
          animate: true,
          elk: {
            'algorithm': 'layered',
            'elk.direction': 'DOWN'
          }
        }
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
        this.props.setGroupNode([...this.props.groupNode, event.target._private.data.id]);
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
        this.props.setSelectedNode(e.target._private.data);
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
  }

  componentDidUpdate(prevProps) {
    if (prevProps.graph !== this.props.graph) {
      this.renderCytoscapeElement();
    }
  }

  componentDidMount() {
    if (this.props.graph) {
      this.renderCytoscapeElement();
    }
  }

  collapseNode(nodeName) {
    const newExpandedNodes = this.props.groupNode.filter(x => x !== nodeName);
    this.props.setGroupNode(newExpandedNodes);
  }

  render() {
    return (
      <div>
        <div className="graph-buttons">
          <GraphButtons parentCallback={this.handleCallback} />
          {this.props.groupNode.length > 0 &&
            <div className="nodes-table-container">
              <table className="nodes-table">
                <tbody>
                  <tr>
                    <td className="header">Expanded groups</td>
                    <td></td>
                  </tr>
                  {this.props.groupNode.map(groupNode => {
                    return (
                      <tr key={groupNode}>
                        <td>{groupNode.replace('node_group_', '')}</td>
                        <td onClick={() => this.collapseNode(groupNode)} className="clickable" tooltip="Collapse this group">&#x2715;</td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          }
        </div>
        <div id="cy">
        </div>
      </div>
    )
  }
}

class GraphButtons extends React.Component {
  onTrigger = (event) => {
    this.props.parentCallback(event.target.id);
    event.preventDefault();
  }

  render() {
    return (
      <div>
        <Button variant="primary" id="fit" className="graph-button" onClick={this.onTrigger}>Fit</Button>
        <Button variant="primary" id="center" className="graph-button" onClick={this.onTrigger}>Center</Button>
        <Button variant="primary" id="reset" className="graph-button" onClick={this.onTrigger}>Reset</Button>
      </div>
    )
  }
}

const getElements = (graph) => {
  const elements = [];
  if (graph.nodes && graph.edges) {
    graph.nodes.forEach(node => {
      elements.push({
        data: {
          id: node.id,
          label: getLabel(node.label),
          parent: node.parent,
          attributes: node.attributes,
          properties: node.properties,
          node_type: node.node_type,
          highlight: String(node.highlight),
          border_color: node.node_type === 'group_node' ? '#5B69FF' : customColor[getHash(node.label)],
          color: node.node_type === 'group_node' ? '#fff' : customColor[getHash(node.label)],
        },
        grabbable: false,
      });
    });
    graph.edges.forEach(edge => {
      elements.push({
        data: {
          source: edge.source,
          target: edge.target,
        }
      });
    });
  }
  return elements;
}

function getHash(input) {
  var hash = 0, len = input.length;
  for (var i = 0; i < len; i++) {
    hash = ((hash << 5) - hash) + input.charCodeAt(i);
    hash |= 0;
  }
  hash = Math.abs(hash);
  return hash % customColor.length;
}

