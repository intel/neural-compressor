import os
import sys
import argparse
import tensorflow as tf
from tensorflow.core.framework import graph_pb2

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('pbfile')
    return parser.parse_args()

def insert_transpose(graph, a, b, to_nchw):
    if not isinstance(b, list):
        b = [b]
    trans_perm = graph.node.add()
    trans_perm.name = a.name + '/transpose/perm'
    trans_perm.op = 'Const'
    trans_perm.attr['dtype'].type = 3 # DT_INT32
    trans_perm.attr['value'].tensor.dtype = 3 # DT_INT32
    trans_perm.attr['value'].tensor.tensor_shape.dim.add()
    trans_perm.attr['value'].tensor.tensor_shape.dim[0].size = 4
    if to_nchw:
        trans_perm.attr['value'].tensor.tensor_content = b'\000\000\000\000\003\000\000\000\001\000\000\000\002\000\000\000'
    else:
        trans_perm.attr['value'].tensor.tensor_content = b'\000\000\000\000\002\000\000\000\003\000\000\000\001\000\000\000'
    
    trans = graph.node.add()
    trans.name = a.name + '/transpose'
    trans.op = 'Transpose'
    trans.input.append(a.name)
    trans.input.append(trans_perm.name)
    trans.attr['T'].type = 1
    trans.attr['Tperm'].type = 3

    for n in b:
        inputs = []
        for i in n.input:
            if i == a.name:
                inputs.append(trans.name)
            else:
                inputs.append(i)
        cnt = len(n.input)
        for i in range(0, cnt):
            del n.input[0]
        for i in range(0, cnt):
            n.input.append(inputs[i])

def convert_list_nhwc(l):
    c = l.i[1]
    h = l.i[2]
    w = l.i[3]
    l.i[1] = h
    l.i[2] = w
    l.i[3] = c
    
def convert_conv_nhwc(node_conv):
    node_conv.attr['data_format'].s = b'NHWC'
    convert_list_nhwc(node_conv.attr['dilations'].list)
    convert_list_nhwc(node_conv.attr['strides'].list)

def convert_general_nhwc(node):
    node.attr['data_format'].s = b'NHWC'

def convert_mp_nhwc(node_mp):
    node_mp.attr['data_format'].s = b'NHWC'
    convert_list_nhwc(node_mp.attr['ksize'].list)
    convert_list_nhwc(node_mp.attr['strides'].list)

def convert_image_nhwc(node_image):
    c = node_image.attr['shape'].shape.dim[1].size
    del node_image.attr['shape'].shape.dim[1]
    d = node_image.attr['shape'].shape.dim.add()
    d.size = c

def init_node(n):
    node = {}
    node['node'] = n
    node['inputs'] = []
    node['outputs'] = []
    return node

def connect_nodes(n1, n2):
    if n2['node'].name not in n1['outputs']:
        n1['outputs'].append(n2['node'].name)
        n2['inputs'].append(n1['node'].name)
    else:
        print('{} -> {} already connected'.format(n1['node'].name, n2['node'].name))

def disconnect_nodes(n1, n2):
    if n1['node'].name not in n2['inputs'] or n2['node'].name not in n1['outputs']:
        print('{} -> {} not connected'.format(n1['node'].name, n2['node'].name))
    for i in range(0, len(n1['outputs'])):
        if n1['outputs'][i] == n2['node'].name:
            del n1['outputs'][i]
            break
    for i in range(0, len(n2['inputs'])):
        if n2['inputs'][i] == n1['node'].name:
            del n2['inputs'][i]
            break
            
def build_graph(graph):
    node_map = {}
    for n in graph.node:
        node = init_node(n)
        node_map[n.name] = node
    for n in node_map:
        for i in node_map[n]['node'].input:
            if ':' in i:
                i = i[:i.find(':')]
            i = i.lstrip('^')
            if i not in node_map:
                print('node {} not found'.format(i))
            else:
                connect_nodes(node_map[i], node_map[n])
    return node_map

def trim_const_from_graph(node_map):
    trim_list = []
    for n in node_map:
        if node_map[n]['node'].op == 'Const':
            trim_list.append(n)
    for n in trim_list:
        print('trimming {}'.format(n))
        for o in node_map[n]['outputs']:
            disconnect_nodes(node_map[n], node_map[o])
        del node_map[n]

    trim_list = []
    for n in node_map:
        if node_map[n]['node'].op == 'Identity' and len(node_map[n]['inputs']) == 0:
            trim_list.append(n)
    for n in trim_list:
        print('trimming {}'.format(n))
        for o in node_map[n]['outputs']:
            disconnect_nodes(node_map[n], node_map[o])
        del node_map[n]


def all_input_in_nhwc(n, node_map, nhwc_nodes):
    for i in node_map[n]['inputs']:
        if i not in nhwc_nodes:
            return False
    return True

def all_output_in_nhwc(n, node_map, nhwc_nodes):
    for o in node_map[n]['outputs']:
        if o not in nhwc_nodes:
            return False
    return True

def find_nhwc_region(node_map):
    transpose_nhwc_nodes = {}
    transpose_nchw_nodes = {}
    nhwc_nodes = []

    transpose_nhwc_nodes_append_list = []
    transpose_nchw_nodes_append_list = []
    for n in node_map:
        if node_map[n]['node'].op == 'Conv2D':
            transpose_nhwc_nodes_append_list.append(n)
            transpose_nchw_nodes_append_list.append(n)
            nhwc_nodes.append(n)
    for n in transpose_nhwc_nodes_append_list:
        if not all_input_in_nhwc(n, node_map, nhwc_nodes):
            transpose_nhwc_nodes[n] = 1
    for n in transpose_nchw_nodes_append_list:
        if not all_output_in_nhwc(n, node_map, nhwc_nodes):
            transpose_nchw_nodes[n] = 1

    prev_cnt_nhwc_nodes = len(nhwc_nodes)
    nhwc_op_list = ['Conv2D', 'Relu', 'FusedBatchNorm', 'MaxPool', 'BiasAdd', 'Add']
    while True:
        transpose_nchw_nodes_append_list = []
        for n in transpose_nchw_nodes:
            for o in node_map[n]['outputs']:
                if o not in nhwc_nodes and node_map[o]['node'].op in nhwc_op_list:
                    if all_input_in_nhwc(o, node_map, nhwc_nodes):
                        nhwc_nodes.append(o)
                        if o not in transpose_nchw_nodes_append_list:
                            transpose_nchw_nodes_append_list.append(o)

        transpose_nhwc_nodes_remove_list = []
        transpose_nchw_nodes_remove_list = []
        for n in transpose_nhwc_nodes:
            if (all_input_in_nhwc(n, node_map, nhwc_nodes) and
                n not in transpose_nhwc_nodes_remove_list):
                transpose_nhwc_nodes_remove_list.append(n)
        for n in transpose_nhwc_nodes_remove_list:
            del transpose_nhwc_nodes[n]

        for n in transpose_nchw_nodes:
            if (all_output_in_nhwc(n, node_map, nhwc_nodes) and
                n not in transpose_nchw_nodes_remove_list):
                transpose_nchw_nodes_remove_list.append(n)
        for n in transpose_nchw_nodes_remove_list:
            del transpose_nchw_nodes[n]

        for n in transpose_nchw_nodes_append_list:
            if not all_output_in_nhwc(n, node_map, nhwc_nodes):
                transpose_nchw_nodes[n] = 1

        if len(nhwc_nodes) == prev_cnt_nhwc_nodes:
            break
        prev_cnt_nhwc_nodes = len(nhwc_nodes)
                        
    print('\n\nTranspose to NHWC at nodes:')
    for n in transpose_nhwc_nodes:
        print('    {}'.format(n))
    
    print('\n\nTranspose to NCHW at nodes:')
    for n in transpose_nchw_nodes:
        print('    {}'.format(n))
    
    return nhwc_nodes, transpose_nhwc_nodes, transpose_nchw_nodes

def main():
    args = get_args()

    graph = graph_pb2.GraphDef()
    with open(args.pbfile, 'rb') as f:
        graph.ParseFromString(f.read())

    node_map = build_graph(graph)
    trim_const_from_graph(node_map)

    nhwc_nodes, transpose_nhwc_nodes, transpose_nchw_nodes = find_nhwc_region(node_map)

    nhwc_op_list = ['Conv2D', 'Relu', 'FusedBatchNorm', 'MaxPool', 'BiasAdd', 'Add']
    for n in nhwc_nodes:
        if node_map[n]['node'].op == 'Conv2D':
            convert_conv_nhwc(node_map[n]['node'])
        elif node_map[n]['node'].op in ['FusedBatchNorm', 'BiasAdd']:
            convert_general_nhwc(node_map[n]['node'])
        elif node_map[n]['node'].op == 'MaxPool':
            convert_mp_nhwc(node_map[n]['node'])
      
    done_nhwc = False
    if len(transpose_nhwc_nodes) == 1:
        for n in transpose_nhwc_nodes:
            if len(node_map[n]['inputs']) == 1 and node_map[n]['inputs'][0] == 'image':
                image_outputs = []
                for o in node_map['image']['outputs']:
                    if o != n:
                        image_outputs.append(node_map[o]['node'])
                insert_transpose(graph, node_map['image']['node'], image_outputs, True)
                convert_image_nhwc(node_map['image']['node'])
                done_nhwc = True

    if not done_nhwc:
        for n in transpose_nhwc_nodes:
            for i in node_map[n]['inputs']:
                if i not in nhwc_nodes:
                    insert_transpose(graph, node_map[i]['node'], node_map[n]['node'], False)

    for n in transpose_nchw_nodes:
        node_outputs = []
        for o in node_map[n]['outputs']:
            if o not in nhwc_nodes:
                node_outputs.append(node_map[o]['node'])
        insert_transpose(graph, node_map[n]['node'], node_outputs, True)

    with open(args.pbfile+'.patch', 'wb') as f:
        f.write(graph.SerializeToString())

if __name__ == '__main__':
    main()

