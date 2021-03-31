import sys
import onnx
from onnx import helper, TensorProto, numpy_helper
import unittest
import numpy as np

sys.path.append('..')
from lpot.adaptor.ox_utils.onnx_model import ONNXModel

def get_onnx_model():
    model = torchvision.models.resnet18()
    x = Variable(torch.randn(1, 3, 224, 224))
    torch_out = torch.onnx.export(model, x, "resnet18.onnx", export_params=True, verbose=True)

def generate_input_initializer(tensor_shape, tensor_dtype, input_name):
    '''
    Helper function to generate initializers for test inputs
    '''
    tensor = np.random.ranf(tensor_shape).astype(tensor_dtype)
    init = numpy_helper.from_array(tensor, input_name)
    return init  


class TestOnnxModel(unittest.TestCase):
    def setUp(self):
        #   Relu      
        #    |      \ 
        #   Conv     \
        #    |        \ 
        #   Relu       |  
        #    |       Conv  
        #   Conv      / 
        #      \     /  
        #         |
        #        Add
    
        input0 = helper.make_tensor_value_info('input0', TensorProto.FLOAT, [1, 3, 1, 3])
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 3, 1, 3])
        
        X1_weight = generate_input_initializer([3, 3, 1, 1], np.float32, 'X1_weight')
        X1_bias = generate_input_initializer([3], np.float32, 'X1_bias')
        X3_weight = generate_input_initializer([3, 3, 1, 1], np.float32, 'X3_weight')
        X3_bias = generate_input_initializer([3],np.float32, 'X3_bias')
        X5_weight = generate_input_initializer([3, 3, 1, 1], np.float32, 'X5_weight')
        X5_bias = generate_input_initializer([3],np.float32,'X5_bias')
       
        relu_node_1 = onnx.helper.make_node('Relu', ['input0'], ['X1'], name='Relu1')
        conv_node_1 = onnx.helper.make_node('Conv', ['X1', 'X1_weight', 'X1_bias'], ['X2'], name='Conv1')
        relu_node_2 = onnx.helper.make_node('Relu', ['X2'], ['X3'], name= 'Relu2')
        conv_node_2 = onnx.helper.make_node('Conv', ['X3', 'X3_weight', 'X3_bias'], ['X4'], name='Conv2')
        conv_node_3 = onnx.helper.make_node('Conv', ['X1', 'X5_weight', 'X5_bias'], ['X5'], name='Conv3')
        add_node = onnx.helper.make_node('Add', ['X4', 'X5'], ['output'], name='Add')
      
        graph = helper.make_graph([relu_node_1, conv_node_1, relu_node_2, conv_node_2, conv_node_3, add_node], 'test_graph_6', [input0], [output])
        graph.initializer.add().CopyFrom(X1_weight)
        graph.initializer.add().CopyFrom(X1_bias)
        graph.initializer.add().CopyFrom(X3_weight)
        graph.initializer.add().CopyFrom(X3_bias)
        graph.initializer.add().CopyFrom(X5_weight)
        graph.initializer.add().CopyFrom(X5_bias)

        model = helper.make_model(graph)
        test_model_path = './test_model_6.onnx'
        onnx.save(model, test_model_path)
        model = onnx.load(test_model_path)
        self.model = ONNXModel(model)

    def test_nodes(self):
        self.assertEqual(len(self.model.nodes()), 6)
        nodes_name = [node.name for node in self.model.nodes()]
        nodes = ["Relu1", "Conv1", "Relu2", "Conv2", "Conv3", "Add"]
        for node in nodes:
            self.assertTrue(node in nodes_name)

    def test_initializer(self):
        self.assertEqual(len(self.model.initializer()), 6)
        inits_name = [init.name for init in self.model.initializer()]
        inits = ['X1_weight', 'X1_bias', 'X3_weight', 'X3_bias', 'X5_weight', 'X5_bias']
        for init in inits:
            self.assertTrue(init in inits_name)
        

    def test_remove_node(self):
        for node in self.model.nodes():
            if node.op_type == "Add":
                self.model.remove_node(node)
        self.assertEqual(len(self.model.nodes()), 5)
        nodes_name = [node.name for node in self.model.nodes()]
        nodes = ["Relu1", "Conv1", "Relu2", "Conv2", "Conv3"]
        for node in nodes:
            self.assertTrue(node in nodes_name)

    def test_remove_nodes(self):
        nodes_to_remove = []
        for node in self.model.nodes():
            if node.name == "Conv3" or node.name == "Add":
                nodes_to_remove.append(node)
        self.model.remove_nodes(nodes_to_remove)
        self.assertEqual(len(self.model.nodes()), 4)
        nodes_name = [node.name for node in self.model.nodes()]
        nodes = ["Relu1", "Conv1", "Relu2", "Conv2"]
        for node in nodes:
            self.assertTrue(node in nodes_name)

    def test_add_node(self):
        node_to_add = onnx.helper.make_node('Relu', ['output'], ['output1'], keepdims=0)
        self.model.add_node(node_to_add)
        last_node = self.model.nodes()[-1]
        self.assertEqual(last_node.op_type, 'Relu')

    def test_add_nodes(self):
        nodes_to_add = []
        for i in range(2):
            node_to_add = onnx.helper.make_node('Relu', ["add_node{}_input".format(str(i))], ["add_node{}_output".format(str(i))], keepdims=0)
            nodes_to_add.append(node_to_add)
        self.model.add_nodes(nodes_to_add)
        self.assertEqual(self.model.nodes()[-1].input, ['add_node1_input'])
        self.assertEqual(self.model.nodes()[-2].input, ['add_node0_input'])
        self.assertEqual(self.model.nodes()[-1].output, ['add_node1_output'])
        self.assertEqual(self.model.nodes()[-2].output, ['add_node0_output'])

    def test_get_initializer(self):
        inits = ['X1_weight', 'X1_bias', 'X3_weight', 'X3_bias', 'X5_weight', 'X5_bias']
        for init in inits:
            self.assertIsNotNone(self.model.get_initializer(init))

    def test_remove_initializer(self):
        for init in self.model.initializer():
            if init.name == "X1_weight":
                self.model.remove_initializer(init)
        self.assertEqual(len(self.model.initializer()), 5)
        inits_name = [init.name for init in self.model.initializer()]
        inits = ['X1_bias', 'X3_weight', 'X3_bias', 'X5_weight', 'X5_bias']
        for init in inits:
            self.assertTrue(init in inits_name)

    def test_remove_initializers(self):
        init_to_remove = []
        for init in self.model.initializer():
            if "bias" in init.name:
                init_to_remove.append(init)
        self.model.remove_initializers(init_to_remove)
        self.assertEqual(len(self.model.initializer()), 3)
        inits_name = [init.name for init in self.model.initializer()]
        inits = ['X1_weight', 'X3_weight', 'X5_weight']
        for init in inits:
            self.assertTrue(init in inits_name)

    def test_input_name_to_nodes(self):
        self.assertEqual(len(self.model.input_name_to_nodes()), 12)
        ipts_name = [name for name in self.model.input_name_to_nodes()]
        ipts = ['input0', 'X1',  'X2', 'X3', 'X3_weight', 'X3_bias','X5_weight', 'X5_bias', 'X4', 'X5']
        for ipt in ipts:
            self.assertTrue(ipt in ipts_name)

    def test_output_name_to_node(self):
        self.assertEqual(len(self.model.output_name_to_node()), 6)
        opts_name = [name for name in self.model.output_name_to_node()]
        opts = ['X1', 'X2', 'X3', 'X4', 'X5', 'output']
        for opt in opts:
            self.assertTrue(opt in opts_name)
    
    def test_get_children(self):
        for node in self.model.nodes():
            if node.name == "Relu1":
                children = self.model.get_children(node)
        self.assertEqual(len(children), 2)
        children_name = [child.name for child in children]
        names = ["Conv1", "Conv3"]
        for name in names:
            self.assertTrue(name in children_name)

    def test_get_parents(self):
        for node in self.model.nodes():
            if node.op_type == "Add":
                parents = self.model.get_parents(node)
        self.assertEqual(len(parents), 2)
        parents_name = [parent.name for parent in parents]
        names = ["Conv2", "Conv3"]
        for name in names:
            self.assertTrue(name in parents_name)

    def test_get_parent(self):
        for node in self.model.nodes():
            if node.op_type == "Add":
                node_to_get_parent = node
        parent = self.model.get_parent(node, 0)
        self.assertEqual(parent.name, "Conv2")
        parent = self.model.get_parent(node, 1)
        self.assertEqual(parent.name, "Conv3")
        parent = self.model.get_parent(node, 2)
        self.assertIsNone(parent)

    def test_find_nodes_by_initializer(self):
        for init in self.model.initializer():
            if init.name == "X1_weight":
                initializer = init
        nodes = self.model.find_nodes_by_initializer(self.model.graph(), initializer)
        self.assertEqual(len(nodes), 1)
        self.assertEqual(nodes[0].name, "Conv1")

    def test_save(self):
        self.model.save_model_to_file('./test_model_6.onnx', use_external_data_format=True)

    
if __name__ == "__main__":
    unittest.main()
