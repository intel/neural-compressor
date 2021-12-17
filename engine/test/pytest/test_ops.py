import unittest
from tensorflow.core.framework import node_def_pb2
from tensorflow.core.framework import attr_value_pb2
from onnx import NodeProto
from onnx.helper import make_attribute
from collections import namedtuple
from engine.compile.ops.op import OPERATORS, Operator
from engine.compile.ops.tensor import Tensor


class TestOps(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        pass

    @classmethod
    def tearDownClass(self):
        pass
    
    def test_all(self):
        all_node = node_def_pb2.NodeDef()
        all_node.name = 'all'
        all_node.op = 'All'
        all_node.attr['keep_dims'].CopyFrom(attr_value_pb2.AttrValue(b=False))

        # all_node_test = ops.all.All()
        all_node_test = OPERATORS['All']()
        all_node_test.set_attr('tensorflow', all_node)
        keep_dims = all_node_test.attr['keep_dims']
        self.assertFalse(keep_dims)

    def test_assert(self):
        assert_node = node_def_pb2.NodeDef()
        assert_node.name = 'assert'
        assert_node.op = 'Assert'
        assert_node.attr['summarize'].CopyFrom(attr_value_pb2.AttrValue(i=3))

        assert_node_test = OPERATORS['Assert']()
        assert_node_test.set_attr('tensorflow', assert_node)
        summarize = assert_node_test.attr['summarize']
        self.assertEqual(3, summarize)
    
    def test_batch_matmul(self):
        batch_matmul_node = node_def_pb2.NodeDef()
        batch_matmul_node.name = 'batch_matmul'
        batch_matmul_node.op = 'BatchMatMul'
        batch_matmul_node.attr['adj_x'].CopyFrom(attr_value_pb2.AttrValue(b=False))
        batch_matmul_node.attr['adj_y'].CopyFrom(attr_value_pb2.AttrValue(b=True))

        batch_matmul_node_test = OPERATORS['BatchMatMul']()
        batch_matmul_node_test.set_attr('tensorflow', batch_matmul_node)
        transpose_a = batch_matmul_node_test.attr['transpose_a']
        transpose_b = batch_matmul_node_test.attr['transpose_b']
        self.assertFalse(transpose_a)
        self.assertTrue(transpose_b)
    
    def test_cast(self):
        cast_node = NodeProto()
        cast_node.name =  'cast'
        cast_node.op_type = 'Cast'
        cast_node.attribute.append(make_attribute('to', 1))

        cast_node_test = OPERATORS['Cast']()
        cast_node_test.set_attr('onnxruntime', cast_node)
        dst_dtype = cast_node_test.attr['DstT']
        self.assertEqual('fp32', dst_dtype)
    
    def test_concat(self):
        concat_node = NodeProto()
        concat_node.name =  'concat'
        concat_node.op_type = 'Concat'
        concat_node.attribute.append(make_attribute('axis', [1, 2]))

        concat_node_test = OPERATORS['Concat']()
        concat_node_test.set_attr('onnxruntime', concat_node)
        axis = concat_node_test.attr['axis']
        self.assertSequenceEqual([1, 2], axis)
    
    def test_add(self):
        add_node = NodeProto()
        add_node.name = 'add'
        add_node.op_type = 'Add'

        add_node_test = OPERATORS['Add']()
        add_node_test.name = add_node.name
        name = add_node_test.name
        self.assertEqual('add', name)
    
    def test_constant_of_shape(self):
        constant_of_shape_node = NodeProto()
        constant_of_shape_node.name = 'constant_of_shape'
        constant_of_shape_node.op_type = 'ConstantOfShape'

        constant_of_shape_node_test = OPERATORS['ConstantOfShape']()
        constant_of_shape_node_test.name = constant_of_shape_node.name
        name = constant_of_shape_node_test.name
        self.assertEqual('constant_of_shape', name)

    def test_dequantize_linear(self):
        dequantize_linear_node = NodeProto()
        dequantize_linear_node.name = 'dequantize_linear'
        dequantize_linear_node.op_type = 'DequantizeLinear'

        dequantize_linear_node_test = OPERATORS['DequantizeLinear']()
        dequantize_linear_node_test.name = dequantize_linear_node.name
        name = dequantize_linear_node_test.name
        self.assertEqual('dequantize_linear', name)
    
    def test_div(self):
        div_node = NodeProto()
        div_node.name = 'div'
        div_node.op_type = 'Div'

        div_node_test = OPERATORS['Div']()
        div_node_test.name = div_node.name
        name = div_node_test.name
        self.assertEqual('div', name)
    
    def test_equal(self):
        equal_node = NodeProto()
        equal_node.name = 'equal'
        equal_node.op_type = 'Equal'

        equal_node_test = OPERATORS['Equal']()
        equal_node_test.name = equal_node.name
        name = equal_node_test.name
        self.assertEqual('equal', name)
    
    def test_expand(self):
        expand_node = NodeProto()
        expand_node.name = 'expand'
        expand_node.op_type = 'Expand'

        expand_node_test = OPERATORS['Expand']()
        expand_node_test.name = expand_node.name
        name = expand_node_test.name
        self.assertEqual('expand', name)
    
    def test_non_zero(self):
        non_zero_node = NodeProto()
        non_zero_node.name = 'non_zero'
        non_zero_node.op_type = 'NonZero'

        non_zero_node_test = OPERATORS['NonZero']()
        non_zero_node_test.name = non_zero_node.name
        name = non_zero_node_test.name
        self.assertEqual('non_zero', name)
    
    def test_qlinear_matmul(self):
        qlinear_matmul_node = NodeProto()
        qlinear_matmul_node.name = 'qlinear_matmul'
        qlinear_matmul_node.op_type = 'QLinearMatMul'

        qlinear_matmul_node_test = OPERATORS['QLinearMatMul']()
        qlinear_matmul_node_test.name = qlinear_matmul_node.name
        name = qlinear_matmul_node_test.name
        self.assertEqual('qlinear_matmul', name)
    
    def test_qlinear_add(self):
        qlinear_add_node = NodeProto()
        qlinear_add_node.name = 'qlinear_add'
        qlinear_add_node.op_type = 'QLinearAdd'

        qlinear_add_node_test = OPERATORS['QLinearAdd']()
        qlinear_add_node_test.name = qlinear_add_node.name
        name = qlinear_add_node_test.name
        self.assertEqual('qlinear_add', name)
    
    def test_qlinear_mul(self):
        qlinear_mul_node = NodeProto()
        qlinear_mul_node.name = 'qlinear_mul'
        qlinear_mul_node.op_type = 'QLinearMul'

        qlinear_mul_node_test = OPERATORS['QLinearMul']()
        qlinear_mul_node_test.name = qlinear_mul_node.name
        name = qlinear_mul_node_test.name
        self.assertEqual('qlinear_mul', name)
    
    def test_where(self):
        where_node = NodeProto()
        where_node.name = 'where'
        where_node.op_type = 'Where'

        where_node_test = OPERATORS['Where']()
        where_node_test.name = where_node.name
        name = where_node_test.name
        self.assertEqual('where', name)

    def test_erf(self):
        erf_node = node_def_pb2.NodeDef()
        erf_node.name = 'erf'
        erf_node.op = 'Erf'

        erf_node_test = OPERATORS['Erf']()
        erf_node_test.name = erf_node.name
        name = erf_node_test.name
        self.assertEqual('erf', name)
    
    def test_fill(self):
        fill_node = node_def_pb2.NodeDef()
        fill_node.name = 'fill'
        fill_node.op = 'Fill'

        fill_node_test = OPERATORS['Fill']()
        fill_node_test.name = fill_node.name
        name = fill_node_test.name
        self.assertEqual('fill', name)
    
    def test_flat_map_dataset(self):
        flat_map_dataset_node = node_def_pb2.NodeDef()
        flat_map_dataset_node.name = 'flat_map_dataset'
        flat_map_dataset_node.op = 'FlatMapDataset'

        flat_map_dataset_node_test = OPERATORS['FlatMapDataset']()
        flat_map_dataset_node_test.name = flat_map_dataset_node.name
        name = flat_map_dataset_node_test.name
        self.assertEqual('flat_map_dataset', name)
    
    def test_identity(self):
        identity_node = node_def_pb2.NodeDef()
        identity_node.name = 'identity'
        identity_node.op = 'Identity'

        identity_node_test = OPERATORS['Identity']()
        identity_node_test.name = identity_node.name
        name = identity_node_test.name
        self.assertEqual('identity', name)
    
    def test_innerproduct(self):
        innerproduct_node = node_def_pb2.NodeDef()
        innerproduct_node.name = 'innerproduct'
        innerproduct_node.op = 'InnerProduct'

        innerproduct_node_test = OPERATORS['InnerProduct']()
        innerproduct_node_test.name = innerproduct_node.name
        name = innerproduct_node_test.name
        self.assertEqual('innerproduct', name)
    
    def test_less_equal(self):
        less_equal_node = node_def_pb2.NodeDef()
        less_equal_node.name = 'less_equal'
        less_equal_node.op = 'LessEqual'

        less_equal_node_test = OPERATORS['LessEqual']()
        less_equal_node_test.name = less_equal_node.name
        name = less_equal_node_test.name
        self.assertEqual('less_equal', name)
    
    def test_make_iterator(self):
        make_iterator_node = node_def_pb2.NodeDef()
        make_iterator_node.name = 'make_iterator'
        make_iterator_node.op = 'MakeIterator'

        make_iterator_node_test = OPERATORS['MakeIterator']()
        make_iterator_node_test.name = make_iterator_node.name
        name = make_iterator_node_test.name
        self.assertEqual('make_iterator', name)

    def test_matmul_with_bias_tanh(self):
        matmul_with_bias_tanh_node = node_def_pb2.NodeDef()
        matmul_with_bias_tanh_node.name = 'matmul_with_bias_tanh'
        matmul_with_bias_tanh_node.op = 'MatMulWithBiasTanh'

        matmul_with_bias_tanh_node_test = OPERATORS['MatMulWithBiasTanh']()
        matmul_with_bias_tanh_node_test.name = matmul_with_bias_tanh_node.name
        name = matmul_with_bias_tanh_node_test.name
        self.assertEqual('matmul_with_bias_tanh', name)
    
    def test_pow(self):
        pow_node = node_def_pb2.NodeDef()
        pow_node.name = 'pow'
        pow_node.op = 'Pow'

        pow_node_test = OPERATORS['Pow']()
        pow_node_test.name = pow_node.name
        name = pow_node_test.name
        self.assertEqual('pow', name)
    
    def test_real_div(self):
        real_div_node = node_def_pb2.NodeDef()
        real_div_node.name = 'real_div'
        real_div_node.op = 'RealDiv'

        real_div_node_test = OPERATORS['RealDiv']()
        real_div_node_test.name = real_div_node.name
        name = real_div_node_test.name
        self.assertEqual('real_div', name)
    
    def test_sqrt(self):
        sqrt_node = node_def_pb2.NodeDef()
        sqrt_node.name = 'sqrt'
        sqrt_node.op = 'Sqrt'

        sqrt_node_test = OPERATORS['Sqrt']()
        sqrt_node_test.name = sqrt_node.name
        name = sqrt_node_test.name
        self.assertEqual('sqrt', name)
    
    def test_square(self):
        square_node = node_def_pb2.NodeDef()
        square_node.name = 'square'
        square_node.op = 'Square'

        square_node_test = OPERATORS['Square']()
        square_node_test.name = square_node.name
        name = square_node_test.name
        self.assertEqual('square', name)
    
    def test_stop_gradient(self):
        stop_gradient_node = node_def_pb2.NodeDef()
        stop_gradient_node.name = 'stop_gradient'
        stop_gradient_node.op = 'StopGradient'

        stop_gradient_node_test = OPERATORS['StopGradient']()
        stop_gradient_node_test.name = stop_gradient_node.name
        name = stop_gradient_node_test.name
        self.assertEqual('stop_gradient', name)
    
    def test_tanh(self):
        tanh_node = node_def_pb2.NodeDef()
        tanh_node.name = 'tanh'
        tanh_node.op = 'Tanh'

        tanh_node_test = OPERATORS['Tanh']()
        tanh_node_test.name = tanh_node.name
        name = tanh_node_test.name
        self.assertEqual('tanh', name)
    
    def test_tensor_slice_dataset(self):
        tensor_slice_dataset_node = node_def_pb2.NodeDef()
        tensor_slice_dataset_node.name = 'tensor_slice_dataset'
        tensor_slice_dataset_node.op = 'TensorSliceDataset'

        tensor_slice_dataset_node_test = OPERATORS['TensorSliceDataset']()
        tensor_slice_dataset_node_test.name = tensor_slice_dataset_node.name
        name = tensor_slice_dataset_node_test.name
        self.assertEqual('tensor_slice_dataset', name)
    
    def test_fused_batch_matmul_v2(self):
        fused_batch_matmul_v2_node = node_def_pb2.NodeDef()
        fused_batch_matmul_v2_node.name = 'fused_batch_matmul_v2'
        fused_batch_matmul_v2_node.op = '_FusedBatchMatMulV2'
        fused_batch_matmul_v2_node.attr['adj_x'].CopyFrom(attr_value_pb2.AttrValue(b=False))
        fused_batch_matmul_v2_node.attr['adj_y'].CopyFrom(attr_value_pb2.AttrValue(b=True))

        fused_batch_matmul_v2_node_test = OPERATORS['_FusedBatchMatMulV2']()
        fused_batch_matmul_v2_node_test.set_attr('tensorflow', fused_batch_matmul_v2_node)
        adj_x = fused_batch_matmul_v2_node_test.attr['transpose_a']
        adj_y = fused_batch_matmul_v2_node_test.attr['transpose_b']
        self.assertFalse(adj_x)
        self.assertTrue(adj_y)
    
    def test_fused_batch_norm_v3(self):
        fused_batch_norm_v3_node = node_def_pb2.NodeDef()
        fused_batch_norm_v3_node.name = 'fused_batch_norm_v3'
        fused_batch_norm_v3_node.op = 'FusedBatchNormV3'
        fused_batch_norm_v3_node.attr['epsilon'].CopyFrom(
                                            attr_value_pb2.AttrValue(f=0.0010000000474974513))
        fused_batch_norm_v3_node.attr['exponential_avg_factor'].CopyFrom(
                                                        attr_value_pb2.AttrValue(i=1))
        fused_batch_norm_v3_node.attr['is_training'].CopyFrom(attr_value_pb2.AttrValue(b=True))

        fused_batch_norm_v3_node_test = OPERATORS['FusedBatchNormV3']()
        fused_batch_norm_v3_node_test.set_attr('tensorflow', fused_batch_norm_v3_node)
        epsilon = fused_batch_norm_v3_node_test.attr['epsilon']
        exponential_avg_factor = fused_batch_norm_v3_node_test.attr['exponential_avg_factor']
        is_training = fused_batch_norm_v3_node_test.attr['is_training']
        self.assertEqual(0.0010000000474974513, epsilon)
        self.assertEqual(1, exponential_avg_factor)
        self.assertTrue(is_training)
    
    def test_fused_gemm(self):
        fused_gemm_node = NodeProto()
        fused_gemm_node.name = 'fused_gemm'
        fused_gemm_node.op_type = 'FusedGemm'
        fused_gemm_node.attribute.append(make_attribute('activation', 'Tanh'))
        fused_gemm_node.attribute.append(make_attribute('transA', 1))
        fused_gemm_node.attribute.append(make_attribute('transB', 1))
        fused_gemm_node.attribute.append(make_attribute('alpha', 2.0))
        fused_gemm_node.attribute.append(make_attribute('beta', 2.0))

        fused_gemm_node_test = OPERATORS['FusedGemm']()
        fused_gemm_node_test.set_attr('onnxruntime', fused_gemm_node)
        src0_perm = fused_gemm_node_test.attr['src0_perm']
        src1_perm = fused_gemm_node_test.attr['src1_perm']
        append_op = fused_gemm_node_test.attr['append_op']
        alpha = fused_gemm_node_test.attr['alpha']
        beta = fused_gemm_node_test.attr['beta']
        op_type = fused_gemm_node_test.op_type
        self.assertEqual('1,0', src0_perm)
        self.assertEqual('0,1', src1_perm)
        self.assertEqual('tanh', append_op)
        self.assertEqual(2, alpha)
        self.assertEqual(2, beta)
        self.assertEqual('InnerProduct', op_type)
    
    def test_gemm(self):
        gemm_node = NodeProto()
        gemm_node.name = 'gemm'
        gemm_node.op_type = 'Gemm'
        gemm_node.attribute.append(make_attribute('transA', 1))
        gemm_node.attribute.append(make_attribute('transB', 1))
        gemm_node.attribute.append(make_attribute('alpha', 2.0))
        gemm_node.attribute.append(make_attribute('beta', 2.0))

        gemm_node_test = OPERATORS['Gemm']()
        gemm_node_test.set_attr('onnxruntime', gemm_node)
        src0_perm = gemm_node_test.attr['src0_perm']
        src1_perm = gemm_node_test.attr['src1_perm']
        alpha = gemm_node_test.attr['alpha']
        beta = gemm_node_test.attr['beta']
        op_type = gemm_node_test.op_type
        self.assertEqual('1,0', src0_perm)
        self.assertEqual('0,1', src1_perm)
        self.assertEqual(2, alpha)
        self.assertEqual(2, beta)
        self.assertEqual('MatMulWithBias', op_type)
    
    def test_quantize_linear(self):
        quantize_linear_node = NodeProto()
        quantize_linear_node.name = 'quantize_linear'
        quantize_linear_node.op_type = 'QuantizeLinear'
        
        quantize_linear_node_test = OPERATORS['QuantizeLinear']()
        quantize_linear_node_test.set_attr('onnxruntime', quantize_linear_node)
        output_dtype = quantize_linear_node_test.attr['output_dtype']
        quant_mode = quantize_linear_node_test.attr['quant_mode']
        op_type = quantize_linear_node_test.op_type
        self.assertEqual('u8', output_dtype)
        self.assertEqual('zp_scale', quant_mode)
        self.assertEqual('Quantize', op_type)

    def test_fused_matmul_onnx(self):
        fused_matmul_node = NodeProto()
        fused_matmul_node.name = 'fused_matmul'
        fused_matmul_node.op_type = 'FusedMatMul'
        fused_matmul_node.attribute.append(make_attribute('transA', 1))
        fused_matmul_node.attribute.append(make_attribute('transB', 0))
        fused_matmul_node.attribute.append(make_attribute('alpha', 0.125))

        fused_matmul_node_test = OPERATORS['FusedMatMul']()
        fused_matmul_node_test.set_attr('onnxruntime', fused_matmul_node)
        transpose_a = fused_matmul_node_test.attr['transpose_a']
        transpose_b = fused_matmul_node_test.attr['transpose_b']
        alpha = fused_matmul_node_test.attr['alpha']
        self.assertTrue(transpose_a)
        self.assertFalse(transpose_b)
        self.assertEqual(0.125, alpha)
    
    def test_fused_matmul_tensorflow(self):
        fused_matmul_node = node_def_pb2.NodeDef()
        fused_matmul_node.name = 'fused_matmul'
        fused_matmul_node.op = '_FusedMatMul'
        fused_matmul_node.attr['transpose_a'].CopyFrom(attr_value_pb2.AttrValue(b=True))
        fused_matmul_node.attr['transpose_b'].CopyFrom(attr_value_pb2.AttrValue(b=True))
        fused_matmul_node.attr['epsilon'].CopyFrom(attr_value_pb2.AttrValue(f=0.0))

        fused_matmul_node_test = OPERATORS['_FusedMatMul']()
        fused_matmul_node_test.set_attr('tensorflow', fused_matmul_node)
        src0_perm = fused_matmul_node_test.attr['src0_perm']
        src1_perm = fused_matmul_node_test.attr['src1_perm']
        self.assertEqual('1,0', src0_perm)
        self.assertEqual('0,1', src1_perm)
    
    def test_gather_onnx(self):
        gather_node = NodeProto()
        gather_node.name = 'gather'
        gather_node.op_type = 'Gather'
        gather_node.attribute.append(make_attribute('axis', 0))

        gather_node_test = OPERATORS['Gather']()
        gather_node_test.set_attr('onnxruntime', gather_node)
        batch_dims = gather_node_test.attr['batch_dims']
        axis = gather_node_test.attr['axis']
        self.assertEqual(0, batch_dims)
        self.assertEqual(0, axis)
    
    def test_gather_tensorflow(self):
        gather_node = node_def_pb2.NodeDef()
        gather_node.name = 'gather'
        gather_node.op = 'Gather'
        gather_node.attr['batch_dims'].CopyFrom(attr_value_pb2.AttrValue(i=0))

        gather_node_test = OPERATORS['Gather']()
        gather_node_test.set_attr('tensorflow', gather_node)
        batch_dims = gather_node_test.attr['batch_dims']
        axis = gather_node_test.attr['axis']
        self.assertEqual(0, batch_dims)
        self.assertEqual(0, axis)
    
    def test_reduce_mean_onnx(self):
        reduce_mean_node = NodeProto()
        reduce_mean_node.name = 'reduce_mean'
        reduce_mean_node.op_type = 'ReduceMean'
        reduce_mean_node.attribute.append(make_attribute('keep_dims', 0))
        reduce_mean_node.attribute.append(make_attribute('axis', [0]))

        reduce_mean_node_test = OPERATORS['ReduceMean']()
        reduce_mean_node_test.set_attr('onnxruntime', reduce_mean_node)
        keep_dims = reduce_mean_node_test.attr['keep_dims']
        axis = reduce_mean_node_test.attr['axis']
        self.assertFalse(keep_dims)
        self.assertEqual(0, axis)
    
    def test_reduce_mean_tensorflow(self):
        reduce_mean_node = node_def_pb2.NodeDef()
        reduce_mean_node.name = 'reduce_mean'
        reduce_mean_node.op = 'ReduceMean'
        reduce_mean_node.attr['keep_dims'].CopyFrom(attr_value_pb2.AttrValue(b=False))

        reduce_mean_node_test = OPERATORS['ReduceMean']()
        reduce_mean_node_test.input_tensors = [Tensor(), Tensor(data=[0])]
        reduce_mean_node_test.set_attr('tensorflow', reduce_mean_node)
        keep_dims = reduce_mean_node_test.attr['keep_dims']
        axis = reduce_mean_node_test.attr['axis']
        self.assertFalse(keep_dims)
        self.assertEqual(0, axis)
    
    def test_squeeze_onnx(self):
        squeeze_node = NodeProto()
        squeeze_node.name = 'squeeze'
        squeeze_node.op_type = 'Squeeze'
        squeeze_node.attribute.append(make_attribute('axis', [0, 1, 2]))

        squeeze_node_test = OPERATORS['Squeeze']()
        squeeze_node_test.set_attr('onnxruntime', squeeze_node)
        axis = squeeze_node_test.attr['axis']
        self.assertEqual('0,1,2', axis)
    
    def test_squeeze_tensorflow(self):
        squeeze_node = node_def_pb2.NodeDef()
        squeeze_node.name = 'squeeze'
        squeeze_node.op = 'Squeeze'
        list_value = attr_value_pb2.AttrValue.ListValue(i=[0])
        squeeze_node.attr['squeeze_dims'].CopyFrom(attr_value_pb2.AttrValue(list=list_value))

        squeeze_node_test = OPERATORS['Squeeze']()
        squeeze_node_test.set_attr('tensorflow', squeeze_node)
        squeeze_dims = squeeze_node_test.attr['squeeze_dims']
        self.assertEqual([0], squeeze_dims)

    def test_iter_dataset_related(self):
        a = namedtuple('fake_list', ['list'])
        b = namedtuple('fake_shape', ['shape'])
        c = namedtuple('fake_dim', ['dim'])
        d = namedtuple('fake_size', ['size'])
        e = namedtuple('fake_dtype', ['type'])
        f = namedtuple('fake_attr', ['attr'])
        shape_list = [c([d(-1), d(128)])]
        attr = {'output_shapes': a(b(shape_list)),
                     'output_types': a(e([3]))}
        fake_node = f(attr)
        
        op_type_list = ['IteratorGetNext', 'IteratorV2', 'OptimizeDataset', 'MapAndBatchDataset']
        for op_type in op_type_list:
            iterator_get_next_node_test = OPERATORS[op_type]()
            iterator_get_next_node_test.output_tensors=[Tensor()]
            iterator_get_next_node_test.set_attr('tensorflow', fake_node)
            output_shapes = iterator_get_next_node_test.attr['output_shapes']
            output_types = iterator_get_next_node_test.attr['output_types']
            self.assertSequenceEqual([[-1, 128]], output_shapes)
            self.assertSequenceEqual(['int32'], output_types)
    
    def test_quantize_v2(self):
        quantize_v2_node = node_def_pb2.NodeDef()
        quantize_v2_node.name = 'quantize_v2'
        quantize_v2_node.op = 'QuantizeV2'

        quantize_v2_node_test = OPERATORS['QuantizeV2']()
        quantize_v2_node_test.set_attr('tensorflow', quantize_v2_node)
        output_dtype = quantize_v2_node_test.attr['output_dtype']
        op_type = quantize_v2_node_test.op_type
        self.assertEqual('u8', output_dtype)
        self.assertEqual('Quantize', op_type)
    
    def test_quantized_fused_matmul_and_dequantize(self):
        quantized_fused_matmul_and_dequantize_node = node_def_pb2.NodeDef()
        quantized_fused_matmul_and_dequantize_node.name = 'quantized_fused_matmul_and_dequantize'
        quantized_fused_matmul_and_dequantize_node.op = '_QuantizedFusedMatMulAndDequantize'
        quantized_fused_matmul_and_dequantize_node.attr['transpose_a'].CopyFrom(
                                                attr_value_pb2.AttrValue(b=True))
        quantized_fused_matmul_and_dequantize_node.attr['transpose_b'].CopyFrom(
                                                attr_value_pb2.AttrValue(b=False))
        quantized_fused_matmul_and_dequantize_node.attr['epsilon'].CopyFrom(
                                                attr_value_pb2.AttrValue(f=1.0))
        list_value = attr_value_pb2.AttrValue.ListValue(s=[b'add'])
        quantized_fused_matmul_and_dequantize_node.attr['fused_ops'].CopyFrom(
                                                attr_value_pb2.AttrValue(list=list_value))

        quantized_fused_matmul_and_dequantize_node_test = OPERATORS[
                                                '_QuantizedFusedMatMulAndDequantize']()
        quantized_fused_matmul_and_dequantize_node_test.set_attr('tensorflow', 
                                                quantized_fused_matmul_and_dequantize_node)
        src0_perm = quantized_fused_matmul_and_dequantize_node_test.attr['src0_perm']
        src1_perm = quantized_fused_matmul_and_dequantize_node_test.attr['src1_perm']
        epsilon = quantized_fused_matmul_and_dequantize_node_test.attr['epsilon']
        fused_ops = quantized_fused_matmul_and_dequantize_node_test.attr['fused_ops']
        output_dtype = quantized_fused_matmul_and_dequantize_node_test.attr['output_dtype']
        self.assertEqual('1,0', src0_perm)
        self.assertEqual('1,0', src1_perm)
        self.assertEqual(1.0, epsilon)
        self.assertEqual(['add'], fused_ops)
        self.assertEqual('fp32', output_dtype)
    
    def test_quantized_matmul_with_bias_and_dequantize(self):
        quantized_matmul_with_bias_and_dequantize_node = node_def_pb2.NodeDef()
        quantized_matmul_with_bias_and_dequantize_node.name = \
                                                    'quantized_matmul_with_bias_and_dequantize'
        quantized_matmul_with_bias_and_dequantize_node.op = 'QuantizedMatMulWithBiasAndDequantize'
        quantized_matmul_with_bias_and_dequantize_node.attr['transpose_a'].CopyFrom(
                                                attr_value_pb2.AttrValue(b=True))
        quantized_matmul_with_bias_and_dequantize_node.attr['transpose_b'].CopyFrom(
                                                attr_value_pb2.AttrValue(b=False))
        

        quantized_matmul_with_bias_and_dequantize_node_test = OPERATORS[
                                                'QuantizedMatMulWithBiasAndDequantize']()
        quantized_matmul_with_bias_and_dequantize_node_test.set_attr('tensorflow', 
                                                quantized_matmul_with_bias_and_dequantize_node)
        transpose_a = quantized_matmul_with_bias_and_dequantize_node_test.attr['transpose_a']
        transpose_b = quantized_matmul_with_bias_and_dequantize_node_test.attr['transpose_b']
        output_dtype = quantized_matmul_with_bias_and_dequantize_node_test.attr['output_dtype']
        self.assertTrue(transpose_a)
        self.assertFalse(transpose_b)
        self.assertEqual('s8', output_dtype)

    def test_layer_normalization(self):
        layer_normalization_node = NodeProto()
        layer_normalization_node.name = 'layer_normalization'
        layer_normalization_node.op_type = 'LayerNormalization'
        layer_normalization_node.attribute.append(make_attribute('stash_type', 1))
        layer_normalization_node.attribute.append(make_attribute('axis', 3))
        layer_normalization_node.attribute.append(make_attribute('epsilon', 1.0))
        
        op_type_list = ['LayerNormalization', '_MklLayerNorm']
        for op_type in op_type_list:
            layer_normalization_node_test = OPERATORS[op_type]()
            layer_normalization_node_test.set_attr('onnxruntime', layer_normalization_node)
            axis = layer_normalization_node_test.attr['axis']
            epsilon = layer_normalization_node_test.attr['epsilon']
            op_type = layer_normalization_node_test.op_type
            self.assertEqual(3, axis)
            self.assertEqual(1.0, epsilon)
            self.assertEqual('LayerNorm', op_type)
        
    def test_one_hot(self):
        one_hot_node = node_def_pb2.NodeDef()
        one_hot_node.name = 'one_hot'
        one_hot_node.op = 'OneHot'
        one_hot_node.attr['axis'].CopyFrom(attr_value_pb2.AttrValue(i=0))

        one_hot_node_test = OPERATORS['OneHot']()
        one_hot_node_test.input_tensors = [Tensor(), Tensor(data=0), Tensor(data=1),
                                            Tensor(data=2)]
        one_hot_node_test.set_attr('tensorflow', one_hot_node)
        axis = one_hot_node_test.attr['axis']
        depth = one_hot_node_test.attr['depth']
        on_value = one_hot_node_test.attr['on_value']
        off_value = one_hot_node_test.attr['off_value']
        self.assertEqual(0, axis)
        self.assertEqual(0, depth)
        self.assertEqual(1, on_value)
        self.assertEqual(2, off_value)
        self.assertEqual(1, len(one_hot_node_test.input_tensors))
    
    def test_onnx_input(self):
        a = namedtuple('fake_node', ['name', 'type'])
        b = namedtuple('type', ['tensor_type'])
        c = namedtuple('tensor_type', ['shape', 'elem_type'])
        d = namedtuple('shape', ['dim'])
        fake_node = a(name='onnx_input', type=b(c(shape=d([1, 1]), elem_type=7)))
        onnx_input_node = fake_node
        outputs = namedtuple('fake_output', ['outputs'])
        fake_nodes_dict = {'onnx_input': outputs(['next'])}

        onnx_input_node_test = OPERATORS['ONNXINPUT']()
        onnx_input_node_test.extract('onnxruntime', onnx_input_node, None, fake_nodes_dict)
        op_type = onnx_input_node_test.op_type
        tensor_name = onnx_input_node_test.output_tensors[0].name
        source_op = onnx_input_node_test.output_tensors[0].source_op
        dest_op = onnx_input_node_test.output_tensors[0].dest_op
        self.assertEqual('ONNXINPUT', op_type)
        self.assertEqual('onnx_input:0', tensor_name)
        self.assertSequenceEqual(['onnx_input'], source_op)
        self.assertSequenceEqual(['next'], dest_op)

    def test_transpose(self):
        transpose_node = NodeProto()
        transpose_node.name = 'transpose'
        transpose_node.op_type = 'Transpose'
        transpose_node.attribute.append(make_attribute('perm', [0, 2, 3, 1]))

        transpose_node_test = OPERATORS['Transpose']()
        transpose_node_test.set_attr('onnxruntime', transpose_node)
        src_perm = transpose_node_test.attr['src_perm']
        dst_perm = transpose_node_test.attr['dst_perm']
        self.assertEqual('0,1,2,3', src_perm)
        self.assertEqual('0,2,3,1', dst_perm)
    
    def test_unpack(self):
        unpack_node = node_def_pb2.NodeDef()
        unpack_node.name = 'unpack'
        unpack_node.op = 'Unpack'
        unpack_node.attr['axis'].CopyFrom(attr_value_pb2.AttrValue(i=0))
        unpack_node.attr['num'].CopyFrom(attr_value_pb2.AttrValue(i=2))

        unpack_node_test = OPERATORS['Unpack']()
        unpack_node_test.set_attr('tensorflow', unpack_node)
        axis = unpack_node_test.attr['axis']
        num = unpack_node_test.attr['num']
        self.assertEqual(0, axis)
        self.assertEqual(2, num)
    
    def test_unsqueeze(self):
        unsqueeze_node = NodeProto()
        unsqueeze_node.name = 'unsqueeze'
        unsqueeze_node.op_type = 'Unsqueeze'
        unsqueeze_node.attribute.append(make_attribute('axis', [0, 2, 3, 1]))

        unsqueeze_node_test = OPERATORS['Unsqueeze']()
        unsqueeze_node_test.set_attr('onnxruntime', unsqueeze_node)
        axis = unsqueeze_node_test.attr['axis']
        self.assertEqual('0,2,3,1', axis)
    
    def test_range(self):
        range_node = NodeProto()
        range_node.name = 'range'
        range_node.op_type = 'Range'

        range_node_test = OPERATORS['Range']()
        range_node_test.name = range_node.name
        name = range_node_test.name
        self.assertEqual('range', name)
    
    def test_relu(self):
        relu_node = NodeProto()
        relu_node.name = 'relu'
        relu_node.op_type = 'Relu'

        relu_node_test = OPERATORS['Relu']()
        relu_node_test.name = relu_node.name
        name = relu_node_test.name
        self.assertEqual('relu', name)
    
    def test_matmul_with_bias_relu(self):
        mat_node = NodeProto()
        mat_node.name = 'matmul_with_bias_relu'
        mat_node.op_type = 'MatMulWithBiasRelu'

        mat_node_test = OPERATORS['MatMulWithBiasRelu']()
        mat_node_test.name = mat_node.name
        name = mat_node_test.name
        self.assertEqual('matmul_with_bias_relu', name)
    
    def test_matmul(self):
        mat_node = NodeProto()
        mat_node.name = 'matmul'
        mat_node.op_type = 'Matmul'

        mat_node_test = OPERATORS['Matmul']()
        mat_node_test.name = mat_node.name
        name = mat_node_test.name
        self.assertEqual('matmul', name)
    
    def test_quantize(self):
        qat_node = NodeProto()
        qat_node.name = 'quantize'
        qat_node.op_type = 'Quantize'

        qat_node_test = OPERATORS['Quantize']()
        qat_node_test.name = qat_node.name
        name = qat_node_test.name
        self.assertEqual('quantize', name)
    
    def test_not(self):
        not_node = NodeProto()
        not_node.name = 'not'
        not_node.op_type = 'Not'

        not_node_test = OPERATORS['Not']()
        not_node_test.name = not_node.name
        name = not_node_test.name
        self.assertEqual('not', name)
    
    def test_cumsum(self):
        cumsum_node = NodeProto()
        cumsum_node.name = 'cumsum'
        cumsum_node.op_type = 'CumSum'

        cumsum_node_test = OPERATORS['CumSum']()
        cumsum_node_test.name = cumsum_node.name
        name = cumsum_node_test.name
        self.assertEqual('cumsum', name)
    
    def test_onehot(self):
        onehot_node = NodeProto()
        onehot_node.name = 'onehot'
        onehot_node.op_type = 'Onehot'

        onehot_node_test = OPERATORS['Onehot']()
        onehot_node_test.name = onehot_node.name
        name = onehot_node_test.name
        self.assertEqual('onehot', name)
    
    def test_toke_type_ids(self):
        token_node = NodeProto()
        token_node.name = 'toke_type_ids'
        token_node.op_type = 'TokenTypeIds'

        token_node_test = OPERATORS['TokenTypeIds']()
        token_node_test.name = token_node.name
        name = token_node_test.name
        self.assertEqual('toke_type_ids', name)
    
    def test_positio_ids(self):
        position_node = NodeProto()
        position_node.name = 'position_ids'
        position_node.op_type = 'PositionIds'

        position_node_test = OPERATORS['PositionIds']()
        position_node_test.name = position_node.name
        name = position_node_test.name
        self.assertEqual('position_ids', name)
    
    def test_loop(self):
        loop_node = NodeProto()
        loop_node.name = 'loop'
        loop_node.op_type = 'Loop'

        loop_node_test = OPERATORS['Loop']()
        loop_node_test.name = loop_node.name
        name = loop_node_test.name
        self.assertEqual('loop', name)
    
    def test_sigmoid(self):
        sigmoid_node = NodeProto()
        sigmoid_node.name = 'sigmoid'
        sigmoid_node.op_type = 'Sigmoid'

        sigmoid_node_test = OPERATORS['Sigmoid']()
        sigmoid_node_test.name = sigmoid_node.name
        name = sigmoid_node_test.name
        self.assertEqual('sigmoid', name)
    
    def test_matmul_with_bias_sigmoid(self):
        mat_node = NodeProto()
        mat_node.name = 'matmul_with_bias_sigmoid'
        mat_node.op_type = 'MatMulWithBiasSigmoid'

        mat_node_test = OPERATORS['MatMulWithBiasSigmoid']()
        mat_node_test.name = mat_node.name
        name = mat_node_test.name
        self.assertEqual('matmul_with_bias_sigmoid', name)
    
    def test_embedding_bag(self):
        eb_node = NodeProto()
        eb_node.name = 'embedding_bag'
        eb_node.op_type = 'EmbeddingBag'

        eb_node_test = OPERATORS['EmbeddingBag']()
        eb_node_test.name = eb_node.name
        name = eb_node_test.name
        self.assertEqual('embedding_bag', name)
    
    def test_flatten(self):
        flatten_node = NodeProto()
        flatten_node.name = 'flatten'
        flatten_node.op_type = 'Flatten'

        flatten_node_test = OPERATORS['Flatten']()
        flatten_node_test.name = flatten_node.name
        name = flatten_node_test.name
        self.assertEqual('flatten', name)

if __name__ == "__main__":
    unittest.main()