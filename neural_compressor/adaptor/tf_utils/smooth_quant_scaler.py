import numpy as np
import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_util

# TODO remove this to bypass np fp error
# np.seterr(all='raise')

class SmoothQuantScaler:
    def __init__(self, model, dataloader, alpha, scales_per_op):
        self.model = model
        self.dataloader = dataloader
        self.alpha = alpha
        self.scales_per_op = scales_per_op
        
    def _adjust_activation(self, scale, input_node_name, output_node_name, w_i):
        """Insert the Mul node after the activation before the weight node
        
        Args:
            w_i: distinguish between different output weight node on different branches
        """
        from neural_compressor.adaptor.tf_utils.graph_util import GraphRewriterHelper as Helper
        node_suffix = str(w_i)
        mul_const_node = Helper.create_constant_node(input_node_name + "/scale_mul" + node_suffix, scale, tf.float32)
        mul_node = Helper.create_node('Mul', input_node_name + "_mul" + node_suffix, [input_node_name + "/scale_mul" + node_suffix, input_node_name])
        Helper.set_attr_dtype(mul_node, "T", dtypes.float32)
        from neural_compressor.adaptor.tf_utils.graph_util import GraphAnalyzer
        g = GraphAnalyzer()
        g.graph = self.model
        g.add_node(mul_node, input_node_name, [output_node_name]) # v0/conv0/conv2d/Conv2D
        g.add_node(mul_const_node, None, [input_node_name + "_mul" + node_suffix])
        # TODO whether to rewrite to the original graph def
        self.model.graph_def = g.dump_graph()
        # print(len(g.dump_graph().node))
        # print(len(self.model.graph_def.node))
        # print(self.model.graph_def == g.dump_graph())

    def _adjust_weight(self, scale, weight_node, original_weight):
        original_shape = original_weight.shape
        if len(original_shape) == 4:
            fh, fw, ic, oc = original_shape
            # TODO Check ? weight is the third dimension!!!
            # should not use transpose here !!!!!!
            # should use reshape here
            # W = np.transpose(original_weight, [0, 1, 3, 2]) # put channel to last dimension
            W = original_weight.reshape(fh,fw,oc,ic)
            # breakpoint()
            # W = np.transpose(original_weight, [0, 3, 1, 2])
            # W = np.reshape(W, (-1, W.shape[-1]))
            W *= scale
            W = np.reshape(W, original_shape)

            weight_node.attr['value'].tensor.CopyFrom(tensor_util.make_tensor_proto(W))
        elif len(original_shape) == 2:
            W = original_weight
            W = np.reshape(W, (original_shape[1], original_shape[0]))
            W *= scale
            W = np.reshape(W, original_shape)
            weight_node.attr['value'].tensor.CopyFrom(tensor_util.make_tensor_proto(W))

    def transform(self, max_vals_per_channel, shape_infos, sq_weight_tensors, sq_weights_nodes, sq_node_names):
        if self.scales_per_op:
            # obtain the smooth scale per op
            # S_j: a list
            # adjust activation
            # adjust weight
            for idx, input_node_name in enumerate(max_vals_per_channel):
                A_max_per_in_channel = max_vals_per_channel[input_node_name]
                W_lst = sq_weight_tensors[input_node_name]
                W_const_node_lst = sq_weights_nodes[input_node_name]  # Use the const nodes before to get weight values
                W_node_lst = sq_node_names[input_node_name] # Get the concrete weight node as the output of Mul insertion
                for w_i, W in enumerate(W_lst):
                    if len(W.shape) == 4:
                        # https://www.tensorflow.org/api_docs/python/tf/nn/conv2d
                        # weight: [filter_height, filter_width, in_channels, out_channels]
                        # activation: NHWC, also batch_shape + [in_height, in_width, in_channels]
                        tensor = np.transpose(W, [0, 1, 3, 2])
                        # reduce weight max to (in_channel, ), aligned with activation max
                        # tensor = W
                        W_max_per_in_channel = np.max(np.reshape(tensor, (-1, tensor.shape[-1])), axis=0)
                    elif len(W.shape) == 2: # matmul
                        # reduce weight max to (in_channel, ), aligned with activation max
                        tensor = W
                        # W_max_per_in_channel = np.max(np.reshape(tensor, (-1, tensor.shape[-1])), axis=0)
                        W_max_per_in_channel = np.max(W, axis=1)
                    else:
                        assert False, "not supported"
                    # breakpoint()
                    scale = np.power(A_max_per_in_channel, self.alpha) / np.power(W_max_per_in_channel, (1-self.alpha))
                    # clip the scales that are too small
                    scale = tf.clip_by_value(scale, clip_value_min=1e-5, clip_value_max=float("inf"))

                    self._adjust_weight(scale, W_const_node_lst[w_i], W)
                    self._adjust_activation(1 / scale, input_node_name, W_node_lst[w_i], w_i)
                    # tf.io.gfile.GFile('one_op.pb', 'wb').write(self.model.graph_def.SerializeToString())
                    # print("Collecting weight, activation and scale for the first op here...")
                    # print("weight can be viewed in the conv2d block filter attr")
                    # print("activation: in calib, p self._sq_output_tensor_dict['v0/conv0/Pad'][0]")
                    # print("scale: p scale")
                    # breakpoint()
                    # # return self.model
                # if idx == 0:
                #     tf.io.gfile.GFile('i_middle_1_with_branch_check.pb', 'wb').write(self.model.graph_def.SerializeToString())
                    # tf.io.gfile.GFile('i_middle_0_with_branch_check.pb', 'wb').write(self.model.graph_def.SerializeToString())
                    # return self.model
        else:
            pass
        # breakpoint()
        tf.io.gfile.GFile('j_new_only_matmul.pb', 'wb').write(self.model.graph_def.SerializeToString())
        return self.model
