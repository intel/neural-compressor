#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import copy
import json
import yaml
import math
import numpy as np
from collections import OrderedDict, UserDict
from .query import QueryBackendCapability
from .adaptor import adaptor_registry, Adaptor
from ..utils.utility import LazyImport, CpuInfo, singleton, Dequantize, dump_elapsed_time
from ..utils.utility import Statistics, GLOBAL_STATE, MODE, version1_lt_version2
from ..utils import logger
from ..conf.dotdict import deep_get
from ..experimental.data.dataloaders.base_dataloader import BaseDataLoader
tf = LazyImport('tensorflow')

def _add_supported_quantized_objects(custom_objects):
  """Map all the quantized objects."""
  from neural_compressor.adaptor.keras_utils.quantizer import Quantize, DeQuantize
  from neural_compressor.adaptor.keras_utils.quantizer import FakeQuant, QConv2D, QDense
  custom_objects["Quantize"] = Quantize
  custom_objects["DeQuantize"] = DeQuantize
  custom_objects["FakeQuant"] = FakeQuant
  custom_objects["QConv2D"] = QConv2D
  custom_objects["QDense"] = QDense
  return custom_objects

@adaptor_registry
class KerasAdaptor(Adaptor):
    '''The keras class of framework adaptor layer.

    '''
    def __init__(self, framework_specific_info):
        super(KerasAdaptor, self).__init__(framework_specific_info)
        self.framework_specific_info = framework_specific_info
        self.approach = deep_get(self.framework_specific_info, 'approach', False)
        self.quantize_config = {'op_wise_config': {}}
        self.device = self.framework_specific_info['device']
        #self.work_dir = os.path.abspath(self.framework_specific_info['workspace_path'])
        self.recipes = deep_get(self.framework_specific_info, 'recipes', {})
        #os.makedirs(self.work_dir, exist_ok=True)

        self.pre_optimized_model = None
        self.pre_optimizer_handle = None
        self.fp32_ops = []
        self.query_handler = KerasQuery(local_config_file=os.path.join(
            os.path.dirname(__file__), 'keras.yaml'))

        self.fp32_results = []
        self.fp32_preds_as_label = False
        self.benchmark = (GLOBAL_STATE.STATE == MODE.BENCHMARK)
        self.callbacks = []
        self.optype_statistics = None

    def tuning_cfg_to_fw(self, tuning_cfg):
        self.quantize_config['calib_iteration'] = tuning_cfg['calib_iteration']
        self.quantize_config['device'] = self.device
        self.quantize_config['advance'] = deep_get(tuning_cfg, 'advance')
        fp32_ops = []
        dispatched_op_names = [j[0] for j in tuning_cfg['op']]
        invalid_op_names = [i for i in self.quantize_config['op_wise_config']
                            if i not in dispatched_op_names]

        for op_name in invalid_op_names:
            self.quantize_config['op_wise_config'].pop(op_name)

        for each_op_info in tuning_cfg['op']:
            op_name = each_op_info[0]
            if tuning_cfg['op'][each_op_info]['activation']['dtype'] == 'fp32':
                if op_name in self.quantize_config['op_wise_config']:
                    self.quantize_config['op_wise_config'].pop(op_name)
                    fp32_ops.append(op_name)
                continue

            is_perchannel = False
            bit = None
            if 'weight' in tuning_cfg['op'][each_op_info]:
                is_perchannel = tuning_cfg['op'][each_op_info]['weight'][
                    'granularity'] == 'per_channel'
                #bit = tuning_cfg['op'][each_op_info]['weight']['bit']
            weight_bit = bit if bit else 7.0
            algorithm = tuning_cfg['op'][each_op_info]['activation']['algorithm']
            is_asymmetric = False
            if 'activation' in tuning_cfg['op'][each_op_info]:
                is_asymmetric = tuning_cfg['op'][each_op_info]['activation']['scheme'] == 'asym'
            self.quantize_config['op_wise_config'][op_name] = (is_perchannel,
                                                               algorithm,
                                                               is_asymmetric,
                                                               weight_bit)
        self.fp32_ops = fp32_ops

    @dump_elapsed_time("Pass quantize model")
    def quantize(self, tune_cfg, model, dataloader, q_func=None):
        '''Execute the quantize process on the specified model.

           Args:
               tune_cfg(dict): The chosen tuning configuration.
               model (object): The model to do quantization.
               dataloader(object): The dataloader used to load quantization dataset.
               q_func (optional): training function for quantization aware training mode.
        '''
        self.tuning_cfg_to_fw(tune_cfg)
        logger.debug("Dump quantization configurations:")
        logger.debug(self.quantize_config)
        calib_sampling_size = tune_cfg.get('calib_sampling_size', 1)
        if isinstance(dataloader, BaseDataLoader):
            batch_size = dataloader.batch_size
            for i in range(batch_size):
                if calib_sampling_size % (batch_size - i) == 0:
                    calib_batch_size = batch_size - i
                    if i != 0:  # pragma: no cover
                        logger.warning("Reset `calibration.dataloader.batch_size` field "
                                       "to {}".format(calib_batch_size) +
                                       " to make sure the sampling_size is "
                                       "divisible exactly by batch size")
                    break
            tmp_iterations = int(math.ceil(calib_sampling_size / calib_batch_size))
            dataloader.batch(calib_batch_size)
            self.quantize_config['calib_iteration'] = tmp_iterations

        else: # pragma: no cover
            if hasattr(dataloader, 'batch_size') and \
              calib_sampling_size % dataloader.batch_size != 0:
                iter = self.quantize_config['calib_iteration']
                logger.warning(
                    "Please note that calibration sampling size {} " \
                    "isn't divisible exactly by batch size {}. " \
                    "So the real sampling size is {}.".
                    format(calib_sampling_size, dataloader.batch_size,
                           dataloader.batch_size * iter))
        q_layers = []
        for idx, layer in enumerate(self.fp32_layers):
          layer_config = layer["config"]
          if layer["class_name"] in ["Conv2D", "Dense"] and \
            layer['config']['name'] in self.quantize_config['op_wise_config']:
              op_config = self.quantize_config['op_wise_config'][layer['config']['name']]
              mode = 'per_channel' if op_config[0] else 'per_tensor'
              #(TODO) support asym/sym
              fake_quant_name = 'fake_quant_' + str(idx)
              q_layers.append({'class_name': 'FakeQuant', 
                  'config': {'mode': 'per_tensor', 'name': fake_quant_name}})
              q_layers.append(layer)
          else:
              q_layers.append(layer)

        keras_object = model._model_object
        json_model = copy.deepcopy(json.loads(keras_object.to_json()))
        json_model['config']['layers'] = q_layers
        quantized_model = self._restore_model_from_json(json_model)

        converted_model = self._calibrate(quantized_model, dataloader, 
                self.quantize_config['calib_iteration'])

        from neural_compressor.model.keras_model import KerasModel
        converted_model = KerasModel(converted_model)
        return converted_model

    def _calibrate(self, model, dataloader, calib_interation):
        # run eagerly to fetch the numpy min/max
        model.compile(run_eagerly=True)
        results = {}
        for idx, (inputs, labels) in enumerate(dataloader):
            outputs = model.predict_on_batch(inputs)
            json_model = copy.deepcopy(json.loads(model.to_json()))
            config = json_model["config"]
            layers = config["layers"]
            for layer in layers:
                if layer['class_name'] == 'FakeQuant':
                    min_value = layer['config']['min_value']
                    max_value = layer['config']['max_value']
                    if layer['config']['name'] not in results:
                        results[layer['config']['name']] = {
                                'min': [min_value], 'max': [max_value]}
                    else:
                        results[layer['config']['name']]['min'].append(min_value)
                        results[layer['config']['name']]['max'].append(max_value)
            if idx + 1  == calib_interation:
                break
        
        # insert the calibrated min/max to Q/DQ
        json_model = copy.deepcopy(json.loads(model.to_json()))
        config = json_model["config"]
        layers = config["layers"]
        q_layers = []
        for layer in layers:
            layer_config = copy.deepcopy(layer['config'])
            if layer['class_name'] == 'FakeQuant':
                min_value = min(results[layer['config']['name']]['min'])
                max_value = max(results[layer['config']['name']]['max'])
                q_layers.append({'class_name': 'Quantize',
                                 'config': {'min_range': min_value,
                                            'max_range': max_value,
                                           }})
                q_layers.append({'class_name': 'DeQuantize',
                                 'config': {'min_range': min_value,
                                            'max_range': max_value,
                                           }})
            elif layer['class_name'] == 'Conv2D' or layer['class_name'] == 'Dense':
                # index 0 is weight, index 1 is bias
                q_layer_name = 'Q' + layer['class_name']
                kernel = self.layer_weights[layer['config']['name']][0]
                layer_config['min_value'] = str(kernel.min())
                layer_config['max_value'] = str(kernel.max())
                q_layers.append({'class_name': q_layer_name, 'config': layer_config})
            else:
                q_layers.append(layer) 

        json_model['config']['layers'] = q_layers
        quantized_model = self._restore_model_from_json(json_model)
        return quantized_model

    def _restore_model_from_json(self, json_model):
        from tensorflow.keras.models import model_from_json
        custom_objects = {}
        # We need to keep a dictionary of custom objects as our quantized library
        # is not recognized by keras.
        custom_objects = _add_supported_quantized_objects(custom_objects)
        qmodel = model_from_json(json.dumps(json_model), custom_objects=custom_objects)
        qmodel = self._set_weights(qmodel, self.layer_weights)
        return qmodel

    # set fp32 weights to qmodel
    def _set_weights(self, qmodel, layer_weights):
        for qlayer in qmodel.layers:
            if qlayer.get_weights():
                if qlayer.name in layer_weights:
                    qlayer.set_weights(layer_weights[qlayer.name])
                else:
                    hit_layer = False
                    for sub_layer in qlayer.submodules: 
                        if sub_layer.name in layer_weights:
                            qlayer.set_weights(layer_weights[sub_layer.name])
                            hit_layer = True
                            break
                    if not hit_layer:
                        raise ValueError('Can not match the module weights....')
        return qmodel

    @dump_elapsed_time(customized_msg="Model inference")
    def evaluate(self, model, dataloader, postprocess=None,
                 metrics=None, measurer=None, iteration=-1,
                 tensorboard=False, fp32_baseline=False):
        '''The function is used to run evaluation on validation dataset.

           Args:
               model (object): The model to do calibration.
               dataloader (generator): generate the data and labels.
               postprocess (object, optional): process the result from the model
               metric (object, optional): Depends on model category. Defaults to None.
               measurer (object, optional): for precise benchmark measurement.
               iteration(int, optional): control steps of mini-batch
               tensorboard (boolean, optional): for tensorboard inspect tensor.
               fp32_baseline (boolen, optional): only for compare_label=False pipeline
        '''
        # use keras object
        keras_model = model.model
        logger.info("Start to evaluate the Keras model.")
        results = []
        for idx, (inputs, labels) in enumerate(dataloader):
            # use predict on batch
            if measurer is not None:
                measurer.start()
                predictions = keras_model.predict_on_batch(inputs)
                measurer.end()
            else:
                predictions = keras_model.predict_on_batch(inputs)

            if self.fp32_preds_as_label:
                self.fp32_results.append(predictions) if fp32_baseline else \
                    results.append(predictions)

            if postprocess is not None:
                predictions, labels = postprocess((predictions, labels))
            if metrics:
                for metric in metrics:
                    if not hasattr(metric, "compare_label") or \
                        (hasattr(metric, "compare_label") and metric.compare_label):
                        metric.update(predictions, labels)
            if idx + 1 == iteration:
                break
        return results

    def query_fw_capability(self, model):
        '''The function is used to return framework tuning capability.

           Args:
               model (object): The model to query quantization tuning capability.
        '''
        self.pre_optimized_model = model
        fp32_config = {'weight': {'dtype': 'fp32'}, 'activation': {'dtype': 'fp32'}}
        int8_type = self.query_handler.get_op_types_by_precision(precision='int8')
        op_capability = self.query_handler.get_quantization_capability()
        conv_config = copy.deepcopy(op_capability['int8']['Conv2D'])
        dense_config = copy.deepcopy(op_capability['int8']['Dense'])
        other_config = copy.deepcopy(op_capability['int8']['default'])

        # get the layers info
        keras_object = model._model_object
        json_model = copy.deepcopy(json.loads(keras_object.to_json()))
        config = json_model["config"]
        self.fp32_layers = config["layers"]

        # get fp32 layer weights
        self.layer_weights = {}
        for layer in keras_object.layers:
            if layer.get_weights():
                self.layer_weights[layer.name] = copy.deepcopy(layer.get_weights())

        quantizable_op_details = OrderedDict()
        for details in self.fp32_layers:
            node_op = details['class_name']
            node_name = details['config']['name']
            if node_op == 'Conv2D': 
                quantizable_op_details[(node_name, node_op)] = [conv_config, fp32_config]
            elif node_op == 'Dense':
                quantizable_op_details[(node_name, node_op)] = [dense_config, fp32_config]
            else:
                quantizable_op_details[(node_name, node_op)] = [fp32_config]

        capability = {
            'opwise': copy.deepcopy(quantizable_op_details),
            'optypewise': self.get_optype_wise_ability(quantizable_op_details),
        }
        logger.debug("Dump framework quantization capability:")
        logger.debug(capability)

        return capability

    def get_optype_wise_ability(self, quantizable_op_details):
        """Get the op type wise capability by generating the union value of each op type.
        Returns:
            [string dict]: the key is op type while the value is the
                           detail configurations of activation and weight for this op type.
        """
        res = OrderedDict()
        for op in quantizable_op_details:
            if op[1] not in res:
                    res[op[1]] = {'activation': quantizable_op_details[op][0]['activation']}
                    if 'weight' in quantizable_op_details[op][0]:
                        res[op[1]]['weight'] = quantizable_op_details[op][0]['weight']
        return res

    def inspect_tensor(self, model, dataloader, op_list=[], iteration_list=[],
                       inspect_type='activation', save_to_disk=False):
        '''The function is used by tune strategy class for dumping tensor info.

           Args:
               model (object): The model to inspect.
               dataloader (object): The dataloader used to feed into.
               op_list (list): The op name in the fp32 model for dumpping.
               iteration_list (list): The iteration list containing iterations to dump.
               inspect_type (str): The valid value are 'weight', 'activation', 'all'.
               save_to_disk (bool): Save to disk or memory.

           Return:
               Numpy Array Dict
               {
                 'weight': {
                   'node0_name': {'weight0_name': numpy.array, 'bias0_name': numpy.array, ...},
                   'node1_name': {'weight1_name': numpy.array, 'bias1_name': numpy.array, ...},
                   ...
                 },
                 'activation': [
                   # iter 0
                   {
                     'node0_name': {'output0_name': numpy.array, 'output1_name': numpy.array, ...}
                     'node1_name': {'output1_name': numpy.array, 'output1_name': numpy.array, ...}
                     ...
                   },
                   # iter 1
                   ...
                 ]
               }
        '''
        pass

    def set_tensor(self, model, tensor_dict):
        '''The function is used by tune strategy class for setting tensor back to model.

           Args:
               model (object): The model to set tensor. Usually it is quantized model.
               tensor_dict (dict): The tensor dict to set. Note the numpy array contains float
                                   value, adaptor layer has the responsibility to quantize to
                                   int8 or int32 to set into the quantized model if needed.
                                   The dict format is something like:
                                   {
                                     'weight0_name': numpy.array,
                                     'bias0_name': numpy.array,
                                     ...
                                   }
        '''
        pass

    def quantize_input(self, model):
        ''' quantize the model to be able to take quantized input

            Args:
                model (object): The model to quantize input

            Return:
                model (object): The quantized input model
                scale (float): The scale for dataloader to generate quantized input
        '''
        return model, 1.

    def _pre_eval_hook(self, model, *args, **kwargs):
        '''The function is used to do some preprocession before evaluation phase.

        Return:
              model
        '''
        return model

    def _post_eval_hook(self, model, *args, **kwargs):
        '''The function is used to do some post process after complete evaluation.
        '''
        pass

    def save(self, model, path):
        '''The function is used by tune strategy class for saving model.

           Args:
               model (object): The model to saved.
               path (string): The path where to save.
        '''
        model.save(path)

    def convert(self, model, source, destinatin):
        '''The function is used to convert a source model format to another.

           Args:
               model (neural_compressor.model): base model to be converted.
               source (string): The source model format.
               destination (string): The destination model format.
        '''
        pass

class KerasQuery(QueryBackendCapability):
    def __init__(self, local_config_file=None):
        super().__init__()
        self.version = tf.version.VERSION
        self.cfg = local_config_file
        self.cur_config = None
        self._one_shot_query()

    def _one_shot_query(self):
        with open(self.cfg) as f:
            content = yaml.safe_load(f)
            try:
                self.cur_config = self._get_specified_version_cfg(content)
            except Exception as e:
                logger.info("Fail to parse {} due to {}.".format(self.cfg, str(e)))
                self.cur_config = None
                raise ValueError("Please check if the format of {} follows Neural Compressor yaml schema.".
                                 format(self.cfg))

    def _get_specified_version_cfg(self, data):
        """Get the configuration for the current runtime.
        If there's no matched configuration in the input yaml, we'll
        use the `default` field of yaml.

        Args:
            data (Yaml content): input yaml file.

        Returns:
            [dictionary]: the content for specific version.
        """
        default_config = None
        for sub_data in data:
            if sub_data['version']['name'] == self.version:
                return sub_data

            if sub_data['version']['name'] == 'default':
                default_config = sub_data

        return default_config

    def get_version(self):
        """Get the current backend version infomation.

        Returns:
            [string]: version string.
        """
        return self.cur_config['version']['name']

    def get_precisions(self):
        """Get supported precisions for current backend.

        Returns:
            [string list]: the precisions' name.
        """
        return self.cur_config['precisions']['names']

    def get_op_types(self):
        """Get the supported op types by all precisions.

        Returns:
            [dictionary list]: A list composed of dictionary which key is precision
            and value is the op types.
        """
        return self.cur_config['ops']

    def get_quantization_capability(self):
        """Get the supported op types' quantization capability.

        Returns:
            [dictionary list]: A list composed of dictionary which key is precision
            and value is a dict that describes all op types' quantization capability.
        """
        return self.cur_config['capabilities']

    def get_op_types_by_precision(self, precision):
        """Get op types per precision

        Args:
            precision (string): precision name

        Returns:
            [string list]: A list composed of op type.
        """
        assert precision in list(self.cur_config['ops'].keys())
        return self.cur_config['ops'][precision]
