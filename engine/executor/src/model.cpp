//  Copyright (c) 2021 Intel Corporation
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

#include "model.hpp"

namespace executor {

Model::Model(const ModelConfig& conf, const string& weight_root) : weight_root_(weight_root) { Init(conf); }

Model::Model(const string& conf_file, const string& weight_root) : weight_root_(weight_root) {
  ModelConfig conf = ModelConfig(conf_file);
  CHECK_EQ(conf.CheckConfig(), true) << "model config not right....";
  Init(conf);
}

void Model::Init(const ModelConfig& conf) {
  name_ = conf.name();
  MemoryAllocator::InitStrategy();
  // For each operator, set up its input and output
  auto op_configs = conf.operators();
  input_vecs_.resize(op_configs.size());
  output_vecs_.resize(op_configs.size());
  // Basically, build all the operators and set up their connections.
  for (int operator_id = 0; operator_id < op_configs.size(); ++operator_id) {
    auto op_conf = op_configs[operator_id];
    auto operator_name = op_conf->name();
    operators_.push_back(OperatorRegistry::CreateOperator(*op_conf));
    operator_names_.push_back(operator_name);
    operator_name_index_[operator_name] = operator_id;
    // handle the input/output tensors to the model
    // we will have an input operator only have output data
    // in a graph the input tensors must come from output tensors
    // so we create output tensors and input tensors all take from output
    // tensors besides, we have two operators, one is Input and the other is
    // Output Input only have output tensors and Output only have input tensors
    // we treat weight tensors as output from Input operator and other
    // operators' input
    auto op_type = op_conf->type();

    int output_size = op_conf->output_tensor_size();
    for (int output_id = 0; output_id < output_size; ++output_id) {
      SetOutput(op_configs, operator_id, output_id, &tensor_name_index_);
    }
    int input_size = op_conf->input_tensor_size();
    for (int input_id = 0; input_id < input_size; ++input_id) {
      SetInput(op_configs, operator_id, input_id, &tensor_name_index_);
    }
  }
  // for debug tensor life
  for (size_t i = 0; i < tensors_.size(); ++i) {
    LOG(INFO) << "tensor name is " << tensors_[i]->name() << " tensor life is  " << tensors_[i]->life();
  }
  // prepare the operator like cache weight
  for (int i = 0; i < operators_.size(); ++i) {
    operators_[i]->Prepare(input_vecs_[i], output_vecs_[i]);
  }
}

void Model::SetInput(const vector<OperatorConfig*>& conf, const int operator_id, const int tensor_id,
                     map<string, int>* tensor_name_index_) {
  // model input tensor not in output tensors
  const OperatorConfig* op_conf = conf[operator_id];
  const string& tensor_name = op_conf->input_tensors(tensor_id)->name();
  if (!tensor_name_index_->count(tensor_name)) {
    LOG(FATAL) << "Unknown input tensor " << tensor_name << ", operator " << op_conf->name() << ", input index "
               << tensor_id;
  }
  const int id = (*tensor_name_index_)[tensor_name];
  // add tensor life count for memory handling
  tensors_[id]->add_tensor_life(1);
  input_vecs_[operator_id].push_back(tensors_[id]);
  // set model output tensors, it maybe a little strange as Output operator only
  // have input and the input is MODEL's output
  const string& op_type = op_conf->type();
  if (op_type == "Output") {
    model_output_tensors_.push_back(tensors_[id]);
    output_tensors_.push_back(Tensor(nullptr, tensors_[id]->shape(), tensors_[id]->dtype()));
  }
}

void Model::SetOutput(const vector<OperatorConfig*>& conf, const int operator_id, const int tensor_id,
                      map<string, int>* tensor_name_index_) {
  const OperatorConfig* op_conf = conf[operator_id];
  const string& tensor_name = op_conf->output_tensors(tensor_id)->name();
  if (tensor_name_index_->count(tensor_name)) {
    LOG(FATAL) << "duplicate output tensor name..." << tensor_name;
  }
  // start from output tensor
  auto tensor_config = op_conf->output_tensors(tensor_id);
  const int id = tensors_.size();
  Tensor* tensor_ptr(new Tensor(*tensor_config));
  tensors_.push_back(tensor_ptr);
  tensor_names_.push_back(tensor_name);
  output_vecs_[operator_id].push_back(tensor_ptr);
  (*tensor_name_index_)[tensor_name] = id;
  // set model input tensors, it maybe a little strange as Input operator only
  // have output and the output is MODEL's input
  const string& op_type = op_conf->type();
  if (op_type == "Input") {
    // parse weight here
    if (tensor_config->location().size() != 0) {
      void* weight_ptr =
          read_file_to_type(weight_root_, tensor_config->dtype(), tensor_config->shape(), tensor_config->location());
      tensor_ptr->set_data(weight_ptr);
      return;
    }
    // set model input tensors
    model_input_tensors_.push_back(tensor_ptr);
    model_input_configs_.push_back(tensor_config);
  }
}

vector<Tensor>& Model::Forward(vector<Tensor>& input_data) {
  CHECK_EQ(input_data.size(), model_input_tensors_.size())
      << "input data size not equal with model input tensor size....";
  // if we want use dynamic input data shape at run time, we should check the
  // input data shape and get the output shape, this should be necessary in each
  // Operator's Forward function
  bool reshape_model = false;
  for (int i = 0; i < input_data.size(); ++i) {
    vector<int64_t> data_shape = input_data[i].shape();
    // here we use model input configs to get the configured shape
    vector<int64_t> model_input_shape = model_input_configs_[i]->shape();
    vector<int64_t> origin_model_input = model_input_tensors_[i]->shape();
    LOG(INFO) << "data shape is " << data_shape[0] << " model config is " << model_input_shape[0] << " origin shape is "
              << origin_model_input[0];
    CHECK_EQ(data_shape.size(), model_input_shape.size()) << "input data should have same "
                                                          << "dimensions with configured model shape....";
    for (int axis = 0; axis < data_shape.size(); ++axis) {
      if (data_shape[axis] != origin_model_input[axis]) {
        // not equal case only happen when model input axis support dynamic in
        // config which axis value should be -1
        CHECK_EQ(model_input_shape[axis], -1) << "data shape mismatch " << data_shape[axis]
                                              << " while model input shape need " << model_input_shape[axis];
        reshape_model = true;
      }
    }
  }
  for (int i = 0; i < input_data.size(); ++i) {
    // model_input_tesnor_[i]->free_data();
    model_input_tensors_[i]->set_data(input_data[i].mutable_data());
    model_input_tensors_[i]->set_shape(input_data[i].shape());
  }

  if (reshape_model) {
    for (int i = 0; i < operators_.size(); ++i) {
      LOG(INFO) << "operator " << operators_[i]->name() << " gonna reshape with type " << operators_[i]->type();
      operators_[i]->Reshape(input_vecs_[i], output_vecs_[i]);
    }
  }
  for (int i = 0; i < operators_.size(); ++i) {
    LOG(INFO) << "operator " << operators_[i]->name() << " gonna forward with type " << operators_[i]->type();
    operators_[i]->Forward(input_vecs_[i], output_vecs_[i]);
  }

  return this->output_tensors();
}

}  // namespace executor
