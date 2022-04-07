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

#include "executor.hpp"

DEFINE_int32(batch_size, 1, "image batch sizes");
DEFINE_int32(seq_len, 384, "default seq len");
DEFINE_int32(iterations, 10, "iterations");
DEFINE_string(dataset, "", "dataset path");
DEFINE_string(mode, "", "model execution mode");
DEFINE_string(config, "", "model configuration using yaml file");
DEFINE_string(weight, "", "init model weight");
DEFINE_int32(w, 2, "warm up times");

void run_net() {
  executor::Model bert_model(FLAGS_config, FLAGS_weight);
  LOG(INFO) << "normal multibatch test begin";
  // 1. inialize input tensors
  vector<executor::Tensor> input_tensors;
  vector<string> input_dtype;
  vector<vector<float>> input_range;
  vector<vector<int64_t>> input_shape;
  auto input_configs = bert_model.input_configs();
  for (int i = 0; i < bert_model.num_inputs(); ++i) {
    input_tensors.push_back(executor::Tensor(*(input_configs[i])));
    input_dtype.push_back(input_tensors[i].dtype());
    input_range.push_back(vector<float>({1, 100}));
    input_shape.push_back(input_tensors[i].shape());
    if (input_shape[i][0] == -1 && input_shape[i][1] == -1) {
      input_shape[i][0] = FLAGS_batch_size;
      input_shape[i][1] = FLAGS_seq_len;
    } else if (input_shape[i][0] == -1) {
      input_shape[i][0] = FLAGS_seq_len;
    }
  }
  executor::DataLoader* dataloader;
  // dataloader = new executor::ConstDataLoader(input_shape, input_dtype, input_range);
  dataloader = new executor::DummyDataLoader(input_shape, input_dtype, input_range);

  // 2. forward the model
  float duration = 0;
  for (auto i = 0; i < FLAGS_iterations; i++) {
    auto raw_data = dataloader->prepare_batch(i);
    for (int j = 0; j < input_tensors.size(); ++j) {
      input_tensors[j].set_data(raw_data[j]);
      input_tensors[j].set_shape(input_shape[j]);
    }
    // float* src_0 = static_cast<float*>(input_tensors[0]->mutable_data());
    // float* src_1 = static_cast<float*>(input_tensors[1]->mutable_data());
    // LOG(INFO) << "src 0 value is: " << *src_0;
    // LOG(INFO) << "src 1 value is: " << *src_1;

    float start_time = executor::Time("start");
    vector<executor::Tensor>& output_data = bert_model.Forward(input_tensors);
    // warmup time not considered
    if (i < FLAGS_w) continue;
    duration += executor::Time("end") - start_time;

    // if (output_data[0].dtype() == "fp32") {
    //   float* data = static_cast<float*>(output_data[0].mutable_data());
    //   executor::PrintToFile(data, output_data[0].name() + "_" + FLAGS_mode, 200);
    // } else if (output_data[0].dtype() == "s8") {
    //   char* data = static_cast<char*>(output_data[0].mutable_data());
    //   executor::PrintToFile(data, output_data[0].name() + "_" + FLAGS_mode, 200);
    // } else if (output_data[0].dtype() == "u8") {
    //   unsigned char* data = static_cast<unsigned char*>(output_data[0].mutable_data());
    //   executor::PrintToFile(data, output_data[0].name() + "_" + FLAGS_mode, 200);
    // }
  }
  delete dataloader;

  float latency = duration / ((FLAGS_iterations - FLAGS_w) * FLAGS_batch_size);
  // LOG(INFO) << " Batch Size is " << FLAGS_batch_size;
  // LOG(INFO) << " Latency is " << latency << " ms";
  // LOG(INFO) << " Throughput is " << 1000./ latency;
  std::cout << " Batch Size is " << FLAGS_batch_size << std::endl;
  std::cout << " Latency is " << latency << " ms" << std::endl;
  std::cout << " Throughput is " << 1000. / latency << std::endl;
}

int main(int argc, char** argv) {
  executor::GlobalInit(&argc, &argv);
  run_net();

  return 0;
}
