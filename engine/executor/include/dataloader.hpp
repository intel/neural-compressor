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

#ifndef ENGINE_EXECUTOR_INCLUDE_DATALOADER_HPP_
#define ENGINE_EXECUTOR_INCLUDE_DATALOADER_HPP_
#include <omp.h>

#include <functional>
#include <numeric>
#include <random>
#include <string>
#include <thread>  // NOLINT
#include <vector>

#include "common.hpp"
#include "gflags/gflags.h"
#include "tensor.hpp"

namespace executor {

class DataLoader {
 public:
  virtual ~DataLoader() {}
  virtual void* load_sample(size_t* index_list, size_t size) { return NULL; }
  virtual vector<void*>& prepare_batch(const int index, const int seq_len = -1) = 0;
  virtual vector<vector<int64_t>> prepare_shape() = 0;

 private:
  vector<vector<void*>> dataset_;
  string dataset_name_;
  int batch_size_ = 1;
  // managed_shared_memory managed_shm_;
  // string shared_memory_;
};

// class BertDataLoader : public DataLoader {
//   public:
//     BertDataLoader(const vector<Tensor*>& input_tensors,
//                    const vector<vector<int64_t>>& input_shape,
//                    const string& input_path)
//       : DataLoader(), input_tensors_(input_tensors), input_shape_(input_shape),
//         input_path_(input_path) {
//           min_data_ = new float[1];
//           max_data_ = new float[1];
//           min_data_[0] = 0.f;
//           max_data_[0] = 1.f;
//           // LOG(INFO) << "min value is: " << *min_data_;
//           // LOG(INFO) << "max value is: " << *max_data_;
//         }
//     ~BertDataLoader() {
//       if(min_data_!= nullptr){
//         delete min_data_;
//         min_data_ = nullptr;
//       }
//       if(max_data_!= nullptr){
//         delete max_data_;
//         max_data_ = nullptr;
//       }
//     }
//
//     vector<void*>& prepare_batch(const int index) override {
//       // for (int i = 0; i < input_tensors_.size(); ++i) {
//       for (int i = 0; i < 3; ++i) {
//         const auto& tensor = input_tensors_[i];
//         auto size = std::accumulate(input_shape_[i].begin(), input_shape_[i].end(),
//                                      (int64_t)1, std::multiplies<int64_t>());
//         if (tensor->dtype() == "int32") {
//           void* data = read_file_to_type(input_path_ + tensor->name(),
//                                           tensor->dtype(), input_shape_[i],
//                                           {0, sizeof(int32_t) * size});
//           batch_data_.push_back(data);
//         } else if (tensor->dtype() == "u8") {
//           void* data = read_file_to_type(input_path_ + tensor->name(),
//                                           tensor->dtype(), input_shape_[i],
//                                           {0, sizeof(int32_t) * size});
//           batch_data_.push_back(data);
//         }
//       }
//
//       // LOG(INFO) << "min_ value is: " << *min_data_;
//       // LOG(INFO) << "max_ value is: " << *max_data_;
//
//       batch_data_.push_back(static_cast<void*>(min_data_));
//       batch_data_.push_back(static_cast<void*>(max_data_));
//       // void* min_p = MemoryAllocator::GetMemory(sizeof(float), INT64_MAX);
//       // void* max_p = MemoryAllocator::GetMemory(sizeof(float), INT64_MAX);
//       // static_cast<float*>(min_p)[0] = (float)0.;
//       // static_cast<float*>(max_p)[0] = (float)1.;
//       // batch_data_.push_back(min_p);
//       // batch_data_.push_back(max_p);
//
//       return batch_data_;
//     }
//
//   private:
//     vector<Tensor*> input_tensors_;
//     vector<vector<int64_t>> input_shape_;
//     string input_path_;
//     vector<void*> batch_data_;
//     float* min_data_ = nullptr;
//     float* max_data_ = nullptr;
// };

class ConstDataLoader : public DataLoader {
 public:
  ConstDataLoader(const vector<vector<int64_t>>& input_shape, const vector<string>& dtype,
                  const vector<vector<float>>& value_range)
      : DataLoader(), shape_(input_shape), dtype_(dtype), value_range_(value_range) {
    CHECK_EQ(input_shape.size(), dtype.size()) << "shape size should equal with dtype size.";
    for (int i = 0; i < input_shape.size(); ++i) {
      CHECK_EQ(value_range[i].size(), 2) << "value range size should be 2 from low to high";
      int64_t size =
          std::accumulate(input_shape[i].begin(), input_shape[i].end(), (int64_t)1, std::multiplies<int64_t>());
      size_.push_back(size);
      if (dtype_[i] == "fp32") {
        float* data = new float[size];
        batch_data_.push_back(data);
      } else if (dtype_[i] == "int32") {
        int32_t* data = new int32_t[size];
        batch_data_.push_back(data);
      } else if (dtype_[i] == "u8") {
        unsigned char* data = new unsigned char[size];
        batch_data_.push_back(data);
      } else if (dtype_[i] == "int32") {
        unsigned int* data = new unsigned int[size];
        batch_data_.push_back(data);
      }
    }
  }
  virtual ~ConstDataLoader() {
    for (int i = 0; i < batch_data_.size(); ++i) {
      delete batch_data_[i];
    }
  }

  virtual vector<vector<int64_t>> prepare_shape() { return shape_; }

  virtual vector<void*>& prepare_batch(const int index, const int seq_len = -1) {
    for (int i = 0; i < shape_.size(); ++i) {
      std::uniform_real_distribution<float> u(value_range_[i][0], value_range_[i][1]);
      if (dtype_[i] == "fp32") {
        auto data = static_cast<float*>(batch_data_[i]);
        for (int idx = 0; idx < size_[i]; ++idx) {
          *(data++) = static_cast<float>(idx % static_cast<int>(value_range_[i][1]));
        }
      } else if (dtype_[i] == "int32") {
        auto data = static_cast<int32_t*>(batch_data_[i]);
        for (int idx = 0; idx < size_[i]; ++idx) {
          *(data++) = static_cast<int32_t>(idx % static_cast<int>(value_range_[i][1]));
        }
      } else if (dtype_[i] == "u8") {
        auto data = static_cast<unsigned char*>(batch_data_[i]);
        for (int idx = 0; idx < size_[i]; ++idx) {
          *(data++) = static_cast<unsigned char>(idx % static_cast<int>(value_range_[i][1]));
        }
      } else if (dtype_[i] == "int32") {
        auto data = static_cast<int*>(batch_data_[i]);
        for (int idx = 0; idx < size_[i]; ++idx) {
          *(data++) = static_cast<int>(idx % static_cast<int>(value_range_[i][1]));
        }
      }
    }
    return batch_data_;
  }

 private:
  vector<vector<int64_t>> shape_;
  vector<string> dtype_;
  vector<void*> batch_data_;
  vector<vector<float>> value_range_;
  std::mt19937 generator_;
  vector<int64_t> size_;
};

class DummyDataLoader : public DataLoader {
 public:
  DummyDataLoader(const vector<vector<int64_t>>& input_shape, const vector<string>& dtype,
                  const vector<vector<float>>& value_range)
      : DataLoader(), shape_(input_shape), dtype_(dtype), value_range_(value_range) {
    CHECK_EQ(input_shape.size(), dtype.size()) << "shape size should equal with dtype size.";
    for (int i = 0; i < input_shape.size(); ++i) {
      CHECK_EQ(value_range[i].size(), 2) << "value range size should be 2 from low to high";
      int64_t size =
          std::accumulate(input_shape[i].begin(), input_shape[i].end(), (int64_t)1, std::multiplies<int64_t>());
      size_.push_back(size);
      if (dtype_[i] == "fp32") {
        float* data = new float[size];
        batch_data_.push_back(data);
      } else if (dtype_[i] == "int32") {
        int32_t* data = new int32_t[size];
        batch_data_.push_back(data);
      } else if (dtype_[i] == "u8") {
        unsigned char* data = new unsigned char[size];
        batch_data_.push_back(data);
      } else if (dtype_[i] == "int32") {
        unsigned int* data = new unsigned int[size];
        batch_data_.push_back(data);
      }
    }
  }
  virtual ~DummyDataLoader() {
    for (int i = 0; i < batch_data_.size(); ++i) {
      delete batch_data_[i];
    }
  }

  virtual vector<vector<int64_t>> prepare_shape() { return shape_; }
  virtual vector<void*>& prepare_batch(const int index, const int seq_len = -1) {
    for (int i = 0; i < shape_.size(); ++i) {
      std::uniform_real_distribution<float> u(value_range_[i][0], value_range_[i][1]);
      if (dtype_[i] == "fp32") {
        auto data = static_cast<float*>(batch_data_[i]);
        for (int idx = 0; idx < size_[i]; ++idx) {
          *(data++) = u(generator_);
        }
      } else if (dtype_[i] == "int32") {
        auto data = static_cast<int32_t*>(batch_data_[i]);
        for (int idx = 0; idx < size_[i]; ++idx) {
          *(data++) = static_cast<int32_t>(u(generator_));
        }
      } else if (dtype_[i] == "u8") {
        auto data = static_cast<unsigned char*>(batch_data_[i]);
        for (int idx = 0; idx < size_[i]; ++idx) {
          *(data++) = static_cast<unsigned char>(u(generator_));
        }
      } else if (dtype_[i] == "int32") {
        auto data = static_cast<int*>(batch_data_[i]);
        for (int idx = 0; idx < size_[i]; ++idx) {
          *(data++) = static_cast<int>(u(generator_));
        }
      }
    }
    return batch_data_;
  }

 private:
  vector<vector<int64_t>> shape_;
  vector<string> dtype_;
  vector<void*> batch_data_;
  vector<vector<float>> value_range_;
  std::mt19937 generator_;
  vector<int64_t> size_;
};

}  // namespace executor
#endif  // ENGINE_EXECUTOR_INCLUDE_DATALOADER_HPP_
