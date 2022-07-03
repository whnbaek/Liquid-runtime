// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef DALI_PIPELINE_DATA_META_H_
#define DALI_PIPELINE_DATA_META_H_

#include <string>
#include "dali/pipeline/data/types.h"

namespace dali {

class DALIMeta {
 public:
  DALIMeta() = default;

  explicit DALIMeta(const TensorLayout &layout) : layout_(layout) {
  }

  inline const TensorLayout &GetLayout() const {
    return layout_;
  }

  inline void SetLayout(const TensorLayout &layout) {
    layout_ = layout;
  }

  inline const std::string &GetSourceInfo() const {
    return source_info_;
  }

  inline void SetSourceInfo(const std::string &source_info) {
    source_info_ = source_info;
  }

  inline void SetSkipSample(bool skip_sample) {
    skip_sample_ = skip_sample;
  }

  inline bool ShouldSkipSample() const {
    return skip_sample_;
  }

  inline void SetShape(const TensorShape<3> &shape) {
    shape_ = shape;
  }

  inline const TensorShape<3> &GetShape() const {
    return shape_;
  }

  inline void SetAlreadyRead(bool already_read) {
    already_read_ = already_read;
  }

  inline bool AlreadyRead() const {
    return already_read_;
  }

  inline void SetSampleIndex(size_t sample_index) {
    sample_index_ = sample_index;
  }

  inline size_t GetSampleIndex() const {
    return sample_index_;
  }

 private:
  TensorLayout layout_;
  std::string source_info_;
  bool skip_sample_ = false;
  TensorShape<3> shape_;
  bool already_read_ = false;
  size_t sample_index_;
};

}  // namespace dali


#endif  // DALI_PIPELINE_DATA_META_H_
