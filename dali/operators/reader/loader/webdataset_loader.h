// Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_OPERATORS_READER_LOADER_WEBDATASET_LOADER_H_
#define DALI_OPERATORS_READER_LOADER_WEBDATASET_LOADER_H_

#include <liburing.h>
#include <fstream>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include "dali/core/bitmask.h"
#include "dali/operators/reader/loader/loader.h"
#include "dali/pipeline/data/tensor.h"
#include "dali/util/file.h"

namespace dali {
namespace detail {
namespace wds {

const std::string kCurrentIndexVersion = "v1.2";  // NOLINT
const std::unordered_set<std::string> kSupportedIndexVersions = {"v1.1", kCurrentIndexVersion};
constexpr char kExtDelim = ';';
const std::set<DALIDataType> kSupportedTypes = {DALI_UINT8,   DALI_UINT16, DALI_UINT32, DALI_UINT64,
                                                DALI_INT8,    DALI_INT16,  DALI_INT32,  DALI_INT64,
                                                DALI_FLOAT16, DALI_FLOAT,  DALI_FLOAT64};

enum class MissingExtBehavior {
  Empty,
  Skip,
  Raise,
  Invalid
};
MissingExtBehavior ParseMissingExtBehavior(std::string);

template <typename T>
class VectorRange {
 private:
  std::vector<T>* data_ = nullptr;

 public:
  size_t start = 0;
  size_t num = 0;
  VectorRange() = default;
  explicit inline VectorRange(std::vector<T>& data, size_t start_idx = 0, size_t count = 0)
      : data_(&data), start(start_idx), num(count) {}

  inline T* begin() {
    return data_->data() + start;
  }

  inline T* end() {
    return begin() + num;
  }
};

struct ComponentDesc {
  std::string filename, ext;
  size_t size = 0;
  int64_t offset = 0;
  TensorShape<3> shape = {0, 0, 0};
  size_t extended_size = 0;
  VectorRange<size_t> outputs;

  ComponentDesc() = default;
};

struct SampleDesc {
  VectorRange<ComponentDesc> components;
  VectorRange<size_t> empty_outputs;
  size_t wds_shard_index;
  int64_t line_number;
  bool kind_flag;
};

}  // namespace wds
}  // namespace detail

class DLL_PUBLIC WebdatasetLoader : public Loader<CPUBackend, vector<Tensor<CPUBackend>>> {
 public:
  using LoadTarget = vector<Tensor<CPUBackend>>;

  explicit WebdatasetLoader(const OpSpec& spec, std::shared_ptr<io_uring>& ring);
  ~WebdatasetLoader() override;

  void PrepareEmpty(std::vector<Tensor<CPUBackend>>&) override;
  void ReadSample(std::vector<Tensor<CPUBackend>>&) override;
  LoadTargetSharedPtr ReadOne(bool is_new_batch) override;

 protected:
  Index SizeImpl() override;
  void PrepareMetadataImpl() override;
  void Reset(bool wrap_to_shard) override;

  std::vector<std::string> paths_;
  std::vector<std::string> index_paths_;
  std::vector<std::set<std::string>> ext_;
  std::vector<DALIDataType> dtypes_;
  detail::wds::MissingExtBehavior missing_component_behavior_;

 private:
  std::vector<detail::wds::SampleDesc> samples_;        // data from the index files
  std::vector<detail::wds::ComponentDesc> components_;  // data about the components held
                                                        // together for space optimization
  std::vector<size_t> empty_outputs_;  // indices of empty outputs to fill in for space optimization
  std::vector<size_t> output_indicies_;  // indices of outputs that a component corresponds to

  std::vector<int> wds_shards_;
  size_t sample_index_ = 0;
  std::once_flag multiple_files_single_component;

  std::vector<bool> is_in_buffer_;
  std::shared_ptr<io_uring> ring_;

  bool kind_flag_;
  size_t shard_size_, buffer_fill_, buffer0_fill_, buffer1_fill_;
  std::vector<LoadTargetSharedPtr> buffer0_, buffer1_, next_buffer0_, next_buffer1_;
  std::shared_ptr<std::vector<LoadTargetUniquePtr>> shared_empty_tensors_;
  std::shared_ptr<std::mutex> shared_empty_tensors_mutex_;

  std::string GetSampleSource(const detail::wds::SampleDesc& sample);
};

}  // namespace dali
#endif  // DALI_OPERATORS_READER_LOADER_WEBDATASET_LOADER_H_
