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

#ifndef DALI_OPERATORS_READER_WEBDATASET_READER_OP_H_
#define DALI_OPERATORS_READER_WEBDATASET_READER_OP_H_

#include <liburing.h>
#include <vector>
#include <string>
#include <memory>
#include "dali/operators/reader/loader/webdataset_loader.h"
#include "dali/operators/reader/reader_op.h"
#include "dali/pipeline/data/tensor.h"

namespace dali {

#define IO_URING_ENTRIES 4096

class DLL_PUBLIC WebdatasetReader : public DataReader<CPUBackend, vector<Tensor<CPUBackend>>> {
 public:
  using LoadTarget = vector<Tensor<CPUBackend>>;

  explicit WebdatasetReader(const OpSpec& spec)
      : DataReader<CPUBackend, vector<Tensor<CPUBackend>>>(spec) {
    ring_ = std::shared_ptr<struct io_uring>(new struct io_uring(),
                            [](struct io_uring* ring) { io_uring_queue_exit(ring); });
    DALI_ENFORCE(io_uring_queue_init(IO_URING_ENTRIES, ring_.get(), 0) == 0,
                 std::string("io_uring_queue_init - ") + std::strerror(errno));
    loader_ = InitLoader<WebdatasetLoader>(spec, ring_);
  }

  bool SetupImpl(std::vector<OutputDesc>& output_desc, const HostWorkspace&) override;
  void RunImpl(HostWorkspace& ws) override;
  bool CanInferOutputs() const override {
    return true;
  }

 protected:
  USE_READER_OPERATOR_MEMBERS(CPUBackend, vector<Tensor<CPUBackend>>);

  std::shared_ptr<struct io_uring> ring_;
};

}  // namespace dali

#endif  // DALI_OPERATORS_READER_WEBDATASET_READER_OP_H_
