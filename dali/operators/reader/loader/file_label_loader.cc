// Copyright (c) 2017-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <memory>

#include "dali/core/common.h"
#include "dali/operators/reader/loader/file_label_loader.h"
#include "dali/util/file.h"
#include "dali/operators/reader/loader/utils.h"

namespace dali {

using filesystem::dir_sep;

inline size_t getFilesize(const char* filename) {
  struct stat st;
  if (stat(filename, &st) != 0) {
    return 0;
  }
  return st.st_size;
}


inline bool is_cached(std::string name) {
  struct stat buffer;
  const char* name_c = ("/dev/shm/cache/" + name).c_str();
  return (stat(name_c, &buffer) == 0);
}

void FileLabelLoader::PrepareEmpty(ImageLabelWrapper &image_label) {
  PrepareEmptyTensor(image_label.image);
}

void FileLabelLoader::ReadSample(ImageLabelWrapper &image_label) {
  auto image_pair = image_label_pairs_[current_index_++];
  int cur_idx = current_index_ - 1;

  // handle wrap-around
  MoveToNextShard(current_index_);

  // copy the label
  image_label.label = image_pair.second;
  DALIMeta meta;
  meta.SetSourceInfo(image_pair.first);
  meta.SetSkipSample(false);

  // if image is cached, skip loading
  if (ShouldSkipImage(image_pair.first)) {
    meta.SetSkipSample(true);
    image_label.image.Reset();
    image_label.image.SetMeta(meta);
    image_label.image.Resize({0}, DALI_UINT8);
    return;
  }

  if (cache_size_ > 0 && !caching_done_) {
    bool must_cache =
        std::binary_search(shm_cache_index_list_.begin(), shm_cache_index_list_.end(), cur_idx);
    if (!caching_done_ && must_cache) {
      shm::CacheEntry *ce = new shm::CacheEntry(image_pair.first);
      int ret = -1;
      ret = ce->create_segment();
      DALI_ENFORCE(ret != -1, "Cache for " + image_pair.first + " could not be created.");
      ret = ce->put_cache_simple(file_root_ + "/" + image_pair.first);
      DALI_ENFORCE(ret != -1, "Cache for " + image_pair.first + " could not be populated.");
      shm_cached_items_.push_back(image_pair.first);
      delete ce;
    }
  }

  // check if cached
  // Change this to be parameter. Hardcoded for now
  bool use_prefix = true;
  std::string prefix;
  prefix = file_root_;
  int node = -1;
  if (cache_size_ > 0 && caching_done_) {
    // Check current node
    if (is_cached(image_pair.first)) {
      prefix = "/dev/shm/cache";
    } else if ((node = shm::is_cached_in_other_node(shm_cache_index_list_other_nodes,
                                                  image_pair.first, node_id_)) >= 0) {
      // Check other nodes
      use_prefix = false;
      string fn = file_root_ + "/" + image_pair.first;
      int64_t image_size = getFilesize(fn.c_str());
      if (image_size < 0) {
        prefix = file_root_;
        use_prefix = true;
      } else {
        // Get a unique server id
        if (image_label.image.shares_data()) {
          image_label.image.Reset();
        }
        image_label.image.Resize({image_size});
        net_mutex_.lock();
        int bytes_read =
            shm::read_from_other_node(server_fd_[node], image_pair.first,
                                      image_label.image.mutable_data<uint8_t>(), image_size);
        net_mutex_.unlock();
        if (bytes_read < 0) {
          prefix = file_root_;
          use_prefix = true;
        }
      }
    }
  } else {
    prefix = file_root_;
  }

  if (use_prefix) {
    auto current_image = FileStream::Open(filesystem::join_path(prefix, image_pair.first),
                                          read_ahead_, !copy_read_data_);
    Index image_size = current_image->Size();

    if (copy_read_data_) {
      if (image_label.image.shares_data()) {
        image_label.image.Reset();
      }
      image_label.image.Resize({image_size}, DALI_UINT8);
      // copy the image
      Index ret = current_image->Read(image_label.image.mutable_data<uint8_t>(), image_size);
      DALI_ENFORCE(ret == image_size, make_string("Failed to read file: ", image_pair.first));
    } else {
      auto p = current_image->Get(image_size);
      DALI_ENFORCE(p != nullptr, make_string("Failed to read file: ", image_pair.first));
      // Wrap the raw data in the Tensor object.
      image_label.image.ShareData(p, image_size, false, {image_size}, DALI_UINT8);
    }

    // close the file handle
    current_image->Close();
  }

  image_label.image.SetMeta(meta);
}

Index FileLabelLoader::SizeImpl() {
  return static_cast<Index>(image_label_pairs_.size());
}
}  // namespace dali
