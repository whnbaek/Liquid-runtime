// Copyright (c) 2017-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_OPERATORS_READER_LOADER_FILE_LABEL_LOADER_H_
#define DALI_OPERATORS_READER_LOADER_FILE_LABEL_LOADER_H_

#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <sys/uio.h>
#include <unistd.h>

#include <fstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
#include <algorithm>
#include <thread>

#include "dali/core/common.h"
#include "dali/operators/reader/loader/loader.h"
#include "dali/operators/reader/loader/filesystem.h"
#include "dali/operators/reader/loader/commands.h"
#include "dali/operators/shmcache/posixshmem.h"
#include "dali/util/file.h"

namespace dali {

struct ImageLabelWrapper {
  Tensor<CPUBackend> image;
  int label;
};

class DLL_PUBLIC FileLabelLoader : public Loader<CPUBackend, ImageLabelWrapper> {
 public:
  explicit inline FileLabelLoader(
    const OpSpec& spec,
    bool shuffle_after_epoch = false)
    : Loader<CPUBackend, ImageLabelWrapper>(spec),
      shuffle_after_epoch_(shuffle_after_epoch),
      current_index_(0),
      current_epoch_(0),
      caching_done_(false) {

      vector<string> files;
      vector<int> labels;

      has_files_arg_ = spec.TryGetRepeatedArgument(files, "files");
      has_labels_arg_ = spec.TryGetRepeatedArgument(labels, "labels");
      has_file_list_arg_ = spec.TryGetArgument(file_list_, "file_list");
      has_file_root_arg_ = spec.TryGetArgument(file_root_, "file_root");
      bool has_file_filters_arg = spec.TryGetRepeatedArgument(filters_, "file_filters");

      // TODO(ksztenderski): CocoLoader inherits after FileLabelLoader and it doesn't work with
      // GetArgument.
      spec.TryGetArgument(case_sensitive_filter_, "case_sensitive_filter");

      DALI_ENFORCE(has_file_root_arg_ || has_files_arg_ || has_file_list_arg_,
        "``file_root`` argument is required when not using ``files`` or ``file_list``.");

      DALI_ENFORCE(has_files_arg_ + has_file_list_arg_ <= 1,
        "File paths can be provided through ``files`` or ``file_list`` but not both.");

      DALI_ENFORCE(has_files_arg_ || !has_labels_arg_,
        "The argument ``labels`` is valid only when file paths "
        "are provided as ``files`` argument.");

      DALI_ENFORCE(!has_file_filters_arg || filters_.size() > 0,
                   "``file_filters`` list cannot be empty.");

      if (has_file_list_arg_) {
        DALI_ENFORCE(!file_list_.empty(), "``file_list`` argument cannot be empty");
        if (!has_file_root_arg_) {
          auto idx = file_list_.rfind(filesystem::dir_sep);
          if (idx != string::npos) {
            file_root_ = file_list_.substr(0, idx);
          }
        }
      }

      if (has_files_arg_) {
        DALI_ENFORCE(files.size() > 0, "``files`` specified an empty list.");
        if (has_labels_arg_) {
          DALI_ENFORCE(files.size() == labels.size(), make_string("Provided ", labels.size(),
            " labels for ", files.size(), " files."));

          for (int i = 0, n = files.size(); i < n; i++)
            image_label_pairs_.emplace_back(std::move(files[i]), labels[i]);
        } else {
            for (int i = 0, n = files.size(); i < n; i++)
              image_label_pairs_.emplace_back(std::move(files[i]), i);
        }
      }

      if ((spec.HasArgument("node_ip_list") ^ spec.HasArgument("node_port_list")) == 1)
        DALI_ENFORCE(1, "node_ip_list and node_port_list must be specified together");

      if (spec.HasArgument("node_ip_list")) {
        node_ip_list_ = spec.GetRepeatedArgument<std::string>("node_ip_list");
        if (node_ip_list_.size() > 0)
          dist_mint = true;
      }
      if (spec.HasArgument("node_port_list")) {
        node_port_list_ = spec.GetRepeatedArgument<int>("node_port_list");
        if (node_port_list_.size() > 0)
          dist_mint = true;
      }

      DALI_ENFORCE(node_ip_list_.size() == node_port_list_.size(),
                   "Length and port and IP list must be same");

      // Init the clients
      if (dist_mint) {
        DALI_ENFORCE(cache_size_orig_ > 0, "Cache size must be non zero in dist mint");
        for (unsigned int i = 0; i < node_port_list_.size(); i++) {
          if (static_cast<int>(i) == node_id_) {
            shard_port_list_.push_back(0);
            server_fd_.push_back(0);
          } else {
            shard_port_list_.push_back(node_port_list_[i] + shard_id_);
            server_fd_.push_back(initialize_socket(shard_port_list_[i], node_ip_list_[i]));
          }
        }
        DALI_ENFORCE(server_fd_.size() == node_ip_list_.size(),
                     "Error in starting client connection");
      }

      /*
      * Those options are mutually exclusive as `shuffle_after_epoch` will make every shard looks differently
      * after each epoch so coexistence with `stick_to_shard` doesn't make any sense
      * Still when `shuffle_after_epoch` we will set `stick_to_shard` internally in the FileLabelLoader so all
      * DALI instances will do shuffling after each epoch
      */
      DALI_ENFORCE(!(shuffle_after_epoch_  && stick_to_shard_),
                   "shuffle_after_epoch and stick_to_shard cannot be both true");
      DALI_ENFORCE(!(shuffle_after_epoch_ && shuffle_),
                   "shuffle_after_epoch and random_shuffle cannot be both true");
      /*
       * Imply `stick_to_shard` from  `shuffle_after_epoch`
       */
      if (shuffle_after_epoch_) {
        stick_to_shard_ = true;
      }
    if (!dont_use_mmap_) {
      mmap_reserver_ = FileStream::MappingReserver(
                                  static_cast<unsigned int>(initial_buffer_fill_));
    }
    copy_read_data_ = dont_use_mmap_ || !mmap_reserver_.CanShareMappedData();
  }

  void PrepareEmpty(ImageLabelWrapper &tensor) override;
  void ReadSample(ImageLabelWrapper &tensor) override;

  ~FileLabelLoader() {
    if (dist_mint) {
      for (unsigned int i = 0; i < node_port_list_.size(); i++) {
        if (server_fd_[i] > 0) {
          close(server_fd_[i]);
          shutdown(server_fd_[i], 0);
        }
      }
    }
  }

 protected:
  Index SizeImpl() override;

  void PrepareMetadataImpl() override {
    if (image_label_pairs_.empty()) {
      if (!has_file_list_arg_ && !has_files_arg_) {
        image_label_pairs_ =
            filesystem::traverse_directories(file_root_, filters_, case_sensitive_filter_);
      } else if (has_file_list_arg_) {
        // load (path, label) pairs from list
        std::ifstream s(file_list_);
        DALI_ENFORCE(s.is_open(), "Cannot open: " + file_list_);

        vector<char> line_buf(16 << 10);  // 16 kB should be more than enough for a line
        char *line = line_buf.data();
        for  (int n = 1; s.getline(line, line_buf.size()); n++) {
          // parse the line backwards:
          // - skip trailing whitespace
          // - consume digits
          // - skip whitespace between label and
          int i = strlen(line) - 1;

          for (; i >= 0 && isspace(line[i]); i--) {}  // skip trailing spaces

          int label_end = i + 1;

          if (i < 0)  // empty line - skip
            continue;

          for (; i >= 0 && isdigit(line[i]); i--) {}  // skip

          int label_start = i + 1;

          for (; i >= 0 && isspace(line[i]); i--) {}

          int name_end = i + 1;
          DALI_ENFORCE(name_end > 0 && name_end < label_start &&
                       label_start >= 2 && label_end > label_start,
                       make_string("Incorrect format of the list file \"",  file_list_, "\":", n,
                       " expected file name followed by a label; got: ", line));

          line[label_end] = 0;
          line[name_end] = 0;

          image_label_pairs_.emplace_back(line, std::atoi(line + label_start));
        }

        DALI_ENFORCE(s.eof(), "Wrong format of file_list: " + file_list_);
      }
    }
    DALI_ENFORCE(SizeImpl() > 0, "No files found.");

    Index size_per_shard = SizeImpl() / num_shards_;
    if (cache_size_orig_ > size_per_shard) {
      extra_cache_size_ = cache_size_orig_ - size_per_shard;
      cache_size_ = size_per_shard;
    } else {
      cache_size_ = cache_size_orig_;
    }

    if (shuffle_) {
      // seeded with hardcoded value to get
      // the same sequence on every shard
      std::mt19937 g(kDaliDataloaderSeed + shuffle_seed_);
      std::shuffle(image_label_pairs_.begin(), image_label_pairs_.end(), g);
    }
    Reset(true);
  }

  void Reset(bool wrap_to_shard) override {
    if (wrap_to_shard) {
      current_index_ = start_index(shard_id_, num_shards_, SizeImpl());
    } else {
      current_index_ = 0;
    }
    index_start_ = current_index_;
    index_end_ = current_index_ + SizeImpl() / num_shards_;

    // current_epoch_++;

    if (shuffle_after_epoch_) {
      std::mt19937 g(kDaliDataloaderSeed + current_epoch_ + shuffle_seed_);
      std::shuffle(image_label_pairs_.begin(), image_label_pairs_.end(), g);
    }

  // If the epoch count is 1 here, it means we have completed
  // epoch 1. SO stop caching beyond this point
  if (current_epoch_ == 1) {
    caching_done_ = true;
    if (num_nodes_ > 1 && cache_size_ > 0 && !resume_) {
      // if we have extra items to cache, handle here
      if (extra_cache_size_ > 0 && dist_mint) {
        vector<string> items_not_in_node;
        for (int it = 0; it < num_nodes_; it++) {
          if (it != node_id_)
            items_not_in_node.insert(items_not_in_node.end(),
                                     shm_cache_index_list_other_nodes[it].begin(),
                                     shm_cache_index_list_other_nodes[it].end());
        }
        // Get a random shuffle order at each node so that what is in cache of each node is random
        // halps balance nodes to remote caches
        std::mt19937 gen_s(shuffle_seed_ + node_id_);
        std::shuffle(items_not_in_node.begin(), items_not_in_node.end(), gen_s);
        Index items_per_shard = items_not_in_node.size() / num_shards_per_node_;
        Index start_idx = (shard_id_ % num_shards_per_node_) * extra_cache_size_;
        Index end_idx = (shard_id_ % num_shards_per_node_ + 1) * extra_cache_size_;
        mint_prefetcher =
            std::thread(shm::prefetch_cache, items_not_in_node, start_idx, end_idx, file_root_);
        prefetcher_running = true;
      }
    }
  }

  // Create a shuffled list for caching
  // Sort it so that search becomes easier
  if (!caching_done_ && cache_size_ > 0) {
    // Get the cache list for other nodes
    if (num_nodes_ > 1) {
      shm_cache_index_list_other_nodes.resize(num_nodes_);
      for (int nid = 0; nid < num_nodes_; nid++) {
        if (nid == node_id_) {
          // We are in the current node; do nothing
          continue;
        }
        vector<string> nid_list = shm_cache_index_list_other_nodes[nid];
        // Resize list to the total size of shards in this node
        for (int sh = 0; sh < num_shards_per_node_; sh++) {
          std::mt19937 gen(shuffle_seed_);
          Index shard_start_idx = start_index(num_shards_per_node_ * nid + sh, num_shards_, Size());
          Index shard_end_idx = shard_start_idx + Size() / num_shards_;
          Index shard_size = shard_end_idx - shard_start_idx;
          vector<int> cache_list_per_shard(shard_size);
          std::iota(cache_list_per_shard.begin(), cache_list_per_shard.end(), shard_start_idx);
          std::shuffle(cache_list_per_shard.begin(), cache_list_per_shard.end(), gen);
          cache_list_per_shard.resize(cache_size_);
          std::sort(cache_list_per_shard.begin(), cache_list_per_shard.end());
          vector<string> cache_list_per_shard_name;
          for (unsigned int k = 0; k < cache_list_per_shard.size(); k++) {
            cache_list_per_shard_name.push_back(image_label_pairs_[cache_list_per_shard[k]].first);
          }
          nid_list.insert(nid_list.end(), cache_list_per_shard_name.begin(),
                          cache_list_per_shard_name.end());
        }
        std::sort(nid_list.begin(), nid_list.end());
        shm_cache_index_list_other_nodes[nid] = nid_list;
      }
    }

    std::mt19937 gen(shuffle_seed_);
    shm_cache_index_list_.resize(Size() / num_shards_);
    std::iota(shm_cache_index_list_.begin(), shm_cache_index_list_.end(), index_start_);
    std::shuffle(shm_cache_index_list_.begin(), shm_cache_index_list_.end(), gen);
    shm_cache_index_list_.resize(cache_size_);
    std::sort(shm_cache_index_list_.begin(), shm_cache_index_list_.end());
    vector<string> shm_cache_name_list_;
    for (unsigned int k = 0; k < shm_cache_index_list_.size(); k++)
      shm_cache_name_list_.push_back(image_label_pairs_[shm_cache_index_list_[k]].first);
  }

  current_epoch_++;

  if (resume_)
    caching_done_ = true;
}

  using Loader<CPUBackend, ImageLabelWrapper>::shard_id_;
  using Loader<CPUBackend, ImageLabelWrapper>::num_shards_;
  using Loader<CPUBackend, ImageLabelWrapper>::shuffle_seed_;
  using Loader<CPUBackend, ImageLabelWrapper>::cache_size_orig_;
  using Loader<CPUBackend, ImageLabelWrapper>::num_nodes_;
  using Loader<CPUBackend, ImageLabelWrapper>::node_id_;
  using Loader<CPUBackend, ImageLabelWrapper>::resume_;
  using Loader<CPUBackend, ImageLabelWrapper>::seed_;
  using Loader<CPUBackend, ImageLabelWrapper>::num_shards_per_node_;

  string file_root_, file_list_, node_ip_;
  vector<std::string> node_ip_list_;
  vector<int> node_port_list_;
  vector<int> shard_port_list_;
  vector<int> server_fd_;
  int extra_cache_size_ = 0;
  int cache_size_ = 0;
  vector<std::pair<string, int>> image_label_pairs_;
  vector<string> filters_;

  bool has_files_arg_ = false;
  bool has_labels_arg_ = false;
  bool has_file_list_arg_ = false;
  bool has_file_root_arg_ = false;
  bool case_sensitive_filter_ = false;

  bool shuffle_after_epoch_;
  Index current_index_;
  int current_epoch_;
  bool caching_done_;
  int index_start_;
  int index_end_;
  bool dist_mint = false;
  std::thread mint_prefetcher;
  bool prefetcher_running = false;
  vector<int> shm_cache_index_list_;
  vector<vector<string>> shm_cache_index_list_other_nodes;
  vector<std::string> shm_cached_items_;
  FileStream::MappingReserver mmap_reserver_;
};

}  // namespace dali

#endif  // DALI_OPERATORS_READER_LOADER_FILE_LABEL_LOADER_H_
