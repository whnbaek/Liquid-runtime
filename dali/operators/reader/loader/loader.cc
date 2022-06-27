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


#include "dali/operators/reader/loader/loader.h"

namespace dali {

DALI_SCHEMA(LoaderBase)
  .AddOptionalArg("random_shuffle",
      R"code(Determines whether to randomly shuffle data.

A prefetch buffer with a size equal to ``initial_fill`` is used to read data sequentially,
and then samples are selected randomly to form a batch.)code", false)
  .AddOptionalArg("initial_fill",
      R"code(Size of the buffer that is used for shuffling.

If ``random_shuffle`` is False, this parameter is ignored.)code", 1024)
  .AddOptionalArg("num_shards",
      R"code(Partitions the data into the specified number of parts (shards).

This is typically used for multi-GPU or multi-node training.)code", 1)
  .AddOptionalArg("num_nodes",
      R"code(Number of physical nodes involved in multi GPU training).)code", 1)
  .AddOptionalArg("node_id",
      R"code(ID of the node in multi GPU training).)code", 0)
  .AddOptionalArg("resume",
      R"code(Resume with old cache.)code", false)
  .AddOptionalArg("shard_id",
      R"code(Index of the shard to read.)code", 0)
  .AddOptionalArg("cache_size",
      R"code(Number of items to cache for this loader.)code", 0)
  .AddOptionalArg("shuffle_seed",
      R"code(Shuffle the initial dataset metadata)code", 0)
  .AddOptionalArg("tensor_init_bytes",
      R"code(Hint for how much memory to allocate per image.)code", 1048576)
  .AddOptionalArg("stick_to_shard",
      R"code(Determines whether the reader should stick to a data shard instead of going through
the entire dataset.

If decoder caching is used, it significantly reduces the amount of data to be cached, but
might affect accuracy of the training.)code", false)
  .AddOptionalArg("read_ahead",
      R"code(Determines whether the accessed data should be read ahead.

For large files such as LMDB, RecordIO, or TFRecord, this argument slows down the first access but
decreases the time of all of the following accesses.)code", false)
  .AddOptionalArg("prefetch_queue_depth",
      R"code(Specifies the number of batches to be prefetched by the internal Loader.

This value should be increased when the pipeline is CPU-stage bound, trading memory
consumption for better interleaving with the Loader thread.)code", 1)
  .AddOptionalArg("skip_cached_images",
      R"code(If set to True, the loading data will be skipped when the sample is
in the decoder cache.

In this case, the output of the loader will be empty.)code", false)
  .AddOptionalArg("lazy_init",
            R"code(Parse and prepare the dataset metadata only during the first run instead of
in the constructor.)code", false)
  .AddOptionalArg("pad_last_batch",
      R"code(If set to True, pads the shard by repeating the last sample.

.. note::
  If the number of batches differs across shards, this option can cause an entire batch of repeated
  samples to be added to the dataset.)code", false)
.AddOptionalArg("dont_use_mmap",
      R"code(If set to True, the Loader will use plain file I/O instead of trying to map
the file in memory.

Mapping provides a small performance benefit when accessing a local file system, but most network file
systems, do not provide optimum performance.
)code", false);

size_t start_index(const size_t shard_id,
                   const size_t shard_num,
                   const size_t size) {
  return size * shard_id / shard_num;
}

size_t start_index_for_node(const size_t node_id, const size_t num_nodes, const size_t size) {
  return size * node_id / num_nodes;
}

Index num_samples(const size_t shard_num,
                  const size_t size) {
  return static_cast<Index>(std::ceil(size * 1.0 / shard_num));
}

int initialize_socket(int port, string ip_addr) {
  int cnct, server_fd;
  struct sockaddr_in server;
  struct hostent *hp;

  server.sin_family = AF_INET;
  server.sin_port = htons(port);
  server.sin_addr.s_addr = INADDR_ANY;
  server_fd = socket(AF_INET, SOCK_STREAM, 0);
  if (server_fd < 0) {
    std::cerr << "Error creating socket\n";
    return -1;
  }

  if (!set_tcp_nodelay(server_fd))
    return -1;

  hp = gethostbyname(ip_addr.c_str());
  bcopy(reinterpret_cast<char *>(hp->h_addr), reinterpret_cast<char *>(&server.sin_addr.s_addr),
        hp->h_length);
  cnct = connect(server_fd, (struct sockaddr *)&server, sizeof(server));
  if (cnct < 0) {
    std::cerr << "Error connecting " << endl;
    return -1;
  }

  return server_fd;
}

}  // namespace dali
