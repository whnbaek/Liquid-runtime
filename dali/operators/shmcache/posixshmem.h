// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef DALI_OPERATORS_SHMCACHE_POSIXSHMEM_H_
#define DALI_OPERATORS_SHMCACHE_POSIXSHMEM_H_

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>


namespace shm {

extern const char prefix[];

/* Reads one sample from a remote node whose tcp port is
 * given by the open server_fd.
 * Returns the number of bytes read
 * This has to be called after validating that the sample
 * exists in the node you're requesting. The GET request is in this func
 */
int read_from_other_node(int server_fd, std::string fname, uint8_t* buf, uint64_t msg_size);

/* Sends a GET request and reads header - file size from remote node
 */
int read_header_from_other_node(int server_fd, std::string fname);

/* Returns the node where this sample is cached in memory
 * If not cached anywhere and needs to be fetched from one's own
 * disk, then returns -1
 */
int is_cached_in_other_node(std::vector<std::vector<std::string>>& cache_lists,
                            std::string sample_name, int node_id);


void prefetch_cache(std::vector<std::string> items_not_in_node, int start_idx, int end_idx,
                    std::string file_root);

class CacheEntry {
 public:
  /* Constructor : initialize segment using path of
   * disk file, and populate name. This does not
   * open the file until create or attach segment
   * is called
   */
  explicit CacheEntry(std::string path);


  /* Given a path to the current file on disk
   * that has to be cached in shared memory
   * this function will return the descriptor
   * for the corresponding shared memory
   * segment. If the segment already exists,
   * it returns the fd. If not, it creates and then
   * returns the handle
   */
  int create_segment();

  /* This function must be used when you
   * know the shm segment exists and you want a
   * handle to it. If the segment does not exist,
   * the function returns EINVAL error
   */
  int attach_segment();

  /* Given the path to the shm segment or a name
   * this function returns the pointer to the
   * contents of the file after mmap in its
   * address space. We don't implement this
   * because DALI already does this
   * It mmaps a path, and reads it. We'll update
   * the path to file -> shm
   * In the tmp test implementation, always
   * unmap after reading
   */
  void* get_cache();

  /* Write to the shm segment that is already
   * created and open by mmap into its address space
   * unmaps on exit.
   */
  int put_cache(std::string from_file);
  int put_cache_simple(std::string from_file);

  /* This function returns the shared memory path
   * to this segment. This path can be
   * updated in the dali file map.
   */
  std::string get_shm_path();

  /* Returns the file size
   */
  int get_size() {
    return size_;
  }

  /* This function must be called after
   * an attach_segment or create_segment
   * It closes the opened segment, but
   * does not delete it.
   */
  int close_segment();

  /* Thisfunction must be called to delete
   * the cache.
   * Ideally, this must be called at the end
   * of experiment when you want to c;lear off
   * all entries
   */
  int remove_segment();

  /* name_ is the identifier of the shared
   * segment in the path indicated by prefix
   * To access the segment use path :
   * prefix + name
   */
  std::string name_;

 private:
  // fd of the open file - tmp if create-segment
  //  final if attach-segment
  int fd_ = -1;

  int size_ = 0;
};

}  // end namespace shm

#endif  // DALI_OPERATORS_SHMCACHE_POSIXSHMEM_H_
