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

// A utility file to create, attach, read and write to shared memory segments
// Change all operation shere to work on a tmp file and then rename it atomically

#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/file.h>
#include <sys/ipc.h>
#include <sys/mman.h>
#include <sys/shm.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "dali/operators/reader/loader/commands.h"
#include "dali/operators/shmcache/posixshmem.h"

namespace shm {

const char prefix[] = "/dev/shm/cache/";

int read_header_from_other_node(int server_fd, std::string fname) {
  char request[REQUEST_SIZE] = "GET ";
  strncpy(request + GET_SIZE, fname.c_str(), fname.length());
  int ret = send(server_fd, request, REQUEST_SIZE, 0);
  if (ret < 0) {
    std::cerr << "Error sending request" << std::endl;
    return -1;
  }
  unsigned char header[HEADER_SIZE];
  int r = recv(server_fd, header, HEADER_SIZE, 0);
  if (strcmp(reinterpret_cast<const char *>(header), "NOFOUND") == 0) {
    std::cerr << "File " << fname << " requested was notfound on server cache" << std::endl;
    return -1;
  }

  uint64_t msg_size = *(reinterpret_cast<uint64_t *>(header));
  return msg_size;
}

int read_from_other_node(int server_fd, std::string fname, uint8_t *buf, uint64_t msg_size) {
  char request[REQUEST_SIZE] = "GET ";
  strncpy(request + GET_SIZE, fname.c_str(), fname.length());
  int ret = send(server_fd, request, REQUEST_SIZE, 0);
  if (ret < 0) {
    std::cerr << "Error sending request" << std::endl;
    return -1;
  }

  int msg_read = 0;
  int rec = 0;
  do {
    rec = recv(server_fd, buf + msg_read, msg_size, 0);
    if (rec == HEADER_SIZE) {
      if (strcmp(reinterpret_cast<const char *>(buf), "NOFOUND") == 0) {
        std::cerr << "File " << fname << " requested was notfound on server cache" << std::endl;
        return -1;
      }
    }
    msg_read += rec;
    if (rec < 0) {
      std::cerr << "Error receiving\n";
      return 0;
    }
  } while (rec != 0 && (msg_read < static_cast<int>(msg_size)));

  return rec;
}

int is_cached_in_other_node(std::vector<std::vector<std::string>> &cache_lists,
                            std::string sample_name, int node_id) {
  int num_nodes = cache_lists.size();
  for (int i = 0; i < num_nodes; i++) {
    if (i == node_id) {
      continue;
    } else {
      // do a binary search on cache_list[i]
      bool found = std::binary_search(cache_lists[i].begin(), cache_lists[i].end(), sample_name);
      if (found)
        return i;
    }
  }
  return -1;
}

void prefetch_cache(std::vector<std::string> items_not_in_node, int start_idx, int end_idx,
                    std::string file_root) {
  for (int i = start_idx; i < end_idx; i++) {
    std::string img = items_not_in_node[i];
    CacheEntry *ce = new CacheEntry(img);
    int ret = -1;
    ret = ce->create_segment();
    if (ret == -1) {
      std::cerr << "Cache for " << img << " could not be created." << std::endl;
      return;
    }
    ret = ce->put_cache_simple(file_root + "/" + img);
    if (ret == -1) {
      std::cerr << "Cache for " << img << " could not be populated." << std::endl;
      return;
    }
    delete ce;
  }
}


bool file_exist(const char *filename) {
  struct stat buffer;
  return (stat(filename, &buffer) == 0);
}


std::string create_name(std::string path) {
  std::replace(path.begin(), path.end(), '/', '_');
  return path;
}

std::string shm_path(std::string name, std::string prefix) {
  // assumes prefix ends with '/'
  return prefix + name;
}

int open_shared_file(const char *path, int flags, mode_t mode) {
  if (!path) {
    errno = ENOENT;
  }

  flags |= O_NOFOLLOW | O_CLOEXEC;
  /* Disable asynchronous cancellation.  */
  int state;
  pthread_setcancelstate(PTHREAD_CANCEL_DISABLE, &state);
  int fd = open(path, flags, mode);
  if (fd == -1) {
    std::cerr << "Cannot open shm segment" << std::endl;
  }
  pthread_setcancelstate(state, NULL);
  return fd;
}


int try_mkdir(const char *path, int mode) {
  typedef struct stat stat_dir;
  stat_dir st;
  int status = 0;
  if (stat(path, &st) != 0) {
    if (mkdir(path, mode) != 0 && errno != EEXIST)
      status = -1;
  } else if (!S_ISDIR(st.st_mode)) {
    errno = ENOTDIR;
    status = -1;
  }

  return status;
}

/**
** ensures all directories in path exist
** We start working top-down to ensure
** each directory in path exists.
*/
int mkdir_path(const char *path, mode_t mode) {
  char *pp;
  char *sp;
  int status;
  char *copypath = strdup(path);

  status = 0;
  pp = copypath;
  // Find all occurences of '/' in path : if /a/c/tmp.jpg
  // we need to mkdir(/a) and mkdir (/a/c)
  while (status == 0 && (sp = strchr(pp, '/')) != 0) {
    if (sp != pp) {
      /* Neither root nor double slash in path */
      *sp = '\0';
      if ((status = try_mkdir(copypath, mode)) < 0) {
        std::cerr << "Error creating directory " << copypath << std::endl;
        return -1;
      }
      *sp = '/';
    }
    pp = sp + 1;
  }
  if (status == 0) {
    status = try_mkdir(path, mode);
    if (status < 0) {
      std::cerr << "Error creating dir " << path << std::endl;
      return -1;
    }
  }

  free(copypath);
  return (status);
}


int get_file_size(std::string filename) {
  struct stat st;
  stat(filename.c_str(), &st);
  int size = st.st_size;
  return size;
}

CacheEntry::CacheEntry(std::string path) {
  /* If tyhe path conatins prefix,
   * (can happen when we are attching
   * to an existing cache segment),
   * then the name of the segment is
   * path - prefix
   */
  name_ = path;
  uint64_t found = name_.find(prefix);
  if (found != std::string::npos) {
    // Prefix is found in path
    name_.erase(found, strlen(prefix));
  }
  // name_ = path_;
  // std::replace( name_.begin(), name_.end(), '/', '_');
}

int CacheEntry::create_segment() {
  // Get the unique name for the shm segment
  // name_ =  create_name(path_);

  // Create directories in the file path if they dont exist
  //  Pass only the dir heirarchy to the function.
  //  Strip off the file name
  std::string dir_path(name_);
  if (dir_path.find('/') != std::string::npos) {
    dir_path = dir_path.substr(0, dir_path.rfind("/"));
    int status = mkdir_path((shm_path(dir_path, prefix)).c_str(), 0777);
    if (status < 0) {
      std::cerr << "Error creating path " << shm_path(dir_path, prefix) << std::endl;
      return -1;
    }
  }

  int flags = O_CREAT | O_RDWR;
  int mode = 511;
  // Get the full shm path and open it
  std::string shm_path_name_tmp = shm_path(name_ + "-tmp", prefix);
  fd_ = open_shared_file(shm_path_name_tmp.c_str(), flags, mode);
  close(fd_);
  return fd_;
}

int CacheEntry::attach_segment() {
  // if the shm segment is already open,
  // return the descriptor
  // We could be attaching to a segment when its
  // being written to. So always try
  // opening from the name
  // if (fd_ != -1)
  //   return fd_;

  // Else, open the file without the O_CREAT
  // flags and return the fd
  int flags = O_RDWR;
  int mode = 511;
  std::string shm_path_name;

  shm_path_name = shm_path(name_, prefix);

  fd_ = open_shared_file(shm_path_name.c_str(), flags, mode);
  return fd_;
}


int CacheEntry::put_cache_simple(std::string from_file) {
  int bytes_to_write = get_file_size(from_file);
  std::string shm_path_name = shm_path(name_, prefix);
  std::string shm_path_name_tmp = shm_path(name_ + "-tmp", prefix);
  size_ = bytes_to_write;
  FILE *source, *target;
  size_t n, m;
  unsigned char buff[4096];

  // Open the tmp file holding a lock
  if ((source = fopen(from_file.c_str(), "rb")) == NULL) {
    std::cerr << "File open error " << from_file.c_str() << std::endl;
    return -1;
  }

  if ((target = fopen(shm_path_name_tmp.c_str(), "wb")) == NULL) {
    std::cerr << "File open error " << shm_path_name_tmp << std::endl;
    return -1;
  }
  do {
    n = fread(buff, 1, sizeof buff, source);
    if (n)
      m = fwrite(buff, 1, n, target);
    else
      m = 0;
  } while ((n > 0) && (n == m));
  if (m) {
    perror("copy");
    return -1;
  }

  fclose(source);
  fclose(target);
  int ret = -1;
  // rename the file

  if ((ret = rename(shm_path_name_tmp.c_str(), shm_path_name.c_str())) < 0) {
    if (file_exist(shm_path_name.c_str())) {
      return size_;
    } else {
      std::cerr << "Caching rename failed : " << strerror(errno) << std::endl;
      return -1;
    }
  }

  return size_;
}


int CacheEntry::put_cache(std::string from_file) {
  int bytes_to_write = get_file_size(from_file);
  size_ = bytes_to_write;
  if (fd_ < 0) {
    errno = EINVAL;
    std::cerr << "File " << name_ << " has invalid decriptor" << std::endl;
    return -1;
  }
  ftruncate(fd_, bytes_to_write);

  // mmap the shm file to get ptr
  void *ptr = nullptr;
  if ((ptr = mmap(0, bytes_to_write, PROT_WRITE, MAP_SHARED, fd_, 0)) == MAP_FAILED) {
    std::cerr << "mmap error" << std::endl;
    return -1;
  }


  // write to shared memory segment
  // We will mmap the file to read from, because
  // in DALI, the file to be read will be mmaped first.
  void *ptr_from = nullptr;
  int fd_from = -1;
  if ((fd_from = open(from_file.c_str(), O_RDONLY)) < 0) {
    std::cerr << "Open failed" << std::endl;
    return -1;
  }
  if ((ptr_from = mmap(0, bytes_to_write, PROT_READ, MAP_SHARED, fd_from, 0)) == MAP_FAILED) {
    std::cerr << "mmap error" << std::endl;
    return -1;
  }
  std::shared_ptr<void> p_;
  p_ = std::shared_ptr<void>(ptr_from, [=](void *) { munmap(ptr_from, bytes_to_write); });

  // Do the memcpy now
  memcpy(ptr, p_.get(), bytes_to_write);
  int ret = 0;

  // Now unmap both files
  if ((ret = munmap(ptr, bytes_to_write)) == -1) {
    std::cerr << "Munmap failed" << std::endl;
    return -1;
  }

  if ((ret = munmap(ptr_from, bytes_to_write)) == -1) {
    std::cerr << "Munmap failed" << std::endl;
    return -1;
  }

  close(fd_from);
  // close the tmp file
  close(fd_);

  std::string shm_path_name = shm_path(name_, prefix);
  std::string shm_path_name_tmp = shm_path(name_ + "-tmp", prefix);
  // rename the file
  if ((ret = rename(shm_path_name_tmp.c_str(), shm_path_name.c_str())) < 0) {
    std::cerr << "Caching rename failed" << std::endl;
    return -1;
  }

  return bytes_to_write;
}


void *CacheEntry::get_cache() {
  std::string from_file = prefix + name_;
  int bytes_to_read = get_file_size(from_file);
  size_ = bytes_to_read;

  // If the descriptor is invalid, you need to sttach the segment.
  if (fd_ < 0) {
    errno = EINVAL;
    std::cerr << "File " << name_ << " has invalid decriptor" << std::endl;
    return nullptr;
  }
  // mmap the shm file to get ptr
  void *ptr = nullptr;
  if ((ptr = mmap(0, bytes_to_read, PROT_READ, MAP_SHARED, fd_, 0)) == MAP_FAILED) {
    std::cerr << "mmap error" << std::endl;
    return nullptr;
  }

  return ptr;
}

std::string CacheEntry::get_shm_path() {
  std::string shm_path_name;
  shm_path_name = shm_path(name_, prefix);
  return shm_path_name;
}

int CacheEntry::close_segment() {
  int ret = 0;
  if (fd_ > -1) {
    if ((ret = close(fd_)) < 0) {
      std::cerr << "File " << prefix + name_ << " close failed" << std::endl;
      return -1;
    }
  }
  return 0;
}

int CacheEntry::remove_segment() {
  std::string shm_path_name;
  shm_path_name = shm_path(name_, prefix);
  int result = unlink(shm_path_name.c_str());
  return result;
}

}  // end namespace shm
