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

#ifndef DALI_OPERATORS_READER_LOADER_COMMANDS_H_
#define DALI_OPERATORS_READER_LOADER_COMMANDS_H_

#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <sys/uio.h>
#include <unistd.h>

#include <iostream>
#include <fstream>

#define REQUEST_SIZE 100
#define GET_SIZE 4
#define HEADER_SIZE 8
#define SOCK_CLOSED -2
#define SOCK_ERROR -3
#define SUCCESS 2
#define NOT_FOUND "NOTFOUND"
#define PORT 5555

namespace dali {

bool set_recv_window(int sockfd, int len_bytes);
bool set_send_window(int sockfd, int len_bytes);
bool set_tcp_nodelay(int sockfd);
}  // namesapce dali
#endif  // DALI_OPERATORS_READER_LOADER_COMMANDS_H_
