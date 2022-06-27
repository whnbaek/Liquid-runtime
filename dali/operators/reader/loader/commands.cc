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

#include "dali/operators/reader/loader/commands.h"

namespace dali {

bool set_recv_window(int sockfd, int len_bytes) {
    socklen_t i;
    i = sizeof(int);

    if (setsockopt(sockfd, SOL_SOCKET, SO_RCVBUF, &len_bytes, i) < 0) {
        std::cerr << "Error setting recvbuf size" << strerror(errno) << std::endl;
        return false;
    }
    return true;
}

bool set_send_window(int sockfd, int len_bytes) {
    socklen_t i;
    i = sizeof(int);

    if (setsockopt(sockfd, SOL_SOCKET, SO_SNDBUF, &len_bytes, i) < 0) {
        std::cerr << "Error setting sendbuf size" << strerror(errno) << std::endl;
        return false;
    }
    return true;
}

bool set_tcp_nodelay(int sockfd) {
    int yes = 1;

    if (setsockopt(sockfd, IPPROTO_TCP, TCP_NODELAY, reinterpret_cast<char*>(&yes), sizeof(int)) <
        0) {
      std::cerr << "Error setting tcp nodel" << strerror(errno) << std::endl;
      return false;
    }
    return true;
}

}  // namespace dali
