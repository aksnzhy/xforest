//------------------------------------------------------------------------------
// Copyright (c) 2018 by contributors. All Rights Reserved.
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
//------------------------------------------------------------------------------

/*!
*  Copyright (c) 2018 by Contributors
* \file socket_communicator.h
* \brief Implementation of Communicator class with TCP socket.
*/
#include <thread>

#include "src/network/socket_communicator.h"
#include "src/base/split_string.h"

namespace xforest {

const int kTimeOut = 5;  // 5 minutes for socket timeout
const int kMaxConnection = 1024;  // 1024 maximal socket connection
const int kMaxBuffer = 200 * 1024 * 1024;  // 200 MB bufferfor each message

bool SocketCommunicator::Initialize(bool is_sender,
                                    const char* ip,
                                    int port,
                                    int num_sender,
                                    int queue_size) {
  if (is_sender) {
    is_sender_ = true;
    return InitSender(ip, port);
  } else {
    is_sender_ = false;
    return InitReceiver(ip, port, num_sender, queue_size);
  }
}

bool SocketCommunicator::InitSender(const char* ip, int port) {
  // Sender only has a client socket
  socket_.resize(1);
  socket_[0] = new TCPSocket();
  TCPSocket* client = socket_[0];
  // Connect to server
  if (client->Connect(ip, port)) {
    LOG(INFO) << "Connected to " << ip << ":" << port;
    return true;
  } else {
    LOG(ERROR) << "Cannot connect to " << ip << ":" << port;
    return false;
  }
}

bool SocketCommunicator::InitReceiver(const char* ip, 
                                      int port, 
                                      int num_sender, 
                                      int queue_size) {
  CHECK(num_sender >= 1);
  CHECK(queue_size > 0);
  // Init message queue
  num_sender_ = num_sender;
  queue_size_ = queue_size;
  queue_ = new MessageQueue(queue_size_, num_sender_);
  // Init socket, and socket_[0] is the server socket
  socket_.resize(num_sender+1);
  thread_.resize(num_sender);
  socket_[0] = new TCPSocket();
  TCPSocket* server = socket_[0];
  server->SetTimeout(kTimeOut * 60 * 1000);  // millsec
  // Bind socket
  if (server->Bind(ip, port) == false) {
    LOG(ERROR) << "Cannot bind to " << ip << ":" << port;
    return false;
  }
  LOG(INFO) << "Bind to " << ip << ":" << port;
  // Listen
  if (server->Listen(kMaxConnection) == false) {
    LOG(ERROR) << "Cannot listen on " << ip << ":" << port;
    return false;
  }
  LOG(INFO) << "Listen on " << ip << ":" << port << ", wait sender connect ...";
  // Accept all sender sockets
  std::string accept_ip;
  int accept_port;
  for (int i = 1; i <= num_sender_; ++i) {
    socket_[i] = new TCPSocket();
    if (server->Accept(socket_[i], &accept_ip, &accept_port) == false) {
      LOG(ERROR) << "Error on accept socket.";
      return false;
    }
    // new thread for the socket
    thread_[i-1] = new std::thread(MsgHandler, socket_[i], queue_);
    LOG(INFO) << "Accept new sender: " << accept_ip << ":" << accept_port;
  }

  return true;
}

void SocketCommunicator::MsgHandler(TCPSocket* socket, MessageQueue* queue) {
  char* buffer = new char[kMaxBuffer];
  for (;;) {
    // First recv the size 
    int received_bytes = 0;
    int data_size = 0;
    while (received_bytes < sizeof(int)) {
      int max_len = sizeof(int) - received_bytes;
      int tmp = socket->Receive((char*)(&data_size)+received_bytes, max_len);
      received_bytes += tmp;
    }
    if (data_size <= 0) {
      LOG(INFO) << "Socket finish job";
      break;
    }
    // Then recv the data
    received_bytes = 0;
    while (received_bytes < data_size) {
      int max_len = data_size - received_bytes;
      int tmp = socket->Receive(buffer, max_len);
      received_bytes += tmp;
    }
    queue->Add(buffer, data_size);
  }
  delete [] buffer;
}

void SocketCommunicator::Finalize() {
  if (is_sender_) {
    FinalizeSender();
  } else {
    FinalizeReceiver();
  }
}

void SocketCommunicator::FinalizeSender() {
  // We send a size = -1 signal to notify
  // receiver to finish its job
  if (socket_[0] != nullptr) {
    int size = -1;
    int sent_bytes = 0;
    while (sent_bytes < sizeof(int)) {
      int max_len = sizeof(int) - sent_bytes;
      int tmp = socket_[0]->Send((char*)(&size)+sent_bytes, max_len);
      sent_bytes += tmp;
    }
    socket_[0]->Close();
    delete socket_[0];
    socket_[0] = nullptr;
  }
}

void SocketCommunicator::FinalizeReceiver() {
  for (int i = 0; i <= num_sender_; ++i) {
    if (socket_[i] != nullptr) {
      socket_[i]->Close();
      delete socket_[i];
      socket_[i] = nullptr;
    }
  }
}

int SocketCommunicator::Send(const char* src, int size) {
  if (!is_sender_) {
    LOG(ERROR) << "Receiver cannot invoke send() API.";
    return -1;
  }
  TCPSocket* client = socket_[0];
  // First sent the size of data
  int sent_bytes = 0;
  while (sent_bytes < sizeof(int)) {
    int max_len = sizeof(int) - sent_bytes;
    int tmp = client->Send((char*)(&size)+sent_bytes, max_len);
    sent_bytes += tmp;
  }
  // Then send the data
  sent_bytes = 0;
  while (sent_bytes < size) {
    int max_len = size - sent_bytes;
    int tmp = client->Send(src+sent_bytes, max_len);
    sent_bytes += tmp;
  }

  return size + sizeof(int);
}

int SocketCommunicator::Receive(char* dest, int max_size) {
  if (is_sender_) {
    LOG(ERROR) << "Sender cannot invoke Receive() API.";
    return -1;
  }
  // Get message from the message queue
  return queue_->Remove(dest, max_size);
}

}  // namespace xforest