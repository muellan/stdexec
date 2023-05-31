/*
 * Copyright (c) 2022 NVIDIA Corporation
 *
 * Licensed under the Apache License Version 2.0 with LLVM Exceptions
 * (the "License"); you may not use this file except in compliance with
 * the License. You may obtain a copy of the License at
 *
 *   https://llvm.org/LICENSE.txt
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <nvexec/stream_context.cuh>
#include <stdexec/execution.hpp>

#include <thrust/device_vector.h>

#include <cstdio>
#include <span>

namespace ex = stdexec;

struct move_only_t {
  static constexpr int invalid() { return -42; }

  move_only_t() = delete;
  move_only_t(const move_only_t&) = delete;
  move_only_t& operator=(move_only_t&&) = delete;
  move_only_t& operator=(const move_only_t&) = delete;

  __host__ __device__ move_only_t(int data)
    : data_(data)
    , self_(this) {
    std::printf("move_only_t(%d), self = %p\n", data_, (void*)self_);
  }

  __host__ __device__ move_only_t(move_only_t&& other)
    : data_(std::exchange(other.data_, invalid()))
    , self_(this) {
    std::printf("move_only_t(move_only_t(%d)), self = %p, other = %p\n", data_, (void*)self_, (void*)&other);
  }

  __host__ __device__ ~move_only_t() {
    std::printf("move_only_t::~move_only_t, this = %p, self = %p\n", (void*)this, (void*)self_);
    if (this != self_) {
      // TODO Trap
      std::printf("Error: move_only_t::~move_only_t failed\n");
    }
    data_ = invalid();
  }

  __host__ __device__ bool contains(int val) {
    if (this != self_) {
      std::printf("Error: move_only_t::contains failed: %p != %p\n", (void*)this, (void*)self_);
      return false;
    }

    return data_ == val;
  }

  int data_{invalid()};
  move_only_t* self_{(move_only_t*)(std::uintptr_t)(std::intptr_t)-1};
};


int main() {
  nvexec::stream_context ctx;

  double* inout = nullptr;
  const int nelems = 10;
  cudaMallocManaged(&inout, nelems*sizeof(double));

  auto task =
      stdexec::transfer_just(ctx.get_scheduler(),std::span<double>{inout,nelems})
  |   stdexec::bulk(nelems,
          [](std::size_t i, std::span<double> out){ out[i] = i; })
  |   stdexec::let_value(
          [](std::span<double> out){ return stdexec::just(out); })
  |   stdexec::bulk(nelems,
          [](std::size_t i, std::span<double> out){ out[i] = 2.0 * out[i]; });

  try {
    stdexec::sync_wait(std::move(task)).value();

    for (int i = 0; i < nelems; ++i) { //
      std::cout << inout[i] << ' ';
    }
    std::cout << '\n';
  }
  catch(cudaError_t err) {
    std::printf("CUDA error: caught \"%s\" in main()\n", cudaGetErrorString(err));
  }

  cudaFree(inout);

  // const int n = 2 * 1024;
  // thrust::device_vector<float> input(n, 1.0f);
  // float* first = thrust::raw_pointer_cast(input.data());
  // float* last = thrust::raw_pointer_cast(input.data()) + input.size();

  // nvexec::stream_context stream_ctx{};

  // auto snd = ex::transfer_just(stream_ctx.get_scheduler(), std::span{first, last})
  //          | nvexec::reduce(42.0f);

  // auto [result] = stdexec::sync_wait(std::move(snd)).value();

  // std::cout << "result: " << result << std::endl;
}
