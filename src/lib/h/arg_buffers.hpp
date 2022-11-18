// Copyright (c) 2022, Marcio Lucca
// All rights reserved.
// 
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// 
// 1. Redistributions of source code must retain the above copyright notice, this
//   list of conditions and the following disclaimer.
// 
// 2. Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
// 
// 3. Neither the name of the copyright holder nor the names of its
//   contributors may be used to endorse or promote products derived from
//   this software without specific prior written permission.
// 
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#ifndef _MDL_COMPUTE_ARG_BUFFERS
#define _MDL_COMPUTE_ARG_BUFFERS

#include <array>
#include <atomic>
#include <type_traits>
#include <utility>
#include <vector>

namespace mdl {
namespace compute {
  extern std::atomic_size_t idSeq;

  enum class BufferType {
    In, Out, InOut, Private, Shared
  };


  template <BufferType BT, class DT = void*>
  struct buffer {
    std::uint64_t id;
    DT data;
    std::size_t size;

    BufferType GetType() const { return BT; }
  };


  typedef buffer<BufferType::In, const void*> in_buffer;
  typedef buffer<BufferType::InOut> inout_buffer;
  typedef buffer<BufferType::Out> out_buffer;
  typedef buffer<BufferType::Private, const void*> private_buffer;
  typedef buffer<BufferType::Shared> shared_buffer;


  template <class T>
  struct sizefn {
    std::size_t operator()(const T& value) {
      if constexpr(std::is_pointer_v<T>) {
        return sizefn<typename std::remove_pointer<T>::type>{}(*value);
      } else {
        return sizeof(value);
      }
    }
  };
  
  template <class T>
  struct sizefn<std::vector<T>> {
    std::size_t operator()(const std::vector<T>& value) {
      return value.size() * sizeof(T);
    }
  };

  template <class T, std::size_t N>
  struct sizefn<std::array<T, N>> {
    std::size_t operator()(const std::array<T, N>& value) {
      return value.size() * sizeof(T);
    }
  };



  template <class T>
  struct addressfn {
    void * operator()(T& value) {
      if constexpr(std::is_pointer_v<T>) {
        return addressfn<typename std::remove_pointer<T>::type>{}(*value);
      } else {
        return static_cast<void *>(&value);
      }
    }

    const void * operator()(const T& value) {
      if constexpr(std::is_pointer_v<T>) {
        typedef typename std::remove_pointer<T>::type cType;
        typedef typename std::remove_const<cType>::type type;
        return addressfn<type>{}(*value);
      } else {
        return static_cast<const void *>(&value);
      }
    }
  };

  template <class T>
  struct addressfn<std::vector<T>> {
    void * operator()(std::vector<T>& value) {
      return value.data();
    }

    const void * operator()(const std::vector<T>& value) {
      return value.data();
    }
  };

  template <class T, std::size_t N>
  struct addressfn<std::array<T, N>> {
    void * operator()(std::array<T, N>& value) {
      return value.data();
    }

    const void * operator()(const std::array<T, N>& value) {
      return value.data();
    }
  };



  template <class T>
  in_buffer in(const T& val, std::size_t size = 0) {
    return {
      .id = ++idSeq, 
      .data = addressfn<T>{}(val), 
      .size = size > 0 ? size : sizefn<T>{}(val)
    };
  }

  template <class T>
  inout_buffer inout(T& val, std::size_t size = 0) {
    return {
      .id = ++idSeq, 
      .data = addressfn<T>{}(val), 
      .size = size > 0 ? size : sizefn<T>{}(val)
    };
  }

  template <class T>
  out_buffer out(T& val, std::size_t size = 0) {
    return {
      .id = ++idSeq, 
      .data = addressfn<T>{}(val), 
      .size = size > 0 ? size : sizefn<T>{}(val)
    };
  }

  private_buffer priv(std::size_t size);

  template <class T>
  shared_buffer shared(T& val, std::size_t size = 0) {
    return {
      .id = ++idSeq, 
      .data = addressfn<T>{}(val), 
      .size = size > 0 ? size : sizefn<T>{}(val)
    };
  }
} // compute
} // mdl

#endif // _MDL_COMPUTE_ARG_BUFFERS