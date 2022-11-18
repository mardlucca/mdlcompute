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

#include <gtest/gtest.h>

#include <mdlcompute.h>
#include <iostream>
#include <string>

using std::cout;
using std::endl; 

namespace mdl {
namespace compute {
namespace compute_test {



  TEST(ArgBuffersTetSuite, TestInBuffer) {
    int x = 10;
    in_buffer buff = in(x);

    ASSERT_EQ(sizeof(int), buff.size);
    ASSERT_TRUE(buff.id > 0);
    const int* ptr = static_cast<const int *>(buff.data);
    ASSERT_EQ(x, ptr[0]);

    in_buffer buff2 = in(100L);

    ASSERT_EQ(sizeof(long), buff2.size);
    ASSERT_EQ(buff.id + 1, buff2.id);
    // cast must be const
    const long * ptr2 = static_cast<const long *>(buff2.data);
    ASSERT_EQ(100L, ptr2[0]);

    in_buffer buff3 = buff2;
    ASSERT_EQ(sizeof(long), buff3.size);
    ASSERT_EQ(buff2.id, buff3.id);
    const long * ptr3 = static_cast<const long *>(buff3.data);
    ASSERT_EQ(100L, ptr3[0]);
  }

  TEST(ArgBuffersTetSuite, TestInBuffer_BoundedArray) {
    int x[] = {1, 2, 3, 4, 5};
    in_buffer buff = in(x);

    ASSERT_EQ(5 * sizeof(int), sizeof(x));
    ASSERT_EQ(sizeof(x), buff.size);
    ASSERT_TRUE(buff.id > 0);
    ASSERT_EQ(&x, buff.data);
  }

  void unboundedArray(int a[], std::size_t size) {
    in_buffer buff = in(a, size);
    ASSERT_EQ(buff.size, size);
    ASSERT_EQ(a, buff.data);
  }

  TEST(ArgBuffersTetSuite, TestInBuffer_UnboundedArray) {
    int x[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    unboundedArray(x, sizeof(x));
  }

  TEST(ArgBuffersTetSuite, TestInBuffer_Pointer) {
    const std::uint8_t x = 1;
    const std::uint8_t * px = &x;

    in_buffer buff = in(px);
    ASSERT_EQ(&x, buff.data);
    ASSERT_EQ(sizeof(x), buff.size);
  }

  TEST(ArgBuffersTetSuite, TestInBuffer_Vector) {
    const std::vector<int> v = {10, 20, 30};

    in_buffer buff = in(v);
    ASSERT_EQ(v.data(), buff.data);
    ASSERT_EQ(v.size() * sizeof(int), buff.size);
  }

  TEST(ArgBuffersTetSuite, TestInBuffer_StdArray) {
    const std::array<std::uint16_t, 10> a = {1, 2, 3};

    in_buffer buff = in(a);
    ASSERT_EQ(a.data(), buff.data);
    ASSERT_EQ(a.size() * sizeof(std::uint16_t), buff.size);
  }

  TEST(ArgBuffersTetSuite, TestInOutBuffer) {
    std::uint8_t x = 1;

    inout_buffer buff = inout(x);
    ASSERT_EQ(&x, buff.data);
    ASSERT_EQ(sizeof(x), buff.size);
  }

  TEST(ArgBuffersTetSuite, TestInOutBuffer_Pointer) {
    std::uint8_t x = 1;
    std::uint8_t * px = &x;

    inout_buffer buff = inout(px);
    ASSERT_EQ(&x, buff.data);
    ASSERT_EQ(sizeof(x), buff.size);
  }

  TEST(ArgBuffersTetSuite, TestInOutBuffer_Vector) {
    std::vector<int> v = {10, 20, 30};

    inout_buffer buff = inout(v);
    ASSERT_EQ(v.data(), buff.data);
    ASSERT_EQ(v.size() * sizeof(int), buff.size);
  }

  TEST(ArgBuffersTetSuite, TestInOutBuffer_StdArray) {
    std::array<std::uint16_t, 10> a;

    inout_buffer buff = inout(a);
    ASSERT_EQ(a.data(), buff.data);
    ASSERT_EQ(a.size() * sizeof(std::uint16_t), buff.size);
  }

} // compute_test
} // compute
} // mdl