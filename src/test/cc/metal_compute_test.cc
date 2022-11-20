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

#include <mdl/compute.h>
#include <iostream>
#include <string>
#include <vector>

using std::cout;
using std::endl; 

namespace mdl {
namespace compute {
namespace compute_test {

  const char* shaderSrc = R"(
      #include <metal_stdlib>
      using namespace metal;

      kernel void add_arrays(device const float* inA,
                            device const float* inB,
                            device float* result,
                            uint index [[thread_position_in_grid]])
      {
          // the for-loop is replaced with a collection of threads, each of which
          // calls this function.
          result[index] = 1.0 + inA[index] + inB[index];
      }

      kernel void init_array(device float* result,
                            uint index [[thread_position_in_grid]])
      {
          // the for-loop is replaced with a collection of threads, each of which
          // calls this function.
          result[index] = 1 + index;
      }
  )";

  const char* shaderSrc2 = R"(
      #include <metal_stdlib>
      using namespace metal;

      kernel void swap(device float* inA [[buffer(0)]], 
                       device float* inB [[buffer(1)]],
                       uint index [[ thread_position_in_grid ]])
      {
          float tmp = inA[index];
          inA[index] = inB[index];
          inB[index] = tmp;
      }
  )";

  const char* shaderSrc3 = R"(
      #include <metal_stdlib>
      using namespace metal;

      kernel void copy(device float* inA [[buffer(0)]], 
                       device float* outB [[buffer(1)]],
                       uint index [[ thread_position_in_grid ]])
      {
          outB[index] = inA[index];
      }

      kernel void set(device float* outA [[buffer(0)]], 
                       const device float& value [[buffer(1)]],
                       uint index [[ thread_position_in_grid ]])
      {
          outA[index] = value;
      }
  )";

  const char* shaderSrcWithError = R"(
      #include <metal_stdlib>
      using namespace metal;

      kernel void add_arrays(device const float* inA,`
                            device const float* inB,
                            device float* result,
                            uint index [[thread_position_in_grid]])
      {
  )";

  TEST(ComputeTestSuite, TestComputeEngineAvailable) {
    MetalComputeEngine engine;
    ASSERT_TRUE(engine.Available());
  }

  TEST(ComputeTestSuite, TestLoadLibrary) {
    MetalComputeEngine engine;
    engine.LoadLibrary(shaderSrc);
    ASSERT_TRUE(engine.ContainsFunction("init_array"));
    ASSERT_TRUE(engine.ContainsFunction("add_arrays"));
    ASSERT_FALSE(engine.ContainsFunction("bogus"));
  }

  TEST(ComputeTestSuite, TestLoadLibrary_WithCompilationError) {
    MetalComputeEngine engine;
    ASSERT_THROW(engine.LoadLibrary(shaderSrcWithError), CompilationException);
  }

  TEST(ComputeTestSuite, TestCall_InOut) {
    MetalComputeEngine engine;
    engine.LoadLibrary(shaderSrc2);

    const int kSize = 10;
    float f1[kSize];
    float f2[kSize];

    for (int i = 0; i < kSize; i++) {
      f1[i] = i;
      f2[i] = kSize - i - 1;
    }
    
    engine.NewBatch()
        .WithGrid(1, kSize, 1, kSize).Call("swap", inout(f1), inout(f2))
        .Dispatch().Wait();

    for (int i = 0; i < kSize; i++) {
      ASSERT_FLOAT_EQ(kSize - i - 1, f1[i]);
      ASSERT_FLOAT_EQ(i, f2[i]);
    }

    engine.NewBatch()
        .WithGrid(1, kSize - 1, 1, kSize - 1).Call("swap", inout(f1), inout(f2))
        .Dispatch().Wait();

    for (int i = 0; i < kSize - 1; i++) {
      ASSERT_FLOAT_EQ(i, f1[i]);
      ASSERT_FLOAT_EQ(kSize - i - 1, f2[i]);
    }
    ASSERT_FLOAT_EQ(9, f2[9]);
    ASSERT_FLOAT_EQ(kSize - 9- 1, f1[9]);
  }

  TEST(ComputeTestSuite, TestCall_In) {
    MetalComputeEngine engine;
    engine.LoadLibrary(shaderSrc2);
    engine.LoadLibrary(shaderSrc3);

    const int kSize = 10;
    float f1[kSize];
    
    for (int i = 0; i < kSize; i++) {
      f1[i] = i;
    }

    // modifying an "in" buffer has no effect
    engine.NewBatch()
        .WithGrid(1, kSize, 1, kSize).Call("set", in(f1), 2.0f)
        .Dispatch().Wait();

    for (int i = 0; i < kSize; i++) {
      ASSERT_FLOAT_EQ(i, f1[i]);
    }
  }  

  TEST(ComputeTestSuite, TestCall_DefaultArgumentType) {
    MetalComputeEngine engine;
    engine.LoadLibrary(shaderSrc2);
    engine.LoadLibrary(shaderSrc3);

    const int kSize = 10;
    float f1[kSize];
    
    for (int i = 0; i < kSize; i++) {
      f1[i] = i;
    }

    // by default, arguments are considered type "in"
    engine.NewBatch()
        .WithGrid(1, kSize, 1, kSize).Call("set", f1, 2.0f)
        .Dispatch().Wait();

    for (int i = 0; i < kSize; i++) {
      ASSERT_FLOAT_EQ(i, f1[i]);
    }
  }  

  TEST(ComputeTestSuite, TestCall_Out) {
    MetalComputeEngine engine;
    engine.LoadLibrary(shaderSrc2);
    engine.LoadLibrary(shaderSrc3);

    const int kSize = 10;
    float f1[kSize];
    
    engine.NewBatch()
        .WithGrid(1, kSize, 1, kSize).Call("set", out(f1), 2.0f)
        .Dispatch().Wait();

    for (int i = 0; i < kSize; i++) {
      ASSERT_FLOAT_EQ(2.0f, f1[i]);
    }
  }  

  TEST(ComputeTestSuite, TestCall_Shared) {
    MetalComputeEngine engine;
    engine.LoadLibrary(shaderSrc2);

    const int kSize = 10;
    float f1[kSize];
    float f2[kSize];

    for (int i = 0; i < kSize; i++) {
      f1[i] = i;
      f2[i] = kSize - i - 1;
    }
    
    engine.NewBatch()
        .WithGrid(1, kSize, 1, kSize).Call("swap", shared(f1), shared(f2))
        .Dispatch().Wait();

    for (int i = 0; i < kSize; i++) {
      ASSERT_FLOAT_EQ(kSize - i - 1, f1[i]);
      ASSERT_FLOAT_EQ(i, f2[i]);
    }
  }

  TEST(ComputeTestSuite, TestCall_PrivateBuffer) {
    MetalComputeEngine engine;
    engine.LoadLibrary(shaderSrc2);
    engine.LoadLibrary(shaderSrc3);

    const int kSize = 10;
    float f1[kSize];
    float f2[kSize];

    for (int i = 0; i < kSize; i++) {
      f1[i] = i;
    }
    
    auto p1 = priv(sizeof(f1));
    engine.NewBatch()
        .WithGrid(1, kSize, 1, kSize).Call("copy", f1, p1)    // same as in(f1)
        .WithGrid(1, kSize, 1, kSize).Call("copy", p1, out(f2))
        .Dispatch().Wait();

    for (int i = 0; i < kSize; i++) {
      ASSERT_FLOAT_EQ(f1[i], f2[i]);
    }
  }  

  TEST(ComputeTestSuite, TestCall_InexistentFn) {
    MetalComputeEngine engine;
    engine.LoadLibrary(shaderSrc2);
    engine.LoadLibrary(shaderSrc3);

    const int kSize = 10;
    float f1[kSize];
    float f2[kSize];

    for (int i = 0; i < kSize; i++) {
      f1[i] = i;
    }
    
    auto p1 = priv(sizeof(f1));
    ASSERT_THROW(
        engine.NewBatch()
          .WithGrid(1, kSize, 1, kSize).Call("copy", f1, p1)    // same as in(f1)
          .WithGrid(1, kSize, 1, kSize).Call("copyBogus", p1, out(f2))
          .Dispatch().Wait(), 
        FunctionNotFoundException);
  }

  TEST(ComputeTestSuite, TestCall_DynamicArray) {
    MetalComputeEngine engine;
    engine.LoadLibrary(shaderSrc2);
    engine.LoadLibrary(shaderSrc3);

    int kSize = 10;
    float * f1 = new float[kSize];
    
    engine.NewBatch()
        .WithGrid(1, kSize, 1, kSize).Call("set", out(f1, kSize * sizeof(float)), 2.0f)
        .Dispatch().Wait();

    for (int i = 0; i < kSize; i++) {
      ASSERT_FLOAT_EQ(2.0f, f1[i]);
    }

    delete[] f1;
  }  

  TEST(ComputeTestSuite, TestCall_Vector) {
    MetalComputeEngine engine;
    engine.LoadLibrary(shaderSrc2);
    engine.LoadLibrary(shaderSrc3);

    std::vector<float> v(10);
    
    engine.NewBatch()
        .WithGrid(1, v.size(), 1, v.size()).Call("set", out(v), 2.0f)
        .Dispatch().Wait();

    for (int i = 0; i < v.size(); i++) {
      ASSERT_FLOAT_EQ(2.0f, v[i]);
    }
  }  

  TEST(ComputeTestSuite, TestCall_PointerToVector) {
    MetalComputeEngine engine;
    engine.LoadLibrary(shaderSrc2);
    engine.LoadLibrary(shaderSrc3);

    std::vector<float> v1(10);
    std::vector<float> v2(10);
    
    engine.NewBatch()
        .WithGrid(1, v1.size(), 1, v1.size()).Call("set", out(v1), 11.0f)
        .WithGrid(1, v2.size(), 1, v2.size()).Call("set", out(v2), 12.0f)
        .Dispatch().Wait();

    for (int i = 0; i < v1.size(); i++) {
      ASSERT_FLOAT_EQ(11.0f, v1[i]);
      ASSERT_FLOAT_EQ(12.0f, v2[i]);
    }

    std::vector<float>* p1 = &v1;
    std::vector<float>* p2 = &v2;

    engine.NewBatch()
        .WithGrid(1, v2.size(), 1, v2.size()).Call("copy", p1, out(p2))
        .Dispatch().Wait();

    for (int i = 0; i < v1.size(); i++) {
      ASSERT_FLOAT_EQ(11.0f, v1[i]);
      ASSERT_FLOAT_EQ(11.0f, v2[i]);
    }
  }  
} // compute_test
} // compute
} // mdl