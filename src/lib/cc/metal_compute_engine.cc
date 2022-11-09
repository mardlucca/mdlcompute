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

#include "../h/compute_exception.h"
#include "../h/metal_compute_engine.h"

#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>

#include <iostream>
#include <string>
#include <string_view>

using std::cout;
using std::endl;

namespace mdl {
namespace compute {

  MetalComputeEngine::MetalComputeEngine() {
    device = MTL::CreateSystemDefaultDevice()->retain();
    commandQueue = device->newCommandQueue();
  }

  MetalComputeEngine::~MetalComputeEngine() {
      for (auto it = libraries.begin(); it != libraries.end(); it++) {
        (*it)->release();
      }
      libraries.clear();
      Release(commandQueue);
      Release(device);
  }

  bool MetalComputeEngine::Available() const {
    return device != nullptr && commandQueue != nullptr;
  }

  void MetalComputeEngine::LoadLibrary(const std::string& sourceCode) {
    NS::Error* error = nullptr;
    MTL::Library* library = device->newLibrary(
        NS::String::string(sourceCode.c_str(), NS::StringEncoding::UTF8StringEncoding), 
        nullptr, 
        &error);
    if (!library) {
      throw CompilationException(error->description()->utf8String());
    }

    libraries.push_back(library);
    NS::Array * functions = library->functionNames();
    for (int i = 0; i < functions->count(); i++) {
      libraryByFn[functions->object(i)->description()->utf8String()] = library;
    }
  }

  bool MetalComputeEngine::ContainsFunction(const std::string& functionName) const {
    return libraryByFn.count(functionName);
  }
} // compute
} // mdl


// int main() {

//   NS::Error* error = nullptr;
//   MTL::Library* library = device->newLibrary(
//        NS::String::string(shaderSrc, NS::StringEncoding::UTF8StringEncoding), nullptr, &error );

//   cout << "Created library" << endl;

//   // MTL::Function * fn = library->newFunction(NS::String::string("add_arrays", NS::StringEncoding::UTF8StringEncoding));
//   MTL::Function * fn2 = library->newFunction(NS::String::string("init_array", NS::StringEncoding::UTF8StringEncoding));
//   cout << "Created function" << endl;

//   MTL::ComputePipelineState * pipeline = device->newComputePipelineState(fn2, &error);
//   cout << "Created pipeline" << endl;

//   if (fn2) { fn2->release(); }
//   if (library) { library->release(); }


//   MTL::CommandBuffer * commandBuffer = queue->commandBuffer();
//   cout << "created command buffer" << endl; 

//   MTL::ComputeCommandEncoder * encoder = commandBuffer->computeCommandEncoder();
//   cout << "created encoder" << endl; 

//   const int len = 20;
//   const int buffSize = len * sizeof(float);
//   MTL::Buffer* result = device->newBuffer(buffSize, MTL::ResourceStorageModeShared);
//   cout << "created data" << endl; 


//   encoder->setComputePipelineState(pipeline);
//   encoder->setBuffer(result, 0, 0);

//   auto maxSize = pipeline->maxTotalThreadsPerThreadgroup();
//   cout << "Max thread group size: " << maxSize << endl;;
//   maxSize = std::min(maxSize, (decltype(maxSize)) buffSize);
//   cout << "Max thread group size: " << maxSize << endl;;
//   cout << "threadExecutionWidth: " << pipeline->threadExecutionWidth() << endl;

//   MTL::Size threadGroupSize(maxSize, 1, 1);
//   MTL::Size gridSize(buffSize, 1, 1);

//   encoder->dispatchThreads(gridSize, threadGroupSize);
//   encoder->endEncoding();
//   cout << "encoder set" << endl; 

//   commandBuffer->commit();
//   cout << "command buffer commit" << endl; 

//   commandBuffer->waitUntilCompleted();
//   cout << "Computation finished" << endl; 

//   double sum = 0.0;
//   float * d = (float *) result->contents();
//   for (int i = 0; i < len; i++) {
//     sum += d[i];
//   }
//   cout << sum << " -> cuzudo" << endl;

//   if (encoder) { encoder->release(); }
//   if (commandBuffer) { commandBuffer->release(); }
//   // if (data1) { data1->release(); }
//   if (result) { result->release(); }
//   if (queue) { queue->release(); }
//   if (pipeline) { pipeline->release(); }
//   if (error) { error->release(); };
//   if (device) { device->release(); }

//   return 0;
// }

