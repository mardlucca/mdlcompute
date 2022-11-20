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

#ifndef _MDL_METAL_COMPUTE_ENGINE
#define _MDL_METAL_COMPUTE_ENGINE

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>

#include <list>
#include <memory>
#include <string>
#include <unordered_map>

#include "arg_buffers.hpp"

namespace mdl {
namespace compute {

  class MetalComputeEngine {
    struct BufferDescriptor {
      MTL::Buffer* mtlBuffer;
      void * appBuffer;
      size_t size;
      BufferType bufferType;
    };

    struct Batch {
      NS::AutoreleasePool* autoReleasePool;
      MetalComputeEngine * engine;
      MTL::CommandBuffer * commandBuffer;
      MTL::ComputeCommandEncoder * encoder;

      std::unordered_map<std::size_t, BufferDescriptor> buffers;

      int argIndex = 0;
      std::size_t numRows = 0;
      std::size_t numCols = 0; 
      std::size_t workGroupRows = 0;
      std::size_t workGroupCols = 0;

      Batch(MetalComputeEngine * engine, bool parallel);
      ~Batch();

      template <class Buff>
      void AddBuffer(const Buff& buff) {
        if (!buffers.contains(buff.id)) {
          buffers[buff.id] = BufferDescriptor {
            .mtlBuffer = engine->GetBuffer(buff),
            .appBuffer = buff.data,
            .size = buff.size,
            .bufferType = buff.GetType()
          };
        }
        encoder->setBuffer(buffers[buff.id].mtlBuffer, 0, argIndex);
        argIndex++;
      }
    };

    public:
      class BatchBuilder;

      class Gate {
        public:
          void Wait() const;
        private:
          std::shared_ptr<Batch> batch;
          friend class MetalComputeEngine::BatchBuilder;

          Gate(const std::shared_ptr<Batch>& batch);
      };

      class CallBuilder {
        public:
          template <class... Args>
          BatchBuilder Call(const std::string& fn, Args&&... args);

        private:
          std::shared_ptr<Batch> batch;
          friend class MetalComputeEngine::BatchBuilder;

          CallBuilder(const std::shared_ptr<Batch>& batch);
          CallBuilder(std::shared_ptr<Batch>&& batch);

          template <class T, class... Args>
          BatchBuilder DoCall(const T& arg1, Args&&... args);

          template <class T>
          BatchBuilder DoCall(const T& arg);

          template <class T>
          void AddBuffer(const T& value);
      };

      class BatchBuilder {
        public:
          CallBuilder WithGrid(
              std::size_t numRows, std::size_t numCols, 
              std::size_t workGroupRows, std::size_t workGroupCols);
          Gate Dispatch();
        private:
          std::shared_ptr<Batch> batch;

          BatchBuilder(const std::shared_ptr<Batch>& batch);
          BatchBuilder(std::shared_ptr<Batch>&& batch);
          friend class MetalComputeEngine;
      };

      MetalComputeEngine();
      virtual ~MetalComputeEngine();

      bool Available() const;
      BatchBuilder NewBatch(bool parallel = false);
      void LoadLibrary(const std::string& sourceCode);
      bool ContainsFunction(const std::string& functionName) const;
    private:
      MTL::Device* device;
      MTL::CommandQueue* commandQueue;
      std::list<MTL::Library*> libraries;
      std::unordered_map<std::string, MTL::Library*> libraryByFn;
      std::unordered_map<std::string, MTL::ComputePipelineState*> pipelinesByFn;
      std::unordered_map<std::size_t, MTL::Buffer *> buffersById;

      template <class Ref>
      void Release(Ref*& referencing);

      MTL::ComputePipelineState* GetPipeline(const std::string& functionName);
      MTL::Buffer * GetBuffer(const in_buffer& buffer);
      MTL::Buffer * GetBuffer(const inout_buffer& buffer);
      MTL::Buffer * GetBuffer(const out_buffer& buffer);
      MTL::Buffer * GetBuffer(const private_buffer& buffer);
      MTL::Buffer * GetBuffer(const shared_buffer& buffer);
      void ReleaseBuffer(std::size_t bufferId);
  };

  template <class Ref>
  void MetalComputeEngine::Release(Ref*& referencing) {
    if (referencing) {
      referencing->release();
      referencing = nullptr;
    }
  }

  template <class... Args>
  MetalComputeEngine::BatchBuilder MetalComputeEngine::CallBuilder::Call(
      const std::string& fn, Args&&... args) {
    batch->encoder->setComputePipelineState(batch->engine->GetPipeline(fn));
    return DoCall(std::forward<Args>(args)...);
  }

  template <class T, class... Args>
  MetalComputeEngine::BatchBuilder  MetalComputeEngine::CallBuilder::DoCall(const T& arg1, Args&&... args) {
    AddBuffer(arg1);
    return DoCall(std::forward<Args>(args)...);
  }

  template <class T>
  MetalComputeEngine::BatchBuilder MetalComputeEngine::CallBuilder::DoCall(const T& arg) {
    AddBuffer(arg);
    MTL::Size threadGroupSize(batch->workGroupCols, batch->workGroupRows, 1);
    MTL::Size gridSize(batch->numCols, batch->numRows, 1);
    batch->encoder->dispatchThreads(gridSize, threadGroupSize);
    return BatchBuilder(batch);
  }

  template <class T>
  void MetalComputeEngine::CallBuilder::AddBuffer(const T& value) {
    AddBuffer(in(value));
  }

  template <>
  void MetalComputeEngine::CallBuilder::AddBuffer<in_buffer>(const in_buffer& value);
  template <>
  void MetalComputeEngine::CallBuilder::AddBuffer<inout_buffer>(const inout_buffer& value);
  template <>
  void MetalComputeEngine::CallBuilder::AddBuffer<out_buffer>(const out_buffer& value);
  template <>
  void MetalComputeEngine::CallBuilder::AddBuffer<private_buffer>(const private_buffer& value);
  template <>
  void MetalComputeEngine::CallBuilder::AddBuffer<shared_buffer>(const shared_buffer& value);
} // comput
} // mdl


#endif  // _MDL_METAL_COMPUTE_ENGINE