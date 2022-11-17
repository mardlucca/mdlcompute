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

  MetalComputeEngine::Batch::Batch(
      MetalComputeEngine * engine) 
      : autoReleasePool(NS::AutoreleasePool::alloc()->init()),
        engine(engine), commandBuffer(engine->commandQueue->commandBuffer()), 
        encoder(commandBuffer->computeCommandEncoder()) {
  }

  MetalComputeEngine::Batch::~Batch() {
    if (encoder) {
      encoder->endEncoding();
    }

    if (autoReleasePool) {
      autoReleasePool->release();
    }
    for (auto it = buffers.begin(); it != buffers.end(); it++) {
      engine->ReleaseBuffer(it->first);
    }
  }

  template <>
  void MetalComputeEngine::Batch::AddBuffer<in_buffer>(const in_buffer& buff) {
    if (!buffers.contains(buff.id)) {
      buffers[buff.id] = BufferDescriptor {
        .mtlBuffer = engine->GetBuffer(buff),
        .appBuffer = nullptr,
        .size = buff.size,
        .bufferType = buff.GetType()
      };
    }
    encoder->setBuffer(buffers[buff.id].mtlBuffer, 0, argIndex);
    argIndex++;
  }

  template <>
  void MetalComputeEngine::Batch::AddBuffer<private_buffer>(const private_buffer& buff) {
    if (!buffers.contains(buff.id)) {
      buffers[buff.id] = BufferDescriptor {
        .mtlBuffer = engine->GetBuffer(buff),
        .appBuffer = nullptr,
        .size = buff.size,
        .bufferType = buff.GetType()
      };
    }
    encoder->setBuffer(buffers[buff.id].mtlBuffer, 0, argIndex);
    argIndex++;
  }


  MetalComputeEngine::MetalComputeEngine() {
    device = MTL::CreateSystemDefaultDevice()->retain();
    commandQueue = device->newCommandQueue();
  }

  MetalComputeEngine::~MetalComputeEngine() {
      for (auto it = pipelinesByFn.begin(); it != pipelinesByFn.end(); it++) {
        it->second->release();
      }
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

  MetalComputeEngine::BatchBuilder MetalComputeEngine::NewBatch() {
    return MetalComputeEngine::BatchBuilder(
        std::shared_ptr<MetalComputeEngine::Batch>(new MetalComputeEngine::Batch(this)));
  }

  MTL::ComputePipelineState* MetalComputeEngine::GetPipeline(const std::string& functionName) {
    if (pipelinesByFn.contains(functionName)) {
      return pipelinesByFn[functionName];
    }

    if (!libraryByFn.count(functionName)) {
      throw FunctionNotFoundException(std::string("Function not found: ") + functionName);
    }

    MTL::Library * library = libraryByFn[functionName];
    MTL::Function * fn = library->newFunction(
        NS::String::string(functionName.c_str(), NS::UTF8StringEncoding));

    if (!fn) {
      throw FunctionNotFoundException(std::string("Could not load function object: ") + functionName);
    }

    NS::Error* error = nullptr;
    MTL::ComputePipelineState * pipeline = device->newComputePipelineState(fn, &error);

    if (!pipeline) {
      throw FunctionNotFoundException(std::string(error->description()->utf8String()));
    }

    pipelinesByFn[functionName] = pipeline;
    Release(fn);

    return pipeline;
  }

  MTL::Buffer * MetalComputeEngine::GetBuffer(const in_buffer& buffer) {
    if (!buffersById.contains(buffer.id)) {
      buffersById[buffer.id] = device->newBuffer(
        buffer.data, buffer.size, MTL::ResourceStorageModeManaged);
    }
    return buffersById[buffer.id];
  }

  MTL::Buffer * MetalComputeEngine::GetBuffer(const inout_buffer& buffer) {
    if (!buffersById.contains(buffer.id)) {
      buffersById[buffer.id] = device->newBuffer(
        buffer.data, buffer.size, MTL::ResourceStorageModeManaged);
    }
    return buffersById[buffer.id];
  }

  MTL::Buffer * MetalComputeEngine::GetBuffer(const out_buffer& buffer) {
    if (!buffersById.contains(buffer.id)) {
      buffersById[buffer.id] = device->newBuffer(
        buffer.data, buffer.size, MTL::ResourceStorageModeManaged);
    }
    return buffersById[buffer.id];
  }

  MTL::Buffer * MetalComputeEngine::GetBuffer(const private_buffer& buffer) {
    if (!buffersById.contains(buffer.id)) {
      buffersById[buffer.id] = device->newBuffer(buffer.size, MTL::ResourceStorageModePrivate);
    }
    return buffersById[buffer.id];
  }

  MTL::Buffer * MetalComputeEngine::GetBuffer(const shared_buffer& buffer) {
    if (!buffersById.contains(buffer.id)) {
      buffersById[buffer.id] = device->newBuffer(
        buffer.data, buffer.size, MTL::ResourceStorageModeShared);
    }
    return buffersById[buffer.id];
  }

  void MetalComputeEngine::ReleaseBuffer(std::size_t bufferId) {
    if (buffersById.contains(bufferId)) {
      buffersById[bufferId]->release();
      buffersById.erase(bufferId);
    }
  }



  MetalComputeEngine::BatchBuilder::BatchBuilder(
      const std::shared_ptr<MetalComputeEngine::Batch>& batch) : batch(batch) {}
  MetalComputeEngine::BatchBuilder::BatchBuilder(
      std::shared_ptr<MetalComputeEngine::Batch>&& batch) : batch(std::move(batch)) {}

  MetalComputeEngine::CallBuilder MetalComputeEngine::BatchBuilder::WithGrid(
      std::size_t numRows, std::size_t numCols, 
      std::size_t workGroupRows, std::size_t workGroupCols) {

    batch->argIndex = 0;
    batch->numRows = numRows;
    batch->numCols = numCols;
    batch->workGroupRows = workGroupRows;
    batch->workGroupCols = workGroupCols;

    return MetalComputeEngine::CallBuilder(batch);
  }

  MetalComputeEngine::Gate MetalComputeEngine::BatchBuilder::Dispatch() {
    batch->encoder->endEncoding();
    batch->encoder = nullptr;


    MTL::BlitCommandEncoder * bltEncoder = batch->commandBuffer->blitCommandEncoder();
    for (auto it = batch->buffers.begin(); it != batch->buffers.end(); it++) {
      BufferDescriptor& desc = it->second;
      if (desc.bufferType == BufferType::InOut || desc.bufferType == BufferType::Out) {
        bltEncoder->synchronizeResource(desc.mtlBuffer);
      }
    }
    bltEncoder->endEncoding();

    batch->commandBuffer->commit();
    return MetalComputeEngine::Gate(batch);
  }
  
  MetalComputeEngine::CallBuilder::CallBuilder(
      const std::shared_ptr<MetalComputeEngine::Batch>& batch) : batch(batch) {}
  MetalComputeEngine::CallBuilder::CallBuilder(
      std::shared_ptr<MetalComputeEngine::Batch>&& batch) : batch(std::move(batch)) {}


  template <>
  void MetalComputeEngine::CallBuilder::AddBuffer<in_buffer>(const in_buffer& value) {
    batch->AddBuffer(value);
  }
  template <>
  void MetalComputeEngine::CallBuilder::AddBuffer<inout_buffer>(const inout_buffer& value) {
    batch->AddBuffer(value);
  }
  template <>
  void MetalComputeEngine::CallBuilder::AddBuffer<out_buffer>(const out_buffer& value) {
    batch->AddBuffer(value);
  }
  template <>
  void MetalComputeEngine::CallBuilder::AddBuffer<private_buffer>(const private_buffer& value) {
    batch->AddBuffer(value);
  }
  template <>
  void MetalComputeEngine::CallBuilder::AddBuffer<shared_buffer>(const shared_buffer& value) {
    batch->AddBuffer(value);
  }


  MetalComputeEngine::Gate::Gate(const std::shared_ptr<MetalComputeEngine::Batch>& batch) 
      : batch(batch) {}

  void MetalComputeEngine::Gate::Wait() const {
    batch->commandBuffer->waitUntilCompleted();

    for (auto it = batch->buffers.begin(); it != batch->buffers.end(); it++) {
      BufferDescriptor& desc = it->second;
      if (desc.bufferType == BufferType::InOut 
          || desc.bufferType == BufferType::Out
          || desc.bufferType == BufferType::Shared) {
        std::memcpy(desc.appBuffer, desc.mtlBuffer->contents(), desc.size);
      }
    }
  }
} // compute
} // mdl

