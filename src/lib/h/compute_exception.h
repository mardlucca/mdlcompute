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

#ifndef _MDL_COMPUTE_EXCEPTION
#define _MDL_COMPUTE_EXCEPTION

#include <stdexcept>

namespace mdl {
namespace compute {
  class CompilationException : public std::runtime_error {
    public:
      CompilationException(const CompilationException& other);
      CompilationException(const char* message);
      CompilationException(const std::string& message);
  };

  class FunctionNotFoundException : public std::runtime_error {
    public:
      FunctionNotFoundException(const FunctionNotFoundException& other);
      FunctionNotFoundException(const char* message);
      FunctionNotFoundException(const std::string& message);
  };

  class RuntimeException : public std::runtime_error {
    public:
      RuntimeException(const RuntimeException& other);
      RuntimeException(const char* message);
      RuntimeException(const std::string& message);
  };

}
}

#endif // _MDL_COMPUTE_EXCEPTION