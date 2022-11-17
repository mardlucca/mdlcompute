# C++ Compute Library

Facilitate doing computations using GPU.

Currently only supports [Apple's Metal](https://developer.apple.com/documentation/metal?language=objc) library, and only tested on MacOS 12

Example Usage:

```c++
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
```

```c++
  int main() {
    MetalComputeEngine engine;
    engine.LoadLibrary(shaderSrc2);

    const int kSize = 10;
    float f1[kSize];
    float f2[kSize];

    for (int i = 0; i < kSize; i++) {
      f1[i] = i;
      f2[i] = kSize - i;
    }
    
    engine.NewBatch()
        .WithGrid(1, kSize, 1, kSize).Call("swap", inout(f1), inout(f2))
        .Dispatch().Wait();

    for (int i = 0; i < kSize; i++) {
      cout << f1[i] << ", ";
    }
    cout << endl;
    for (int i = 0; i < kSize; i++) {
      cout << f2[i] << ", ";
    }
    cout << endl;

    return 0;
  }
```