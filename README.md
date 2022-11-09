# C++ Sample Project

This is a sample C++ Bazel project that can be used as a Git template repo for creating simple C++ 
libraries. Features include:
* Default .bazelrc configured for C++17 support.
* Google test support (see example under src/test/cc)
* Standardized folder structure (src/main/h, src/main/cc, src/test/cc, src/test/resources)
* Support for "test resource" files (under src/test/resources).
* Base .gitignore file ignoring default bazel output folders.
* Pre-configured to allow running/debugging in VS Code.
* Example usage of the library.
* BSD 3 License file. 
* Copyright snippets with BSD 3.


At this point this has only been tested in MacOS 12.

Library can be added to a bazel project by adding the following snippet to WORKSPACE:
```
load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

git_repository(
  name = "mdl_cpp_template",
  remote = "https://github.com/mardlucca/cpp-template.git",
  tag = "22.11.0-1"
)
```
And can be used by other rules, as seen in this BUILD file snippet.

```
load("@rules_cc//cc:defs.bzl", "cc_binary")

cc_binary(
  name = "main",
  srcs = ["src/main/cc/main.cc"],
  deps = [
      "@mdl_cpp_template//:lib",
  ]
)
```