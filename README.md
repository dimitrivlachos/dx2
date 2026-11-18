# dx2 - Clean C++ implementation of dxtbx data model

## Quick Start

- Ensure you have an environment with CMake and compilers available (see environment.yml)
- Dependencies:
    - HDF5, gemmi, [Eigen](https://eigen.tuxfamily.org/),
      [nlohmann/json](https://github.com/nlohmann/json),
      [{fmt}](https://github.com/fmtlib/fmt), LZ4 (Required, found via
      `find_package` or pkg-config)
    - zstd (Optional, enables the bitshuffle+zstd filter variant)
    - [mdspan](https://github.com/kokkos/mdspan),
      [bitshuffle](https://github.com/kiyo-masui/bitshuffle),
      [GoogleTest](https://github.com/google/googletest) (downloaded if
      missing; bitshuffle is always downloaded)

```
mkdir build
cd build
cmake ..
```
You can then build with `make`, and run a test with `./tests/test_crystal`.

### Build options

- `DX2_BUILD_TESTS` (default ON when dx2 is the top-level project, OFF
  when embedded with `add_subdirectory` or FetchContent). Controls the
  test suite and the GoogleTest download.
- `DX2_BITSHUFFLE_SIMD_FLAGS` (default empty). Extra compile flags for
  the bitshuffle decoder, which picks its SIMD kernels from the
  compiler's predefined macros. Empty gives the portable baseline (SSE2
  on x86-64, NEON on aarch64) so that a packaged library runs on any
  machine. Any modern x86-64 host should pass
  `-DDX2_BITSHUFFLE_SIMD_FLAGS=-mavx2` for the wider kernels, which
  decode noticeably faster; `-march=native` goes further when the
  library is built on the machine that runs it. The flag is not the
  default because a library built with it will not run on a CPU that
  lacks those instructions, and it is not valid on aarch64.

You can enable debugging printout by setting the `DX2_DEBUG` environment
variable before running.

## Documentation

API documentation is generated from the Doxygen comments in the source
by Doxygen + Sphinx (via Breathe and Exhale) and published to GitHub
Pages on every push to `main`.

To build it locally you need the `doxygen` and `graphviz` binaries
(e.g. `conda install -c conda-forge doxygen graphviz`) plus the Python
packages:

```
pip install -r docs/requirements.txt
sphinx-build -b html docs docs/_build/html
```

Then open `docs/_build/html/index.html`.

## Writing And Running Tests

See https://google.github.io/googletest/primer.html. The `tests` subdirectory
has an example test.

You can run test executables directly, run `ctest`, or run `make test` (or
via your preferred build tool).

## Using dx2 from other projects

### Option 1: Already installed into the environment

In which case the normal CMake package finding should work:

```cmake
find_package(DX2)
```

### Option 2: As a subdirectory/submodule

If you want to integrate dx2 development into your current project, you can
check dx2 out into a subfolder (or submodule) of your build, and include it with:

```cmake
add_subdirectory(path/to/dx2)
```

### Option 3: Automatically use with FetchContent

If you don't need to develop dx2 but just want to use it, you can get CMake to
automatically download and include it in your project with:

```cmake
include(FetchContent)
FetchContent_Declare(
    dx2
    GIT_REPOSITORY git@github.com:dials/dx2.git
    GIT_TAG main
    FIND_PACKAGE_ARGS
)
FetchContent_MakeAvailable(dx2)

...

# Example adding to a target
target_link_libraries(my-target PUBLIC dx2::dx2)
```

> [!CAUTION]
> This is linking directly to the `main` branch. This is bad practice as it
> makes it non-trivial to ensure updates, but while the project is in the
> incubation stages we don't have a public release tag to base it on.
