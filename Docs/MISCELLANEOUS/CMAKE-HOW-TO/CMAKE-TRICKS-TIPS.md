## 1. How to print/log a `cmake` variable

```cmake
# How to print in cmake variable
message("${CMAKE_EXE_LINKER_FLAGS}")
```

## 2. How to access an OS `ENV` var in `cmake` as `cmake` variable

- [Official doc on CMake Variables](https://cmake.org/cmake/help/book/mastering-cmake/chapter/Writing%20CMakeLists%20Files.html#variables)
```cmake
# Rules
$ENV{VAR}

# Example case for $HOME:
$ENV{HOME}
```