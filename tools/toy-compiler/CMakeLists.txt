# Set Toy Compiler binary filename
set(TOY_COMPILER_BIN_NAME "toy-compiler")


# "${CMAKE_CURRENT_SOURCE_DIR}" always reads where the current `CMakeLists.txt` is located
# Right now ${CMAKE_CURRENT_SOURCE_DIR} is called from this "/path/to/toy-mlir-out-of-tree-tutorial/tools/toy-compiler/CMakeLists.txt" file
# So, ${CMAKE_CURRENT_SOURCE_DIR} == ${TOY_MAIN_SRC_DIR} == /path/to/mlir-project/tools/toy-compiler


# Set all the required dirs
set(TOY_MAIN_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(TOY_INCLUDE_DIR "${TOY_MAIN_SRC_DIR}/include")
set(TOY_LIB_DIR "${TOY_MAIN_SRC_DIR}/lib")


# Point header files location (i.e. /path/to/mlir-project/tools/toy-compiler/include)
include_directories("${TOY_INCLUDE_DIR}")


# Point lib files location (i.e. /path/to/mlir-project/tools/toy-compiler/lib)
# Why use "add_subdirectory", go to "Docs/MISCELLANEOUS/CMAKE-HOW-TO/CMAKE-KNOWLEDGE.md" and read "1."
add_subdirectory("${TOY_LIB_DIR}")


# add new lib `MLIRToyParser` (i.e. alias to `AST.cpp`)
set(LIBS
        MLIRToyParser
)


add_llvm_executable("${TOY_COMPILER_BIN_NAME}" toy-compiler.cpp)

# src- https://stackoverflow.com/a/71388662
# llvm_update_compile_flags adds -fno-exceptions to property COMPILE_FLAGS not COMPILE_DEFINITIONS
# Second, llvm_update_compile_flags inserts a space in front of the flag
llvm_update_compile_flags("${TOY_COMPILER_BIN_NAME}")

target_link_libraries("${TOY_COMPILER_BIN_NAME}" PRIVATE ${LIBS})

mlir_check_all_link_libraries("${TOY_COMPILER_BIN_NAME}")

install(TARGETS "${TOY_COMPILER_BIN_NAME}" DESTINATION bin)
