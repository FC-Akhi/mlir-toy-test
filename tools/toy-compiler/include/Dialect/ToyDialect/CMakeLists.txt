set(LLVM_TARGET_DEFINITIONS ToyDialectBase.td)
mlir_tablegen(ToyDialectBase.h.inc -gen-dialect-decls)
mlir_tablegen(ToyDialectBase.cpp.inc -gen-dialect-defs)
# To-do: tablegen for ToyOps declarations & definitions



# Defining the target alias "ToyCh2OpsIncGen". This alias will be called as a dependency to tools/toy-compiler/lib/Dialect/ToyDialect/ToyDialectBase.cpp

# Means, before the actual compilation starts, all the mlir_tablegen() commands will be executed by cmake; in order to produce those .h.inc, .cpp.inc type headers.

# They will be generated at build/tools/toy-compiler/include/Dialect/ToyDialect/ dir. Donot be confused.

# That's why we have to point this include location at "tools/toy-compiler/CMakeLists.txt" as "include_directories("${STANDALONE_BINARY_DIR}/tools/toy-compiler/include")"
add_public_tablegen_target(ToyCh2OpsIncGen)