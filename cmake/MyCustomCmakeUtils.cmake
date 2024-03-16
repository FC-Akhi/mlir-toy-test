# Collected from - llvm-src-18-build/mlir/cmake/modules/AddMLIR.cmake - https://github.com/llvm/llvm-project/blob/e302950023cd99251371c5dc8a1e3b609dd5a8fe/mlir/cmake/modules/AddMLIR.cmake#L139
# This function autogenerate the .inc type headers
# Arguments:
# dialect_name: Name of our dialect. i.e. "Toy"
# dialect_namespace: This can be found at "toy-compiler/include/Dialect/ToyDialect/ToyDialectBase.td" files "def Toy_Dialect : Dialect { let name = "toy";...}". So the value is "toy"
# tuto_chapter: This is only for that tutorial purpose. So that it is included in Ch 2.0, the value may be "Ch20"
# Outputs:
# Generate .inc for ToyDialectBase
# Inc_gen_name: "${dialect_name}Ops${tuto_chapter}IncGen"
function(add_mlir_dialect_customized dialect_name dialect_namespace tuto_chapter)
    
    set(LLVM_TARGET_DEFINITIONS "${dialect_name}DialectBase.td")

    # generated header name for ${dialect_name} == "Toy", ToyDialectBase.h.inc and ToyDialectBase.cpp.inc
    mlir_tablegen("${dialect_name}DialectBase.h.inc" -gen-dialect-decls -dialect="${dialect_namespace}")
    mlir_tablegen("${dialect_name}DialectBase.cpp.inc" -gen-dialect-defs -dialect="${dialect_namespace}")


    # To-do: tablegen for ToyOps declarations & definitions
    # mlir_tablegen("${dialect_name}Ops.h.inc" -gen-op-decls)
    # mlir_tablegen("${dialect_name}Ops.cpp.inc" -gen-op-defs)



    # Defining the target alias "ToyOpsCh20IncGen". This alias will be called as a dependency to tools/toy-compiler/lib/Dialect/ToyDialect/ToyDialectBase.cpp
    # Means, before the actual compilation starts, all the mlir_tablegen() commands will be executed by cmake; in order to produce those .h.inc, .cpp.inc type headers.
    # They will be generated at build/tools/toy-compiler/include/Dialect/ToyDialect/ dir. Donot be confused.
    # That's why we have to point this include location at "tools/toy-compiler/CMakeLists.txt" as "include_directories("${STANDALONE_BINARY_DIR}/tools/toy-compiler/include")"
    # For ${dialect_name} == "Toy" & ${tuto_chapter} == Ch20, ${INC_GEN_TARGET_NAME} == ToyOpsCh20IncGen
    set(INC_GEN_TARGET_NAME "${dialect_name}Ops${tuto_chapter}IncGen")
    add_public_tablegen_target("${INC_GEN_TARGET_NAME}")

    add_dependencies(mlir-headers "${INC_GEN_TARGET_NAME}")

endfunction()



