# Collected from - llvm-src-18-build/mlir/cmake/modules/AddMLIR.cmake - https://github.com/llvm/llvm-project/blob/e302950023cd99251371c5dc8a1e3b609dd5a8fe/mlir/cmake/modules/AddMLIR.cmake#L139
# This function autogenerate the .inc type headers for 
# This only deals with "-gen-dialect-" & "-gen-op-" generated .inc
# Arguments:
# dialect_name: Name of our dialect. i.e. "Toy"
# dialect_namespace: This can be found at "toy-compiler/include/Dialect/ToyDialect/ToyDialectBase.td" files "def Toy_Dialect : Dialect { let name = "toy";...}". So the value is "toy"
# tuto_chapter: This is only for that tutorial purpose. So that it is included in Ch 2.0, the value may be "Ch20"
# Outputs:
# Generate .inc for ToyDialectBase (i.e. "ToyDialectBase.h.inc" & "ToyDialectBase.cpp.inc")
# Generate .inc for ToyOps (i.e. "ToyOps.h.inc" & "ToyOps.cpp.inc")
# Inc_gen_target_name: "${dialect_name}Ops${tuto_chapter}IncGen" or e.g. "ToyOpsCh20IncGen"
function(add_mlir_dialect_customized dialect_name dialect_namespace tuto_chapter)

    # Not mandatory, if you have Ops defined in seperate .td files (e.g. ToyOps.td)
    set(LLVM_TARGET_DEFINITIONS "${dialect_name}DialectBase.td")
    
    # Mandatory, because the Ops are defined in ToyOps.td
    set(LLVM_TARGET_DEFINITIONS "${dialect_name}Ops.td")

    # generated header name for ${dialect_name} == "Toy", ToyDialectBase.h.inc and ToyDialectBase.cpp.inc
    mlir_tablegen("${dialect_name}DialectBase.h.inc" -gen-dialect-decls -dialect="${dialect_namespace}")
    mlir_tablegen("${dialect_name}DialectBase.cpp.inc" -gen-dialect-defs -dialect="${dialect_namespace}")


    # To-do: tablegen for ToyOps declarations & definitions
    mlir_tablegen("${dialect_name}Ops.h.inc" -gen-op-decls)
    mlir_tablegen("${dialect_name}Ops.cpp.inc" -gen-op-defs)



    # Defining the target alias "ToyOpsCh20IncGen". This alias will be called as a dependency to tools/toy-compiler/lib/Dialect/ToyDialect/ToyDialectBase.cpp
    # Means, before the actual compilation starts, all the mlir_tablegen() commands will be executed by cmake; in order to produce those .h.inc, .cpp.inc type headers.
    # They will be generated at build/tools/toy-compiler/include/Dialect/ToyDialect/ dir. Donot be confused.
    # That's why we have to point this include location at "tools/toy-compiler/CMakeLists.txt" as "include_directories("${STANDALONE_BINARY_DIR}/tools/toy-compiler/include")"
    # For ${dialect_name} == "Toy" & ${tuto_chapter} == Ch20, ${INC_GEN_TARGET_NAME} == ToyOpsCh20IncGen
    set(INC_GEN_TARGET_NAME "${dialect_name}Ops${tuto_chapter}IncGen")
    add_public_tablegen_target("${INC_GEN_TARGET_NAME}")

    add_dependencies(mlir-headers "${INC_GEN_TARGET_NAME}")

endfunction()






# Collected from - https://github.com/llvm/llvm-project/blob/e302950023cd99251371c5dc8a1e3b609dd5a8fe/mlir/cmake/modules/AddMLIR.cmake#L162C1-L176C14
# Generate ".md" Documentation
# Arguments:
# doc_filename: The name of your dialect file. i.e. "${dialect_name}DialectBase.td"
# output_file_name: The name of the output doc file. Whatever you want, you can set it.
# output_directory: Where you want to have it. Default prefix path is "build/docs/your-set-dir/". i.e. set it to "your-set-dir/".
# command: i.e. "-gen-op-doc"
# Example: add_mlir_doc_customized("ToyOps" "ToyOps" "ToyDialect/" "-gen-op-doc")
# Output:
# "${output_file_name}.md" in "build/docs/${output_directory}" dir
# How to activate the doc gen?
# In "build-mlir-18.sh" file, set "cmake --build . --target mlir-doc"
function(add_mlir_doc_customized doc_filename output_file_name output_directory command)

    set(LLVM_TARGET_DEFINITIONS ${doc_filename}.td)

    # The MLIR docs use Hugo, so we allow Hugo specific features here.
    tablegen(MLIR ${output_file_name}.md ${command} -allow-hugo-specific-features ${ARGN})

    set(GEN_DOC_FILE ${STANDALONE_BINARY_DIR}/docs/${output_directory}${output_file_name}.md)

    # build/docs/output_directory/
    # message(STATUS "${GEN_DOC_FILE}")

    add_custom_command(
            OUTPUT ${GEN_DOC_FILE}
            COMMAND ${CMAKE_COMMAND} -E copy
                    ${CMAKE_CURRENT_BINARY_DIR}/${output_file_name}.md
                    ${GEN_DOC_FILE}
            DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${output_file_name}.md)
    add_custom_target(${output_file_name}DocGen DEPENDS ${GEN_DOC_FILE})
    add_dependencies(mlir-doc ${output_file_name}DocGen)

endfunction()
