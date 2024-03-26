
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
add_mlir_dialect_customized("Toy" "toy" "Ch20")


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
add_mlir_doc_customized("ToyDialectBase" "ToyDialectBase" "ToyDialect/" "-gen-dialect-doc")
add_mlir_doc_customized("ToyOps" "ToyOps" "ToyDialect/" "-gen-op-doc")


