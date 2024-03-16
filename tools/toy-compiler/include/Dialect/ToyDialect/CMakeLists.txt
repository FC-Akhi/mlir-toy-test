# Arguments:
# dialect_name: Name of our dialect. i.e. "Toy"
# dialect_namespace: This can be found at "toy-compiler/include/Dialect/ToyDialect/ToyDialectBase.td" files "def Toy_Dialect : Dialect { let name = "toy";...}". So the value is "toy"
# tuto_chapter: This is only for that tutorial purpose. So that it is included in Ch 2.0, the value may be "Ch20"
# Outputs:
# Generate .inc for ToyDialectBase
# Inc_gen_name: "${dialect_name}Ops${tuto_chapter}IncGen"
add_mlir_dialect_customized("${DIALECT_NAME}" "${DIALECT_NAMESPACE}" "Ch20")
