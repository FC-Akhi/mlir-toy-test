# "MLIRToyDialectCPP" is the alias through which tools/toy-compiler/CMakeLists.txt will target this lib
add_mlir_dialect_library(MLIRToyDialectBaseCPP
    ToyDialectBase.cpp

    DEPENDS
    "${DIALECT_NAME}Ops${DIALECT_NAME_SPACE}Ch20IncGen" # <= Telling cmake to generate .inc headers first. Add in Ch 2.0. Check /project-root/cmake/MyCustomCmakeUtils.cmake file's "add_mlir_dialect_customized()" cmake function to know how this target name is set.
)