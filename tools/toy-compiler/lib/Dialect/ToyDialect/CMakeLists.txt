# "MLIRToyDialectCPP" is the alias through which tools/toy-compiler/CMakeLists.txt will target this lib
add_mlir_dialect_library(MLIRToyDialectBaseCPP
    ToyDialectBase.cpp

    DEPENDS
    ToyCh2OpsIncGen # <= Telling cmake to generate .inc headers first. Add in Ch 2.0.
)