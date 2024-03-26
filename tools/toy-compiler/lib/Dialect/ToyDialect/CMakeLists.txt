# "MLIRToyDialectCPP" is the alias through which tools/toy-compiler/CMakeLists.txt will target this lib
add_mlir_dialect_library(MLIRToyDialectBaseCPP
    ToyDialectBase.cpp
    ToyOps.cpp
    MLIRGen.cpp

    # Standard practise. But without it, build still works
    # ADDITIONAL_HEADER_DIRS
    # ${TOY_MAIN_SRC_DIR}/include/Dialect/ToyDialect

    DEPENDS
    ToyOpsCh20IncGen # <= Telling cmake to generate .inc headers first. Add in Ch 2.0.

    LINK_LIBS PUBLIC
    MLIRFunctionInterfaces
    MLIRParser
    MLIRSideEffectInterfaces

    
)