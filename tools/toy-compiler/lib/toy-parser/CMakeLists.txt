# You can check the standard template from "lib/Standalone/CMakeLists.txt"


# How it looks like with AST.cpp
# add_mlir_dialect_library(MLIRToy
#         AST.cpp

#         ADDITIONAL_HEADER_DIRS
#         ${PROJECT_SOURCE_DIR}/include/toy

#         DEPENDS

# 	LINK_LIBS PUBLIC
# 	MLIRIR
# )


# "MLIRToyParser" is the alias through which tools/toy-compiler/CMakeLists.txt will target this lib
add_mlir_dialect_library(MLIRToyParser
    AST.cpp
)