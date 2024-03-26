# MLIR Toy Tutorial as an out-of-tree project

```sh
version: LLVM 18.0.0
commit: ecf881838045985f381003cc27569c73a207d0cc
Date: Tue Jan 2 12:06:27 2024 +0100
```

## TUTORIAL INDEX (`git` branch wise)
- `ch-0-0-build-llvm`: gives the overview of this tutorial and how to build [llvm-project](https://github.com/llvm/llvm-project) from source.
- `ch-0-1-prep-mlir-template`: How to collect and setup the `MLIR out-of-tree` template.
- `ch-0-2-prep-toy-scaffold`: How to setup just `Toy` compiler project scaffold. [Note: Without using/collecting the lexer, parser codes.]
- `ch-1-toy-parser`: How to collect & setup lexer, parser from `llvm-18-src-build` (i.e. `llvm-project`) for Toy language.
- `ch-2-version-1-dialect-declaration`: Setting up the Dialect headers and lib. No changes in `tools/toy-compiler/toy-compiler.cpp`
- `ch-2-version-2-mlir-gen`: Setting up the mlir gen for module in MLIRGen.cpp. No changes in `tools/toy-compiler/toy-compiler.cpp`
- `ch-2-version-3-mlir-gen`: Setting up the mlir gen for module & function prototype with entry block having dummy function return in MLIRGen.cpp. No changes in `tools/toy-compiler/toy-compiler.cpp`
- `ch-2-version-4-mlir-gen`: Setting up the mlir gen for module, function prototype, function return & variable declaration and value assignment in MLIRGen.cpp. No changes in `tools/toy-compiler/toy-compiler.cpp`
- More coming....


## ====== CHAPTER 2-0 Starts ======


## Objective

- ### How to write MLIR gen for MODULE only


## Output
- A blank module


## Git Branch name

- `ch-2-version-4-mlir-gen`


## Newly added files and dirs

```sh
# Newly added Docs Dir
Docs/MLIR-KNOWLEDGE-BASE/
Docs/TOY-TUTO/2.SETUP-TOY-DIALECT-&-EMIT-BASIC-MLIR/


# Newly added Docs
Docs/MLIR-KNOWLEDGE-BASE/1.WHAT-WHY-OF-MLIR.md
Docs/MLIR-KNOWLEDGE-BASE/2.WHAT-IS-DIALECT.md
Docs/TOY-TUTO/2.SETUP-TOY-DIALECT-and-EMIT-BASIC-MLIR/2.0.INIT-SETUP-OF-TOY-DIALECT.md

# Modified docs
Docs/MISCELLANEOUS/CMAKE-HOW-TO/CMAKE-KNOWLEDGE.md


# Newly added dirs for toy-compiler
tools/toy-compiler/include/Dialect/
tools/toy-compiler/include/Dialect/ToyDialect/
tools/toy-compiler/lib/Dialect/
tools/toy-compiler/lib/Dialect/ToyDialect/



# Newly added code files
tools/toy-compiler/include/Dialect/ToyDialect/MLIRGen.h
tools/toy-compiler/include/Dialect/ToyDialect/ToyOps.h
tools/toy-compiler/include/Dialect/ToyDialect/ToyOps.td

tools/toy-compiler/lib/Dialect/ToyDialect/MLIRGen.cpp
tools/toy-compiler/lib/Dialect/ToyDialect/ToyOps.cpp



# Modified
MLIRGEN.cpp
README.md


# Example Toy code dir (e.g. ast.toy, codegen.toy, etc. )
# Test Code Used in this tuto
test/Examples/Toy/Ch2/codegen.toy


# Compile
./build-mlir-18.sh
```

# How to Run
```sh
./build/bin/toy-compiler test/Examples/Toy/Ch2/codegen.toy -emit=mlir
```

# Expected output
```sh
module {
  toy.func @multiply_transpose(%arg0: tensor<*xf64>, %arg1: tensor<*xf64>, %arg2: tensor<*xf64>) -> tensor<*xf64> {
    %0 = toy.constant dense<0.000000e+00> : tensor<f64>
    toy.return %0 : tensor<f64>
  }
  toy.func @main() -> tensor<*xf64> {
    %0 = toy.constant dense<5.000000e+00> : tensor<f64>
    %1 = toy.constant dense<6.000000e+00> : tensor<f64>
    %2 = toy.constant dense<0.000000e+00> : tensor<f64>
    toy.return %2 : tensor<f64>
  }
}
```

# Toy project scaffold upto this point
```sh
└── tools
    ├── CMakeLists.txt
    └── toy-compiler
        ├── CMakeLists.txt
        ├── include
        │   ├── CMakeLists.txt 
        │   ├── Dialect 
        │   │   ├── CMakeLists.txt
        │   │   ├── Ops.td
        │   │   └── ToyDialect
        │   │       ├── CMakeLists.txt 
        │   │       ├── ToyDialectBase.h
        │   │       ├── ToyDialectBase.td
        |   |       ├── MLIRGen.h  # <== Newly added
        |   |       ├── ToyOps.h   # <== Newly added
        |   |       ├── ToyOps.td  # <== Newly added
        │   └── toy-analysis-parser 
        │       ├── AST.h
        │       ├── Lexer.h
        │       └── Parser.h
        ├── lib
        │   ├── CMakeLists.txt
        │   ├── Dialect 
        │   │   ├── CMakeLists.txt
        │   │   └── ToyDialect
        │   │       ├── CMakeLists.txt 
        │   │       ├── ToyDialectBase.cpp
        │   │       ├── MLIRGen.cpp # <== Newly added
        │   │       ├── ToyOps.cpp  # <== Newly added
        │   └── toy-parser
        │       ├── AST.cpp
        │       └── CMakeLists.txt
        └── toy-compiler.cpp

```



## ====== CHAPTER 2-0 Ends ======




