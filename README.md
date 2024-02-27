# MLIR Toy Test

```sh
version: LLVM 18.0.0
```

## INDEX (`git` branch wise)
- `ch-1-version-1-lexer`: gives an overview and how to test the lexer. Source code of the lexer is taken from [llvm-project](https://github.com/llvm/llvm-project).



## ====== CHAPTER 1 VERSION 1 Starts ======


## Objective

- How to collect & setup lexer from `llvm-18-src-build/mlir/examples/toy` (i.e. `llvm-project`) for Toy language.
- Run the lexer for different examples


## Output
- Given a `filename.toy` as input, will generate a list of token of the code written in input file


## Git Branch name

- `ch-1-version-1-lexer`


## How to RUN?

- 


## Newly added files and dirs

```sh

# Newly added Docs
README.md
LLVM_style_guide.md


# Newly added dir for toy-compiler
tools/toy-compiler/include/toy-analysis-parser/
c++_fundamentals



# Newly added code files
tools/toy-compiler/include/toy-analysis-parser/Lexer.h
tools/toy-compiler/toy-compiler.cpp
tools/toy-compiler/include/toy-analysis-parser/AST.h (Coming soon...) 
tools/toy-compiler/include/toy-analysis-parser/Parser.h (Coming soon...) 

# Example Toy code dir (e.g. ast.toy, empty.toy, etc. )
test/Examples/Toy/


# Src code for toy compiler
llvm-project/mlir/examples/toy/


# Compile
./build-mlir-18.sh


# Test
./build/bin/toy-compiler test/Examples/Toy/Ch1/ast.toy -emit=ast


# Toy project scaffold upto this point

└── tools
    └── toy-compiler
        ├── include
        │   └── toy-analysis-parser
        │       ├── AST.h (EMPTY FILE)
        │       ├── Lexer.h 
        │       └── Parser.h (EMPTY FILE)
        └── toy-compiler.cpp  # <==== This is your toy-compiler entry point (i.e. where main() exists for testing lexer)

```

## Key things

**`build/bin/` contains the built binary `toy-compiler`. Now it has the tokenizing feature which can emit `TOKENS`. progressively it will be filled by code**


## ====== CHAPTER 1 VERSION 1 Ends ======


## 