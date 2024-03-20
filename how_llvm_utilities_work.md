## How llvm utilities work???

**I will be explaning llvm utilities I have used inside `mlir-toy-test/tools/toy-compiler`**



 ```sh
namespace cl = llvm::cl;

/// our compiler take an input filename (i.e. filename.toy).
static cl::opt<std::string> inputFilename(
    cl::Positional,
    cl::desc("<input toy file>"),
    cl::init("-"),
    cl::value_desc("filename")
);
```

### This code snippet is from LLVM's command-line (cl) utility, which is part of the LLVM Project. 

The command-line utility specifically helps in parsing and managing command-line arguments in a structured and easy-to-use manner. Here's a breakdown of what each part of the snippet does:

### Namespace Alias

`namespace cl = llvm::cl;`

This line creates an alias cl for the namespace llvm::cl. It's a convenience that allows you to use cl as a shorthand for llvm::cl throughout the rest of the code where this alias is visible. 