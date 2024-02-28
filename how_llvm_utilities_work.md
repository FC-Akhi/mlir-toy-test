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


/// Returns a Toy AST resulting from parsing the file or a nullptr on error.
std::unique_ptr<toy::ModuleAST> parseInputFile(llvm::StringRef filename) {
    
    llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
        llvm::MemoryBuffer::getFileOrSTDIN(filename);
    
    if (std::error_code ec = fileOrErr.getError()) {
        
        return nullptr;
    }

    auto buffer = fileOrErr.get()->getBuffer();
    LexerBuffer lexer(buffer.begin(), buffer.end(), std::string(filename));
    Parser parser(lexer);
    
    return parser.parseModule();
}



/// Driver or Entry point for checking Lexer
int main(int argc, char **argv) {


    // Parse the command line arguments & flags
    cl::ParseCommandLineOptions(argc, argv, "toy compiler\n");

    
    auto testData = parseInputFile(inputFilename);
}
```

### This code snippet is from LLVM's command-line (cl) utility, which is part of the LLVM Project. 

The command-line utility specifically helps in parsing and managing command-line arguments in a structured and easy-to-use manner. Here's a breakdown of what each part of the snippet does:

### Namespace Alias means;

`namespace cl = llvm::cl;`: This line creates an alias cl for the namespace llvm::cl. It's a convenience that allows you to use cl as a shorthand for llvm::cl throughout the rest of the code where this alias is visible.


### Explanation starts from main() driver function

### Arguments passed in main;

`int main(int argc, char **argv)`: The entry point of the program. `argc` is the **argument count**, and `argv` is the **argument vector (array of C-style strings)**. argv[0] is the path to the executable, and argv[1] onwards are the arguments provided by the user.

### The Mechanics of cl::ParseCommandLineOptions:


