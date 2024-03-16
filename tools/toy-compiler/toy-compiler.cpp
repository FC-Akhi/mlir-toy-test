//===- toyc.cpp - The Toy Compiler ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the entry point for the Toy compiler.
//
//===----------------------------------------------------------------------===//


// "Parser.h" is calling "AST.h"
// "AST.h" is calling "Lexer.h"
// "Lexer.h" is top and calling no one.
#include "toy-analysis-parser/Parser.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include <memory>
#include <string>
#include <system_error>

using namespace toy;
namespace cl = llvm::cl;


/// our compiler take an input filename (i.e. filename.toy).
static cl::opt<std::string> inputFilename(
    cl::Positional,
    cl::desc("<input toy file>"),
    cl::init("-"),
    cl::value_desc("filename")
);


namespace {
enum Action { None, DumpAST };
} // namespace

/// This declaration defines a variable “emitAction” of the “Action” enum type.
/// For more - https://llvm.org/docs/CommandLine.html#selecting-an-alternative-from-a-set-of-possibilities
static cl::opt<enum Action> emitAction(
    "emit",
    cl::desc("Select the kind of output desired"),
    cl::values(
        clEnumValN(DumpAST, "ast", "output the AST dump")
    )
);


/// Returns a Toy AST resulting from parsing the file or a nullptr on error.
std::unique_ptr<toy::ModuleAST> parseInputFile(llvm::StringRef filename) {
    
    llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
        llvm::MemoryBuffer::getFileOrSTDIN(filename);
    
    if (std::error_code ec = fileOrErr.getError()) {
        llvm::errs() << "Could not open input file: " << ec.message() << "\n";
        return nullptr;
    }

    auto buffer = fileOrErr.get()->getBuffer();
    LexerBuffer lexer(buffer.begin(), buffer.end(), std::string(filename));
    Parser parser(lexer);
    
    return parser.parseModule();
}




int main(int argc, char **argv) {

    // Parse the command line arguments & flags
    cl::ParseCommandLineOptions(argc, argv, "toy compiler\n");
    
    // Parse code file (i.e. code into LLVM module)
    // Lexer & Parser are called inside
    auto moduleAST = parseInputFile(inputFilename);
    

    if (!moduleAST)
        return 1;


    switch (emitAction) {

        // Then Dump the AST, if -emit=ast argument is passed
        case Action::DumpAST:
            dump(*moduleAST);
            return 0;
        
        default:
            llvm::errs() << "No action specified (parsing only?), use -emit=<action>\n";
    }

    return 0;
}
