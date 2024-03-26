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
#include "Dialect/ToyDialect/ToyDialectBase.h"
#include "Dialect/ToyDialect/MLIRGen.h"


#include <string>
#include <system_error>
#include <utility>


#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"


#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"



using namespace mlir::toy;
namespace cl = llvm::cl;

static cl::opt<std::string> inputFilename(cl::Positional,
                                          cl::desc("<input toy file>"),
                                          cl::init("-"),
                                          cl::value_desc("filename"));

namespace {

enum InputType { Toy, MLIR };

} // namespace



static cl::opt<enum InputType> inputType(

    "x",
    cl::init(Toy),
    cl::desc("Decided the kind of output desired"),

    cl::values(
        clEnumValN(
            Toy,
            "toy",
            "load the input file as a Toy source."
        )
    ),

    cl::values(
        clEnumValN(
            MLIR,
            "mlir",
            "load the input file as an MLIR file"
        )
    )
);



namespace {

enum Action { None, DumpAST, DumpMLIR };

} // namespace



static cl::opt<enum Action> emitAction(

    "emit",
    cl::desc("Select the kind of output desired"),

    cl::values(
        clEnumValN(
            DumpAST,
            "ast",
            "output the AST dump"
        )
    ),

    cl::values(
        clEnumValN(
            DumpMLIR,
            "mlir",
            "output the MLIR dump"
        )
    )
);


/// Returns a Toy AST resulting from parsing the file or a nullptr on error.
std::unique_ptr<toy::ModuleAst> parseInputFile(llvm::StringRef filename) {
    
    llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr = llvm::MemoryBuffer::getFileOrSTDIN(filename);
    
    if (std::error_code ec = fileOrErr.getError()) {

        llvm::errs() << "Could not open input file: " << ec.message() << "\n";

        return nullptr;

    }
    
    auto buffer = fileOrErr.get()->getBuffer();
    
    toy::LexerBuffer lexer(buffer.begin(), buffer.end(), std::string(filename));
    toy::Parser parser(lexer);
    
    return parser.parseModule();

}




int dumpMLIR() {

    mlir::MLIRContext context;
    
    // Load our Dialect in this MLIR Context.
    context.getOrLoadDialect<mlir::toy::ToyDialect>();

    // Handle '.toy' input to the compiler.
    if (inputType != InputType::MLIR && !llvm::StringRef(inputFilename).ends_with(".mlir")) {
        
        auto moduleAst = parseInputFile(inputFilename);
        
        if (!moduleAst)
            return 6;
        
        mlir::OwningOpRef<mlir::ModuleOp> module = mlirGen(context, *moduleAst);
        
        if (!module)
            return 1;

        module->dump();
        
        return 0;
    
    }

    // Otherwise, the input is '.mlir'.
    llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr = llvm::MemoryBuffer::getFileOrSTDIN(inputFilename);
    
    if (std::error_code ec = fileOrErr.getError()) {
        
        llvm::errs() << "Could not open input file: " << ec.message() << "\n";
        return -1;
    
    }

    // Parse the input mlir.
    llvm::SourceMgr sourceMgr;

    sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
    
    mlir::OwningOpRef<mlir::ModuleOp> module = mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &context);
    
    if (!module) {
        
        llvm::errs() << "Error can't load file " << inputFilename << "\n";
        return 3;
    
    }

    module->dump();
    return 0;

}



int dumpAst() {
    
    if (inputType == InputType::MLIR) {
    
        llvm::errs() << "Can't dump a Toy Ast when the input is MLIR\n";
        return 5;
    
    }

    auto moduleAst = parseInputFile(inputFilename);
    
    if (!moduleAst)
        return 1;

    dump(*moduleAst);
    
    return 0;

}

int main(int argc, char **argv) {

    // Register any command line options.
    mlir::registerAsmPrinterCLOptions();
    mlir::registerMLIRContextCLOptions();
    cl::ParseCommandLineOptions(argc, argv, "toy compiler\n");


    switch (emitAction) {

    case Action::DumpAST:
    
        return dumpAst();
    
    case Action::DumpMLIR:
    
        return dumpMLIR();
    
    default:
    
        llvm::errs() << "No action specified (parsing only?), use -emit=<action>\n";
    
    }

    return 0;
}