///===- MLIRGen.cpp - MLIR Generation(Only a blank module will generate) from a Toy AST -----------------------===///
///
/// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
/// See https:///llvm.org/LICENSE.txt for license information.
/// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
///
///===----------------------------------------------------------------------===///
///
/// This file implements a simple IR generation targeting MLIR from a Module AST
/// for the Toy language.
///
///===--------------------------------------------------------------------------------------------------------===///


/// This is for mlir namespace. Check this header file to understand
#include "Dialect/ToyDialect/MLIRGen.h"

/// This header is for AST
#include "toy-analysis-parser/AST.h"


/// This header is for 'mlir::OpBuilder
#include "mlir/IR/Builders.h"


/// This header is for mlir::ModuleOp
#include "mlir/IR/BuiltinOps.h"     // Without it, this file compilation fails


/// This header is for mlir::verify
#include "mlir/IR/Verifier.h"     // Without it, this file compilation fails


/// Declaration of toy namespace. Defination is at the end of the script
using namespace toy;


namespace {


    /// Implementation of a simple MLIR emission from the Toy AST.
    /// This will emit operations that are specific to the Toy language, preserving
    /// the semantics of the language and (hopefully) allow to perform accurate
    /// analysis and transformation based on these high level semantics.
    class MLIRGenImpl {


    public:


        /// @brief Constructor of the class which instantiated with a context
        /// @param context 
        MLIRGenImpl(mlir::MLIRContext &context) : builder(&context) {}


        /// @brief Public API: convert the AST for a Toy module (source file) to an MLIR
        /// @param moduleAst 
        /// @return theModule of type mlir::ModuleOp
        mlir::ModuleOp mlirGen(ModuleAst &moduleAst) {

            
            /// Initializes an empty new MLIR module
            /// @param: create() method requires location parameter, which is provided by `builder.getUnknownLoc()`
            /// The `builder` is an instance of mlir::OpBuilder, a utility in MLIR that simplifies the creation of operations.
            /// getUnknownLoc(): gives a placeholder for the location. And it gives the room for future enhancement
            theModule = mlir::ModuleOp::create(builder.getUnknownLoc());


            /// Verifying the module
            if (failed(mlir::verify(theModule))) {

                theModule.emitError("module verification error");

                return nullptr;

            }


            /// Returning the module
            return theModule;


        }


    private:


        /// A "module" matches a Toy source file: containing a list of functions
        mlir::ModuleOp theModule;


        /// The builder is a helper class to create IR inside a function. The builder
        /// is stateful, in particular it keeps an "insertion point": this is where
        /// the next operations will be introduced.
        mlir::OpBuilder builder;


    };


} /// namespace ends here



/// namespace toy where the public API for codegen will be present
namespace toy {

    /// The public API for codegen
    mlir::OwningOpRef<mlir::ModuleOp> mlirGen(mlir::MLIRContext &context, ModuleAst &moduleAst) {

        return MLIRGenImpl(context).mlirGen(moduleAst);

    }


} /// namespace toy ends here