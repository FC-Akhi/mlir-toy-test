///===---------- MLIRGen.cpp - MLIR Generation(Only a blank module will generate) from a Toy AST --------------===///
///
/// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
/// See https:///llvm.org/LICENSE.txt for license information.
/// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
///
///===---------------------------------------------------------------------------------------------------------===///
///
/// This file implements a simple IR generation targeting MLIR from a Module AST
/// for the Toy language.
///
///===--------------------------------------------------------------------------------------------------------===///


/// This header file is required for using the Toy language's MLIR dialect
#include "Dialect/ToyDialect/MLIRGen.h"


/// This header file provides the definitions for the abstract syntax tree (AST) of the Toy language
#include "toy-analysis-parser/AST.h"


/// This header file is necessary for constructing MLIR operations easily
#include "mlir/IR/Builders.h"


/// This header file includes definitions for creating and manipulating MLIR modules
#include "mlir/IR/BuiltinOps.h"     // Without it, this file compilation fails


/// This header file is used for verifying the correctness of the MLIR module
#include "mlir/IR/Verifier.h"     // Without it, this file compilation fails


/// We're using the 'toy' namespace which contains definitions related to the Toy language
using namespace toy;


namespace {


    /// MLIRGenImpl is a class that converts the Toy AST to MLIR, 
    /// making it possible to analyze and transform the Toy program within the MLIR framework
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


            /// Check the newly created module for structural correctness
            if (failed(mlir::verify(theModule))) {

                /// If verification fails, report an error on the module
                theModule.emitError("module verification error");

                return nullptr;

            }


            /// Successfully return the generated MLIR module
            return theModule;


        }


    private:


        /// Holds the generated MLIR module, corresponding to a source file in the Toy language
        mlir::ModuleOp theModule;


        /// A utility to help build MLIR operations.
        mlir::OpBuilder builder;


    };


} /// namespace ends here



/// namespace toy where the public API for codegen will be present
namespace toy {
    
    /// Public API for generating MLIR from the Toy AST. This function bridges the Toy language and the MLIR framework
    mlir::OwningOpRef<mlir::ModuleOp> mlirGen(mlir::MLIRContext &context, ModuleAst &moduleAst) {
        
        /// Create a generator instance with the given context
        MLIRGenImpl mlirGenerator = MLIRGenImpl(context);

        /// Generate MLIR for the given AST
        auto generatedIR = mlirGenerator.mlirGen(moduleAst);

        /// Return the generated MLIR
        return generatedIR;
    
    }

} /// namespace toy ends here