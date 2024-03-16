//===- ToyDialectBase.td - Toy dialect definition ----------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Definition of the Toy dialect.
//
// This file doesn't exist in official Toy tutorial
// Created for better organization and seperation
//
//
//===----------------------------------------------------------------------===//


#ifndef TOY_DIALECT
#define TOY_DIALECT

include "mlir/IR/OpBase.td"



// ODS facilitates a more efficient and error-resistant way of 
// defining operations in MLIR by abstracting the complexities 
// into a structured format that's both easy to write and maintain. 
// Provide a definition of the 'toy' dialect in the ODS framework so that we
// can define our operations.
def Toy_Dialect : Dialect {
    
    let name = "toy";
    let summary = "A high-level dialect for analyzing and optimizing the Toy language";

    // A much longer description of our dialect.
    let description = [{
        The Toy language is a tensor-based language that allows you to define functions, perform some math computation, and print results. This dialect provides a representation of the language that is amenable to analysis and optimization.
    }];

    let cppNamespace = "::mlir::toy";


}


//===----------------------------------------------------------------------===//
// Base Toy operation definition.
//===----------------------------------------------------------------------===//

// Base class for toy operations extended from ToyDialect. This dialect inherits from the base
// `Op` class in OpBase.td, and provides:
//   * The parent dialect of the operation.
//   * The mnemonic for the operation, or the name without the dialect prefix.
//   * A list of traits for the operation.
class Toy_Op<string mnemonic, list<Trait> traits = []> :
    Op<Toy_Dialect, mnemonic, traits>;
    

#endif // TOY_DIALECT