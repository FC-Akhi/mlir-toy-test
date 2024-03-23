//===- Dialect.h - Dialect declaration for the Toy IR ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the IR Dialect for the Toy language.
// See docs/Tutorials/Toy/Ch-2.md for more information.
//
//===----------------------------------------------------------------------===//

#ifndef TOY_DIALECT_H_
#define TOY_DIALECT_H_

/// In, "ToyDialectBase.h.inc", "ToyDialect" class is created from "public ::mlir::Dialect" class.
/// So you need it
#include "mlir/IR/Dialect.h"


/// Include the auto-generated header file containing the declaration of the toy dialect.
/// You will find it in "build/tools/include/Dialect/ToyDialect/" dir
#include "Dialect/ToyDialect/ToyDialectBase.h.inc"



#endif // TOY_DIALECT_H_