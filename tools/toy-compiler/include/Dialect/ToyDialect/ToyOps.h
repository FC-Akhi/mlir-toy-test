//===- ToyOps.h - Ops declaration for the Toy IR ----------------------===//
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


#ifndef TOY_OPS_H_
#define TOY_OPS_H_




/// An "Interface" in MLIR terminology is a mechanism to attach additional functionality to MLIR operations (Ops) without modifying their definitions.
/// All kind of "Interfaces" are like Traits.
/// This allows for extending the capabilities of operations in a decoupled manner.
/// Interfaces can be used to define common behaviors across different operations, such as shape inference, buffer allocation, and more.


/// "BytecodeOpInterface" could be understood as an interface for operations that are specifically designed to interact with or generate bytecode.
/// Bytecode is a form of instruction set designed for efficient execution by a software interpreter or a virtual machine, rather than direct execution on hardware.
/// In the context of MLIR, an operation implementing the "BytecodeOpInterface" would likely provide functionalities or abide by contracts that are relevant when the operation is to be translated to or interacted with at the bytecode level. This might include serialization to bytecode, interpretation of bytecode operations, or optimizations relevant at the bytecode level.
#include "mlir/Bytecode/BytecodeOpInterface.h"


/// "CallInterfaces" is a mechanism for abstracting the behavior of function calls within the MLIR framework.
/// It can be used to define a set of functionalities or properties that operations (Ops) related to function calls must or can implement.
/// It allows various dialects (domain-specific or general-purpose languages and representations within MLIR) to define operations that behave in a certain way without enforcing a rigid inheritance structure.
/// This interface facilitates tasks like identifying call sites, managing call arguments, and applying optimizations such as inlining, across various levels of abstraction in the compiler's intermediate representation.
#include "mlir/Interfaces/CallInterfaces.h"


/// Abstracts over the definition, declaration, and manipulation of functions across different dialects.
/// This can include accessing and modifying function signatures (arguments and return types), as well as dealing with function attributes and metadata.
/// Used by compiler developers to work with function definitions and declarations in a dialect-agnostic manner, enabling analyses and transformations that involve functions at a more structural level than individual call sites.
#include "mlir/Interfaces/FunctionInterfaces.h"

/// Diff between "CallInterfaces" & "FunctionInterfaces":
/// "CallInterfaces" Scope: Specifically targets operations that represent function calls.
/// "FunctionInterfaces" Scope: Deal with a broader range of functionalities related to functions as entities within the MLIR ecosystem.
/// "CallInterfaces" Purpose: Provides a unified way to interact with and manipulate call operations across different dialects. This includes identifying call sites, handling arguments passed to calls, and applying call-related optimizations (e.g., inlining).
/// "FunctionInterfaces" Purpose: Abstracts over the definition, declaration, and manipulation of functions across different dialects. This can include accessing and modifying function signatures (arguments and return types), as well as dealing with function attributes and metadata.
/// "CallInterfaces" Usage: Used by compiler passes and transformations that need to generically handle function call operations without being tied to a specific dialect's syntax or semantics.
/// "FunctionInterfaces" Usage: Used by compiler developers to work with function definitions and declarations in a dialect-agnostic manner, enabling analyses and transformations that involve functions at a more structural level than individual call sites.


/// For handling side effects of Operations.
/// Essential for understanding and managing the side effects that operations may have, such as modifying global state or accessing I/O. This interface allows for analysis and optimization passes to reason about and optimize code sequences while considering their potential side effects, making it crucial for ensuring correct transformations and maintaining program semantics.
/// "SideEffectInterfaces" are critical for passes that need to consider the side effects of operations to preserve program semantics during transformations and optimizations.
#include "mlir/Interfaces/SideEffectInterfaces.h"




/// Include the auto-generated header file containing the declarations of the toy operations.
#define GET_OP_CLASSES
#include "Dialect/ToyDialect/ToyOps.h.inc"


#endif // TOY_OPS_H_