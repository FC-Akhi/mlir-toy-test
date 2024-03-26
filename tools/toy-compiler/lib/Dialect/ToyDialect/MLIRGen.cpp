///===---------- MLIRGen.cpp - MLIR Generation(Blank module and function prototype will generate) from a Toy AST --------------===///
///
/// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
/// See https:///llvm.org/LICENSE.txt for license information.
/// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
///
///===-------------------------------------------------------------------------------------------------------------------------===///
///
/// This file implements a simple IR generation targeting MLIR from a Module AST
/// for the Toy language.
///
///===-------------------------------------------------------------------------------------------------------------------------===///

#include<stdio.h>

/// Required for Toy language's MLIR dialect operations
#include "Dialect/ToyDialect/MLIRGen.h"

/// Definitions for the Toy language's Abstract Syntax Tree (AST)
#include "toy-analysis-parser/AST.h"

/// 
#include "Dialect/ToyDialect/ToyOps.h"



/// This is where we have Operations builder OpBuilder
#include "mlir/IR/Builders.h"

/// Necessary for using MLIR's Module Operations
#include "mlir/IR/BuiltinOps.h"

/// Checks if the MLIR code we create is valid
#include "mlir/IR/Verifier.h"




/// Additional headers for working with MLIR and LLVM libraries
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"



#include <numeric>    // Without it, this file compilation fails



/// Shortcuts to use some LLVM functionalities more easily
using llvm::ArrayRef;
using llvm::dyn_cast;
using llvm::ScopedHashTableScope;
using llvm::StringRef;

using llvm::cast;
using llvm::Twine;
using llvm::SmallVector;


/// We're going to use functions and types from the Toy dialect
using namespace mlir::toy;
/// We're using the 'toy' namespace which contains definitions related to the Toy language
using namespace toy;




namespace {


    /// MLIRGenImpl is a class that converts the Toy AST to MLIR, 
    /// making it possible to analyze and transform the Toy program within the MLIR framework
    class MLIRGenImpl {


    public:


        /// @brief : Prepares for MLIR generation by initializing a builder with the provided context
        /// @param context : The MLIR context, which is a container for all MLIR-related 
        ///                  information during this generation process
        MLIRGenImpl(mlir::MLIRContext &context) : builder(&context) {}



        /// @brief : Converts the AST of a Toy module into an MLIR module, 
        ///          essentially translating the high-level Toy code into a lower-level intermediate representation
        /// @param moduleAst : The abstract syntax tree of the Toy module that needs to be converted
        /// @return theModule of type mlir::ModuleOp, which is an MLIR representation of the given Toy module AST
        mlir::ModuleOp mlirGen(ModuleAst &moduleAst) {

            

            /// Start with an empty MLIR module. The unknown location is a placeholder, 
            /// indicating this is generated code without a specific source file location.
            theModule = mlir::ModuleOp::create(builder.getUnknownLoc());


            /// Go through each function in the AST and generate its MLIR representation
            for (FunctionAst &f : moduleAst)
                mlirGen(f);


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


        /// @brief : Holds the generated MLIR module, corresponding to a source file in the Toy language
        mlir::ModuleOp theModule;



        /// @brief : A utility to help build MLIR operations.
        mlir::OpBuilder builder;



        /// @brief : Tracks variables and their MLIR representations
        llvm::ScopedHashTable<StringRef, mlir::Value> symbolTable;



        /// @brief : Adds a variable and its corresponding MLIR value to the symbol table,
        ///          if it's not already present
        /// @param var : The name of the variable to be added to the symbol table.
        /// @param value : The MLIR value that corresponds to the variable. This is what the
        ///                variable will be represented as in the generated MLIR code.
        /// @return : A LogicalResult indicating success or failure. It returns success if the 
        ///           variable was successfully added to the symbol table, and failure if the 
        ///           variable is already present in the table.
        mlir::LogicalResult declare(llvm::StringRef var, mlir::Value value) {
            
            if (symbolTable.count(var))

                return mlir::failure();

            symbolTable.insert(var, value);

            return mlir::success();

        }



        /// @brief : Determines the MLIR type based on a shape. Used for variables with defined or undefined shapes
        /// @param shape : An array defining the shape of the tensor. If empty, the tensor is unranked
        /// @return : The MLIR type for the variable, either ranked or unranked
        mlir::Type getType(ArrayRef<int64_t> shape) {
            
            /// If the shape is empty, then this type is unranked. 
            /// For example unranked will look like; tensor<*xf64>
            if (shape.empty())
                return mlir::UnrankedTensorType::get(builder.getF64Type());


            // Otherwise, we use the given shape
            return mlir::RankedTensorType::get(shape, builder.getF64Type());

        }



        /// @brief : This function acts as an overload of the first getType method
        /// @param type : A variable type representation that includes tensor shape information
        /// @return : The corresponding MLIR tensor type, which may be ranked 
        ///           (with a fixed shape) or unranked (without a fixed shape)
        mlir::Type getType(const VarType &type) {

            return getType(type.shape);
        
        }



        /// @brief : Converts a Toy AST location to an MLIR location. Useful for error reporting and diagnostics
        /// @param loc : The location in the Toy AST
        /// @return : An MLIR location corresponding to the Toy AST location
        mlir::Location loc(const Location &loc) {

            return mlir::FileLineColLoc::get(builder.getStringAttr(*loc.file), loc.line, loc.col);
        
        }



        /// @brief : Generates MLIR for a Number Expression AST node.
        ///        This function takes a NumberExprAst node, representing a numerical constant in the AST,
        ///        and generates the corresponding MLIR Constant operation to represent that number in the IR
        /// @param num : The NumberExprAst node representing a numerical constant in the source program
        /// @return : Returns an MLIR Value representing the generated Constant operation in the MLIR module
        mlir::Value mlirGen(NumberExprAst &num) {

            /// The 'builder' is an instance of OpBuilder, which is used to create MLIR operations, 
            /// Here, it's used to create an instance of the 'ConstantOp' operation, which represents
            /// a constant value in MLIR
            
            /// The 'ConstantOp' operation is defined within the Toy dialect using TableGen syntax, 
            /// typically found in a file like 'ToyOps.td'

            /// 'loc(num.loc())' converts the location information from the AST node 'num'
            /// into an MLIR Location object. Aiding in debugging and error reporting

            /// 'num.getValue()' retrieves the actual numeric value contained within the 'num' AST node.
            /// This value with location is then passed to the object of 'ConstantOp' while object instantiate

            /// The 'create<ConstantOp>' method, constructs a 'ConstantOp' operation in the current
            /// MLIR context. This operation encapsulates the constant value ('num.getValue()') as part
            /// of the MLIR intermediate representation, ready for further compilation or transformation
            /// stages. The operation is inserted into the MLIR module or function at the current
            /// insertion point managed by 'builder'
            return builder.create<ConstantOp>(loc(num.loc()), num.getValue());

        }



        /// This is a reference to a variable in an expression. The variable is
        /// expected to have been declared and so should have a value in the symbol
        /// table, otherwise emit an error and return nullptr.
        mlir::Value mlirGen(VariableExprAst &expr) { //**********************FIX COMMENT

            if (auto variable = symbolTable.lookup(expr.getName()))
                return variable;

            emitError(loc(expr.loc()), "error: unknown variable '") << expr.getName() << "'";
            
            return nullptr;
        }



        /// Emit a binary operation **********************FIX COMMENT
        mlir::Value mlirGen(BinaryExprAst &binop) {

            // First emit the operations for each side of the operation before emitting
            // the operation itself. For example if the expression is `a + foo(a)`
            // 1) First it will visiting the LHS, which will return a reference to the
            //    value holding `a`. This value should have been emitted at declaration
            //    time and registered in the symbol table, so nothing would be
            //    codegen'd. If the value is not in the symbol table, an error has been
            //    emitted and nullptr is returned.
            // 2) Then the RHS is visited (recursively) and a call to `foo` is emitted
            //    and the result value is returned. If an error occurs we get a nullptr
            //    and propagate.
            //

            mlir::Value lhs = mlirGen(*binop.getLHS());

            if (!lhs)
                return nullptr;
            
            mlir::Value rhs = mlirGen(*binop.getRHS());
            
            if (!rhs)
                return nullptr;
            
            auto location = loc(binop.loc());

            // Derive the operation name from the binary operator. At the moment we only
            // support '+' and '*'.
            switch (binop.getOp()) {
            
            case '+':
                return builder.create<AddOp>(location, lhs, rhs);
            
            case '*':
                return builder.create<MulOp>(location, lhs, rhs);
            
            }

            emitError(location, "invalid binary operator '") << binop.getOp() << "'";
            
            return nullptr;

        }



        /// Emit a call expression. It emits specific operations for the `transpose`//**********************FIX COMMENT
        /// builtin. Other identifiers are assumed to be user-defined functions.
        mlir::Value mlirGen(CallExprAst &call) {

            llvm::StringRef callee = call.getCallee();
            auto location = loc(call.loc());

            // Codegen the operands first.
            SmallVector<mlir::Value, 4> operands;
            for (auto &expr : call.getArgs()) {
                
                auto arg = mlirGen(*expr);
                
                if (!arg)
                    return nullptr;
                
                operands.push_back(arg);
            }

            // Builtin calls have their custom operation, meaning this is a
            // straightforward emission.
            if (callee == "transpose") {
                if (call.getArgs().size() != 1) {
                    emitError(location, "MLIR codegen encountered an error: toy.transpose "
                                        "does not accept multiple arguments");
                    return nullptr;
                }
                return builder.create<TransposeOp>(location, operands[0]);
            }

            // Otherwise this is a call to a user-defined function. Calls to
            // user-defined functions are mapped to a custom call that takes the callee
            // name as an attribute.
            return builder.create<GenericCallOp>(location, callee, operands);
        }



        /// @brief //**********************FIX COMMENT
        /// @param expr 
        /// @return         
        mlir::Value mlirGen(ExprAst &expr) {

            switch (expr.getKind()) {
            
            case toy::ExprAst::Expr_Num: 
                
                return mlirGen(cast<NumberExprAst> (expr));

            case toy::ExprAst::Expr_Var: //**********************FIX COMMENT
                return mlirGen(cast<VariableExprAst>(expr));

            case toy::ExprAst::Expr_BinOp: //**********************FIX COMMENT

                return mlirGen(cast<BinaryExprAst>(expr));

            case toy::ExprAst::Expr_Call: //**********************FIX COMMENT
                return mlirGen(cast<CallExprAst>(expr));

            default:

                emitError(loc(expr.loc()))
                    << "MLIR codegen encountered and unhandled expr kind '"
                    << Twine(expr.getKind()) << "'";

                return nullptr;
            }

        }



        /// @brief 
        /// @param vardecl 
        /// @return 
        mlir::Value mlirGen(VarDeclExprAst &vardecl) {

            /// vardecl is a node which holding the variable name and the value assigned to it
            /// Retrieves the initial value assigned to the variable, if any, during its declaration
            auto *init = vardecl.getInitVal();


            /// Checks if it retrived the value, if not it will emit an error
            if (!init) {

                emitError(loc(vardecl.loc()), "missing initializer in variable declaration");

                return nullptr;

            }


            /// Call the overload mlirGen which is responsible to handle ir gen for initializer
            /// Returns the ir of the initializer to value
            mlir::Value value = mlirGen(*init);

            
            /// Check if we got the ir of initializer or not, if not it returns nullptr
            if (!value) {
                
                return nullptr;
            }
            
            
            /// 
            if (!vardecl.getType().shape.empty()) {

                value = builder.create<ReshapeOp>(loc(vardecl.loc()), getType(vardecl.getType()), value);

            }

            
            /// By vardecl.getName() it retrives the variable name out of node and 
            /// also take the ir of the initializer together store in symbolTable for further processing
            if (failed(declare(vardecl.getName(), value)))

                return nullptr;

            
            return value;

        }




        /// @brief 
        /// @param ret 
        /// @return 
        mlir::LogicalResult mlirGen(ReturnExprAst &ret) {

            auto location = loc(ret.loc());

            mlir::Value expr = nullptr;

            if (ret.getExpr().has_value()) {

                if (!(expr = mlirGen(**ret.getExpr())))

                    return mlir::failure();

            }

            builder.create<ReturnOp>(location, expr ? ArrayRef(expr) : ArrayRef<mlir::Value>());

            return mlir::success();

        }



        

        /// Emit a print expression. It emits specific operations for two builtins:
        /// transpose(x) and print(x).
        mlir::LogicalResult mlirGen(PrintExprAst &call) {

            auto arg = mlirGen(*call.getArg());
            if (!arg)
                return mlir::failure();

            builder.create<PrintOp>(loc(call.loc()), arg);
            return mlir::success();
        }


        /// @brief 
        /// @param blockAst 
        /// @return 
        mlir::LogicalResult mlirGen(ExprAstList &blockAst) { // need to fix the serial if possible

            
            ScopedHashTableScope<StringRef, mlir::Value> varScope(symbolTable);
            

            for (auto &expr : blockAst) {


                // Specific handling for variable declarations, return statement, and
                // print. These can only appear in block list and not in nested
                // expressions.
                if (auto *vardecl = dyn_cast<VarDeclExprAst>(expr.get())) {
                
                    if (!mlirGen(*vardecl))
                        return mlir::failure();
                    
                    continue;
                }
                
                
                if (auto *ret = dyn_cast<ReturnExprAst>(expr.get()))
                    return mlirGen(*ret);

                
                // Fir builtin function like print() or transpose() part **********************FIX COMMENT
                if (auto *print = dyn_cast<PrintExprAst>(expr.get())) {
                
                    if (mlir::failed(mlirGen(*print)))
                        return mlir::success();
                    
                    continue;

                }

                // Generic expression dispatch codegen.**********************FIX COMMENT
                // For var c = a * b or user defined function etc...
                if (!mlirGen(*expr))
                    
                    return mlir::failure();
            
            }
            
            
            return mlir::success();
        
        }



        /// @brief : Generates an MLIR function prototype from a Toy function prototype
        /// @param proto : The AST node for a function prototype from the Toy AST
        /// @return : An MLIR representing the prototype of function
        mlir::toy::FuncOp mlirGen(PrototypeAst &proto) {

            
            /// Location information of function prototype
            auto location = loc(proto.loc());


            /// @brief : Creating a SmallVector named argTypes, to hold the MLIR types for the arguments of a function
            /// @note : Parameter proto.getArgs().size() determines the number of arguments the function has
            ///         & getType(VarType{}) get type of every parameter passed to the function
            llvm::SmallVector<mlir::Type, 4> argTypes(proto.getArgs().size(), getType(VarType{}));

            /// *****Beginner friendly way of above line*****
            /// Uncomment below beginner versions of above line to test
            /// Determine the number of arguments in the function prototype
            /// int numberOfArguments = proto.getArgs().size(); 

            /// Create a default MLIR type for the arguments
            /// getType(VarType{}) returns a type for function arguments
            /// mlir::Type argumentType = getType(VarType{});

            /// Initialize a SmallVector with the default type for all arguments
            /// This vector will have as many elements as there are arguments, and each element is initialized to its type
            /// llvm::SmallVector<mlir::Type, 4> argTypes(numberOfArguments, argumentType);
            

            // Create the type for the function, specifying its arguments and return type 
            auto funcType = builder.getFunctionType(argTypes, std::nullopt);

            /// Generates an MLIR function operation based on the defined function type, name, and location
            /// This operation adds the function to the MLIR module, making it part of the program's structure
            return builder.create<mlir::toy::FuncOp>(location, proto.getName(), funcType);


            /// *****Beginner friendly way of above line*****
            /// Uncomment below beginner versions of above line to test. Remember to comment out the above line
            /// Get the name of the function from its prototype
            /// auto functionName = proto.getName();


            /// Use the MLIR builder to create a new function operation.
            /// This operation is added to the MLIR module and represents the function in MLIR.
            /// auto createdFunction = builder.create<mlir::toy::FuncOp>(location, functionName, funcType);


            /// Return the newly created function operation.
            /// return createdFunction;


        }



        /// @brief : Generates an MLIR function proto and body from a Toy function proto and body
        /// @param funcAst : The AST node for a function from the Toy AST
        /// @return : An MLIR representing the function
        mlir::toy::FuncOp mlirGen(FunctionAst &funcAst) {
            

            /// @brief : ScopedHashTableScope automatically handles the scopes. 
            ///          When you create a new instance(VarScope) of ScopedHashTableScope, 
            ///          you're effectively saying, "Here begins a new scope." 
            ///          Variables can then be added to the symbol table under 
            ///          this scope. When the ScopedHashTableScope instance goes 
            ///          out of scope, it automatically removes any variables that 
            ///          were added to the symbol table under its scope
            /// @note :  ScopedHashTableScope uses llvm::StringRef for variable names (keys) 
            ///          and mlir::Value for their corresponding SSA values in the symbol table
            ScopedHashTableScope<llvm::StringRef, mlir::Value> varScope(symbolTable);


            /// @brief : Sets the current insertion point to the end of the module's body.
            ///          This means any new operations will be added at the end of the module
            /// @note :  The 'builder' is a tool used for constructing MLIR operations. 
            ///          'setInsertionPointToEnd' moves where new operations are added, 
            ///          ensuring they're placed at the end of the module's list of operations.
            ///          'theModule.getBody()' retrieves the main block of the module, 
            ///          which is where all top-level operations reside
            builder.setInsertionPointToEnd(theModule.getBody());


            /// @brief : Creates an MLIR function that includes both the function's signature and a structure for its body.
            ///          The signature (name, return type, parameters) is set up according to `funcAst.getProto()`
            /// @note : Upon creation, the `mlir::toy::FuncOp` object not only captures the function's signature but also 
            ///         but also preparing a space for its content (body). This block serves as the starting point for defining 
            ///         the function's body, where you'll add the actual operations the function performs
            mlir::toy::FuncOp function = mlirGen(*funcAst.getProto());


            /// @brief : Checks if the function was successfully created
            /// @note : If 'function' is null, it means the function creation failed, and we exit with 'nullptr'
            if (!function)

                return nullptr;


            /// @brief : Accesses the first block in the function, where operations are to be added
            /// @note : The 'entryBlock' serves as the starting line for defining what the function does. 
            ///         It's the place to begin adding operations, even though it starts empty
            mlir::Block &entryBlock = function.front();

            
            /// @brief : Gathers the arguments of the function prototype for further processing
            /// @note : 'protoArgs' holds the list of arguments defined in the function's prototype, 
            ///          used for creating the function parameters in MLIR
            auto protoArgs = funcAst.getProto()->getArgs();


            /// @brief : Maps each argument from the Toy prototype 
            ///          to its corresponding MLIR representation in the function's entry block
            /// @note : This loop pairs each argument name with its MLIR value, 
            ///         declaring them in the MLIR function scope. 
            /// For example: protoArgs Elements (simplified for explanation): 
            /// Could be represented internally as something like ["a", "b", "c"] if you're extracting just the names from the AST.
            /// Corresponding MLIR Values in the function could be : %arg0, %arg1, and %arg2 in the MLIR code
            /// Zipped list could be, nameValue: [("a", %arg0), ("b", %arg1), ("c", %arg2)]
            for (const auto nameValue : llvm::zip(protoArgs, entryBlock.getArguments())) {

                /// Extract the argument name from the protoArgs list or vector
                auto argName = std::get<0>(nameValue)->getName();

                /// Extract the corresponding MLIR representation of the function argument
                auto mlirValue = std::get<1>(nameValue);

                /// Attempt to declare the argument in the current MLIR scope
                if (failed(declare(argName, mlirValue)))

                    return nullptr;

            }


            /// @brief : Sets where new operations will be added in the MLIR function
            /// @note : By using 'setInsertionPointToStart', we specify that any new commands 
            ///         or operations we add next will go at the beginning of the 'entryBlock'
            builder.setInsertionPointToStart(&entryBlock);


            /// @brief : It tries to generate the MLIR representation for the function's body.
            ///        If this process fails, the function definition is removed from the MLIR module to maintain consistency
            /// @note This step is crucial for ensuring that only successfully translated functions are included in the final MLIR module.
            if (mlir::failed(mlirGen(*funcAst.getBody()))) {

                function.erase(); /// Remove the function from the module to avoid inconsistencies
            
                return nullptr;   /// Indicate failure by returning nullptr

            }


            /// Attempts to find a return operation in the function's entry block
            /// 'returnOp' will hold a return operation if one exists at the end of the entry block
            ReturnOp returnOp;


            bool functionEndsWithReturn = !entryBlock.empty(); /// Check if there are any operations in the entry block.
            
            if (functionEndsWithReturn) {
                
                /// Attempt to treat the last operation as a return. If it's not a return, this will be null.
                returnOp = dyn_cast<ReturnOp>(entryBlock.back());
          
            }


            /// @brief : Adds a return operation if none exists
            /// @note : This step is vital for functions meant to end without returning any specific value, ensuring they terminate correctly
            if (!returnOp)

                builder.create<ReturnOp>(loc(funcAst.getProto()->loc()));


            /// @brief : Adjusts the function's type if the return operation has operands
            /// @note : In case the function should return a value, this modifies the 
            ///         function's signature to reflect the correct return type based on the operation's operands
            else if (returnOp.hasOperand()) {

                function.setType(

                    builder.getFunctionType(

                        function.getFunctionType().getInputs(), getType(VarType{})
                    
                    )
                
                );

            }


            /// Returns function with its proto and body with only dummy return
            return function;

        }

        
    };


} /// namespace ends here



/// namespace toy where the public API for codegen will be present
namespace toy {
    
    /// @brief : Translates the AST of a Toy module into an MLIR module
    /// @param context : An environment that holds all the MLIR operations, types, and other entities 
    ///                  needed during the conversion process. It's like a workspace for MLIR tasks
    /// @param moduleAst : The root of the abstract syntax tree (AST) for a Toy program
    /// @return : An object of type mlir::OwningOpRef<mlir::ModuleOp>. This object is a smart pointer that 
    ///           owns an MLIR module, representing the entire converted Toy program. If the conversion 
    ///           is successful, this module contains the MLIR representation of the Toy program. If there's
    ///           an error during conversion, the function returns a nullptr wrapped in the smart pointer,
    ///           indicating the conversion process failed
    mlir::OwningOpRef<mlir::ModuleOp> mlirGen(mlir::MLIRContext &context, ModuleAst &moduleAst) {
        
        /// @brief : Create a instance of MLIRGenImpl class with input context as argument, 
        ///          generate MLIR from ASt & return
        return MLIRGenImpl(context).mlirGen(moduleAst);


        /// *****Beginner friendly way of above line*****
        /// Uncomment below beginner versions of above line to test. Make you comment out previous line
        /// Create a instance of MLIRGenImpl class with input context as argument 
        /// MLIRGenImpl mlirGenerator = MLIRGenImpl(context);

        /// Generate MLIR for the given AST using method called mlirGen()
        /// auto generatedIR = mlirGenerator.mlirGen(moduleAst);

        /// Return the generated MLIR
        /// return generatedIR;
    
    }


} /// namespace toy ends here