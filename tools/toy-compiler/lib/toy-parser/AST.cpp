//===- AST.cpp - Helper for printing out the Toy AST ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the AST dump for the Toy language.
//
//===----------------------------------------------------------------------===//



#include "toy-analysis-parser/AST.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include <string>



using namespace toy;


namespace {

	/// RAII helper to manage increasing/decreasing the identation as we traverse
	/// The AST
	struct Indent {
		
		Indent(int &level) : level(level) {

			++level;

		}


		~Indent() {

			--level;

		}


		int &level;

	};



	class AstDumper {


	public:

		void dump(ModuleAst *node);


	private:

		void dump(const VarType &type);
		void dump(VarDeclExprAst *varDecl);
		void dump(ExprAst *expr);
		void dump(ExprAstList *exprList);
		void dump(NumberExprAst *num);
		void dump(LiteralExprAst *node);
		void dump(VariableExprAst *node);
		void dump(ReturnExprAst *node);
		void dump(BinaryExprAst *node);
		void dump(CallExprAst *node);
		void dump(PrintExprAst *node);
		void dump(PrototypeAst *node);
		void dump(FunctionAst *node);
		

		/// Actually print spaces matching the current indentation level
		void indent() {

			for (int i =0; i < curIndent; i++)

				llvm::errs() << " ";

		}


		int curIndent = 0;


	};


} /// namespace



/// Return a formatted string for the location of any node
template <typename T>
static std::string loc(T *node) {

	const auto &loc = node->loc();

	return (llvm::Twine("@") 
			+ *loc.file + ":" 
			+ llvm::Twine(loc.line) 
			+ ":" 
			+ llvm::Twine(loc.col)).str();


}



/// Helper Macro to bump the indentation level and print the leading spaces for
/// the current indentations
#define INDENT()                                                               \
    Indent level_(curIndent);                                                  \
    indent();



/// Dispatch to a generic expressions to the appropriate subclass using RTTI
void AstDumper::dump(ExprAst *expr) {

	llvm::TypeSwitch<ExprAst *>(expr)
		.Case<BinaryExprAst, CallExprAst, LiteralExprAst, NumberExprAst,
			PrintExprAst, ReturnExprAst, VarDeclExprAst, VariableExprAst>(
				[&](auto *node) {

					this->dump(node);

				}
			);

}



/// A variable declaration is printing the variable name, the type, and then
/// recurse in the initializer value
void AstDumper::dump(VarDeclExprAst *varDecl) {

	INDENT();
	

	llvm::errs() << "VarDecl " << varDecl->getName();
	

	dump(varDecl->getType());
	

	llvm::errs() << " " << loc(varDecl) << "\n";
	

	dump(varDecl->getInitVal());

}



/// A "block", or a list of expression
void AstDumper::dump(ExprAstList *exprList) {

	INDENT();
	

	llvm::errs() << "Block {\n";
	

	for (auto &expr : *exprList)
		dump(expr.get());

	indent();
	

	llvm::errs() << "} // Block\n";

}



/// A literal number, just print the value
void AstDumper::dump(NumberExprAst *num) {

	INDENT();
	llvm::errs() << num->getValue() << " " << loc(num) << "\n";

}



/// Helper to print recursively a literal. This handles nested array like:
/// [[1, 2], [3, 4]]
/// We print out such array with the dimensions spelled out at every level:
/// <2, 2>[<2>[1, 2], <2>[3, 4]]
void printLitHelper(ExprAst *litOrNum) {

	///Inside a literal expression we can have either a number or another literal
	if (auto *num = llvm::dyn_cast<NumberExprAst>(litOrNum)) {

		llvm::errs() << num->getValue();
		return;

	}


	auto *literal = llvm::cast<LiteralExprAst>(litOrNum);


	/// Print the dimension for this literal first
	llvm::errs() << "<";
	llvm::interleaveComma(literal->getDims(), llvm::errs());
	llvm::errs() << ">";


	/// Now print the content, recursing on every element of the list
	llvm::errs() << "[";
	llvm::interleaveComma(literal->getValues(), llvm::errs(), [&](auto &elt) {
		
		printLitHelper(elt.get());

	});


	llvm::errs() << "]";


}



/// Print a literal, see the recursive helper above for the implementation
void AstDumper::dump(LiteralExprAst *node) {

	INDENT();
	llvm::errs() << "Literal: ";
	printLitHelper(node);
	llvm::errs() << " " << loc(node) << "\n";

}



/// Print a variable reference (jsut a name)
void AstDumper::dump(VariableExprAst *node) {

	INDENT();
	llvm::errs() << "var: " << node->getName() << " " << loc(node) << "\n";

}



/// Return statement print the return and its (optional) argument
void AstDumper::dump(ReturnExprAst *node) {

	INDENT();
	llvm::errs() << "Return\n";
	if (node->getExpr().has_value())

		return dump(*node->getExpr());

	{
		INDENT();
		llvm::errs() << "(void)\n";

	}

}



/// Print a binary operation, first the operator, then recurse into LHS and RHS
void AstDumper::dump(BinaryExprAst *node) {

	INDENT();
	llvm::errs() << "BinOp: " << node->getOp() << " " << loc(node) << "\n";
	dump(node->getLHS());
	dump(node->getRHS());

}



/// Print a call expression, first the callee name and the list of args by
/// recursing into each individual argument
void AstDumper::dump(CallExprAst *node) {

	INDENT();
	

	llvm::errs() << "Call '" << node->getCallee() << "'' [ " << loc(node) << "\n";
	

	for (auto &arg : node->getArgs())
		dump(arg.get());

	indent();
	

	llvm::errs() << "]\n"; 

}



/// Print a builtin print call, first the builtin name and then the argument
void AstDumper::dump(PrintExprAst *node) {

	INDENT();
	llvm::errs() << "Print [ " << loc(node) << "\n";
	dump(node->getArg());
	indent();
	llvm::errs() << "]\n";

}



/// Print type: only the shape is printed in between '<' and '>'
void AstDumper::dump(const VarType &type) {

	llvm::errs() << "<";
	llvm::interleaveComma(type.shape, llvm::errs());
	llvm::errs() << ">";

}



/// Print a function prototype, first the function name, and then the list of 
/// parameters names
void AstDumper::dump(PrototypeAst *node) {

	INDENT();


	llvm::errs() << "Proto '" << node->getName() << "' " << loc(node) << "\n";
	

	indent();
	

	llvm::errs() << "Params: [";
	llvm::interleaveComma(node->getArgs(), llvm::errs(), [](auto &arg) {

		llvm::errs() << arg->getName();
 
	});

	llvm::errs() << "]\n"; 

}



/// Print a function, first the prototype and then the body
void AstDumper::dump(FunctionAst *node) {


	/// There's a macro or a function called INDENT being called, 
	/// which probably manages indentation levels for pretty-printing the AST structure. 
	INDENT()


	llvm::errs() << "Function:\n";
	

	dump(node->getProto());
    dump(node->getBody()); 

}



/// Print a module, actually loop over the functions and print them in sequence
void AstDumper::dump(ModuleAst *node) {


	/// There's a macro or a function called INDENT being called, 
	/// which probably manages indentation levels for pretty-printing the AST structure. 
	INDENT();


	llvm::errs() << "Module:\n";


	/// This loop iterates over all the functions 
	/// (or possibly other constructs) contained 
	/// within the module pointed to by node.
	for (auto &f : *node)
		dump(&f);

	
}



namespace toy {

	///Public API
	void dump(ModuleAst &module) {

		AstDumper().dump(&module);

	}

} /// namespace toy