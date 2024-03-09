//===- AST.h - Node definition for the Toy AST ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the AST for the Toy language. It is optimized for
// simplicity, not efficiency. The AST forms a tree structure where each node
// references its children using std::unique_ptr<>.
//
//===----------------------------------------------------------------------===//

#ifndef TOY_AST_H
#define TOY_AST_H


#include "toy-analysis-parser/Lexer.h"


#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include <utility>
#include <vector>
#include <optional>


namespace toy {


	/// A variable type with shape information
	struct VarType {

		std::vector<int64_t> shape;

	};



	/// Base class for all expression nodes.
	class ExprAst{
	

	public:
		

		enum ExprAstKind {

			Expr_VarDecl,
			Expr_Return,
			Expr_Num,
			Expr_Literal,
			Expr_Var,
			Expr_BinOp,
			Expr_Call,
			Expr_Print,

		}; 



		ExprAst(ExprAstKind kind, Location location)
				: kind(kind), location(std::move(location)) {}



		virtual ~ExprAst() = default;


		ExprAstKind getKind() const {

			return kind; 

		}

		
		const Location &loc() {

			return location;

		}


	private:


		const ExprAstKind kind;

		Location location;


	};



	/// Ablock-st of expressions
	using ExprAstList = std::vector<std::unique_ptr<ExprAst>>;



	/// Expression class for a literal value
	class NumberExprAst : public ExprAst {

		double val;


	public:


		NumberExprAst(Location loc, double val)
					: ExprAst(Expr_Num, std::move(loc)), val(val) {}


		double getValue() {

			return val;

		}


		///LVM style RTTI
		static bool classof(const ExprAst *c) {

			return c->getKind() == Expr_Num;

		}


	};



	/// Expression class for a literal value
	class LiteralExprAst : public ExprAst {

		std::vector<std::unique_ptr<ExprAst>> values;
		std::vector<int64_t> dims;


	public:


		LiteralExprAst(Location loc, std::vector<std::unique_ptr<ExprAst>> values, std::vector<int64_t> dims)
						: ExprAst(Expr_Literal, std::move(loc)), values(std::move(values)), dims(std::move(dims)) {}


		llvm::ArrayRef<std::unique_ptr<ExprAst>> getValues() {

			return values;

		}


		llvm::ArrayRef<int64_t> getDims() {

			return dims;

		}


		/// LLVM style RTTI
		static bool classof(const ExprAst *c) {

			return c->getKind() == Expr_Literal;

		}


	};



	/// Expression class for referencing a variable, like "a"
	class VariableExprAst : public ExprAst {

		std::string name;


	public:


		VariableExprAst(Location loc, llvm::StringRef name)
						: ExprAst(Expr_Var, std::move(loc)), name(name) {}


		llvm::StringRef getName() {

			return name;

		}


		/// LLVM style RTTI
		static bool classof(const ExprAst *c) {

			return c->getKind() == Expr_Var;

		}


	};



	/// Expression class for defining a varaible
	class VarDeclExprAst : public ExprAst {

		std::string name;
		VarType type;
		std::unique_ptr<ExprAst> initVal;


	public:


		VarDeclExprAst(Location loc, llvm::StringRef name, VarType type, std::unique_ptr<ExprAst> initVal)
					: ExprAst(Expr_VarDecl, std::move(loc)), name(name), type(std::move(type)), initVal(std::move(initVal)) {}


		llvm::StringRef getName() {

			return name;

		}


		ExprAst *getInitVal() {

			return initVal.get();

		}


		const VarType &getType() {

			return type;

		}


		/// LLVM style RTTI
		static bool classof(const ExprAst *c) {

			return c->getKind() == Expr_VarDecl;

		}


	};



	/// Expression class for a return operator
	class ReturnExprAst : public ExprAst {

		std::optional<std::unique_ptr<ExprAst>> expr;


	public:


		ReturnExprAst(Location loc, std::optional<std::unique_ptr<ExprAst>> expr)
					: ExprAst(Expr_Return, std::move(loc)), expr(std::move(expr)) {}



		std::optional<ExprAst *> getExpr() {

			if (expr.has_value())

				return expr->get();

			return std::nullopt;

		}


		/// LLVM style RTTI
		static bool classof(const ExprAst *c) {

			return c->getKind() == Expr_Return;

		}



	};



	/// Expression class for a binary operator
	class BinaryExprAst : public ExprAst {

		char op;
		std::unique_ptr<ExprAst> lhs,rhs;


	public:


		char getOp() {

			return op;

		}


		ExprAst *getLHS() {

			return lhs.get();

		}



		ExprAst *getRHS() {

			return rhs.get();

		}



		BinaryExprAst(Location loc, char op, std::unique_ptr<ExprAst> lhs, std::unique_ptr<ExprAst> rhs)
					: ExprAst(Expr_BinOp, std::move(loc)), op(op), lhs(std::move(lhs)), rhs(std::move(rhs)) {}



		/// LLVM style RTTI
		static bool classof(const ExprAst *c) {

			return c->getKind() == Expr_BinOp;

		}
	
	};



	/// Expression class for function calls
	class CallExprAst : public ExprAst {

		std::string callee;
		std::vector<std::unique_ptr<ExprAst>> args;


	public:


		CallExprAst(Location loc, const std::string &callee, std::vector<std::unique_ptr<ExprAst>> args)
					: ExprAst(Expr_Call, std::move(loc)), callee(callee), args(std::move(args)) {}



		llvm::StringRef getCallee() {

			return callee;

		}


		llvm::ArrayRef<std::unique_ptr<ExprAst>> getArgs() {

			return args;

		}


		/// LLVM style RTTI
		static bool classof(const ExprAst *c) {

			return c->getKind() == Expr_Call;

		}

	};



	/// Expression class for builtin print calls
	class PrintExprAst : public ExprAst {

		std::unique_ptr<ExprAst> arg;


	public:


		PrintExprAst(Location loc, std::unique_ptr<ExprAst> arg)
					: ExprAst(Expr_Print, std::move(loc)), arg(std::move(arg)) {}



		ExprAst *getArg() {

			return arg.get();

		}


		/// LLVM style RTTI
		static bool classof(const ExprAst *c) {

			return c->getKind() == Expr_Print;

		}

	};



	/// This class represents the "prototype" for a function, which captures its
	/// name, and its argument names (thus implicitly the number of arguments the
	/// function takes)
	class PrototypeAst {

		Location location;
		std::string name;
		std::vector<std::unique_ptr<VariableExprAst>> args;


	public:


		PrototypeAst(Location location, const std::string &name, std::vector<std::unique_ptr<VariableExprAst>> args)
					: location(std::move(location)), name(name), args(std::move(args)) {}


		const Location &loc() {

			return location;

		}


		llvm::StringRef getName() const {

			return name;

		}


		llvm::ArrayRef<std::unique_ptr<VariableExprAst>> getArgs() {

			return args;

		}

	};



	/// This class represents a function defination itself
	class FunctionAst {

		std::unique_ptr<PrototypeAst> proto;
		std::unique_ptr<ExprAstList> body;


	public:


		FunctionAst(std::unique_ptr<PrototypeAst> proto, std::unique_ptr<ExprAstList> body)
					: proto(std::move(proto)), body(std::move(body)) {}

		PrototypeAst *getProto() {

			return proto.get();

		}


		ExprAstList *getBody() {

			return body.get();

		}

	};



	/// This class represents a list of functions to be processed together
	class ModuleAst {

		std::vector<FunctionAst> functions;


	public:


		ModuleAst(std::vector<FunctionAst> functions)
				: functions(std::move(functions)) {}


		auto begin() {

			return functions.begin();

		}


		auto end() {

			return functions.end();

		}


	};



	void dump(ModuleAst &);



} /// namespace toy




#endif // TOY_AST_H
