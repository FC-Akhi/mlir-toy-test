//===- Parser.h - Toy Language Parser -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the parser for the Toy language. It processes the Token
// provided by the Lexer and returns an AST.
//
//===----------------------------------------------------------------------===//

#ifndef TOY_PARSER_H
#define TOY_PARSER_H



#include "toy-analysis-parser/AST.h"


#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <map>
#include <utility>
#include <vector>
#include <optional>



namespace toy {

	class Parser {


	public:


		/// Create a Parser for the supplied lexer
		Parser(Lexer &lexer) : lexer(lexer) {}


		/// Parse a full Module. A module is a list of function definations
		std::unique_ptr<ModuleAst> parseModule() {

			lexer.getNextToken(); /// prime the lexer

			printf("0.lastChar:%d\n", lexer.lastChar);

			printf("0.curToken:%d\n", lexer.getCurToken());

			/// Parse functions one at a time and accumulate in this vector
			std::vector<FunctionAst> functions;

			while (auto f = parseDefinition()) {

				functions.push_back(std::move(*f));
				
				if (lexer.getCurToken() == TK_eof)
					
					break;

			}


			/// If we didn't reach EOF, there was an error during parsing
			if (lexer.getCurToken() != TK_eof)

				return parseError<ModuleAst>("nothing", "at the end of module");

			return std::make_unique<ModuleAst>(std::move(functions)); 

		}


	private:


		Lexer &lexer;



		/// Helper function to signal errors while parsing, it takes an argument
    	/// indicating the expected token and another argument giving more context.
   		/// Location is retrieved from the lexer to enrich the error message.
    	template <typename R, typename T, typename U = const char *>

    	std::unique_ptr<R> parseError(T &&expected, U &&context = "") {

    		auto curToken = lexer.getCurToken();

    		llvm::errs() << "Parse error (" << lexer.getLastLocation().line << ", "
    					 << lexer.getLastLocation().col << "): expected '" << expected
    					 << "' " << context << "but has Token" << curToken;


    		if (isprint(curToken))

    			llvm::errs() << " '" << (char)curToken << "'";

    		llvm::errs() << "\n";

    		return nullptr;


    	}



		/// Parse a literal number
		/// numberexpr ::= number
		std::unique_ptr<ExprAst> parseNumberExpr() {

			auto loc = lexer.getLastLocation();
			auto result = std::make_unique<NumberExprAst>(std::move(loc), lexer.getValue());

			lexer.consume(TK_number);

			return std::move(result);

		}



		/// identifierexpr
    	///   ::= identifier
    	///   ::= identifier '(' expression ')'
		std::unique_ptr<ExprAst> parseIdentifierExpr() {

			std::string name(lexer.getIdentifier());


			auto loc = lexer.getLastLocation();
			lexer.getNextToken(); ///eat identifier

			if (lexer.getCurToken() != '(') // simple variable ref

				return std::make_unique<VariableExprAst>(std::move(loc), name);

			/// This is function call
			lexer.consume(Token('('));

			std::vector<std::unique_ptr<ExprAst>> args;

			if (lexer.getCurToken() != ')') {

				while (true) {

					if (auto arg = parseExpression())

						args.push_back(std::move(arg));

					else

						return nullptr;

					if (lexer.getCurToken() == ')')

						break;

					if (lexer.getCurToken() != ',')

						return parseError<ExprAst>(", or )", "in argument list");

					lexer.getNextToken();
				}

			}

			lexer.consume(Token(')'));


			/// It can be builtin call to print
			if (name == "print") {

				if (args.size() != 1)

					return parseError<ExprAst>("<single arg>", "as argument to print()");

				return std::make_unique<PrintExprAst>(std::move(loc), std::move(args[0]));

			}


			/// Call to a user-defined function
			return std::make_unique<CallExprAst>(std::move(loc), name, std::move(args));


		}



    	/// parenexpr ::= '(' expression ')'
		std::unique_ptr<ExprAst> parseParenExpr() {

			lexer.getNextToken(); /// eat

			auto v = parseExpression();

			if (!v)

				return nullptr;

			if (lexer.getCurToken() != ')')

				return parseError<ExprAst>(")", "to close expression with parentheses");

			lexer.consume(Token(')'));

			return v;

		}




		/// Parse a literal array expression.
    	/// tensorLiteral ::= [ literalList ] | number
    	/// literalList ::= tensorLiteral | tensorLiteral, literalList
		std::unique_ptr<ExprAst> parseTensorLiteralExpr() {

			auto loc = lexer.getLastLocation();
			lexer.consume(Token('['));


			/// Hold the list of values at this nesting level
			std::vector<std::unique_ptr<ExprAst>> values;


			/// Hold the dimensions for all the nesting inside this level
			std::vector<int64_t> dims;

			do {

				/// We can have either another nested array or a number literal
				if (lexer.getCurToken() == '[') {

					values.push_back(parseTensorLiteralExpr());


					if (!values.back())

						return nullptr; /// parse error in the nested array

				} else {

					if (lexer.getCurToken() != TK_number)
						
						return parseError<ExprAst>("<num> or [", "in literal expression");

					values.push_back(parseNumberExpr());

				}


				/// End of this list  on ']'
				if (lexer.getCurToken() == ']')

					break;


				/// Elements are separated by a comma
				if (lexer.getCurToken() != ',')

					return parseError<ExprAst>("] or ,", "in literal expression");

				lexer.getNextToken(); /// eat


			} while (true);


			if (values.empty())

				return parseError<ExprAst>("<something>", "to fill literal expression");

			lexer.getNextToken(); /// eat ]


			/// Fill in the dimensions now. First the current nesting level
			dims.push_back(values.size());


			/// If there is any nested array, process all of them and ensure that
			/// dimensions are uniform
			if (llvm::any_of(values, [](std::unique_ptr<ExprAst> &expr) {

				return llvm::isa<LiteralExprAst>(expr.get());

			}))

			{

				auto *firstLiteral = llvm::dyn_cast<LiteralExprAst>(values.front().get());

				if (!firstLiteral)

					return parseError<ExprAst>("uniform well-nested dimensions", "inside literal expression");

				/// Append the nested dimensions to the current level
				auto firstDims = firstLiteral->getDims();
				dims.insert(dims.end(), firstDims.begin(), firstDims.end());


				/// Sanity check that shape is uniformacross all elements of the list
				for (auto &expr : values) {

					auto *exprLiteral = llvm::cast<LiteralExprAst>(expr.get());

					if (!exprLiteral)

						return parseError<ExprAst>("uniform well-nested dimensions",
												   "inside literal expression");

					if (exprLiteral->getDims() != firstDims)

						return parseError<ExprAst>("uniform well-nested dimensions",
												   "inside literal expression");


				}

			}

			return std::make_unique<LiteralExprAst>(std::move(loc), std::move(values), std::move(dims));




		}


		/// primary
    	///   ::= identifierexpr
    	///   ::= numberexpr
    	///   ::= parenexpr
    	///   ::= tensorliteral
    	std::unique_ptr<ExprAst> parsePrimary() {


    		switch (lexer.getCurToken()) {


    		case TK_number:

    			return parseNumberExpr();


    		case TK_identifier:

    			return parseIdentifierExpr();


    		case '(':

    			return parseParenExpr();


    		case '[':

    			return parseTensorLiteralExpr();


    		case '}':

    			return nullptr;


    		case ';':

    			return nullptr;


    		default:

    			llvm::errs() << "unknown token '" << lexer.getCurToken()
    						 << "' when expecting and expression\n";

    			return nullptr;


    		}


    	}




    	


		/// Get the precedence of the pending binary operator token.
    	int getTokPrecedence() {

    		if (!isascii(lexer.getCurToken()))

    			return -1;

    		/// -1 is the lowest precedence
    		switch (static_cast<char>(lexer.getCurToken())) {

    		case '-':

    			return 20;

    		case '+':

    			return 20;


    		case '*':

    			return 40;


    		default:

    			return -1;

    		}

    	}


	


		/// Recursively parse the right hand side of a binary expression, the ExprPrec
    	/// argument indicates the precedence of the current binary operator.
    	///
    	/// binoprhs ::= ('+' primary)*
    	std::unique_ptr<ExprAst> parseBinOpRHS(int exprPrec, std::unique_ptr<ExprAst> lhs) {

    		/// If this is a binop, find its precedence
    		while (true) {

    			int tokPrec = getTokPrecedence();


    			/// If this is a binop that binds at least as tightly as the current binop
    			/// consume it, otherwise we are done
    			if (tokPrec < exprPrec) 

    				return lhs;

    			/// Okay, we know this is a binop
    			int binOp = lexer.getCurToken();
    			lexer.consume(Token(binOp));
    			auto loc = lexer.getLastLocation();


    			/// Parse the primary expression after the binary operator
    			auto rhs = parsePrimary();


    			if (!rhs)

    				return parseError<ExprAst>("expression", "to complete binary operator");

    			/// If BinOp binds less tightly with rhs than the operator after rhs, let
    			/// the pending operator take rhs as its lhs
    			int nextPrec = getTokPrecedence();


    			if (tokPrec < nextPrec) {

    				rhs =  parseBinOpRHS(tokPrec + 1, std::move(rhs));

    				if (!rhs)

    					return nullptr;
    			}


    			/// Merge lhs/rhs
    			lhs = std::make_unique<BinaryExprAst>(std::move(loc), binOp, std::move(lhs), std::move(rhs));


    		}

    	}



    	/// expression::= primary binop rhs
    	std::unique_ptr<ExprAst> parseExpression() {

    		auto lhs = parsePrimary();

    		if(!lhs)

    			return nullptr;

    		return parseBinOpRHS(0, std::move(lhs));

    	}




		/// Parse a return statement
		/// return := return ; | return expression ;
		std::unique_ptr<ReturnExprAst> parseReturn() {

			auto loc = lexer.getLastLocation();
			lexer.consume(TK_eof);


			/// return takes an optional argument
			std::optional<std::unique_ptr<ExprAst>> expr;


			
			if (lexer.getCurToken() != ';') {

				expr = parseExpression();

				if (!expr)

					return nullptr;
			}


			return std::make_unique<ReturnExprAst>(std::move(loc), std::move(expr));

		}



		

    	/// type ::= < shape_list >
    	/// shape_list ::= number | number , shape_list
    	std::unique_ptr<VarType> parseType() {

    		if (lexer.getCurToken() != '<')

    			return parseError<VarType>("<", "to begin type");


    		lexer.getNextToken(); /// eat <


    		auto type = std::make_unique<VarType>();


    		while (lexer.getCurToken() == TK_number) {

    			type->shape.push_back(lexer.getValue());

    			lexer.getNextToken();


    			if (lexer.getCurToken() == ',')

    				lexer.getNextToken();

    		}


    		if (lexer.getCurToken() != '>')

    			return parseError<VarType>(">", "to end type");


    		lexer.getNextToken(); /// eat >

    		return type;

    	}



    	/// Parse a variable declaration, it starts with a `var` keyword followed by
    	/// and identifier and an optional type (shape specification) before the
    	/// initializer.
    	/// decl ::= var identifier [ type ] = expression
    	std::unique_ptr<VarDeclExprAst> parseDeclaration() {

    		if (lexer.getCurToken() != TK_var)

    			return parseError<VarDeclExprAst>("var", "to begin declaration");

    		auto loc = lexer.getLastLocation();

    		lexer.getNextToken(); // eat var

    		if (lexer.getCurToken() != TK_identifier)

    			return parseError<VarDeclExprAst>("identified", "after 'var' declaration");

    		std::string id(lexer.getIdentifier());

    		lexer.getNextToken(); /// eat Id


    		std::unique_ptr<VarType> type; /// Type is optional, it can be inferred

    		if (lexer.getCurToken() == '<') {

    			type = parseType();
    			if (!type)

    				return nullptr;

    		}

    		if (!type)

    			type = std::make_unique<VarType>();

    		lexer.consume(Token('='));
    		auto expr = parseExpression();


    		return std::make_unique<VarDeclExprAst>(std::move(loc), std::move(id), 
    												std::move(*type), std::move(expr));

    	}



    	/// Parse a block: a list of expression separated by semicolons and wrapped in
	    /// curly braces.
	    ///
	    /// block ::= { expression_list }
	    /// expression_list ::= block_expr ; expression_list
	    /// block_expr ::= decl | "return" | expression
    	std::unique_ptr<ExprAstList> parseBlock() {

    		if (lexer.getCurToken() != '{')

    			return parseError<ExprAstList>("{", "to begin block");

    		lexer.consume(Token('{'));

    		auto exprList = std::make_unique<ExprAstList>();


    		/// Ignore empty expressions: swallow sequences of semicolons
    		while (lexer.getCurToken() == ';')

    			lexer.consume(Token(';'));


    		/// Checks if current token is not eof or it is '}'. Which means end of function
    		while (lexer.getCurToken() != '}' && lexer.getCurToken() != TK_eof) {


    			/// If above condition of while is true that means we are still inside function
    			/// If we are still inside function that means there could be first case
    			/// current token could be variable declaration
    			/// For example: var a = 2 + 3;
    			/// Below part will take care of var a only
    			if (lexer.getCurToken() == TK_var) {

    				/// Variable declaration
    				auto varDecl = parseDeclaration();

    				if (!varDecl)

    					return nullptr;

    				exprList->push_back(std::move(varDecl));

    			} 


    			/// If we are still inside function that means there could be second case
    			/// current token could be return
    			else if (lexer.getCurToken() == TK_return) {

    				/// Return statement
    				auto ret = parseReturn();

    				if (!ret)

    					return nullptr;

    				exprList->push_back(std::move(ret));

    			} 


    			/// If we are still inside function that means there could be third case
    			/// current token could be general expression
    			/// For example: var a = 2 + 3;
    			/// Below part will take care of 2 + 3 only
    			else {

    				/// General expression
    				auto expr = parseExpression();

    				if (!expr)

    					return nullptr;

    				exprList->push_back(std::move(expr));

    			}

    			/// Ensure that elements are separated by a semicolon
    			if (lexer.getCurToken() != ';')

    				return parseError<ExprAstList>(";", "after expression");


    			/// Ignore empty expression: swallow sequences of semicolon 
    			while (lexer.getCurToken() == ';')

    				lexer.consume(Token(';'));

    		}

    		if (lexer.getCurToken() != '}')

    			return parseError<ExprAstList>("}", "to close block");

    		lexer.consume(Token('}'));

    		return exprList;


    	}



		/// prototype ::= def id '(' decl_list ')'
	    /// decl_list ::= identifier | identifier, decl_list
    	std::unique_ptr<PrototypeAst> parsePrototype() {

    		auto loc = lexer.getLastLocation();


    		if (lexer.getCurToken() != TK_def) {

    			printf("1CurToken: %d\n", lexer.getCurToken());

    			return parseError<PrototypeAst>("def", "in prototype");

    		}

    		printf("2CurToken: %d\n", lexer.getCurToken());
    		lexer.consume(TK_eof);



    		if (lexer.getCurToken() != TK_identifier) {

    			printf("Inside 2nd if======\n");

    			return parseError<PrototypeAst>("function name", "in prototype");

    		}

    		std::string fnName(lexer.getIdentifier());

    		lexer.consume(TK_identifier);



    		if (lexer.getCurToken() != '(')

    			return parseError<PrototypeAst>("(", "in prototype");

    		

    		lexer.consume(Token('('));

    		std::vector<std::unique_ptr<VariableExprAst>> args;


    		
    		if (lexer.getCurToken() != ')') {

    			do {

    				std::string name(lexer.getIdentifier());

    				auto loc = lexer.getLastLocation();
    				lexer.consume(TK_identifier);

    				auto decl = std::make_unique<VariableExprAst>(std::move(loc), name);
    				args.push_back(std::move(decl));

    				if (lexer.getCurToken() != ',')

    					break;

    				lexer.consume(Token(','));

    				if (lexer.getCurToken() != TK_identifier)

    					return parseError<PrototypeAst>("identifier", "after ',' in function parameter list");


    			} while (true);

    		}

    		

    		if (lexer.getCurToken() != ')')

    			return parseError<PrototypeAst>(")", "to end function prototype");


    		

    		/// success
    		lexer.consume(Token(')'));

    		

    		return std::make_unique<PrototypeAst>(std::move(loc), fnName, std::move(args)); 



    	}





	    /// Parse a function definition, we expect a prototype initiated with the
	    /// `def` keyword, followed by a block containing a list of expressions.
	    ///
	    /// definition ::= prototype block
    	std::unique_ptr<FunctionAst> parseDefinition() {
		    
		    auto proto = parsePrototype();
		    
		    
		    if (proto) {
		    	
		    	auto block = parseBlock();

		        if (block)
		    
		            return std::make_unique<FunctionAst>(std::move(proto), std::move(block));
		    
		    }
		    

		    return nullptr;
		}


    };
    	




} /// namespace toy



#endif // TOY_PARSER_H
