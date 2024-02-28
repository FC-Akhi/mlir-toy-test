///==- Lexer.h - Lexer for the Toy language -------------------------------===///
///
/// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
/// See https://llvm.org/LICENSE.txt for license information.
/// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
///
///===----------------------------------------------------------------------===///
///
/// This file implements a simple Lexer for the Toy language.
///
///===----------------------------------------------------------------------===///

#ifndef TOY_LEXER_H
#define TOY_LEXER_H



/// Include LLVM's StringRef for efficient string handling
#include "llvm/ADT/StringRef.h"



/// Include headers for memory management, string manipulation, and IO operations
#include <memory>
#include <string>
#include <iostream>
#include <sstream>



/// Define NDEBUG to disable debug features in release builds
#define NDEBUG

/// Include assert for debug checks
#include <cassert>



/// Define the 'toy' namespace to encapsulate the lexer components
/// Its same as std namespace we used normally. 
/// But toy is user defined one
namespace toy {



	/// Structure to represent a source code location (file, line, and column)
	struct Location {

		std::shared_ptr<std::string> file; /// filename
		int line;						   /// line number								 
		int col;						   /// column number

	};



	/// Enum to represent different types of tokens in the language
	enum Token : int {

	    /// Single character tokens represented by their ASCII value
        TK_semicolon = ';',
        TK_parenthese_open = '(',
        TK_parenthese_close = ')',
        TK_cbracket_open = '{',
        TK_cbracket_close = '}',
        TK_sbracket_open = '[',
        TK_sbracket_close = ']',

        /// Special tokens represented by negative values
        TK_eof = -1,           /// End of file
        
        /// Commands
        TK_def = -2,           /// 'def' keyword
        TK_return = -3,        /// 'return' keyword
        TK_var = -4,           /// 'var' keyword
        
        /// Primary
        TK_identifier = -5,    /// Identifier
        TK_number = -6,        /// Numeric literal

	};



	/// Base class for the lexer
	class Lexer {


	public:


		/// Constructor initializes the lexer with a filename
		Lexer(std::string filename) : 
			 lastLocation({std::make_shared<std::string>(std::move(filename)), 0, 0}) {}


	    /// Virtual destructor for safe polymorphic deletion 
		virtual ~Lexer() = default;


		/// Returns the current token without consuming it
		Token getCurToken() { 
			
			return curTok; 
		}


		/// Advances to the next token and returns it
		Token getNextToken() {

			return curTok = getTok();
		
		}


		/// Consumes the expected token or asserts mismatch
		void consume(Token tok) {

			assert(tok == curTok && "consume Token mismatch expectation");

			getNextToken();

		}


		/// Returns the identifier string if the current token is an identifier
		llvm::StringRef getIdentifier() {

			/// check explanation of assert in file.cpp; line no:
			assert(curTok == TK_identifier);

			
			return identifierStr;

		}


		/// Returns the numeric value if the current token is a number
		double getValue() {

			/// check explanation of assert in file.cpp; line no:
			assert(curTok == TK_number);
				

			return numVal;

		}


		/// Returns the location of the last token
		Location getLastLocation() {

			return lastLocation;

		}


		/// Gets the line number of the current token
		int getLine() {

			return curLineNum;

		}


		/// Gets the column number of the current token
		int getCol() {

			return curCol;

		}


	// private:


		std::string identifierStr;  // Stores the last parsed identifier
        double numVal = 0;          // Stores the last parsed number value
        
        Token lastChar = Token(' ');// Stores the last character read
        Token curTok = TK_eof;      // Stores the current token
        
        virtual llvm::StringRef readNextLine() = 0; // Pure virtual function to read the next line
        llvm::StringRef curLineBuffer = "\n"; // Current line buffer

        // Source code location tracking
        Location lastLocation;
        int curLineNum = 0;
        int curCol = 0;


        /// Reads the next character from the input stream
        /// Its main job is to read character one by one and returns that character
		int getNextChar() {

			if (curLineBuffer.empty()) {

				return EOF;

			}

			++curCol;

			/// At the starting or on the first iteration, It gives '\n' to nextChar
			/// In C++ (and many other programming languages), 
			/// certain characters that cannot be directly typed or 
			/// represented in a string (like a newline, tab, etc.) 
			/// are represented using escape sequences. So we will not see '\n'
			/// if we print.
			/// In next ietrations we will start getting real characters, if this getNextChar()
			/// method is called again
			/// To check how this part works uncomment run the code by uncommenting main()
			auto nextChar = curLineBuffer.front();


			/// empty the curLineBuffer completely
			curLineBuffer = curLineBuffer.drop_front();


			/// If curLineBuffer is completely empty
			if (curLineBuffer.empty()) {

				/// Start reading next line
				/// Even though getNextChar() is defined in the base class Lexer, 
				/// when it calls readNextLine(), the runtime system checks the 
				/// actual type of the object (LexerBuffer in this case) and calls 
				/// the overridden version of readNextLine() in LexerBuffer.
				curLineBuffer = readNextLine();


			}


			if (nextChar == '\n') {

				++curLineNum;

				curCol = 0;

			}


			return nextChar;
		
		}


		/// Tokenizes the input stream
		Token getTok() {


			// Skip any whitespace
			while (isspace(lastChar))

				lastChar = Token(getNextChar());


			lastLocation.line = curLineNum;
			lastLocation.col = curCol;


			// Number: [0-9]+
			if (isdigit(lastChar) || lastChar == '.') {

				std::string numStr;

				do {

					numStr += lastChar;
					lastChar = Token(getNextChar());
				
				} while (isdigit(lastChar) || lastChar == '.');
			

				numVal = strtod(numStr.c_str(), nullptr);

				return TK_number;
			
			}


			// Identifier: [a-zA-Z][a-zA-Z0-9_]*
			if (isalpha(lastChar)) {

				identifierStr = (char)lastChar;


				while (isalnum((lastChar = Token(getNextChar()))) || lastChar == '_')

					identifierStr += (char)lastChar;

				if (identifierStr == "return")

					return TK_return;

				if (identifierStr == "def")

					return TK_def;

				if (identifierStr == "var")

					return TK_var;

				return TK_identifier;


			}


			//  Check for the comment
			if (lastChar == '#') {

				// Comment until end of line
				do {

					lastChar = Token(getNextChar());

				} while (lastChar != EOF && lastChar != '\n' && lastChar != '\r');


				if (lastChar != EOF)

					return getTok();

			}


			// Check fo rthe end of file. Don't eat the EOF
			if (lastChar == EOF)

				return TK_eof;


			// Otherwise, just return the character as its ascii value. 
			Token thisChar = Token(lastChar);
			lastChar = Token(getNextChar());
			return thisChar;

		
		}


	};


	class LexerBuffer final : public Lexer {


	public:


		/// Constructor initializes the LexerBuffer with a filename and start and end of the string or file
		LexerBuffer(std::string filename, const char *begin, const char *end)
			: Lexer(std::move(filename)), current(begin), end(end) {}


	private:


		/// Declare current and end of the line
		const char *current, *end;


		/// Read a single line out of file and returns that line to getNextChar() in Base Lexer class
		llvm::StringRef readNextLine() override {

			
			auto *begin = current;


			while (current <= end && *current && *current != '\n')

				++current;

			if (current <= end && *current)

				++current;
			

			llvm::StringRef result{
				
				begin, static_cast<size_t>(current - begin)
			};


			return result;


		}


	};

}



#endif // TOY_LEXER_H
