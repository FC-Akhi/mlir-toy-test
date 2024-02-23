//===- Lexer.h - Lexer for the Toy language -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a simple Lexer for the Toy language.
//
//===----------------------------------------------------------------------===//

#ifndef TOY_LEXER_H
#define TOY_LEXER_H


#include "llvm/ADT/StringRef.h"

#include <memory>
#include <string>



#include <iostream>
#include <sstream>

#define NDEBUG

#include <cassert>



namespace toy {


	struct Location {

		std::shared_ptr<std::string> file; /// filename
		int line;						   /// line number								 
		int col;						   /// column number

	};



	enum Token : int {

	    TK_semicolon = ';',
	    TK_parenthese_open = '(',
	    TK_parenthese_close = ')',
	    TK_cbracket_open = '{',
	    TK_cbracket_close = '}',
	    TK_sbracket_open = '[',
	    TK_sbracket_close = ']',

	    TK_eof = -1,

	    /// commands
	    TK_def = -2,
	    TK_return = -3,
	    TK_var = -4,

	    /// primary
	    TK_identifier = -5,
	    TK_number = -6,

	};




	class Lexer {


	public:

		/// Constructor
		Lexer(std::string filename) : 
			 lastLocation({std::make_shared<std::string>(std::move(filename)), 0, 0}) {}


	    /// Destructor 
		virtual ~Lexer() = default;


		Token getCurToken() { 
			
			return curTok; 
		}

		Token getNextToken() {

			return curTok = getTok();
		
		}


		void consume(Token tok) {

			assert(tok == curTok && "consume Token mismatch expectation");

			getNextToken();

		}


		llvm::StringRef getIdentifier() {

			/// check explanation of assert in file.cpp; line no:
			assert(curTok == TK_identifier);

			

			return identifierStr;

		}


		double getValue() {

			/// check explanation of assert in file.cpp; line no:
			assert(curTok == TK_number);

			
			
				
			return numVal;

		}


		Location getLastLocation() {

			return lastLocation;

		}


		int getLine() {

			return curLineNum;

		}


		int getCol() {

			return curCol;

		}




		std::string identifierStr;

		double numVal = 0;

		Token lastChar = Token(' ');

		Token curTok = TK_eof;
		

		virtual llvm::StringRef readNextLine() = 0;

		llvm::StringRef curLineBuffer = "\n";


		/// Below three are for file
		Location lastLocation;
		int curLineNum = 0;
		int curCol = 0;




		int getNextChar() {

			if (curLineBuffer.empty()) {

				return EOF;

			}

			++curCol;

			/// Gives '\n' to nextChar
			/// In C++ (and many other programming languages), 
			/// certain characters that cannot be directly typed or 
			/// represented in a string (like a newline, tab, etc.) 
			/// are represented using escape sequences. 
			auto nextChar = curLineBuffer.front();


			/// empty the curLineBuffer completely
			curLineBuffer = curLineBuffer.drop_front();

			std::cout << curLineBuffer.str() << "\n";

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

		LexerBuffer(std::string filename, const char *begin, const char *end)
			: Lexer(std::move(filename)), current(begin), end(end) {}


	private:

		const char *current, *end;

		llvm::StringRef readNextLine() override {

			// printf("Inside LexerBuffer\n");

			// printf("Current: %c\n", *current);

			// printf("End: %c\n", *(end - 1));

			auto *begin = current;

			// printf("begin: %c\n", *begin);

			while (current <= end && *current && *current != '\n')

				++current;

			if (current <= end && *current)

				++current;

			// printf("Current: %c\n", *current);

			llvm::StringRef result{
				
				begin, static_cast<size_t>(current - begin)
			};

			return result;

		}

		


	};

}


int main() {
    
    std::string testData = "First line.\nSecond line.\nThird line.";
    
    // Create an instance of LexerBuffer with testData

    // printf("Begin: %c\n", *testData.begin());

    // printf("End: %c\n", *(testData.end() - 1));

    toy::LexerBuffer lexer("testData", &*testData.begin(), &*testData.end());

    int ch;
    
    ch = lexer.getNextChar();



    while (ch != EOF) {
    
        // std::cout << static_cast<char>(ch); // Print each character
        // printf("ch: %d\n", ch);
    	
        if (ch == '\n') {

        	std::cout << std::endl;
        }

        ch = lexer.getNextChar();
    }

    return 0;
}


#endif // TOY_LEXER_H
