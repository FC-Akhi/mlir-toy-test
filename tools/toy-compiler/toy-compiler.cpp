//===- toyc.cpp - The Toy Compiler ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the entry point for the Toy compiler.
//
//===----------------------------------------------------------------------===//



//  Calling "Lexer.h".
#include "include/toy-analysis-parser/Lexer.h"



#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include <memory>
#include <string>
#include <system_error>



using namespace toy;
namespace cl = llvm::cl;



/// our compiler take an input filename (i.e. filename.toy).
/// Check how below code snippet works from how_llvm_utilities_work.md
static cl::opt<std::string> inputFilename(

    "input",
    cl::desc("<input toy file>"),
    cl::init("-"),
    cl::value_desc("filename")


);





/// Driver or Entry point for checking Lexer
int main(int argc, char **argv) {


    // Parse the command line arguments & flags
    cl::ParseCommandLineOptions(argc, argv, "toy compiler\n");

    
    auto testData = parseInputFile(inputFilename);


    /// Input string for testing
    std::string testData = "def foo(x) {\n return x + 1;\n}\nvar y = foo(5);";
    

    /// Instantiate object of LexerBuffer
    toy::LexerBuffer lexer("testData", &*testData.begin(), &*testData.end());


    /// Print input string
    std::cout << "Input string" << std::endl;
    std::cout << testData << std::endl;


    ///======================
    /// If you want to test how lexer reads character by character, then uncomment Test1 and comment out Test2
    /// If you want to test how lexer tokenize, then uncomment Test1 and comment out Test1
    ///======================


    /// Test1: Check how getNextChar() working
    int ch;
    ch = lexer.getNextChar();


    while (ch != EOF) {
        
        if (ch == '\n') {

            std::cout << std::endl;
        }

        /// Prints each character
        std::cout <<"ch:"<< static_cast<char>(ch) << std::endl;


        /// Reads each character
        ch = lexer.getNextChar();
    

    }


    /// Test2: Check how getTok() working
    toy::Token token;
    

    // Repeatedly call getNextToken until EOF token is reached
    do {
        token = lexer.getNextToken(); // Get the next token

        // Output the type of token and any associated value
        switch(token) {

            case toy::TK_def:
                
                std::cout << "Line number: " << lexer.lastLocation.line;
                std::cout << " Token: def" << std::endl;

                break;
            
            case toy::TK_return:

                std::cout << "Line number: " << lexer.lastLocation.line;
                std::cout << " Token: return" << std::endl;
                
                break;
            
            case toy::TK_var:

                std::cout << "Line number: " << lexer.lastLocation.line;
                std::cout << " Token: var" << std::endl;
                
                break;
            
            case toy::TK_identifier:

                std::cout << "Line number: " << lexer.lastLocation.line;
                std::cout << " Token: Identifier (" << lexer.getIdentifier().str() << ")" << std::endl;
                
                break;
            
            case toy::TK_number:

                std::cout << "Line number: " << lexer.lastLocation.line;
                std::cout << " Token: Number (" << lexer.getValue() << ")" << std::endl;
                
                break;
            
            case toy::TK_eof:

                std::cout << "Line number: " << lexer.lastLocation.line;
                std::cout << " Token: EOF" << std::endl;
                
                break;
            
            default:
                
                // If the token is a single character (e.g., ';', '{', '}')
                if(token > 0) {
                    
                    std::cout << "Line number: " << lexer.lastLocation.line;
                    std::cout << " Token: " << static_cast<char>(token) << std::endl;
                
                }
                
                break;

        }


    } while(token != toy::TK_eof);


    return 0;


}