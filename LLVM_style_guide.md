# Best practices from LLVM style guide

## enum, classes, structs, typedefs

	1. Should be noun
	2. Start with capital letters


## enum and enumerator

### enum declaration

	1. Use suffix 'Kind' in the name. Exp: TokenKind 

### enumerator declaration

	1. Start with capital letters
	2. Unless the enumerators are defined within their own small namespace or inside a class, they should have a prefix corresponding to the enum declaration name. Example: For enum TokenKind { ... };, use TK_Argument, TK_BasicBlock, etc.
	3. Exception case: enum for convenience constant does need to follow 2nd rule on prefix
	For example:
		enum {
	  		MaxSize = 42,
	  		Density = 12
		};
	



## Variables
	
	1. Should be nouns (as they represent state). 
	2. The name should be camel case
	3. Start with an upper case letter (e.g. Leader or Boats).


## Header Guard

	1. The header file’s guard should be the all-caps path that a user of this header would #include, using ‘_’ instead of path separator and extension marker. 2. For example, the header file llvm/include/llvm/Analysis/Utils/Local.h would be #include-ed as #include "llvm/Analysis/Utils/Local.h", so its guard is LLVM_ANALYSIS_UTILS_LOCAL_H.

## Function
	1. Function names should be verb phrases (as they represent actions), and command-like function should be imperative. 
	2. The name should be camel case
	3. Start with a lower case letter (e.g. openFile() or isFoo()).
