#include <iostream>
#include <memory>



int main() {


	/// Shared pointer declaration for an array
	std::shared_ptr<int []> ptr1(new int [10]);

	/// declaring and initializing another new shared pointer
	auto ptr2 = ptr1;


	/// For counting how many pointers are pointing to a same object
	printf("Reference count: %ld\n", ptr1.use_count());


	/// Example to understand the Lexer class constructor input
	std::shared_ptr<int> ptr3 = std::make_shared<int> (10);


	/// For counting how many pointers are pointing to a same object
	printf("Reference count: %ld\n", ptr3.use_count());


	return 0;

}