#include <iostream>
#include <memory>



int main() {

	std::shared_ptr<int []> ptr1(new int [10]);

	auto ptr2 = ptr1;

	printf("Reference count: %ld\n", ptr1.use_count());


	/// Example to understand the Lexer class constructor input
	std::shared_ptr<int> ptr3 = std::make_shared<int> (10);


	printf("Reference count: %ld\n", ptr3.use_count());


	return 0;

}