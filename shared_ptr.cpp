#include <iostream>
#include <memory>

// #define NDEBUG

#include <cassert>

void display_number(int *p) {

	/// For development purpose. It will be disabled after development will be completed. 
	/// If this assert is false then it function will abort from this point. It will not move forward anymore
	assert(p != NULL);

	printf("%d\n", *p);


}



int main() {

	std::shared_ptr<int []> ptr1(new int [10]);

	auto ptr2 = ptr1;

	printf("Reference count: %ld\n", ptr1.use_count());


	/// Example to understand the Lexer class constructor input
	std::shared_ptr<int> ptr3 = std::make_shared<int> (10);


	printf("Reference count: %ld\n", ptr3.use_count());


	int a = 10;

	display_number(&a);

	int *k = NULL;

	display_number(k);



	


	return 0;

}