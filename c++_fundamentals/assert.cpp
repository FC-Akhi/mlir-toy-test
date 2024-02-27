#include <stdio.h>


/// When debugging is completed uncomment below line. To deactivate assert
/// #define NDEBUG

#include <cassert>

void display_number(int *p) {

	/// For development purpose. It will be disabled after development will be completed. 
	/// If this assert is false then it function will abort from this point. It will not move forward anymore
	assert(p != NULL);

	printf("%d\n", *p);


}



int main() {

	/// For checking assert for true condition
	int a = 10;

	display_number(&a);


	/// For checking assert for false condition
	int *k = NULL;

	display_number(k);

	return 0;

}
