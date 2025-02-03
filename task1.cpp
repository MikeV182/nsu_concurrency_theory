#include <iostream>
#include <cmath>
#include <vector>

#define ARRAY_SIZE 10000000
// acos(0.0) will return the value for Pi/2. 
// To get the value of Pi we need to multiply it on 2:
#define PI (2*acos(0.0))
#define STEP_SIZE (2 * PI / ARRAY_SIZE)

#if build == FLOAT
    #define ARRAY_TYPE float
#elif build == DOUBLE
    #define ARRAY_TYPE double
#endif

int main() {
    ARRAY_TYPE sum = 0;
    std::vector<ARRAY_TYPE> array(ARRAY_SIZE);

    for (int i = 0; i < ARRAY_SIZE; i++) {
        array[i] = sin(i * STEP_SIZE);
        sum += array[i];
    }

    std::cout << "Sum: " << sum << std::endl;
    return 0;
}
