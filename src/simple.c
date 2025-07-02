#include <stdio.h>

int main(){
    #pragma omp target
    {
        printf("Hello from GPU");

    }
    return 0;
}