#include <cstdlib>
#include <omp.h>
#include <time.h>
#include <stdio.h>
#include <inttypes.h>
#include <vector>

#include "pbPlots.hpp"
#include "supportLib.hpp"

#define NUM_THREADS 8
#define MATRIX_SIZE 20000


/* matrix_vector_product_omp: Compute matrix-vector product c[m] = a[m][n] * b[n] */
void matrix_vector_product_omp(double *a, double *b, double *c, const int m, const int n) {
    #pragma omp parallel num_threads(NUM_THREADS)
    {
        int nthreads = omp_get_num_threads();
        int threadid = omp_get_thread_num();
        int items_per_thread = m / nthreads;
        int lb = threadid * items_per_thread;
        int ub = (threadid == nthreads - 1) ? (m - 1) : (lb + items_per_thread - 1);
        for (int i = lb; i <= ub; i++) {
            c[i] = 0.0;
            for (int j = 0; j < n; j++)
                c[i] += a[i * n + j] * b[j];
        }
    }
}

/*
* matrix_vector_product: Compute matrix-vector product c[m] = a[m][n] * b[n]
*/
void matrix_vector_product(double *a, double *b, double *c, int m, int n) {
    for (int i = 0; i < m; i++) {
        c[i] = 0.0;
        for (int j = 0; j < n; j++)
            c[i] += a[i * n + j] * b[j];
    }
}

void run_parallel(const int m, const int n) {
    double *a, *b, *c;
    a = (double*)malloc(sizeof(double) * m * n);
    b = (double*)malloc(sizeof(double) * n);
    c = (double*)malloc(sizeof(double) * m);

    double t1 = omp_get_wtime();
    #pragma omp parallel num_threads(NUM_THREADS)
    {
        int nthreads = omp_get_num_threads();
        int threadid = omp_get_thread_num();
        int items_per_thread = m / nthreads;
        int lb = threadid * items_per_thread;
        int ub = (threadid == nthreads - 1) ? (m - 1) : (lb + items_per_thread - 1);
        for (int i = lb; i <= ub; i++) {
            for (int j = 0; j < n; j++)
                a[i * n + j] = i + j;
            c[i] = 0.0;
        }
        for (int j = 0; j < n; j++)
            b[j] = j;
    }
    printf("Elapsed allocation time (parallel): %.2f sec.\n", omp_get_wtime()-t1);
    
    double t = omp_get_wtime();
    matrix_vector_product_omp(a, b, c, m, n);
    printf("Elapsed time (parallel): %.2f sec.\n", omp_get_wtime()-t);

    free(a);
    free(b);
    free(c);
}

void run_serial(const int m, const int n) {
    double *a, *b, *c;
    a = (double*)malloc(sizeof(double) * m * n);
    b = (double*)malloc(sizeof(double) * n);
    c = (double*)malloc(sizeof(double) * m);

    double t1 = omp_get_wtime();
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++)
            a[i * n + j] = i + j;
    }
    for (int j = 0; j < n; j++)
        b[j] = j;
    printf("Elapsed allocation time (serial): %.2f sec.\n", omp_get_wtime()-t1);

    double t = omp_get_wtime();
    matrix_vector_product(a, b, c, m, n);
    printf("Elapsed time (serial): %.2f sec.\n", omp_get_wtime()-t);

    free(a);
    free(b);
    free(c);
}

void render_graph(std::vector<double> &x, std::vector<double> &y) {
    RGBABitmapImageReference *imageRef = CreateRGBABitmapImageReference();
    StringReference *errorMessage = CreateStringReferenceLengthValue(0, L' ');

    bool result = DrawScatterPlot(imageRef, 600, 400, &x, &y, errorMessage);

    if (result) {
        std::vector<double> *pngData = ConvertToPNG(imageRef->image);
        WriteToFile(pngData, "./plot.png");
        DeleteImage(imageRef->image);
    } else {
        std::cerr << "Error: ";
		for(wchar_t c : *errorMessage->string){
			std::wcerr << c;
		}
		std::cerr << std::endl;
    }

    FreeAllocations();
}

int main(int argc, char **argv) {
    const int m = MATRIX_SIZE, n = MATRIX_SIZE;
    
    printf("Matrix-vector product (c[m] = a[m, n] * b[n]; m = %d, n = %d)\n", m, n);
    printf("Memory used: %" PRIu64 " MiB\n", ((m * n + m + n) * sizeof(double)) >> 20);
    
    run_serial(m, n);
    run_parallel(m, n);
    
    std::vector<double> x{-2, -1, 0, 1, 2};
    std::vector<double> y{2, -1, -2, -1, 2};
    render_graph(x, y);
    
    return 0;
}
