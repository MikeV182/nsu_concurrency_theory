#include <cstdlib>
#include <omp.h>
#include <time.h>
#include <stdio.h>
#include <inttypes.h>
#include <vector>
#include <cmath>

#define NUM_THREADS 40
#define PI 3.14159265358979323846

double func(double x) {
    return exp(-x * x);
}

double integrate(double a, double b, int n) {
    double h = (b - a) / n;
    double sum = 0.0;

    for (int i = 0; i < n; i++)
        sum += func(a + h * (i + 0.5));
    
    sum *= h;
    return sum;
}

double integrate_omp(double (*func)(double), double a, double b, int n) {
    double h = (b - a) / n;
    double sum = 0.0;

    #pragma omp parallel num_threads(NUM_THREADS)
    {
        int nthreads = omp_get_num_threads();
        int threadid = omp_get_thread_num();
        int items_per_thread = n / nthreads;
        int lb = threadid * items_per_thread;
        int ub = (threadid == nthreads - 1) ? (n - 1) : (lb + items_per_thread - 1);
        double sumloc = 0.0; // local sum variable for each thread

        for (int i = lb; i <= ub; i++)
            sumloc += func(a + h * (i + 0.5));
        
        // without "omp critical" or "omp atomic" there will be so called "race conditions" when many of threads
        // are reading and writing to the same variable "sum" so that final result depends on 
        // order in which threads are writing to "sum". This section called "critical section"
        #pragma omp atomic
        sum += sumloc; 
    }
    sum *= h;
    return sum;
}

double run_serial(double a, double b, int nsteps) {
    double t = omp_get_wtime();
    double res = integrate(a, b, nsteps);
    t = omp_get_wtime() - t;
    printf("Result (serial): %.12f; error %.12f\n", res, fabs(res - sqrt(PI)));
    return t;
}

double run_parallel(double a, double b, int nsteps) {
    double t = omp_get_wtime();
    double res = integrate_omp(func, a, b, nsteps);
    t = omp_get_wtime() - t;
    printf("Result (parallel): %.12f; error %.12f\n", res, fabs(res - sqrt(PI)));
    return t;
}

int main(int argc, char **argv) {
    const double a = -4.0; /* [a, b] */
    const double b = 4.0;
    const int nsteps = 40000000; /* n */

    printf("Integration f(x) on [%.12f, %.12f], nsteps = %d\n", a, b, nsteps);
    double tserial = run_serial(a, b, nsteps);
    double tparallel = run_parallel(a, b, nsteps);

    printf("Execution time (serial): %.6f\n", tserial);
    printf("Execution time (parallel): %.6f\n", tparallel);
    printf("Speedup: %.2f\n", tserial / tparallel);
    return 0;
}