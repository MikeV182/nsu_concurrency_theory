#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h>

using namespace std;


double *gauss_scheduled(double **a, double *y, int n, int num_threads, const string schedule_type, int chunk_size)
{
    double *x = new double[n];
    const double eps = 1e-5;

    #pragma omp parallel num_threads(num_threads)
    {
        for (int k = 0; k < n; k++)
        {
            int index = k;
            double max = abs(a[k][k]);

            #pragma omp for schedule(static) nowait
            for (int i = k + 1; i < n; i++)
            {
                if (abs(a[i][k]) > max)
                {
                    max = abs(a[i][k]);
                    index = i;
                }
            }

            #pragma omp single
            {
                if (max < eps)
                {
                    cout << "Решение невозможно из-за нулевого столбца " << index << " матрицы A" << endl;
                    delete[] x;
                    x = nullptr;
                }
                else
                {
                    swap(a[k], a[index]);
                    swap(y[k], y[index]);
                }
            }

            #pragma omp barrier
            if (x == nullptr) continue;

            if (schedule_type == "static") {
                #pragma omp for schedule(static, chunk_size)
                for (int i = k + 1; i < n; i++)
                {
                    double factor = a[i][k] / a[k][k];
                    for (int j = k; j < n; j++)
                        a[i][j] -= factor * a[k][j];
                    y[i] -= factor * y[k];
                }
            } else if (schedule_type == "dynamic") {
                #pragma omp for schedule(dynamic, chunk_size)
                for (int i = k + 1; i < n; i++)
                {
                    double factor = a[i][k] / a[k][k];
                    for (int j = k; j < n; j++)
                        a[i][j] -= factor * a[k][j];
                    y[i] -= factor * y[k];
                }
            } else if (schedule_type == "guided") {
                #pragma omp for schedule(guided, chunk_size)
                for (int i = k + 1; i < n; i++)
                {
                    double factor = a[i][k] / a[k][k];
                    for (int j = k; j < n; j++)
                        a[i][j] -= factor * a[k][j];
                    y[i] -= factor * y[k];
                }
            } else {
                #pragma omp for schedule(auto)
                for (int i = k + 1; i < n; i++)
                {
                    double factor = a[i][k] / a[k][k];
                    for (int j = k; j < n; j++)
                        a[i][j] -= factor * a[k][j];
                    y[i] -= factor * y[k];
                }
            }

            #pragma omp barrier
        }

        #pragma omp single
        {
            for (int k = n - 1; k >= 0; k--)
            {
                x[k] = y[k];
                for (int j = k + 1; j < n; j++)
                    x[k] -= a[k][j] * x[j];
                x[k] /= a[k][k];
            }
        }
    }

    return x;
}


int main()
{
    int n = 2500;
    int num_threads = 8;
    
    double **a = new double *[n];
    double *y = new double[n];

    #pragma omp parallel for
    for (int i = 0; i < n; i++)
    {
        a[i] = new double[n];
        for (int j = 0; j < n; j++)
            a[i][j] = (i == j) ? 2.0 : 1.0;
    }
    
    #pragma omp parallel for
    for (int i = 0; i < n; i++)
        y[i] = n + 1;


    vector<string> schedules = {"static", "dynamic", "guided", "auto"};
    vector<int> chunk_sizes = {1, 10, 100, 1000};
    
    cout << "Стратегия\tЧанк\tВремя (сек)" << endl;

    for (const auto& schedule : schedules)
    {
        for (const auto& chunk : chunk_sizes)
        {
            double t = omp_get_wtime();
            double *x = gauss_scheduled(a, y, n, num_threads, schedule, chunk);
            double exec_time = omp_get_wtime() - t;

            cout << schedule << "\t" << chunk << "\t" << exec_time << endl;

            if (x) delete[] x;
        }
    }

    for (int i = 0; i < n; i++)
        delete[] a[i];
    delete[] a;
    delete[] y;

    return 0;
}
