#include <iostream>
#include <cmath>
#include <omp.h>

using namespace std;

void sysout(double **a, double *y, int n)
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            cout << a[i][j] << "*x" << j;
            if (j < n - 1)
                cout << " + ";
        }
        cout << " = " << y[i] << endl;
    }
}

double *gauss(double **a, double *y, int n)
{
    double *x = new double[n];
    const double eps = 1e-5;

    for (int k = 0; k < n; k++)
    {
        int index = k;
        double max = abs(a[k][k]);

        #pragma omp parallel
        {
            int local_index = k;
            double local_max = max;

            // После завершения обработки этой секции потоком, последний не будет ждать
            // завершения работы других потоков, а продолжит выполнение программы дальше
            #pragma omp for nowait 
            for (int i = k + 1; i < n; i++)
            {
                if (abs(a[i][k]) > local_max)
                {
                    local_max = abs(a[i][k]);
                    local_index = i;
                }
            }

            #pragma omp critical
            {
                if (local_max > max)
                {
                    max = local_max;
                    index = local_index;
                }
            }
        }

        if (max < eps)
        {
            cout << "Решение невозможно из-за нулевого столбца " << index << " матрицы A" << endl;
            delete[] x;
            return nullptr;
        }

        swap(a[k], a[index]);
        swap(y[k], y[index]);

        #pragma omp parallel for
        for (int i = k + 1; i < n; i++)
        {
            double factor = a[i][k] / a[k][k];
            for (int j = k; j < n; j++)
                a[i][j] -= factor * a[k][j];
            y[i] -= factor * y[k];
        }
    }

    // Обратный ход (не распараллеливается из-за зависимости между итерациями)
    for (int k = n - 1; k >= 0; k--)
    {
        x[k] = y[k];
        for (int j = k + 1; j < n; j++)
            x[k] -= a[k][j] * x[j];
        x[k] /= a[k][k];
    }
    
    return x;
}

int main()
{
    int n;
    cout << "Введите количество уравнений: ";
    cin >> n;

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

    sysout(a, y, n);
    double *x = gauss(a, y, n);

    if (x)
    {
        for (int i = 0; i < n; i++)
            cout << "x[" << i << "] = " << x[i] << endl;
        delete[] x;
    }

    #pragma omp parallel for
    for (int i = 0; i < n; i++)
        delete[] a[i];

    delete[] a;
    delete[] y;

    return 0;
}
