#include <iostream>
#include <cmath>

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
        for (int i = k + 1; i < n; i++)
        {
            if (abs(a[i][k]) > max)
            {
                max = abs(a[i][k]);
                index = i;
            }
        }
        if (max < eps)
        {
            cout << "Решение невозможно из-за нулевого столбца " << index << " матрицы A" << endl;
            return nullptr;
        }
        swap(a[k], a[index]);
        swap(y[k], y[index]);
        
        for (int i = k; i < n; i++)
        {
            double temp = a[i][k];
            if (abs(temp) < eps) continue;
            for (int j = k; j < n; j++) a[i][j] /= temp;
            y[i] /= temp;
            if (i == k) continue;
            for (int j = 0; j < n; j++) a[i][j] -= a[k][j];
            y[i] -= y[k];
        }
    }
    
    for (int k = n - 1; k >= 0; k--)
    {
        x[k] = y[k];
        for (int i = 0; i < k; i++) y[i] -= a[i][k] * x[k];
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
    for (int i = 0; i < n; i++)
    {
        a[i] = new double[n];
        for (int j = 0; j < n; j++) a[i][j] = (i == j) ? 2.0 : 1.0;
    }
    for (int i = 0; i < n; i++) y[i] = n + 1;
    
    sysout(a, y, n);
    double *x = gauss(a, y, n);
    
    if (x)
    {
        for (int i = 0; i < n; i++)
            cout << "x[" << i << "] = " << x[i] << endl;
        delete[] x;
    }
    
    for (int i = 0; i < n; i++) delete[] a[i];
    delete[] a;
    delete[] y;
    
    return 0;
}
