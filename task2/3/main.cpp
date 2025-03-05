#include <iostream>
#include <vector>
#include <cmath>
#include <time.h>
#include <omp.h>

using namespace std;

const double EPSILON = 1e-5;
const double TAU = 0.000001; // 0.000001 and 10 000 for Execution time (serial): 173.691134

using Vector = vector<double>;
using Matrix = vector<Vector>;

double norm(const Vector &v)
{
    double sum = 0.0;
    for (double val : v)
    {
        sum += val * val;
    }
    return sqrt(sum);
}

Vector multiply(const Matrix &A, const Vector &x)
{
    int n = A.size();
    Vector result(n, 0.0);
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            result[i] += A[i][j] * x[j];
        }
    }
    return result;
}

Vector simpleIterationMethod(const Matrix &A, const Vector &b)
{
    int n = A.size();
    Vector x(n, 0.0);
    Vector Ax(n);
    while (true)
    {
        Ax = multiply(A, x);
        Vector r(n);
        for (int i = 0; i < n; ++i)
        {
            r[i] = Ax[i] - b[i];
        }

        if (norm(r) / norm(b) < EPSILON)
        {
            break;
        }

        for (int i = 0; i < n; ++i)
        {
            x[i] -= TAU * r[i];
        }
    }

    return x;
}

void initializeSystem(Matrix &A, Vector &b, int N)
{
    A.assign(N, Vector(N, 1.0));
    for (int i = 0; i < N; ++i)
    {
        A[i][i] = 2.0;
    }
    b.assign(N, N + 1);
}

void printVector(const Vector &v)
{
    for (double val : v)
    {
        cout << val << " ";
    }
    cout << endl;
}

int main()
{
    int N;
    cout << "Enter the number of equations (N): ";
    cin >> N;

    Matrix A;
    Vector b;

    initializeSystem(A, b, N);

    double t = omp_get_wtime();
    Vector solution = simpleIterationMethod(A, b);
    t = omp_get_wtime() - t;

    printf("Execution time (serial): %.6f\n", t);

    return 0;
}