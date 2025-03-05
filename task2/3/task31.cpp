#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <time.h>
#include <omp.h>

using namespace std;

const double EPSILON = 1e-5;
const double TAU = 0.000001;

using Vector = vector<double>;
using Matrix = vector<Vector>;

double norm(const Vector &v, int n_threads)
{
    double sum = 0.0;

    // директива reduction позволяет собрать в главном потоке результаты вычислений 
    // частных сумм (разностей, произведений, ...) из параллельных потоков последующего
    // параллельного структурного блока
    #pragma omp parallel for reduction(+:sum) num_threads(n_threads)
    for (size_t i = 0; i < v.size(); ++i)
    {
        sum += v[i] * v[i];
    }
    return sqrt(sum);
}

Vector simpleIterationMethod(const Matrix &A, const Vector &b, int n_threads)
{
    int n = A.size();
    Vector x(n, 0.0);
    Vector Ax(n);

    while (true)
    {
        Vector result(n, 0.0);
        #pragma omp parallel for num_threads(n_threads)
        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j < n; ++j)
            {
                result[i] += A[i][j] * x[j];
            }
        }
        Ax = result;

        Vector r(n);
        #pragma omp parallel for num_threads(n_threads)
        for (int i = 0; i < n; ++i)
        {
            r[i] = Ax[i] - b[i];
        }

        if (norm(r, n_threads) / norm(b, n_threads) < EPSILON)
        {
            break;
        }

        #pragma omp parallel for num_threads(n_threads)
        for (int i = 0; i < n; ++i)
        {
            x[i] -= TAU * r[i];
        }
    }

    return x;
}

int main()
{
    int N;
    cout << "Enter the number of equations (N = 5000 as an example): ";
    cin >> N;

    if (N <= 0)
    {
        cout << "Error: N must be greater than 0" << endl;
        return 1;
    }

    std::ofstream out;
    out.open("Out_task31.txt");

    for (int n_threads = 2; n_threads <= min(80, N); n_threads++)
    {
        Matrix A(N, Vector(N, 1.0));
        Vector b(N, N + 1);

        #pragma omp parallel for num_threads(n_threads)
        for (int i = 0; i < N; ++i)
        {
            A[i][i] = 2.0;
        }

        double t = omp_get_wtime();
        Vector solution = simpleIterationMethod(A, b, n_threads);
        t = omp_get_wtime() - t;

        printf("n_threads: %d Execution time (parallel): %.6f\n", n_threads, t);
        out << n_threads << "   " << t << "   " << 85.948490 / t << "\n";
    }

    out.close();
    cout << "File has been written" << std::endl;
    return 0;
}
