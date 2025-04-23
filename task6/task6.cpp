#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <boost/program_options.hpp>

using namespace std;
namespace po = boost::program_options;

double interpolate(double a, double b, int i, int n) {
    return a + (b - a) * i / static_cast<double>(n - 1);
}

int main(int argc, char** argv) {
    int N = 128, max_iter = 1000000;
    double eps = 1e-6;

    po::options_description desc("Allowed options");
    desc.add_options()
        ("help", "produce help message")
        ("size", po::value<int>(&N)->default_value(128), "grid size (NxN)")
        ("eps", po::value<double>(&eps)->default_value(1e-6), "convergence threshold")
        ("max_iter", po::value<int>(&max_iter)->default_value(1e6), "maximum iterations");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);
    if (vm.count("help")) {
        cout << desc << "\n";
        return 1;
    }

    vector<vector<double>> u(N, vector<double>(N, 0.0));
    vector<vector<double>> u_new = u;

    // Углы
    double tl = 10, tr = 20, br = 30, bl = 20;

    // Верх и низ
    for (int j = 0; j < N; ++j) {
        u[0][j] = interpolate(tl, tr, j, N);
        u[N - 1][j] = interpolate(bl, br, j, N);
    }

    // Левый и правый
    for (int i = 0; i < N; ++i) {
        u[i][0] = interpolate(tl, bl, i, N);
        u[i][N - 1] = interpolate(tr, br, i, N);
    }

    int iter = 0;
    double max_diff;
    do {
        max_diff = 0.0;

        for (int i = 1; i < N - 1; ++i) {
            for (int j = 1; j < N - 1; ++j) {
                u_new[i][j] = 0.25 * (u[i - 1][j] + u[i + 1][j] + u[i][j - 1] + u[i][j + 1]);
                max_diff = max(max_diff, abs(u_new[i][j] - u[i][j]));
            }
        }

        u.swap(u_new);
        ++iter;
    } while (max_diff > eps && iter < max_iter);

    cout << "Iterations: " << iter << "\n";
    cout << "Final error: " << max_diff << "\n";

    // Save result
    ofstream fout("result.dat", ios::binary);
    for (int i = 0; i < N; ++i)
        fout.write(reinterpret_cast<char*>(u[i].data()), N * sizeof(double));

    // Print matrix for small N
    if (N == 10 || N == 13) {
        for (const auto& row : u) {
            for (double val : row)
                printf("%6.2f ", val);
            printf("\n");
        }
    }

    return 0;
}
