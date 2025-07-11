#include <iostream>
#include <iomanip>
#include <cmath>
#include <chrono>
#include <cstring>
#include <boost/program_options.hpp>
#include <nvtx3/nvToolsExt.h>

namespace po = boost::program_options;

void initialize_grid(double* __restrict__ grid, double* __restrict__ new_grid, size_t N) {
    std::memset(grid, 0, N * N * sizeof(double));
    std::memset(new_grid, 0, N * N * sizeof(double));

    grid[0]           = 10.0;
    grid[N - 1]       = 20.0;
    grid[N * (N - 1)] = 30.0;
    grid[N * N - 1]   = 20.0;

    double tl = grid[0];
    double tr = grid[N - 1];
    double bl = grid[N * (N - 1)];
    double br = grid[N * N - 1];

    for (int i = 1; i < N - 1; ++i) {
        grid[i]               = tl + (tr - tl) * i / (N - 1);           // top
        grid[N * (N - 1) + i] = bl + (br - bl) * i / (N - 1);           // bottom
        grid[N * i]           = tl + (bl - tl) * i / (N - 1);           // left
        grid[N * i + N - 1]   = tr + (br - tr) * i / (N - 1);           // right
    }

    #pragma acc enter data copyin(grid[0:N*N], new_grid[0:N*N]) // copies data from the host to the accelerator when entering the 
                                                                // data region indicated by the directive; however, 
                                                                // it does not copy the data back to the host on exiting the data region.
}

double update_grid(double* __restrict__ grid, double* __restrict__ new_grid, size_t N, bool check_error) {
    double error = 0.0;

    if (check_error) {
        #pragma acc parallel loop collapse(2) reduction(max:error) present(grid, new_grid)  // present(list) - when entering the region, the data must be present
        for (int i = 1; i < N - 1; ++i) {                                                   // in device memory, and the structured reference count is incremented
            for (int j = 1; j < N - 1; ++j) {
                int idx = i * N + j;
                double up    = grid[(i - 1) * N + j];
                double down  = grid[(i + 1) * N + j];
                double left  = grid[i * N + (j - 1)];
                double right = grid[i * N + (j + 1)];
                double center = grid[idx];

                new_grid[idx] = 0.25 * (up + down + left + right);

                error = fmax(error, fabs(new_grid[idx] - center));
            }
        }
    } else {
        #pragma acc parallel loop collapse(2) reduction(max:error) present(grid, new_grid)
        for (int i = 1; i < N - 1; ++i) {
            for (int j = 1; j < N - 1; ++j) {
                int idx = i * N + j;
                double up    = grid[(i - 1) * N + j];
                double down  = grid[(i + 1) * N + j];
                double left  = grid[i * N + (j - 1)];
                double right = grid[i * N + (j + 1)];

                new_grid[idx] = 0.25 * (up + down + left + right);
            }
        }
    }

    return error;
}

void copy_grid(double* __restrict__ grid, const double* __restrict__ new_grid, size_t N) {
    #pragma acc parallel loop collapse(2) present(grid, new_grid)
    for (int i = 1; i < N - 1; ++i) {
        for (int j = 1; j < N - 1; ++j) {
            grid[i * N + j] = new_grid[i * N + j];
        }
    }
}

void deallocate(double* __restrict__ grid, double* __restrict__ new_grid) {
    #pragma acc exit data delete(grid[0:0], new_grid[0:0])  // for data created with "enter data" the "exit data" moves data from device memory
    free(grid);                                             // and deallocates the memory. With "delete" the dynamic reference count is decremented. 
    free(new_grid);                                         // If reference counts are zero, the device memory is deallocated.
}

void print_grid(const double* __restrict__ grid, size_t N) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            std::cout << std::setprecision(4) << grid[i * N + j] << "  ";
        }
        std::cout << '\n';
    }
}

int main(int argc, char* argv[]) {
    int N;
    double accuracy;
    int max_iterations;

    po::options_description desc("Options");
    desc.add_options()
        ("help", "show help")
        ("size", po::value<int>(&N)->default_value(256), "Grid size (NxN)")
        ("accuracy", po::value<double>(&accuracy)->default_value(1e-6), "Convergence threshold")
        ("max_iterations", po::value<int>(&max_iterations)->default_value(1e6), "Max iterations");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
        std::cout << desc << "\n";
        return 0;
    }

    std::cout << "Running GPU simulation...\n";

    double* grid     = (double*)malloc(N * N * sizeof(double));
    double* new_grid = (double*)malloc(N * N * sizeof(double));

    nvtxRangePushA("Initialization");
    initialize_grid(grid, new_grid, N);
    nvtxRangePop();

    double error = accuracy + 1.0;
    int iter = 0;

    auto start = std::chrono::steady_clock::now();

    nvtxRangePushA("Main Loop");
    while (error > accuracy && iter < max_iterations) {
        nvtxRangePushA("Compute");

        if (iter % 1000 == 0) {
            error = update_grid(grid, new_grid, N, true);
        } else {
            update_grid(grid, new_grid, N, false);
        }

        nvtxRangePop();

        nvtxRangePushA("Copy");
        copy_grid(grid, new_grid, N);
        nvtxRangePop();

        iter++;
    }
    nvtxRangePop();

    auto end = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(end - start).count();

    std::cout << "Time:       " << elapsed << " sec\n";
    std::cout << "Iterations: " << iter << "\n";
    std::cout << "Final error:" << error << "\n";

    // #pragma acc update self(grid[0:N*N])
    // print_grid(grid, N);

    deallocate(grid, new_grid);

    return 0;
}
