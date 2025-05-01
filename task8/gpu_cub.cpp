#include <iostream>
#include <cmath>
#include <cstdlib>
#include <chrono>

#include <cuda_runtime.h>
#include <cub/block/block_reduce.cuh>
#include <boost/program_options.hpp>

namespace po = boost::program_options;

#define BLOCK_SIZE 256

__device__ double atomicMaxDouble(double* address, double val) { // вызывается из другого GPU-кода и выполняется на GPU
    unsigned long long int* address_as_ull =
        (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(fmax(val, __longlong_as_double(assumed))));
    } while (assumed != old);
    return __longlong_as_double(old);
}

__global__ void updateGridKernel(double* grid, double* gridNew, int size) { // выполняется на GPU, вызывается с хоста
    int i = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int j = blockIdx.x * blockDim.x + threadIdx.x + 1;

    if (i < size - 1 && j < size - 1) {
        gridNew[i * size + j] = 0.25 * (
            grid[(i + 1) * size + j] +
            grid[(i - 1) * size + j] +
            grid[i * size + j + 1] +
            grid[i * size + j - 1]
        );
    }
}

__global__ void copyGridKernel(double* grid, double* gridNew, int size) {
    int i = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int j = blockIdx.x * blockDim.x + threadIdx.x + 1;

    if (i < size - 1 && j < size - 1) {
        grid[i * size + j] = gridNew[i * size + j];
    }
}

__global__ void computeErrorKernel(const double* grid, const double* gridNew, double* error, int size) {
    typedef cub::BlockReduce<double, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage; // BlockReduce использует __shared__ память, чтобы потоки внутри блока могли обмениваться промежуточными результатами.

    double localMax = 0.0;
    for (int idx = threadIdx.x + blockIdx.x * blockDim.x;
         idx < size * size;
         idx += gridDim.x * blockDim.x) {
        int i = idx / size;
        int j = idx % size;
        if (i > 0 && i < size - 1 && j > 0 && j < size - 1) {
            double diff = fabs(grid[idx] - gridNew[idx]);
            if (diff > localMax)
                localMax = diff; // каждый поток находит своё локальное значение.
        }
    }

    double blockMax = BlockReduce(temp_storage).Reduce(localMax, cub::Max()); // агрегирует все эти значения в один максимум — blockMax для блока.
    if (threadIdx.x == 0) {
        atomicMaxDouble(error, blockMax); // один поток делает atomicMaxDouble(...), чтобы обновить глобальную ошибку.
    }
}

void initializeGrid(double* grid, double* gridNew, int size) {
    std::memset(grid, 0, sizeof(double) * size * size);
    std::memset(gridNew, 0, sizeof(double) * size * size);

    grid[0] = 10.0;
    grid[size - 1] = 20.0;
    grid[size * (size - 1)] = 30.0;
    grid[size * size - 1] = 20.0;

    double tl = grid[0];
    double tr = grid[size - 1];
    double bl = grid[size * (size - 1)];
    double br = grid[size * size - 1];

    for (int i = 1; i < size - 1; ++i) {
        double factor = i / static_cast<double>(size - 1);
        grid[i] = tl + (tr - tl) * factor;
        grid[size * (size - 1) + i] = bl + (br - bl) * factor;
        grid[size * i] = tl + (bl - tl) * factor;
        grid[size * i + size - 1] = tr + (br - tr) * factor;
    }
}

void solve(int size, double accuracy, int maxIterations) {
    size_t gridBytes = sizeof(double) * size * size;
    double *grid, *gridNew, *d_grid, *d_gridNew, *d_error;

    grid = (double*)malloc(gridBytes);
    gridNew = (double*)malloc(gridBytes);
    initializeGrid(grid, gridNew, size);

    cudaMalloc(&d_grid, gridBytes);
    cudaMalloc(&d_gridNew, gridBytes);
    cudaMallocManaged(&d_error, sizeof(double)); // Allocates memory that will be automatically managed by the Unified Memory system.

    cudaMemcpy(d_grid, grid, gridBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_gridNew, gridNew, gridBytes, cudaMemcpyHostToDevice);

    // параметры CUDA-ядра
    dim3 blockDim(16, 16);                                  // размеры одного блока (логической группы потоков) в потоках (blockDim.x = 16 | blockDim.y = 16 | 256 потоков в одном блоке)
    dim3 gridDim((size + blockDim.x - 3) / (blockDim.x),    // размер сетки в блоках (-3 т.к. границы сетки не обрабатываются)
                 (size + blockDim.y - 3) / (blockDim.y));

    int iter = 0;
    double error = accuracy + 1.0;

    cudaStream_t stream;
    cudaStreamCreate(&stream); // создание потока CUDA который управляет порядком выполнения задач на GPU. Не зависит от default stream. Нужен чтобы захватить последовательность операций в CUDA Graph.

    cudaGraph_t graph;
    cudaGraphExec_t instance;

    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    updateGridKernel<<<gridDim, blockDim, 0, stream>>>(d_grid, d_gridNew, size);
    cudaStreamEndCapture(stream, &graph);
    cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);

    auto start = std::chrono::steady_clock::now();

    while (error > accuracy && iter < maxIterations) {
        cudaGraphLaunch(instance, stream);

        // можно безопасно вызвать std::swap(...) и перейти к следующему шагу. А без этой строки GPU бы продолжил работать над ядром,
        // в то время как CPU пошёл бы дальше. !!! CUDA не ждёт, пока kernel завершится, а просто ставит его в очередь !!!
        cudaStreamSynchronize(stream);

        std::swap(d_grid, d_gridNew);

        if (iter % 1000 == 0) {
            *d_error = 0.0;

            // запускаем ((size * size + BLOCK_SIZE - 1) / BLOCK_SIZE) блоков, 
            // в каждом по BLOCK_SIZE потоков
            computeErrorKernel<<<(size * size + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
                d_grid, d_gridNew, d_error, size
            );
            cudaDeviceSynchronize(); // блокируем вызовы на хосте до того как все вычисления на GPU не закончатся. Нужно чтобы корректно проверять условие завершения цикла по ошибке.
            error = *d_error;
        }

        ++iter;
    }

    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    std::cout << "Time:        " << elapsed.count() << " sec\n"
              << "Iterations:  " << iter << "\n"
              << "Error value: " << error << std::endl;

    cudaFree(d_grid);
    cudaFree(d_gridNew);
    cudaFree(d_error);
    free(grid);
    free(gridNew);
    cudaStreamDestroy(stream);
}

int main(int argc, char* argv[]) {
    int gridSize;
    double accuracy;
    int maxIterations;

    po::options_description desc("Options");
    desc.add_options()
        ("help", "Show help")
        ("size", po::value<int>(&gridSize)->default_value(256), "Grid size")
        ("accuracy", po::value<double>(&accuracy)->default_value(1e-6), "Target accuracy")
        ("max_iterations", po::value<int>(&maxIterations)->default_value(1e6), "Max iterations");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);
    if (vm.count("help")) {
        std::cout << desc << "\n";
        return 1;
    }

    std::cout << "Running CUDA version with CUDA Graph and CUB...\n";
    solve(gridSize, accuracy, maxIterations);
    return 0;
}
