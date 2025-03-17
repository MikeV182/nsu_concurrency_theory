#include <iostream>
#include <vector>
#include <thread>
#include <chrono>
#include <mutex>

void initialize(std::vector<double>& vector, int startIndex, int endIndex, int n, std::vector<double>& matrix) {
    for (int i = startIndex; i < endIndex; ++i) {
        for (int j = 0; j < n; j++) {
            matrix[i * n + j] = static_cast<double>(i + j);
        }
    }
    if (startIndex == 0) {
        for (int j = 0; j < n; j++) {
            vector[j] = static_cast<double>(j);
        }
    }
}

void multiplication(const std::vector<double>& vector, const std::vector<double>& matrix, std::vector<double>& result, int startIndex, int endIndex, int n) {
    for (int i = startIndex; i < endIndex; i++) {
        result[i] = 0;
        for (int j = 0; j < n; j++) {
            result[i] += matrix[i * n + j] * vector[j];
        }
    }
}

int main() {
    std::vector<int> threadCounts = {1, 2, 4, 7, 8, 16, 20, 40};
    std::vector<int> matrixSizes = {20000, 40000};
    
    for (int n : matrixSizes) {
        std::cout << "Matrix size: " << n << "x" << n << std::endl;
        for (int numThreads : threadCounts) {
            std::vector<double> vector(n);
            std::vector<double> matrix(n * n);
            std::vector<double> result(n, 0);
            std::vector<std::jthread> threads; // https://stackoverflow.com/questions/62325679/what-is-stdjthread-in-c20
            
            int chunkSize = n / numThreads;
            int startIndex = 0;
            
            auto initStart = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < numThreads; ++i) {
                int endIndex = (i == numThreads - 1) ? n : startIndex + chunkSize;
                threads.emplace_back(initialize, ref(vector), startIndex, endIndex, n, ref(matrix));
                startIndex = endIndex;
            }
            threads.clear();
            auto initEnd = std::chrono::high_resolution_clock::now();
            
            auto start = std::chrono::high_resolution_clock::now();
            startIndex = 0;
            for (int i = 0; i < numThreads; ++i) {
                int endIndex = (i == numThreads - 1) ? n : startIndex + chunkSize;
                threads.emplace_back(multiplication, cref(vector), cref(matrix), ref(result), startIndex, endIndex, n);
                startIndex = endIndex;
            }
            
            threads.clear();
            auto end = std::chrono::high_resolution_clock::now();
            
            std::chrono::duration<double> initTime = initEnd - initStart;
            std::chrono::duration<double> computeTime = end - start;
            
            std::cout << "Threads: " << numThreads << " | Init Time: " << initTime.count() << "s | Compute Time: " << computeTime.count() << "s" << std::endl;
        }
        std::cout << "------------------------------------------" << std::endl;
    }
    return 0;
}
