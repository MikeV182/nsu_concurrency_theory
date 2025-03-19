#include <iostream>
#include <thread>
#include <unordered_map>
#include <cmath>
#include <fstream>
#include <random>
#include <mutex>
#include <functional>
#include <string>
#include <condition_variable>

struct Task {
    std::string type;
    double argument;
    int iter;
    int argument2 = 0;
};

template <typename T>
class Server {
public:
    Server() : running(true) {}

    void start() {
        worker_thread = std::thread(&Server::run, this);
    }

    void stop() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            running = false;
        }
        cv.notify_all();
        worker_thread.join();
    }

    size_t add_task(Task task) {
        std::lock_guard<std::mutex> lock(mutex_);
        size_t id = task_id_++;
        tasks_[id] = task;
        cv.notify_one();
        return id;
    }

    T request_result(size_t id) {
        std::unique_lock<std::mutex> lock(mutex_);
        cv.wait(lock, [this, id] { return results_.find(id) != results_.end(); });
        T result = results_[id];
        results_.erase(id);
        return result;
    }

private:
    void run() {
        while (true) {
            Task task;
            size_t id;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                cv.wait(lock, [this] { return !tasks_.empty() || !running; });
                if (!running && tasks_.empty()) break;
                auto it = tasks_.begin();
                id = it->first;
                task = it->second;
                tasks_.erase(it);
            }

            T result;
            if (task.type == "Sinus") {
                result = std::sin(task.argument);
            } else if (task.type == "Square") {
                result = std::sqrt(task.argument);
            } else {
                result = std::pow(task.argument, task.argument2);
            }

            {
                std::lock_guard<std::mutex> lock(mutex_);
                results_[id] = result;
            }
            cv.notify_all();
        }
    }

    std::thread worker_thread;
    std::unordered_map<size_t, Task> tasks_;
    std::unordered_map<size_t, T> results_;
    size_t task_id_ = 0;
    std::mutex mutex_;
    std::condition_variable cv;
    bool running;
};

void client(Server<double>& server, Task task, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Не удалось открыть файл " << filename << std::endl;
        return;
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(0.0, 10.0);
    std::uniform_int_distribution<int> dist2(0, 10);

    for (size_t i = 0; i < task.iter; ++i) {
        double argument = dist(gen);
        int argument2 = (task.type == "Power") ? dist2(gen) : 0;
        
        Task current_task = {task.type, argument, task.iter, argument2};
        size_t task_id = server.add_task(current_task);
        double result = server.request_result(task_id);

        file << "Task: " << task_id << " arg1: " << argument;
        if (task.type == "Power") {
            file << " arg2: " << argument2;
        }
        file << " Result: " << result << std::endl;
    }
}

int main(int argc, char* argv[]) {
    if (argc != 7) {
        std::cerr << "Введите аргументы в формате: функция (Sinus, Square, Power), количество операций (от 5 до 10000)" << std::endl;
        return 1;
    }

    Task client1 = {argv[1], 0, std::stoi(argv[2])};
    Task client2 = {argv[3], 0, std::stoi(argv[4])};
    Task client3 = {argv[5], 0, std::stoi(argv[6])};

    std::string file1 = client1.type + ".txt";
    std::string file2 = client2.type + ".txt";
    std::string file3 = client3.type + ".txt";

    Server<double> server;
    server.start();

    std::thread client_serv1(client, std::ref(server), client1, file1);
    std::thread client_serv2(client, std::ref(server), client2, file2);
    std::thread client_serv3(client, std::ref(server), client3, file3);

    client_serv1.join();
    client_serv2.join();
    client_serv3.join();

    server.stop();

    return 0;
}
