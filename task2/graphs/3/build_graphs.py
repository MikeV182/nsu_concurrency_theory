import matplotlib.pyplot as plt

def read_data(filename):
    threads, times, speedups = [], [], []
    with open(filename) as f:
        for line in f:
            cols = line.split()
            threads.append(int(cols[0]))
            times.append(float(cols[1]))
            speedups.append(float(cols[2]))
    return threads, times, speedups

def plot_and_save(threads, times, speedups, filename_prefix):
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(threads, times, marker='o', label="Время")
    plt.xlabel("Число потоков")
    plt.ylabel("Время (сек)")
    plt.grid(True)
    plt.legend()
    plt.title(f"{filename_prefix}: Время выполнения")
    plt.savefig(f"{filename_prefix}_time.png")
    
    plt.subplot(1, 2, 2)
    plt.plot(threads, speedups, marker='s', label="Ускорение")
    plt.xlabel("Число потоков")
    plt.ylabel("Ускорение")
    plt.grid(True)
    plt.legend()
    plt.title(f"{filename_prefix}: Ускорение")
    plt.savefig(f"{filename_prefix}_speedup.png")
    
    plt.show()

for filename in ["Out_task31.txt", "Out_task32.txt"]:
    threads, times, speedups = read_data(filename)
    plot_and_save(threads, times, speedups, filename.split('.')[0])
