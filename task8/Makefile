TARGET = gpu_cub
NVCC = nvcc
CXXFLAGS = -O3 -std=c++17
BOOST_FLAGS = -lboost_program_options -lcudart

all: $(TARGET)

$(TARGET): gpu_cub.cu
	$(NVCC) gpu_cub.cu -o $(TARGET) $(CXXFLAGS) $(BOOST_FLAGS)

clean:
	rm -f $(TARGET)
