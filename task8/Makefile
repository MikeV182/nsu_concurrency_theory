gpu_cub:
	nvc++ -o gpu_cub gpu_cub.cpp -std=c++17 -Minfo=accel -cuda -fast \
		-I/opt/nvidia/hpc_sdk/Linux_x86_64/23.11/cuda/12.3/include \
		-I/usr/local/cuda/include \
		-L/usr/local/cuda/lib64 \
		-I/opt/nvidia/hpc_sdk/Linux_x86_64/23.11/math_libs/12.3/targets/x86_64-linux/include \
		-L/opt/nvidia/hpc_sdk/Linux_x86_64/23.11/math_libs/12.3/targets/x86_64-linux/lib \
		-L/opt/nvidia/hpc_sdk/Linux_x86_64/23.11/cuda/12.3/lib64 \
		-lcublas -lcudart -lnvToolsExt -lboost_program_options

clear:
	rm gpu_cub *.nsys-rep *.sqlite

profile:
	nsys profile --trace=nvtx,cuda --stats=true ./gpu_cub --max_iterations=50
