host: main.cpp
	pgc++ -acc -ta=multicore -Minfo=accel -O2 -o solver_cpu task6_openacc.cpp -lboost_program_options

gpu: main.cpp
	pgc++ -acc -ta=tesla:managed -Minfo=accel -O2 -o solver_gpu task6_openacc.cpp -lboost_program_options
