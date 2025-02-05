cmake_minimum_required(VERSION 3.10)
project(task1)

# Добавляем опцию для выбора типа массива
option(USE_FLOAT "Use float for array type" OFF)

# Создаём макрос в зависимости от выбранного типа
if(USE_FLOAT)
    add_compile_definitions(FLOAT=1)
else()
    add_compile_definitions(FLOAT=0)
endif()

# Создаём исполняемый файл
add_executable(task1 task1.cpp)
