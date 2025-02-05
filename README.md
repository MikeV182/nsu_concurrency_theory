## Как выбрать тип массива?

для double (по умолчанию) 
> cmake -B build .

для float
> cmake -B build . -DUSE_FLOAT=ON

Затем сборка и запуск
> cmake --build build

> ./build/task1

## Результаты

> float: -0.0277862

> double: 3.68912e-10
