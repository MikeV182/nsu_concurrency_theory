# How to launch

```bash
make host && ./solver_cpu --size=512
make gpu && ./solver_gpu --size=512
```

### Nsight Systems

```bash
nsys profile --stats=true -o profile_gpu ./solver_gpu --size=256 --max_iter=50 # -acc=multicore for CPU activity monitoring && keep --max_iter small
nsys-ui # open .qdrep file with GUI
```
