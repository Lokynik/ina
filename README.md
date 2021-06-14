# INa_full_trace

How to create conda environment:
```sh
conda env create -f environment.yml --prefix ./env
conda activate ./env
conda deactivate  # if you need
```

How to remove the environment:
```sh
conda remove --prefix ./env --all
```
How to build ina.so
```sh
cd src/model_ctypes/ina/
make clean && make
```
How to run in N threads
```sh
cd src/python_scripts/
mpirun -n N python run_mpi.py
```
