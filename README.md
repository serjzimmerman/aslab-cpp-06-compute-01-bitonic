# Having fun with GPGPU. Bitonic sort and Tiled matrix multiplication using OpenCL and C++

Experiments with OpenCL. Several implementations of bitonic sort for CPU/GPU. Perfomance measurements, comparisons with std::sort/__gnu_parallel::sort.
Matrix multiplication (naive, with tiling) and comparisons to baseline naive implemtation/Eigen/

## 1. How to build

We rely on Python for simple C++ codegen from OpenCL kernels. Make sure to have it installed, as well as boost::program_options which is used in driver programs. 

### Linux
```sh
git submodule init && git submodule update

cmake -S ./ -B build/ -DCMAKE_BUILD_TYPE=Release
# You can specify the type to use with the option -DTYPE (by default int is used)
cmake -S ./ -B build/ -DCMAKE_BUILD_TYPE=Release -DTYPE=float

# To enable Eigen make sure you have it installed systemwide and provide the following flag:
cmake -S ./ -B build/ -DCMAKE_BUILD_TYPE=Release -DEIGEN_MAT_MULT=ON
# Similarly, you can enable __gnu_parallel::sort. It relies on OpenMP:
cmake -S ./ -B build/ -DCMAKE_BUILD_TYPE=Release -DPAR_CPU_SORT=ON

cd build/
make -j12
```

### Windows
```sh
git submodule init && git submodule update
# For some reason MSVC does not like linking dynamically to boost libs
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBOOST_ROOT="path/to/boost" -DBoost_USE_STATIC_LIBS=ON
cmake --build build
```

## 2. Bitonic
To run bitonic sort use __bitonic__ target. 

```sh
./bitonic -h
Avaliable options:
#  -h [ --help ]                     Print this help message
#  -p [ --print ]                    Print on failure
#  -s [ --skip ]                     Skip comparing with std::sort
#  -l [ --lower ] arg (=-2147483648) Lower bound
#  -u [ --upper ] arg (=2147483647)  Upper bound
#  -n [ --num ] arg (=22)            Length of the array to sort = 2^n
#  -k [ --kernel ] arg (=naive)      Which kernel to use: naive, cpu, local
#  --lsz arg (=256)                  Local iteration size
```

## 3. Matmult
To run bitonic sort use __matmult__ target. 

```sh
./matmult -h
# Available options:
#  -h [ --help ]                Print this help message
#  -p [ --print ]               Print on failure
#  -e [ --eigen ]               Compare with Eigen matrix multiplication
#  -s [ --skip ]                Skip cpu calculation
#  -l [ --lower ] arg (=-32)    Low bound for random integer
#  -u [ --upper ] arg (=32)     Upper bound for random integer
#  --ax arg (=512)              Number of rows in matrix A
#  --ay arg (=512)              Number of cols in matrix A
#  --by arg (=512)              Number of cols in matrix B
#  -k [ --kernel ] arg (=naive) Which kernel to use: naive, tiled, tiledarb
#  --lsz arg (=8)               Local iteration size
```