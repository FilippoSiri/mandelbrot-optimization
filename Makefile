CC = icpx
NVCC = nvc++
FLAGS = -fast -xHost -ggdb
OMP_FLAGS = -qopenmp -DTHREAD_NO=24
FMA_FLAGS = -DFMA
DOUBLE_FLAGS = -DDOUBLE
RESOLUTION = 5000
#
CPU_REF_SRC = src/mandelbrot.cpp
CPU_OMP_SRC = src/mandelbrot_cpu_omp.cpp
CPU_VECT_SRC = src/mandelbrot_cpu_vect_omp.cpp
GPU_SRC = src/mandelbrot_gpu.cu

all: ref ref-norm-optim omp vect-st-f vect-st-d vect-f vect-d vect-f-static vect-d-static gpu-f gpu-d

clean:
    rm -f bin/*

ref:
    $(CC) $(FLAGS) $(CPU_REF_SRC) -o bin/ref

ref-norm-optim:
    $(CC) $(FLAGS) -DNORM_OPTIM -qopenmp -DTHREAD_NO=1 $(CPU_OMP_SRC) -o bin/ref-norm-optim

omp:
    $(CC) $(FLAGS) $(OMP_FLAGS) -DNORM_OPTIM $(CPU_OMP_SRC) -o bin/omp

vect-st-f:
    $(CC) $(FLAGS) $(FMA_FLAGS) $(CPU_VECT_SRC) -o bin/vect-st-f

vect-st-d:
    $(CC) $(FLAGS) $(FMA_FLAGS) $(DOUBLE_FLAGS) $(CPU_VECT_SRC) -o bin/vect-st-d

vect-f:
    $(CC) $(FLAGS) $(FMA_FLAGS) $(OMP_FLAGS) $(CPU_VECT_SRC) -o bin/vect-f

vect-d:
    $(CC) $(FLAGS) $(FMA_FLAGS) $(DOUBLE_FLAGS) $(OMP_FLAGS) $(CPU_VECT_SRC) -o bin/vect-d

vect-f-static:
    $(CC) $(FLAGS) $(FMA_FLAGS) $(OMP_FLAGS) -DOMP_SCHEDULE=static -DRESOLUTION=$(RESOLUTION) $(CPU_VECT_SRC) -o bin/vect-f-static

vect-d-static:
    $(CC) $(FLAGS) $(FMA_FLAGS) $(DOUBLE_FLAGS) $(OMP_FLAGS) -DOMP_SCHEDULE=static -DRESOLUTION=$(RESOLUTION) $(CPU_VECT_SRC) -o bin/vect-d-static

gpu-f:
    $(NVCC) $(GPU_SRC) -o bin/gpu-f

gpu-d:
    $(NVCC) -DDOUBLE $(GPU_SRC) -o bin/gpu-d
