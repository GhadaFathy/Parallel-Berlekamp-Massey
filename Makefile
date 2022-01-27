CUDA_PATH       := / The_path_to_cuda_toolkit
CUDA_INC_PATH   := $(CUDA_PATH)/include
CUDA_BIN_PATH   := $(CUDA_PATH)/bin
CUDA_LIB_PATH   := $(CUDA_PATH)/lib64
NVCC            := $(CUDA_BIN_PATH)/nvcc
GCC             := g++

LDFLAGS   := -L$(CUDA_LIB_PATH) -lcudart
CCFLAGS   := -m64
NVCCFLAGS := -m64 -arch=sm_20
INCLUDES  := -I$(CUDA_INC_PATH) -I$(CUDA_PATH)/samples/common/inc -I.

CC=gcc
CFLAGS=-Wall -O3

all: parallel_bma 

parallel_bma: parallel_bma.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(LDFLAGS) -O3 -o $@ $<

clean:
	rm parallel_bma 
