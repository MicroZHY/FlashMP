# Include PETSc configuration
include ${PETSC_DIR}/lib/petsc/conf/variables

# Paths for HIP and CUDA
CUDA_PATH = /software/nvidia/cuda/10.0
HIP_BASE_PATH = /public/software/compiler/rocm/dtk-23.10
#HIP_BASE_PATH = /public/software/compiler/rocm/dtk-22.10.1
# Include paths
HIP_INCLUDE_PATH = -I$(HIP_BASE_PATH)/include
CUDA_INCLUDE_PATH = -I$(CUDA_PATH)/include
HIPBLAS_INCLUDE_PATH = -I$(HIP_BASE_PATH)/hipblas/include  -I$(HIP_BASE_PATH)/rocblas/include
HIPSPARSE_INCLUDE_PATH = -I$(HIP_BASE_PATH)/hipsparse/include
PETSC_CC_INCLUDES = -I$(PETSC_DIR)/include -I$(PETSC_DIR)/test/include
PB_BASE_PATH = /public/home/suyuexinghen/zhanghy/FDTD-DMDA/fdtd
PB_INCLUDES = -I$(PB_BASE_PATH)/inc

# Library paths with rpath
CUDA_LIB_PATH = -Wl,-rpath,$(CUDA_PATH)/lib64 -L$(CUDA_PATH)/lib64
HIP_LIB_PATH = -Wl,-rpath,$(HIP_BASE_PATH)/hip/lib -L$(HIP_BASE_PATH)/hip/lib
HIPBLAS_LIB_PATH = -Wl,-rpath,$(HIP_BASE_PATH)/hipblas/lib -L$(HIP_BASE_PATH)/hipblas/lib -Wl,-rpath,$(HIP_BASE_PATH)/rocblas/lib -L$(HIP_BASE_PATH)/rocblas/lib
HIPSPARSE_LIB_PATH = -Wl,-rpath,$(HIP_BASE_PATH)/hipsparse/lib -L$(HIP_BASE_PATH)/hipsparse/lib

# Libraries
HIP_LIBS = -lhipblas -lhipsparse -lamdhip64 -lrocblas 

# Compilers
CC = gcc
HIPCC = hipcc
LINKER = mpif90

# Compilation flags
CFLAGS = -DAMD_PLATFORM -DNVARS=3 -fno-common -DGEOASM_NO_IMPL
HIPCC_FLAGS = -DAMD_PLATFORM -DNVARS=3 -fno-common -DGEOASM_NO_IMPL

ILUPACK_INCLUDES = -I../milu_inc -I../inc
ILUPACK_LIBPATH = -L$(PB_BASE_PATH)/milu_lib/GNU64
# ILUPACK_LIBS = -lilupack -lamd -lmetis -lsparspak -llapack -lblaslike -lblas -lstdc++

# Linking flags
LDFLAGS = -L/opt/rh/devtoolset-7/root/usr/lib64 -lgfortran -Wl,--allow-multiple-definition

# Source and object file paths
PB_SRC_PATH = $(PB_BASE_PATH)/src
PB_OBJ_PATH = $(PB_BASE_PATH)/obj

# Source and object files (排除 geoasm.c)
PB_SRC_CPP_FILES = $(wildcard $(PB_SRC_PATH)/*.cpp)
PB_SRC_C_FILES = $(filter-out $(PB_SRC_PATH)/geoasm.c, $(wildcard $(PB_SRC_PATH)/*.c))
PB_OBJ_CPP_FILES = $(patsubst $(PB_SRC_PATH)/%.cpp, $(PB_OBJ_PATH)/%.o, $(PB_SRC_CPP_FILES))
PB_OBJ_C_FILES = $(patsubst $(PB_SRC_PATH)/%.c, $(PB_OBJ_PATH)/%.o, $(PB_SRC_C_FILES))

# Application specific
MY_APP_FILE = solveFDTD-DMDA-mg-geoAsm.cpp
# MY_APP_FILE = solveFDTD-DMDA.cpp
# MY_APP_FILE = solveFDTD-DMDA.cpp

MY_APP_OBJ_FILE = $(patsubst %.cpp, %.o, $(MY_APP_FILE))
MY_APP = $(basename $(MY_APP_FILE))

# Ensure object directory exists
$(shell mkdir -p $(PB_OBJ_PATH))

# Compilation rules
$(PB_OBJ_PATH)/%.o: $(PB_SRC_PATH)/%.cpp
	$(HIPCC) $(HIPCC_FLAGS) -fvisibility=hidden $(PB_INCLUDES) $(PETSC_CC_INCLUDES) \
	$(HIP_INCLUDE_PATH) $(HIPBLAS_INCLUDE_PATH) $(HIPSPARSE_INCLUDE_PATH) $(ILUPACK_INCLUDES) \
	-c $< -o $@

$(PB_OBJ_PATH)/%.o: $(PB_SRC_PATH)/%.c
	$(CC) $(CFLAGS) $(PB_INCLUDES) $(PETSC_CC_INCLUDES) \
	$(HIP_INCLUDE_PATH) $(HIPBLAS_INCLUDE_PATH) $(HIPSPARSE_INCLUDE_PATH) $(ILUPACK_INCLUDES) \
	-c $< -o $@

# Special rule for geoasm.c with -fvisibility=hidden
$(PB_OBJ_PATH)/geoasm.o: $(PB_SRC_PATH)/geoasm.c
	$(CC) $(CFLAGS) -fvisibility=hidden $(PB_INCLUDES) $(PETSC_CC_INCLUDES) \
	$(HIP_INCLUDE_PATH) $(HIPBLAS_INCLUDE_PATH) $(HIPSPARSE_INCLUDE_PATH) $(ILUPACK_INCLUDES) \
	-c $< -o $@

# Special rule for KSPSolve_GMRES_GPU.cpp with pre-processing output
$(PB_OBJ_PATH)/KSPSolve_GMRES_GPU.o: $(PB_SRC_PATH)/KSPSolve_GMRES_GPU.cpp
	$(HIPCC) $(HIPCC_FLAGS) -fvisibility=hidden $(PB_INCLUDES) $(PETSC_CC_INCLUDES) \
	$(HIP_INCLUDE_PATH) $(HIPBLAS_INCLUDE_PATH) $(HIPSPARSE_INCLUDE_PATH) $(ILUPACK_INCLUDES) \
	-E $< > $(PB_OBJ_PATH)/KSPSolve_GMRES_GPU.i
	$(HIPCC) $(HIPCC_FLAGS) -fvisibility=hidden $(PB_INCLUDES) $(PETSC_CC_INCLUDES) \
	$(HIP_INCLUDE_PATH) $(HIPBLAS_INCLUDE_PATH) $(HIPSPARSE_INCLUDE_PATH) $(ILUPACK_INCLUDES) \
	-c $< -o $@

# Create static library for geoasm
libgeoasm.a: $(PB_OBJ_PATH)/geoasm.o
	ar rcs libgeoasm.a $(PB_OBJ_PATH)/geoasm.o

# Application compilation
$(MY_APP_OBJ_FILE): $(MY_APP_FILE)
	$(HIPCC) $(CFLAGS) -fvisibility=hidden $(PB_INCLUDES) $(PETSC_CC_INCLUDES) \
	$(HIP_INCLUDE_PATH) $(HIPBLAS_INCLUDE_PATH) $(HIPSPARSE_INCLUDE_PATH) $(ILUPACK_INCLUDES) \
	-E $< > $(PB_OBJ_PATH)/solveFDTD-DMDA-mg-geoAsm.i
	$(HIPCC) $(CFLAGS) -fvisibility=hidden $(PB_INCLUDES) $(PETSC_CC_INCLUDES) \
	$(HIP_INCLUDE_PATH) $(HIPBLAS_INCLUDE_PATH) $(HIPSPARSE_INCLUDE_PATH) $(ILUPACK_INCLUDES) \
	-c $< -o $@

# Linking
$(MY_APP): $(PB_OBJ_CPP_FILES) $(PB_OBJ_C_FILES) $(MY_APP_OBJ_FILE) libgeoasm.a
	$(LINKER) $(LDFLAGS) -o $(MY_APP) \
	$(PB_OBJ_CPP_FILES) $(PB_OBJ_C_FILES) $(MY_APP_OBJ_FILE) \
	$(PETSC_LIB) -L. -lgeoasm \
	$(HIPBLAS_LIB_PATH) $(HIPSPARSE_LIB_PATH) $(ILUPACK_LIBPATH) $(ILUPACK_LIBS) $(HIP_LIBS) $(HIP_LIB_PATH)

# Clean target
clean:
	rm -f $(MY_APP) $(MY_APP_OBJ_FILE) $(PB_OBJ_C_FILES) $(PB_OBJ_CPP_FILES) libgeoasm.a $(PB_OBJ_PATH)/geoasm.o $(PB_OBJ_PATH)/*.i