#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <hip/hip_runtime.h>
#include <hipsparse.h>
#include <rocblas.h>
#include <hipblas.h>
#include "KSPSolve_GMRES_GPU.h"
#include "fast_solve.h"
 

extern GMRES_INFO *info;

// Error checking macros
#define HIP_CHECK(err) do { \
    if (err != hipSuccess) { \
        fprintf(stderr, "HIP error: %s at line %d\n", hipGetErrorString(err), __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

#define HIPSPARSE_CHECK(err) do { \
    if (err != HIPSPARSE_STATUS_SUCCESS) { \
        fprintf(stderr, "HIPSPARSE error: %d at line %d\n", err, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

#define ROCBLAS_CHECK(err) do { \
    if (err != rocblas_status_success) { \
        fprintf(stderr, "ROCBLAS error: %d at line %d\n", err, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

#define HIPBLAS_CHECK(status) do { \
    if (status != HIPBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "hipBLAS error: %d at line %d\n", status, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// Global rocblas handle and parameters
 
double alpha = 1.0, beta = 0.0;
rocblas_int x_m = N, x_n = N * N, x_k = N;
rocblas_int y_m = N, y_n = N * N, y_k = N;
rocblas_int z_m = N, z_n = N * N, z_k = N;

 

// Function to read binary file containing doubles
double* read_double_bin_file(const char* filename, size_t num_elements) {
    char full_path[256] = "./port_demo_data_ngrid_36/";
    strcat(full_path, filename);

    double* data = (double*)malloc(num_elements * sizeof(double));
    if (data == NULL) {
        fprintf(stderr, "Memory allocation failed for %s\n", full_path);
        exit(1);
    }

    FILE* file = fopen(full_path, "rb");
    if (file == NULL) {
        fprintf(stderr, "Failed to open %s\n", full_path);
        free(data);
        exit(1);
    }

    size_t read_count = fread(data, sizeof(double), num_elements, file);
    if (read_count != num_elements) {
        fprintf(stderr, "Failed to read %zu elements from %s, read %zu\n", num_elements, full_path, read_count);
        free(data);
        fclose(file);
        exit(1);
    }

    fclose(file);
    return data;
}

// Function to read binary file containing integers
int* read_int_bin_file(const char* filename, size_t num_elements) {
    char full_path[256] = "./port_demo_data_ngrid_36/";
    strcat(full_path, filename);

    int* data = (int*)malloc(num_elements * sizeof(int));
    if (data == NULL) {
        fprintf(stderr, "Memory allocation failed for %s\n", full_path);
        exit(1);
    }

    FILE* file = fopen(full_path, "rb");
    if (file == NULL) {
        fprintf(stderr, "Failed to open %s\n", full_path);
        free(data);
        exit(1);
    }

    size_t read_count = fread(data, sizeof(int), num_elements, file);
    if (read_count != num_elements) {
        fprintf(stderr, "Failed to read %zu elements from %s, read %zu\n", num_elements, full_path, read_count);
        free(data);
        fclose(file);
        exit(1);
    }

    fclose(file);
    return data;
}

// Helper function: Allocate 1D array
double* allocate_double_array(size_t size) {
    double* arr = (double*)malloc(size * sizeof(double));
    if (arr == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(1);
    }
    return arr;
}

// Print first few elements of a double array
void print_double_array(const char* name, double* data, size_t num_elements) {
    printf("%s (前 %zu 个元素):\n", name, num_elements < 10 ? num_elements : 10);
    for (size_t i = 0; i < num_elements && i < 10; i++) {
        printf("%.6f ", data[i]);
    }
    printf("\n\n");
}

// Print first few elements of an integer array
void print_int_array(const char* name, int* data, size_t num_elements) {
    printf("%s (前 %zu 个元素):\n", name, num_elements < 10 ? num_elements : 10);
    for (size_t i = 0; i < num_elements && i < 10; i++) {
        printf("%d ", data[i]);
    }
    printf("\n\n");
}

// Build Z in Block CSR format on CPU (called during preprocessing)
void build_Z_block_csr(double* ds, int grid_size, double alpha, int** row_ptr, int** col_ind, double** vals) {
    int num_blocks = PN;

    *row_ptr = (int*)malloc((num_blocks + 1) * sizeof(int));
    if (*row_ptr == NULL) {
        fprintf(stderr, "Memory allocation failed for row_ptr\n");
        exit(1);
    }

    *col_ind = (int*)malloc(num_blocks * sizeof(int));
    if (*col_ind == NULL) {
        fprintf(stderr, "Memory allocation failed for col_ind\n");
        free(*row_ptr);
        exit(1);
    }

    *vals = read_double_bin_file("vals_inv.bin", num_blocks * BLOCK_SIZE * BLOCK_SIZE);
    if (*vals == NULL) {
        fprintf(stderr, "Failed to read vals_inv.bin\n");
        free(*row_ptr);
        free(*col_ind);
        exit(1);
    }

    for (int i = 0; i <= num_blocks; i++) {
        (*row_ptr)[i] = i;
    }
    for (int i = 0; i < num_blocks; i++) {
        (*col_ind)[i] = i;
    }
}

// Preprocessing function to initialize static GPU data
void preprocess_data(PreprocessedData* data, double* U, double* V, double* U_t, double* V_t, double* ds, double* MEFG, int* boundary_indices, int grid_size, double alpha) {
    // Initialize rocblas handle
    rocblas_handle handle;
    ROCBLAS_CHECK(rocblas_create_handle(&handle));

    data->handle = handle;
    data->pn = PN;
    data->num_boundary = NUM_BOUNDARY;

 
    // Allocate GPU memory for d_tEFG, d_boundary_values, d_corrected_values, d_tEFG_new
    HIP_CHECK(hipMalloc(&data->d_tEFG, 3 * PN * sizeof(double)));
    HIP_CHECK(hipMalloc(&data->d_boundary_values, NUM_BOUNDARY * sizeof(double)));
    HIP_CHECK(hipMalloc(&data->d_corrected_values, NUM_BOUNDARY * sizeof(double)));
    HIP_CHECK(hipMalloc(&data->d_tEFG_new, 3 * PN * sizeof(double)));

    // Allocate GPU memory for d_tE, d_tF, d_tG 
    HIP_CHECK(hipMalloc(&data->d_tE, PN * sizeof(double)));
    HIP_CHECK(hipMalloc(&data->d_tF, PN * sizeof(double)));
    HIP_CHECK(hipMalloc(&data->d_tG, PN * sizeof(double)));

    // Allocate GPU memory for d_extracted_x
    HIP_CHECK(hipMalloc(&data->d_extracted_x, 3 * PN * sizeof(double)));

    // Allocate GPU memory for temporary arrays in solve function
    HIP_CHECK(hipMalloc(&data->d_Et, PN * sizeof(double)));
    HIP_CHECK(hipMalloc(&data->d_Ft, PN * sizeof(double)));
    HIP_CHECK(hipMalloc(&data->d_Gt, PN * sizeof(double)));
    HIP_CHECK(hipMalloc(&data->d_Ep, PN * sizeof(double)));
    HIP_CHECK(hipMalloc(&data->d_Fp, PN * sizeof(double)));
    HIP_CHECK(hipMalloc(&data->d_Gp, PN * sizeof(double)));
    HIP_CHECK(hipMalloc(&data->d_E1p, PN * sizeof(double)));
    HIP_CHECK(hipMalloc(&data->d_F1p, PN * sizeof(double)));
    HIP_CHECK(hipMalloc(&data->d_G1p, PN * sizeof(double)));
    HIP_CHECK(hipMalloc(&data->d_E2, PN * sizeof(double)));
    HIP_CHECK(hipMalloc(&data->d_F2, PN * sizeof(double)));
    HIP_CHECK(hipMalloc(&data->d_G2, PN * sizeof(double)));
    HIP_CHECK(hipMalloc(&data->d_r, 3 * PN * sizeof(double)));
    HIP_CHECK(hipMalloc(&data->d_c, 3 * PN * sizeof(double)));
    

    // Initialize hipsparse handle
    hipsparseHandle_t sparse_handle;
    HIPSPARSE_CHECK(hipsparseCreate(&sparse_handle));
    data->sparse_handle = sparse_handle;

    // Initialize hipsparse matrix descriptor
    hipsparseMatDescr_t descr;
    HIPSPARSE_CHECK(hipsparseCreateMatDescr(&descr));
    HIPSPARSE_CHECK(hipsparseSetMatType(descr, HIPSPARSE_MATRIX_TYPE_GENERAL));
    HIPSPARSE_CHECK(hipsparseSetMatIndexBase(descr, HIPSPARSE_INDEX_BASE_ZERO));
    data->descr = descr;

    // Allocate GPU memory for U, V, U_t, V_t
    HIP_CHECK(hipMalloc(&data->d_U, N * N * sizeof(double)));
    HIP_CHECK(hipMalloc(&data->d_V, N * N * sizeof(double)));
    HIP_CHECK(hipMalloc(&data->d_U_t, N * N * sizeof(double)));
    HIP_CHECK(hipMalloc(&data->d_V_t, N * N * sizeof(double)));

    // Copy U, V, U_t, V_t to GPU
    HIP_CHECK(hipMemcpy(data->d_U, U, N * N * sizeof(double), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(data->d_V, V, N * N * sizeof(double), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(data->d_U_t, U_t, N * N * sizeof(double), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(data->d_V_t, V_t, N * N * sizeof(double), hipMemcpyHostToDevice));

    // Build Z matrix in Block CSR format on CPU
    int* row_ptr;
    int* col_ind;
    double* vals;
    build_Z_block_csr(ds, grid_size, alpha, &row_ptr, &col_ind, &vals);

    // Allocate GPU memory for Z matrix (Block CSR format)
    int num_blocks = PN;
    int val_size = num_blocks * BLOCK_SIZE * BLOCK_SIZE;
    HIP_CHECK(hipMalloc(&data->d_row_ptr, (num_blocks + 1) * sizeof(int)));
    HIP_CHECK(hipMalloc(&data->d_col_ind, num_blocks * sizeof(int)));
    HIP_CHECK(hipMalloc(&data->d_val, val_size * sizeof(double)));

    // Copy Z matrix data to GPU
    HIP_CHECK(hipMemcpy(data->d_row_ptr, row_ptr, (num_blocks + 1) * sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(data->d_col_ind, col_ind, num_blocks * sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(data->d_val, vals, val_size * sizeof(double), hipMemcpyHostToDevice));

    // Allocate GPU memory for MEFG and boundary_indices
    size_t MEFG_size = NUM_BOUNDARY * NUM_BOUNDARY;
    size_t boundary_indices_size = NUM_BOUNDARY;
    HIP_CHECK(hipMalloc(&data->d_MEFG, MEFG_size * sizeof(double)));
    HIP_CHECK(hipMalloc(&data->d_boundary_indices, boundary_indices_size * sizeof(int)));

    // Copy MEFG and boundary_indices to GPU
    HIP_CHECK(hipMemcpy(data->d_MEFG, MEFG, MEFG_size * sizeof(double), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(data->d_boundary_indices, boundary_indices, boundary_indices_size * sizeof(int), hipMemcpyHostToDevice));

    // Free CPU memory for Z matrix
    free(row_ptr);
    free(col_ind);
    free(vals);
   
}

// Cleanup function to free preprocessed GPU data
void cleanup_preprocessed_data(PreprocessedData* data) {
    if (data->d_U) HIP_CHECK(hipFree(data->d_U));
    if (data->d_V) HIP_CHECK(hipFree(data->d_V));
    if (data->d_U_t) HIP_CHECK(hipFree(data->d_U_t));
    if (data->d_V_t) HIP_CHECK(hipFree(data->d_V_t));
    if (data->d_row_ptr) HIP_CHECK(hipFree(data->d_row_ptr));
    if (data->d_col_ind) HIP_CHECK(hipFree(data->d_col_ind));
    if (data->d_val) HIP_CHECK(hipFree(data->d_val));
    if (data->d_MEFG) HIP_CHECK(hipFree(data->d_MEFG));
    if (data->d_boundary_indices) HIP_CHECK(hipFree(data->d_boundary_indices));
    if (data->handle) ROCBLAS_CHECK(rocblas_destroy_handle(data->handle));
    if (data->sparse_handle) HIPSPARSE_CHECK(hipsparseDestroy(data->sparse_handle));
    if (data->descr) HIPSPARSE_CHECK(hipsparseDestroyMatDescr(data->descr));
   
 
    if (data->d_tEFG) HIP_CHECK(hipFree(data->d_tEFG));
    if (data->d_boundary_values) HIP_CHECK(hipFree(data->d_boundary_values));
    if (data->d_corrected_values) HIP_CHECK(hipFree(data->d_corrected_values));
    if (data->d_tEFG_new) HIP_CHECK(hipFree(data->d_tEFG_new));
    if (data->d_tE) HIP_CHECK(hipFree(data->d_tE));
    if (data->d_tF) HIP_CHECK(hipFree(data->d_tF));
    if (data->d_tG) HIP_CHECK(hipFree(data->d_tG));
    if (data->d_extracted_x) HIP_CHECK(hipFree(data->d_extracted_x));
    // temporary arrays in  solve function
    if (data->d_Et) HIP_CHECK(hipFree(data->d_Et));
    if (data->d_Ft) HIP_CHECK(hipFree(data->d_Ft));
    if (data->d_Gt) HIP_CHECK(hipFree(data->d_Gt));
    if (data->d_Ep) HIP_CHECK(hipFree(data->d_Ep));
    if (data->d_Fp) HIP_CHECK(hipFree(data->d_Fp));
    if (data->d_Gp) HIP_CHECK(hipFree(data->d_Gp));
    if (data->d_E1p) HIP_CHECK(hipFree(data->d_E1p));
    if (data->d_F1p) HIP_CHECK(hipFree(data->d_F1p));
    if (data->d_G1p) HIP_CHECK(hipFree(data->d_G1p));
    if (data->d_E2) HIP_CHECK(hipFree(data->d_E2));
    if (data->d_F2) HIP_CHECK(hipFree(data->d_F2));
    if (data->d_G2) HIP_CHECK(hipFree(data->d_G2));
    if (data->d_r) HIP_CHECK(hipFree(data->d_r));
    if (data->d_c) HIP_CHECK(hipFree(data->d_c));
    
}
// HIP kernel for Interleave
__global__ void interleave_kernel(double* d_E1p, double* d_F1p, double* d_G1p, double* d_r, int pn) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < pn) {
        d_r[3*idx] = d_E1p[idx];
        d_r[3*idx + 1] = d_F1p[idx];
        d_r[3*idx + 2] = d_G1p[idx];
    }
}

// HIP kernel for Extract
__global__ void extract_kernel(double* d_c, double* d_E2, double* d_F2, double* d_G2, int pn) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < pn) {
        d_E2[idx] = d_c[3*idx];
        d_F2[idx] = d_c[3*idx + 1];
        d_G2[idx] = d_c[3*idx + 2];
    }
}

// HIP kernel to construct tEFG
__global__ void construct_tEFG_kernel(double* d_E3, double* d_F3, double* d_G3, double* d_tEFG, int pn) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < pn) {
        d_tEFG[idx] = d_E3[idx];
        d_tEFG[pn + idx] = d_F3[idx];
        d_tEFG[2 * pn + idx] = d_G3[idx];
    }
}

// HIP kernel to collect boundary values
__global__ void collect_boundary_values_kernel(double* d_tEFG, int* d_boundary_indices, double* d_boundary_values, int num_boundary) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_boundary) {
        d_boundary_values[idx] = d_tEFG[d_boundary_indices[idx] - 1];
    }
}

// HIP kernel to scatter corrected values
__global__ void scatter_corrected_values_kernel(double* d_tEFG_new, int* d_boundary_indices, double* d_corrected_values, int num_boundary, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_boundary) {
        d_tEFG_new[d_boundary_indices[idx] - 1] = d_corrected_values[idx];
    }
}

// HIP kernel for vector subtraction
__global__ void vector_subtraction_kernel(double* E4, double* E3, double* tE,
                                         double* F4, double* F3, double* tF,
                                         double* G4, double* G3, double* tG,
                                         int pn) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < pn) {
        E4[i] = E3[i] - tE[i];
        F4[i] = F3[i] - tF[i];
        G4[i] = G3[i] - tG[i];
    }
}

// Optimized SpMV for Block CSR matrix
void spmv_block_csr_gpu(int* d_row_ptr, int* d_col_ind, double* d_val, double* d_r, double* d_c, int num_blocks) {
    const int block_dim = BLOCK_SIZE;
    const int mb = num_blocks;
    const int nb = num_blocks;
    const int nnzb = num_blocks;
    const double alpha_spmv = 1.0;
    const double beta_spmv = 0.0;
    PreprocessedData* preprocessed_ptr = (PreprocessedData*)info->preprocessed_ptr;

    HIPSPARSE_CHECK(hipsparseDbsrmv(
        preprocessed_ptr->sparse_handle, HIPSPARSE_DIRECTION_COLUMN, HIPSPARSE_OPERATION_NON_TRANSPOSE,
        mb, nb, nnzb, &alpha_spmv, preprocessed_ptr->descr, d_val, d_row_ptr, d_col_ind, block_dim,
        d_r, &beta_spmv, d_c));

}

// Optimized Solve function
void Solve(double* d_rE, double* d_rF, double* d_rG, PreprocessedData* preprocessed_data, int grid_size, double alpha, double* d_E3, double* d_F3, double* d_G3) {
    int pn = PN;
    alpha = 1.0;
    double beta = 0.0;
    rocblas_handle handle = preprocessed_data->handle;

    // Allocate temporary arrays on GPU
    double *d_Et, *d_Ft, *d_Gt, *d_Ep, *d_Fp, *d_Gp, *d_E1p, *d_F1p, *d_G1p, *d_E2, *d_F2, *d_G2;
    d_Et = preprocessed_data->d_Et;
    d_Ft = preprocessed_data->d_Ft;
    d_Gt = preprocessed_data->d_Gt;
    d_Ep = preprocessed_data->d_Ep;
    d_Fp = preprocessed_data->d_Fp;
    d_Gp = preprocessed_data->d_Gp;
    d_E1p = preprocessed_data->d_E1p;
    d_F1p = preprocessed_data->d_F1p;
    d_G1p = preprocessed_data->d_G1p;
    d_E2 = preprocessed_data->d_E2;
    d_F2 = preprocessed_data->d_F2;
    d_G2 = preprocessed_data->d_G2;


    // Allocate d_r and d_c on GPU
    double *d_r, *d_c;
    d_r = preprocessed_data->d_r;
    d_c = preprocessed_data->d_c;
  

    // Use preprocessed static GPU pointers
    double* d_U = preprocessed_data->d_U;
    double* d_V = preprocessed_data->d_V;
    double* d_U_t = preprocessed_data->d_U_t;
    double* d_V_t = preprocessed_data->d_V_t;
    int* d_row_ptr = preprocessed_data->d_row_ptr;
    int* d_col_ind = preprocessed_data->d_col_ind;
    double* d_val = preprocessed_data->d_val;

    // Forward transformations
    ROCBLAS_CHECK(rocblas_dgemm(handle, rocblas_operation_transpose, rocblas_operation_transpose,
                                x_n, x_k, x_m, &alpha, d_rE, x_k, d_U_t, x_m, &beta, d_Et, x_n));
    ROCBLAS_CHECK(rocblas_dgemm(handle, rocblas_operation_transpose, rocblas_operation_transpose,
                                y_n, y_k, y_m, &alpha, d_Et, y_k, d_V_t, y_m, &beta, d_Ep, y_n));
    ROCBLAS_CHECK(rocblas_dgemm(handle, rocblas_operation_transpose, rocblas_operation_transpose,
                                z_n, z_k, z_m, &alpha, d_Ep, z_k, d_V_t, z_m, &beta, d_E1p, z_n));

    ROCBLAS_CHECK(rocblas_dgemm(handle, rocblas_operation_transpose, rocblas_operation_transpose,
                                x_n, x_k, x_m, &alpha, d_rF, x_k, d_V_t, x_m, &beta, d_Ft, x_n));
    ROCBLAS_CHECK(rocblas_dgemm(handle, rocblas_operation_transpose, rocblas_operation_transpose,
                                y_n, y_k, y_m, &alpha, d_Ft, y_k, d_U_t, y_m, &beta, d_Fp, y_n));
    ROCBLAS_CHECK(rocblas_dgemm(handle, rocblas_operation_transpose, rocblas_operation_transpose,
                                z_n, z_k, z_m, &alpha, d_Fp, z_k, d_V_t, z_m, &beta, d_F1p, z_n));

    ROCBLAS_CHECK(rocblas_dgemm(handle, rocblas_operation_transpose, rocblas_operation_transpose,
                                x_n, x_k, x_m, &alpha, d_rG, x_k, d_V_t, x_m, &beta, d_Gt, x_n));
    ROCBLAS_CHECK(rocblas_dgemm(handle, rocblas_operation_transpose, rocblas_operation_transpose,
                                y_n, y_k, y_m, &alpha, d_Gt, y_k, d_V_t, y_m, &beta, d_Gp, y_n));
    ROCBLAS_CHECK(rocblas_dgemm(handle, rocblas_operation_transpose, rocblas_operation_transpose,
                                z_n, z_k, z_m, &alpha, d_Gp, z_k, d_U_t, z_m, &beta, d_G1p, z_n));

    // Interleave on GPU
    int threads_per_block = 256;
    int blocks = (pn + threads_per_block - 1) / threads_per_block;
    interleave_kernel<<<blocks, threads_per_block>>>(d_E1p, d_F1p, d_G1p, d_r, pn);
    HIP_CHECK(hipGetLastError());

    // SpMV on GPU using preprocessed Z matrix
    // spmv_block_csr_gpu(d_row_ptr, d_col_ind, d_val, d_r, d_c, pn);
  
    const double alpha_spmv = 1.0;
    const double beta_spmv = 0.0;
    HIPSPARSE_CHECK(hipsparseDbsrmv(
        preprocessed_data->sparse_handle, HIPSPARSE_DIRECTION_COLUMN, HIPSPARSE_OPERATION_NON_TRANSPOSE,
        pn, pn, pn, &alpha_spmv, preprocessed_data->descr, d_val, d_row_ptr, d_col_ind, BLOCK_SIZE,
        d_r, &beta_spmv, d_c));



    // Extract on GPU
    extract_kernel<<<blocks, threads_per_block>>>(d_c, d_E2, d_F2, d_G2, pn);
    HIP_CHECK(hipGetLastError());

    // Inverse transformations
    ROCBLAS_CHECK(rocblas_dgemm(handle, rocblas_operation_transpose, rocblas_operation_transpose,
                                x_n, x_k, x_m, &alpha, d_E2, x_k, d_U, x_m, &beta, d_Et, x_n));
    ROCBLAS_CHECK(rocblas_dgemm(handle, rocblas_operation_transpose, rocblas_operation_transpose,
                                y_n, y_k, y_m, &alpha, d_Et, y_k, d_V, y_m, &beta, d_Ep, y_n));
    ROCBLAS_CHECK(rocblas_dgemm(handle, rocblas_operation_transpose, rocblas_operation_transpose,
                                z_n, z_k, z_m, &alpha, d_Ep, z_k, d_V, z_m, &beta, d_E3, z_n));

    ROCBLAS_CHECK(rocblas_dgemm(handle, rocblas_operation_transpose, rocblas_operation_transpose,
                                x_n, x_k, x_m, &alpha, d_F2, x_k, d_V, x_m, &beta, d_Ft, x_n));
    ROCBLAS_CHECK(rocblas_dgemm(handle, rocblas_operation_transpose, rocblas_operation_transpose,
                                y_n, y_k, y_m, &alpha, d_Ft, y_k, d_U, y_m, &beta, d_Fp, y_n));
    ROCBLAS_CHECK(rocblas_dgemm(handle, rocblas_operation_transpose, rocblas_operation_transpose,
                                z_n, z_k, z_m, &alpha, d_Fp, z_k, d_V, z_m, &beta, d_F3, z_n));

    ROCBLAS_CHECK(rocblas_dgemm(handle, rocblas_operation_transpose, rocblas_operation_transpose,
                                x_n, x_k, x_m, &alpha, d_G2, x_k, d_V, x_m, &beta, d_Gt, x_n));
    ROCBLAS_CHECK(rocblas_dgemm(handle, rocblas_operation_transpose, rocblas_operation_transpose,
                                y_n, y_k, y_m, &alpha, d_Gt, y_k, d_V, y_m, &beta, d_Gp, y_n));
    ROCBLAS_CHECK(rocblas_dgemm(handle, rocblas_operation_transpose, rocblas_operation_transpose,
                                z_n, z_k, z_m, &alpha, d_Gp, z_k, d_U, z_m, &beta, d_G3, z_n));

}


