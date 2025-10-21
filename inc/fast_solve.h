 
#ifndef fast_solve_H
#define fast_solve_H

// #define N 38
// #define NUM_BOUNDARY 8550

#define N 36
#define NUM_BOUNDARY 7668


// #define N 34
// #define NUM_BOUNDARY 6834

// #define N 32
// #define NUM_BOUNDARY 6048
 
#define PN (N * N * N)
#define BLOCK_SIZE 3


 

// HIP kernel for Interleave
__global__ void interleave_kernel(double* d_E1p, double* d_F1p, double* d_G1p, double* d_r, int pn);

// HIP kernel for Extract
__global__ void extract_kernel(double* d_c, double* d_E2, double* d_F2, double* d_G2, int pn);

// HIP kernel to construct tEFG
__global__ void construct_tEFG_kernel(double* d_E3, double* d_F3, double* d_G3, double* d_tEFG, int pn);

// HIP kernel to collect boundary values
__global__ void collect_boundary_values_kernel(double* d_tEFG, int* d_boundary_indices, double* d_boundary_values, int num_boundary);

// HIP kernel to scatter corrected values
__global__ void scatter_corrected_values_kernel(double* d_tEFG_new, int* d_boundary_indices, double* d_corrected_values, int num_boundary, int size);

// HIP kernel for vector subtraction
__global__ void vector_subtraction_kernel(double* E4, double* E3, double* tE,
                                         double* F4, double* F3, double* tF,
                                         double* G4, double* G3, double* tG,
                                         int pn);

 

// Function signatures
double* read_double_bin_file(const char* filename, size_t num_elements);
int* read_int_bin_file(const char* filename, size_t num_elements);
double* allocate_double_array(size_t size);
void print_double_array(const char* name, double* data, size_t num_elements);
void print_int_array(const char* name, int* data, size_t num_elements);
void build_Z_block_csr(double* ds, int grid_size, double alpha, int** row_ptr, int** col_ind, double** vals);
void preprocess_data(PreprocessedData* data, double* U, double* V, double* U_t, double* V_t, double* ds, double* MEFG, int* boundary_indices, int grid_size, double alpha);
void cleanup_preprocessed_data(PreprocessedData* data);
void spmv_block_csr_gpu(int* d_row_ptr, int* d_col_ind, double* d_val, double* d_r, double* d_c, int num_blocks);
void Solve(double* d_rE, double* d_rF, double* d_rG, PreprocessedData* preprocessed_data, int grid_size, double alpha, double* d_E3, double* d_F3, double* d_G3);


#endif // fast_solve_H