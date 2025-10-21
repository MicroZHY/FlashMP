#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
// #include <mpi.h>
#include <../src/ksp/pc/impls/asm/asm.c>
#include <../src/ksp/ksp/impls/gmres/gmresimpl.h>
#include <../src/ksp/ksp/impls/bcgs/bcgsimpl.h>
#include <../src/mat/impls/baij/mpi/mpibaij.h>
#include <../src/mat/impls/baij/seq/baij.h>
#include <../src/ksp/pc/impls/factor/ilu/ilu.h>
#include <petsc/private/vecscatterimpl.h>
#include "KSPSolve_GMRES_GPU.h"
#include "CudaTimer.h"
#include <petsc/private/sfimpl.h>
extern char result_file[256];

extern double pack_time;
extern double unpack_time;
extern double comm_time;
extern double ASMLvecToVec_time;

extern double solve_time;
extern double gemv_time;
extern double construct_tEFG_time;
extern double collect_boundary_values_time;

void matrix_iverse(double *a, double out[][NVARS]);
int FIRSTONE = 0; // for debugging
GMRES_INFO *info = NULL;

// Error checking macros
#define HIP_CHECK(err)                                                                       \
    do                                                                                       \
    {                                                                                        \
        if (err != hipSuccess)                                                               \
        {                                                                                    \
            fprintf(stderr, "HIP error: %s at line %d\n", hipGetErrorString(err), __LINE__); \
            exit(EXIT_FAILURE);                                                              \
        }                                                                                    \
    } while (0)

#define ROCBLAS_CHECK(err)                                                    \
    do                                                                        \
    {                                                                         \
        if (err != rocblas_status_success)                                    \
        {                                                                     \
            fprintf(stderr, "ROCBLAS error: %d at line %d\n", err, __LINE__); \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

// Warmup function for rocblas_dgemm
void warmup_rocblas_dgemm()
{
    const int m = 2048;
    const int n = 2048;
    const int k = 2048;
    const double alpha = 1.0;
    const double beta = 0.0;

    rocblas_handle handle;
    ROCBLAS_CHECK(rocblas_create_handle(&handle));

    // Allocate device memory
    double *d_rE, *d_U_t, *d_Et;
    HIP_CHECK(hipMalloc(&d_rE, n * k * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_U_t, k * m * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_Et, n * m * sizeof(double)));

    // Initialize host matrices with random data
    double *h_rE = (double *)malloc(n * k * sizeof(double));
    double *h_U_t = (double *)malloc(k * m * sizeof(double));
    double *h_Et = (double *)malloc(n * m * sizeof(double));

    for (int i = 0; i < n * k; i++)
        h_rE[i] = (double)rand() / RAND_MAX;
    for (int i = 0; i < k * m; i++)
        h_U_t[i] = (double)rand() / RAND_MAX;
    for (int i = 0; i < n * m; i++)
        h_Et[i] = 0.0;

    // Copy data to device
    HIP_CHECK(hipMemcpy(d_rE, h_rE, n * k * sizeof(double), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_U_t, h_U_t, k * m * sizeof(double), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_Et, h_Et, n * m * sizeof(double), hipMemcpyHostToDevice));

    // Warmup: Call rocblas_dgemm 10 times
    for (int i = 0; i < 10; i++)
    {
        ROCBLAS_CHECK(rocblas_dgemm(handle, rocblas_operation_transpose, rocblas_operation_transpose,
                                    n, k, m, &alpha, d_rE, k, d_U_t, m, &beta, d_Et, n));
    }

    // Synchronize to ensure all calls are complete
    HIP_CHECK(hipDeviceSynchronize());

    // Clean up
    free(h_rE);
    free(h_U_t);
    free(h_Et);
    HIP_CHECK(hipFree(d_rE));
    HIP_CHECK(hipFree(d_U_t));
    HIP_CHECK(hipFree(d_Et));
    ROCBLAS_CHECK(rocblas_destroy_handle(handle));
}

__global__ void VecDot_kernel_1(double *xin, double *y0, int len, double *res)
{
    __shared__ double cache[THREADS_PER_BLOCK_DOT];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride;
    double sum0 = 0.0;
    while (i < len)
    {
        sum0 += xin[i] * y0[i];
        i += gridDim.x * blockDim.x;
    }
    cache[tid] = sum0;
    for (stride = THREADS_PER_BLOCK_DOT / 2; stride > 0; stride /= 2)
    {
        __syncthreads();
        if (tid < stride)
        {
            cache[tid] += cache[tid + stride];
        }
    }
    if (tid == 0)
    {
        res[blockIdx.x] = cache[0];
    }
}
__global__ void VecDot_kernel_2(double *xin, double *y0, double *y1, int len, double *res)
{
    __shared__ double cache[2 * THREADS_PER_BLOCK_DOT];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride;
    double sum0 = 0.0;
    double sum1 = 0.0;
    double x;
    while (i < len)
    {
        x = xin[i];
        sum0 += x * y0[i];
        sum1 += x * y1[i];
        i += gridDim.x * blockDim.x;
    }
    cache[tid] = sum0;
    cache[tid + THREADS_PER_BLOCK_DOT] = sum1;
    for (stride = THREADS_PER_BLOCK_DOT / 2; stride > 0; stride /= 2)
    {
        __syncthreads();
        if (tid < stride)
        {
            cache[tid] += cache[tid + stride];
            cache[tid + THREADS_PER_BLOCK_DOT] += cache[tid + stride + THREADS_PER_BLOCK_DOT];
        }
    }
    if (tid == 0)
    {
        res[blockIdx.x] = cache[0];
        res[blockIdx.x + BLOCKS_PER_GRID_DOT] = cache[THREADS_PER_BLOCK_DOT];
    }
}
__global__ void VecDot_kernel_3(double *xin, double *y0, double *y1, double *y2, int len, double *res)
{
    __shared__ double cache[3 * THREADS_PER_BLOCK_DOT];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride;
    double sum0 = 0.0;
    double sum1 = 0.0;
    double sum2 = 0.0;
    double x;
    while (i < len)
    {
        x = xin[i];
        sum0 += x * y0[i];
        sum1 += x * y1[i];
        sum2 += x * y2[i];
        i += gridDim.x * blockDim.x;
    }
    cache[tid] = sum0;
    cache[tid + THREADS_PER_BLOCK_DOT] = sum1;
    cache[tid + 2 * THREADS_PER_BLOCK_DOT] = sum2;
    for (stride = THREADS_PER_BLOCK_DOT / 2; stride > 0; stride /= 2)
    {
        __syncthreads();
        if (tid < stride)
        {
            cache[tid] += cache[tid + stride];
            cache[tid + THREADS_PER_BLOCK_DOT] += cache[tid + stride + THREADS_PER_BLOCK_DOT];
            cache[tid + 2 * THREADS_PER_BLOCK_DOT] += cache[tid + stride + 2 * THREADS_PER_BLOCK_DOT];
        }
    }
    if (tid == 0)
    {
        res[blockIdx.x] = cache[0];
        res[blockIdx.x + BLOCKS_PER_GRID_DOT] = cache[THREADS_PER_BLOCK_DOT];
        res[blockIdx.x + 2 * BLOCKS_PER_GRID_DOT] = cache[2 * THREADS_PER_BLOCK_DOT];
    }
}
__global__ void VecDot_kernel_4(double *xin, double *y0, double *y1, double *y2, double *y3, int len, double *res)
{
    __shared__ double cache[4 * THREADS_PER_BLOCK_DOT];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride;
    double sum0 = 0.0;
    double sum1 = 0.0;
    double sum2 = 0.0;
    double sum3 = 0.0;
    double x;
    while (i < len)
    {
        x = xin[i];
        sum0 += x * y0[i];
        sum1 += x * y1[i];
        sum2 += x * y2[i];
        sum3 += x * y3[i];
        i += gridDim.x * blockDim.x;
    }
    cache[tid] = sum0;
    cache[tid + THREADS_PER_BLOCK_DOT] = sum1;
    cache[tid + 2 * THREADS_PER_BLOCK_DOT] = sum2;
    cache[tid + 3 * THREADS_PER_BLOCK_DOT] = sum3;
    for (stride = THREADS_PER_BLOCK_DOT / 2; stride > 0; stride /= 2)
    {
        __syncthreads();
        if (tid < stride)
        {
            cache[tid] += cache[tid + stride];
            cache[tid + THREADS_PER_BLOCK_DOT] += cache[tid + stride + THREADS_PER_BLOCK_DOT];
            cache[tid + 2 * THREADS_PER_BLOCK_DOT] += cache[tid + stride + 2 * THREADS_PER_BLOCK_DOT];
            cache[tid + 3 * THREADS_PER_BLOCK_DOT] += cache[tid + stride + 3 * THREADS_PER_BLOCK_DOT];
        }
    }
    if (tid == 0)
    {
        res[blockIdx.x] = cache[0];
        res[blockIdx.x + BLOCKS_PER_GRID_DOT] = cache[THREADS_PER_BLOCK_DOT];
        res[blockIdx.x + 2 * BLOCKS_PER_GRID_DOT] = cache[2 * THREADS_PER_BLOCK_DOT];
        res[blockIdx.x + 3 * BLOCKS_PER_GRID_DOT] = cache[3 * THREADS_PER_BLOCK_DOT];
    }
}

void VecMdot_GPU_compare(double *xin, int len, int nv, double **y, hipStream_t *sms, double **ddotres, double **hdotres, double *lhh)
{
    int yidx = 0;
    int count = 0;
    int i, j;
    double sum = 0.0;
    while (yidx < nv)
    {
        switch (nv - yidx)
        {
        case 3:
            hipLaunchKernelGGL(VecDot_kernel_3, BLOCKS_PER_GRID_DOT, THREADS_PER_BLOCK_DOT, 0, 0, xin,
                               y[yidx], y[yidx + 1], y[yidx + 2], len, ddotres[0]);
            hipDeviceSynchronize();
            hipMemcpy(hdotres[0], ddotres[0], 3 * BLOCKS_PER_GRID_DOT * sizeof(double),
                      hipMemcpyDeviceToHost);
            for (i = 0; i < 3; i++)
            {
                sum = 0.0;
                for (j = 0; j < BLOCKS_PER_GRID_DOT; j++)
                {
                    sum += hdotres[0][i * BLOCKS_PER_GRID_DOT + j];
                }
                lhh[count + i] = sum;
            }
            count = count + 3;
            yidx += 3;
            break;

        case 2:
            hipLaunchKernelGGL(VecDot_kernel_2, BLOCKS_PER_GRID_DOT, THREADS_PER_BLOCK_DOT, 0, 0, xin,
                               y[yidx], y[yidx + 1], len, ddotres[0]);
            hipDeviceSynchronize();
            hipMemcpy(hdotres[0], ddotres[0], 2 * BLOCKS_PER_GRID_DOT * sizeof(double),
                      hipMemcpyDeviceToHost);
            for (i = 0; i < 2; i++)
            {
                sum = 0.0;
                for (j = 0; j < BLOCKS_PER_GRID_DOT; j++)
                {
                    sum += hdotres[0][i * BLOCKS_PER_GRID_DOT + j];
                }
                lhh[count + i] = sum;
            }
            count = count + 2;
            yidx += 2;
            break;
        case 1:
            hipLaunchKernelGGL(VecDot_kernel_1, BLOCKS_PER_GRID_DOT, THREADS_PER_BLOCK_DOT, 0, 0, xin,
                               y[yidx], len, ddotres[0]);
            hipDeviceSynchronize();
            hipMemcpy(hdotres[0], ddotres[0], BLOCKS_PER_GRID_DOT * sizeof(double),
                      hipMemcpyDeviceToHost);
            for (i = 0; i < 1; i++)
            {
                sum = 0.0;
                for (j = 0; j < BLOCKS_PER_GRID_DOT; j++)
                {
                    sum += hdotres[0][i * BLOCKS_PER_GRID_DOT + j];
                }
                lhh[count + i] = sum;
            }
            count = count + 1;
            yidx += 1;
            break;
        default:
            hipLaunchKernelGGL(VecDot_kernel_4, BLOCKS_PER_GRID_DOT, THREADS_PER_BLOCK_DOT, 0, 0, xin,
                               y[yidx], y[yidx + 1], y[yidx + 2], y[yidx + 3], len, ddotres[0]);
            hipDeviceSynchronize();
            hipMemcpy(hdotres[0], ddotres[0], 4 * BLOCKS_PER_GRID_DOT * sizeof(double),
                      hipMemcpyDeviceToHost);
            for (i = 0; i < 4; i++)
            {
                sum = 0.0;
                for (j = 0; j < BLOCKS_PER_GRID_DOT; j++)
                {
                    sum += hdotres[0][i * BLOCKS_PER_GRID_DOT + j];
                }
                lhh[count + i] = sum;
            }
            count = count + 4;
            yidx += 4;
            break;
        }
    }
}
void VecMdot_GPU(double *xin, int len, int nv, double **y, hipStream_t *sms, double **ddotres, double **hdotres, double *lhh)
{
    int yidx = 0;
    int count = 0;
    int lhh_st = 0;
    int last_len;
    int cur_sm, last_sm;
    double sum = 0.0;
    int i, j;
    while (yidx < nv)
    {
        switch (nv - yidx)
        {
        case 3:
            cur_sm = count % 2;
            hipLaunchKernelGGL(VecDot_kernel_3, BLOCKS_PER_GRID_DOT, THREADS_PER_BLOCK_DOT, 0, sms[cur_sm], xin,
                               y[yidx], y[yidx + 1], y[yidx + 2], len, ddotres[cur_sm]);
            hipMemcpyAsync(hdotres[cur_sm], ddotres[cur_sm], 3 * BLOCKS_PER_GRID_DOT * sizeof(double),
                           hipMemcpyDeviceToHost, sms[cur_sm]);
            if (count >= 1) // do something to overlap
            {
                // wait stream
                // compute
                last_sm = (count - 1) % 2;
                // wait for last_sm
                hipStreamSynchronize(sms[last_sm]);
                for (i = 0; i < last_len; i++)
                {
                    sum = 0.0;
                    for (j = 0; j < BLOCKS_PER_GRID_DOT; j++)
                    {
                        sum += hdotres[last_sm][i * BLOCKS_PER_GRID_DOT + j];
                    }
                    lhh[lhh_st + i] = sum;
                }
                lhh_st += last_len;
            }
            last_len = 3;
            count++;
            yidx += 3;
            break;
        case 2:
            cur_sm = count % 2;
            hipLaunchKernelGGL(VecDot_kernel_2, BLOCKS_PER_GRID_DOT, THREADS_PER_BLOCK_DOT, 0, sms[cur_sm], xin,
                               y[yidx], y[yidx + 1], len, ddotres[cur_sm]);
            hipMemcpyAsync(hdotres[cur_sm], ddotres[cur_sm], 2 * BLOCKS_PER_GRID_DOT * sizeof(double),
                           hipMemcpyDeviceToHost, sms[cur_sm]);
            if (count >= 1) // do something to overlap
            {
                last_sm = (count - 1) % 2;
                // wait for last_sm
                hipStreamSynchronize(sms[last_sm]);
                for (i = 0; i < last_len; i++)
                {
                    sum = 0.0;
                    for (j = 0; j < BLOCKS_PER_GRID_DOT; j++)
                    {
                        sum += hdotres[last_sm][i * BLOCKS_PER_GRID_DOT + j];
                    }
                    lhh[lhh_st + i] = sum;
                }
                lhh_st += last_len;
            }
            last_len = 2;
            count++;
            yidx += 2;
            break;
        case 1:
            // printf("rank=%d,before VecDot_kernel_1\n",info->rank);
            cur_sm = count % 2;
            hipLaunchKernelGGL(VecDot_kernel_1, BLOCKS_PER_GRID_DOT, THREADS_PER_BLOCK_DOT, 0, sms[cur_sm], xin,
                               y[yidx], len, ddotres[cur_sm]);
            hipMemcpyAsync(hdotres[cur_sm], ddotres[cur_sm], BLOCKS_PER_GRID_DOT * sizeof(double),
                           hipMemcpyDeviceToHost, sms[cur_sm]);

            // printf("rank=%d,after VecDot_kernel_1\n",info->rank);
            if (count >= 1)
            {
                last_sm = (count - 1) % 2;
                // wait for last_sm
                hipStreamSynchronize(sms[last_sm]);
                for (i = 0; i < last_len; i++)
                {
                    sum = 0.0;
                    for (j = 0; j < BLOCKS_PER_GRID_DOT; j++)
                    {
                        sum += hdotres[last_sm][i * BLOCKS_PER_GRID_DOT + j];
                    }
                    lhh[lhh_st + i] = sum;
                }
                lhh_st += last_len;
            }
            last_len = 1;
            count++;
            yidx += 1;
            break;
        default:
            cur_sm = count % 2;
            hipLaunchKernelGGL(VecDot_kernel_4, BLOCKS_PER_GRID_DOT, THREADS_PER_BLOCK_DOT, 0, sms[cur_sm], xin,
                               y[yidx], y[yidx + 1], y[yidx + 2], y[yidx + 3], len, ddotres[cur_sm]);
            hipMemcpyAsync(hdotres[cur_sm], ddotres[cur_sm], 4 * BLOCKS_PER_GRID_DOT * sizeof(double),
                           hipMemcpyDeviceToHost, sms[cur_sm]);
            if (count >= 1)
            {
                last_sm = (count - 1) % 2;
                // wait for last_sm
                hipStreamSynchronize(sms[last_sm]);
                for (i = 0; i < last_len; i++)
                {
                    sum = 0.0;
                    for (j = 0; j < BLOCKS_PER_GRID_DOT; j++)
                    {
                        sum += hdotres[last_sm][i * BLOCKS_PER_GRID_DOT + j];
                    }
                    lhh[lhh_st + i] = sum;
                }
                lhh_st += last_len;
            }
            last_len = 4;
            count++;
            yidx += 4;
            break;
        }
    }
    hipDeviceSynchronize();
    // we need to handle the last one
    count--;
    last_sm = count % 2;
    for (i = 0; i < last_len; i++)
    {
        sum = 0.0;
        for (j = 0; j < BLOCKS_PER_GRID_DOT; j++)
        {
            sum += hdotres[last_sm][i * BLOCKS_PER_GRID_DOT + j];
        }
        lhh[lhh_st + i] = sum;
    }
    // printf("rank=%d,after devicesynchronize\n",info->rank);
}

__global__ void VecAXPY_kernel_1(double *y, int len, double a1, double *x)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    while (tid < len)
    {
        y[tid] += a1 * x[tid];
        tid += gridDim.x * blockDim.x;
    }
}
__global__ void VecAXPY_kernel_2(double *y, int len, double a1, double a2, double *x1, double *x2)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    while (tid < len)
    {
        y[tid] += a1 * x1[tid] + a2 * x2[tid];
        tid += gridDim.x * blockDim.x;
    }
}
__global__ void VecAXPY_kernel_3(double *y, int len, double a1, double a2, double a3, double *x1,
                                 double *x2, double *x3)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    while (tid < len)
    {
        y[tid] += a1 * x1[tid] + a2 * x2[tid] + a3 * x3[tid];
        tid += gridDim.x * blockDim.x;
    }
}
__global__ void VecAXPY_kernel_4(double *y, int len, double a1, double a2, double a3, double a4,
                                 double *x1, double *x2, double *x3, double *x4)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    while (tid < len)
    {
        y[tid] += a1 * x1[tid] + a2 * x2[tid] + a3 * x3[tid] + a4 * x4[tid];
        tid += gridDim.x * blockDim.x;
    }
}
__global__ void VecAXPY_kernel_5(double *y, int len, double a1, double a2, double a3, double a4, double a5,
                                 double *x1, double *x2, double *x3, double *x4, double *x5)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    while (tid < len)
    {
        y[tid] += a1 * x1[tid] + a2 * x2[tid] + a3 * x3[tid] + a4 * x4[tid] + a5 * x5[tid];
        tid += gridDim.x * blockDim.x;
    }
}
__global__ void VecAXPY_kernel_6(double *y, int len, double a1, double a2, double a3, double a4, double a5, double a6,
                                 double *x1, double *x2, double *x3, double *x4, double *x5, double *x6)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    while (tid < len)
    {
        y[tid] += a1 * x1[tid] + a2 * x2[tid] + a3 * x3[tid] + a4 * x4[tid] + a5 * x5[tid] + a6 * x6[tid];
        tid += gridDim.x * blockDim.x;
    }
}
__global__ void VecAXPY_kernel_7(double *y, int len, double a1, double a2, double a3, double a4, double a5, double a6, double a7,
                                 double *x1, double *x2, double *x3, double *x4, double *x5, double *x6, double *x7)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    while (tid < len)
    {
        y[tid] += a1 * x1[tid] + a2 * x2[tid] + a3 * x3[tid] + a4 * x4[tid] + a5 * x5[tid] + a6 * x6[tid] + a7 * x7[tid];
        tid += gridDim.x * blockDim.x;
    }
}
__global__ void VecAXPY_kernel_8(double *y, int len, double a1, double a2, double a3, double a4, double a5, double a6, double a7, double a8,
                                 double *x1, double *x2, double *x3, double *x4, double *x5, double *x6, double *x7, double *x8)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    while (tid < len)
    {
        y[tid] += a1 * x1[tid] + a2 * x2[tid] + a3 * x3[tid] + a4 * x4[tid] + a5 * x5[tid] + a6 * x6[tid] + a7 * x7[tid] + a8 * x8[tid];
        tid += gridDim.x * blockDim.x;
    }
}
// y=alpha*x1+ alpha2 * x2 + alpha3 * x3 +...... +y
void VecMAXPY_GPU(double *y, int len, double *alpha, double **x, int nv)
{
    // int nblocks=(len-1)/THREADS_PER_BLOCK_VX+1;
    // int nblocks=len/THREADS_PER_BLOCK_VX+1;
    int nblocks = 256;
    int idx = 0;
    int j_rem, j;
    // switch(j_rem=nv & 3)
    switch (j_rem = nv & 7)
    {
    case 7:
        hipLaunchKernelGGL(VecAXPY_kernel_7, nblocks, THREADS_PER_BLOCK_VX, 0, 0, y, len, alpha[idx], alpha[idx + 1],
                           alpha[idx + 2], alpha[idx + 3], alpha[idx + 4], alpha[idx + 5], alpha[idx + 6],
                           x[idx], x[idx + 1], x[idx + 2], x[idx + 3], x[idx + 4], x[idx + 5], x[idx + 6]);
        hipDeviceSynchronize();
        idx += 7;
        break;
    case 6:
        hipLaunchKernelGGL(VecAXPY_kernel_6, nblocks, THREADS_PER_BLOCK_VX, 0, 0, y, len, alpha[idx], alpha[idx + 1],
                           alpha[idx + 2], alpha[idx + 3], alpha[idx + 4], alpha[idx + 5], x[idx], x[idx + 1], x[idx + 2], x[idx + 3], x[idx + 4], x[idx + 5]);
        hipDeviceSynchronize();
        idx += 6;
        break;
    case 5:
        hipLaunchKernelGGL(VecAXPY_kernel_5, nblocks, THREADS_PER_BLOCK_VX, 0, 0, y, len, alpha[idx], alpha[idx + 1],
                           alpha[idx + 2], alpha[idx + 3], alpha[idx + 4], x[idx], x[idx + 1], x[idx + 2], x[idx + 3], x[idx + 4]);
        hipDeviceSynchronize();
        idx += 5;
        break;
    case 4:
        hipLaunchKernelGGL(VecAXPY_kernel_4, nblocks, THREADS_PER_BLOCK_VX, 0, 0, y, len, alpha[idx], alpha[idx + 1],
                           alpha[idx + 2], alpha[idx + 3], x[idx], x[idx + 1], x[idx + 2], x[idx + 3]);
        hipDeviceSynchronize();
        idx += 4;
        break;
    case 3:
        hipLaunchKernelGGL(VecAXPY_kernel_3, nblocks, THREADS_PER_BLOCK_VX, 0, 0, y, len, alpha[idx], alpha[idx + 1],
                           alpha[idx + 2], x[idx], x[idx + 1], x[idx + 2]);
        hipDeviceSynchronize();
        idx += 3;
        break;
    case 2:
        hipLaunchKernelGGL(VecAXPY_kernel_2, nblocks, THREADS_PER_BLOCK_VX, 0, 0, y, len, alpha[idx], alpha[idx + 1],
                           x[idx], x[idx + 1]);
        hipDeviceSynchronize();
        idx += 2;
        break;
    case 1:
        hipLaunchKernelGGL(VecAXPY_kernel_1, nblocks, THREADS_PER_BLOCK_VX, 0, 0, y, len, alpha[idx], x[idx]);
        hipDeviceSynchronize();
        idx += 1;
        break;
    }
    // for(j=j_rem;j<nv;j+=4)
    //{
    //	hipLaunchKernelGGL(VecAXPY_kernel_4, nblocks, THREADS_PER_BLOCK_VX, 0, 0, y,len,alpha[idx],alpha[idx+1],
    //	alpha[idx+2],alpha[idx+3],x[idx],x[idx+1],x[idx+2],x[idx+3]);
    //	hipDeviceSynchronize();
    //	idx+=4;
    // }
    for (j = j_rem; j < nv; j += 8)
    {
        hipLaunchKernelGGL(VecAXPY_kernel_8, nblocks, THREADS_PER_BLOCK_VX, 0, 0, y, len, alpha[idx], alpha[idx + 1],
                           alpha[idx + 2], alpha[idx + 3], alpha[idx + 4], alpha[idx + 5], alpha[idx + 6], alpha[idx + 7],
                           x[idx], x[idx + 1], x[idx + 2], x[idx + 3], x[idx + 4], x[idx + 5], x[idx + 6], x[idx + 7]);
        hipDeviceSynchronize();
        idx += 8;
    }
}
// Ax=y
__global__ void Ax_GPU(int *row_ptr, int *col_idx, double *Aval, int n, double *x, double *y)
{
    __shared__ double s_out[THREADS_PER_BLOCK_AX];
    int t_idx = threadIdx.x;
    int target_block_row = (threadIdx.x + blockDim.x * blockIdx.x) / 32;
    int lane;
    int first_block;
    int last_block;
    int target_block;
    int c;
    int r;
    int col;
    int stride;
    double local_out;
    double x_elem;
    double A_elem;
    s_out[t_idx] = 0.0;
    if (target_block_row < n)
    {
        lane = t_idx % 32;
        first_block = row_ptr[target_block_row];
        last_block = row_ptr[target_block_row + 1];
        target_block = first_block + lane / (NVARS * NVARS);
        c = (lane / NVARS) % NVARS;
        r = lane % NVARS;
        if (lane < (32 / (NVARS * NVARS)) * (NVARS * NVARS))
        {
            local_out = 0.0;
            for (; target_block < last_block; target_block += 32 / (NVARS * NVARS))
            {
                col = col_idx[target_block];
                x_elem = x[col * NVARS + c];
                A_elem = Aval[target_block * NVARS * NVARS + c * NVARS + r];
                local_out += x_elem * A_elem;
            }
            s_out[t_idx] = local_out;
            stride = 8; // only for NVARS=3 cased
            for (; stride >= 1; stride /= 2)
            {
                if (lane < stride * NVARS && lane + stride * NVARS < 32)
                {
                    s_out[t_idx] += s_out[t_idx + stride * NVARS];
                }
            }
            if (lane < NVARS)
            {
                y[target_block_row * NVARS + lane] = s_out[t_idx];
            }
        }
    }
}
__global__ void SetSendbuffer(double *inarray, int ar_len, int *indices, double *dsend_buf, int buf_len, int bs)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    // int idx=tid/NVARS;
    // int offset=tid%NVARS;
    int idx = tid / bs;
    int offset = tid % bs;
    int xidx;
    if (tid < buf_len)
    {
        xidx = indices[idx] + offset;
        dsend_buf[tid] = inarray[xidx];
    }
}

// parallel matrix vector multiplication  communications
void AxStartComm()
{
    int i;
    int snp = info->snp;
    int rnp = info->rnp;
    int bs = info->bs;
    int tag = info->rank;
    //	printf("before send snp=%d,rnp=%d,bs=%d\n",snp,rnp,bs);
    //      GPU DIRECT
    /*	for(i=0;i<snp;i++)
        {
            MPI_Isend(info->dsend_buf+bs*info->sstarts[i],bs*(info->sstarts[i+1]-info->sstarts[i]),
            MPI_DOUBLE,info->sprocs[i],tag,MPI_COMM_WORLD,info->swaits+i);
                  // printf("rank %d is sending %d pack to rank %d, bs=%d, info->sstarts[i+1]=%d,info->sstarts[i]=%d\n",info->rank,bs*(info->sstarts[i+1]-info->sstarts[i]),info->sprocs[i],bs,info->sstarts[i+1],info->sstarts[i]);
        }
        for(i=0;i<rnp;i++)
        {
            MPI_Irecv(info->dlvec+bs*info->rstarts[i],bs*(info->rstarts[i+1]-info->rstarts[i]),
            MPI_DOUBLE,info->rprocs[i],info->rprocs[i],MPI_COMM_WORLD,info->rwaits+i);
        //	printf("rank %d is receiving %d pack from rank %d\n",info->rank,bs*(info->rstarts[i+1]-info->rstarts[i]),info->rprocs[i]);
        }
    */
    for (i = 0; i < snp; i++)
    {
        MPI_Isend(info->hsend_buf + bs * info->sstarts[i], bs * (info->sstarts[i + 1] - info->sstarts[i]),
                  MPI_DOUBLE, info->sprocs[i], tag, MPI_COMM_WORLD, info->swaits + i);
        // printf("rank %d is sending %d pack to rank %d, bs=%d, info->sstarts[i+1]=%d,info->sstarts[i]=%d\n",info->rank,bs*(info->sstarts[i+1]-info->sstarts[i]),info->sprocs[i],bs,info->sstarts[i+1],info->sstarts[i]);
    }
    for (i = 0; i < rnp; i++)
    {
        MPI_Irecv(info->hlvec + bs * info->rstarts[i], bs * (info->rstarts[i + 1] - info->rstarts[i]),
                  MPI_DOUBLE, info->rprocs[i], info->rprocs[i], MPI_COMM_WORLD, info->rwaits + i);
        //	printf("rank %d is receiving %d pack from rank %d\n",info->rank,bs*(info->rstarts[i+1]-info->rstarts[i]),info->rprocs[i]);
    }
}

void AxEndComm()
{
    // here we need to wait until all MPI communication has completed
    MPI_Status stat;
    int i;
    for (i = 0; i < info->rnp; i++)
    {
        // printf("before MPI_Wait: rank=%d, sendbuf_len=%d,lvsz=%d\n",info->rank,info->buf_len,info->lvsz);
        MPI_Wait(info->rwaits + i, &stat);
        // printf("wait successfully, rank=%d\n",info->rank);
    }
}

void CreateGMRES_INFO()
{
    if (!info)
    {
        info = (GMRES_INFO *)malloc(sizeof(GMRES_INFO));
        memset(info, 0, sizeof(GMRES_INFO));
    }
    // printf("CreateGMRES_INFO successfully\n");
}

void InitialGMRES_INFO_GPU()
{
    // This subroutine must be called after InitialGMRES_INFO_CPU!
    if (!info->init_gpu_called)
    {
        hipSetDevice(info->idev);
    }
    PetscInt main_nnz, off_nnz, bs;
    PetscInt i;
    PetscReal *hsol = NULL;
    const PetscReal *hrhs = NULL;
    if (!info->hMainRowPtr || !info->hMainColVal || !info->hMainBlkVal || !info->hOffRowPtr || !info->hOffColVal || !info->hOffBlkVal)
    {
        //	printf("ERROR:host pointers for local main-diag or off-diag matrix is NULL!\n");
        //	return;
        printf("WARNING:host pointers for local main-diag or off-diag matrix is NULL!\n");
    }
    main_nnz = info->main_nnz;
    off_nnz = info->off_nnz;
    bs = info->bs;

    // allocate GPU memory
    if (!info->dMainRowPtr)
    {
        hipMalloc((void **)&info->dMainRowPtr, sizeof(PetscInt) * (info->main_nrows + 1));
    }
    if (!info->dMainColVal)
    {
        hipMalloc((void **)&info->dMainColVal, sizeof(PetscInt) * main_nnz);
    }
    if (!info->dMainBlkVal)
    {
        hipMalloc((void **)&info->dMainBlkVal, sizeof(PetscReal) * main_nnz * bs * bs);
    }
    if (!info->dOffRowPtr)
    {
        hipMalloc((void **)&info->dOffRowPtr, sizeof(PetscInt) * (info->off_nrows + 1));
    }
    if (!info->dOffColVal)
    {
        hipMalloc((void **)&info->dOffColVal, sizeof(PetscInt) * off_nnz);
    }
    if (!info->dOffBlkVal)
    {
        hipMalloc((void **)&info->dOffBlkVal, sizeof(PetscReal) * off_nnz * bs * bs);
    }

    // copy host data to GPU:  KSP case: copy only once, SNES case: Amat is jacobi matrix, need to copy every time
    // memory copies for mainRowPtr, MainColVal, MainBlkVal, OffRowPtr, OffColVal, LRowPtr, LColVal, LBlkVal, URowPtr, UColVal, UBlkVal
    if (info->dMainRowPtr)
    {
        hipMemcpy(info->dMainRowPtr, info->hMainRowPtr, sizeof(PetscInt) * (info->main_nrows + 1), hipMemcpyHostToDevice);
    }
    if (info->dMainColVal)
    {
        hipMemcpy(info->dMainColVal, info->hMainColVal, sizeof(PetscInt) * main_nnz, hipMemcpyHostToDevice);
    }
    if (info->dMainBlkVal)
    {
        hipMemcpy(info->dMainBlkVal, info->hMainBlkVal, sizeof(PetscReal) * main_nnz * bs * bs, hipMemcpyHostToDevice);
    }
    if (info->dOffRowPtr)
    {
        hipMemcpy(info->dOffRowPtr, info->hOffRowPtr, sizeof(PetscInt) * (info->off_nrows + 1), hipMemcpyHostToDevice);
    }
    if (info->dOffColVal)
    {
        hipMemcpy(info->dOffColVal, info->hOffColVal, sizeof(PetscInt) * off_nnz, hipMemcpyHostToDevice);
    }
    if (info->dOffBlkVal)
    {
        hipMemcpy(info->dOffBlkVal, info->hOffBlkVal, sizeof(PetscReal) * off_nnz * bs * bs, hipMemcpyHostToDevice);
    }

    if (!info->dvv)
    {
        info->dvv = (PetscReal **)malloc(info->vvdim * sizeof(PetscReal *));
        for (i = 0; i < info->vvdim; i++)
        {
            hipMalloc((void **)&info->dvv[i], sizeof(PetscReal) * info->vsz);
        }
    }
    if (!info->drhs)
    {
        hipMalloc((void **)&info->drhs, sizeof(PetscReal) * info->vsz);
    }
    if (!info->dsol)
    {
        hipMalloc((void **)&info->dsol, sizeof(PetscReal) * info->vsz);
    }
    //
    if (!info->dsindices)
    {
        hipMalloc((void **)&info->dsindices, sizeof(PetscInt) * info->indices_len);
    }
    if (!info->dsend_buf)
    {
        hipMalloc((void **)&info->dsend_buf, sizeof(PetscReal) * info->buf_len);
    }
    if (!info->dlvec)
    {
        hipMalloc((void **)&info->dlvec, sizeof(PetscReal) * info->lvsz);
    }

    if (info->use_asm)
    {
        // we add ASM structure on GPUs
        if (!info->asm_dsindices)
        {
            hipMalloc((void **)&info->asm_dsindices, sizeof(PetscInt) * info->asm_sindices_len);
        }
        if (!info->asm_drindices)
        {
            hipMalloc((void **)&info->asm_drindices, sizeof(PetscInt) * info->asm_rindices_len);
        }
        if (!info->asm_self_dsindices)
        {
            hipMalloc((void **)&info->asm_self_dsindices, sizeof(PetscInt) * info->asm_self_sindices_len);
        }
        if (!info->asm_self_drindices)
        {
            hipMalloc((void **)&info->asm_self_drindices, sizeof(PetscInt) * info->asm_self_rindices_len);
        }
        if (!info->asm_dsend_buf)
        {
            hipMalloc((void **)&info->asm_dsend_buf, sizeof(PetscReal) * info->asm_sendbuf_len);
        }
        if (!info->asm_drecv_buf)
        {
            hipMalloc((void **)&info->asm_drecv_buf, sizeof(PetscReal) * info->asm_recvbuf_len);
        }
        if (!info->asm_dlx)
        {
            hipMalloc((void **)&info->asm_dlx, sizeof(PetscReal) * info->asm_lxsz);
        }
        if (!info->asm_dly)
        {
            hipMalloc((void **)&info->asm_dly, sizeof(PetscReal) * info->asm_lxsz);
        }
        if (!info->asm_dltmp)
        {
            hipMalloc((void **)&info->asm_dltmp, sizeof(PetscReal) * info->asm_lxsz);
        }
    }

    // pure GPU implementations, allocate memory for dlhh
    if (!info->dlhh)
    {
        hipMalloc((void **)&info->dlhh, sizeof(PetscReal) * 32);
    }
    //
    if (!info->hdotres[0])
    {
        hipMallocHost((void **)&info->hdotres[0], sizeof(PetscReal) * 4 * BLOCKS_PER_GRID_DOT);
    }
    if (!info->hdotres[1])
    {
        hipMallocHost((void **)&info->hdotres[1], sizeof(PetscReal) * 4 * BLOCKS_PER_GRID_DOT);
    }
    if (!info->ddotres[0])
    {
        hipMalloc((void **)&info->ddotres[0], sizeof(PetscReal) * 4 * BLOCKS_PER_GRID_DOT);
    }
    if (!info->ddotres[1])
    {
        hipMalloc((void **)&info->ddotres[1], sizeof(PetscReal) * 4 * BLOCKS_PER_GRID_DOT);
    }

    if (info->dsol)
    {
        VecGetArray(info->sol, &hsol);
        hipMemcpy(info->dsol, hsol, sizeof(PetscReal) * info->vsz, hipMemcpyHostToDevice);
        VecRestoreArray(info->sol, &hsol);
    }
    if (info->drhs)
    {
        VecGetArrayRead(info->rhs, &hrhs);
        // for(i=0;i<info->vsz;i++)
        //{
        //	printf("rank=%d,rhs[%d]=%20.15lf\n",info->rank,i,hrhs[i]);
        // }

        hipMemcpy(info->drhs, hrhs, sizeof(PetscReal) * info->vsz, hipMemcpyHostToDevice);
        VecRestoreArrayRead(info->rhs, &hrhs);
    }
    if (info->dsindices && !info->init_gpu_called)
    {
        hipMemcpy(info->dsindices, info->sindices, sizeof(PetscInt) * info->indices_len, hipMemcpyHostToDevice);
    }

    // if(USE_ASM && !info->init_gpu_called)
    if (info->use_asm && !info->init_gpu_called)
    {
        if (info->asm_dsindices)
        {
            hipMemcpy(info->asm_dsindices, info->asm_sindices, sizeof(PetscInt) * info->asm_sindices_len, hipMemcpyHostToDevice);
        }
        if (info->asm_drindices)
        {
            hipMemcpy(info->asm_drindices, info->asm_rindices, sizeof(PetscInt) * info->asm_rindices_len, hipMemcpyHostToDevice);
        }
        if (info->asm_self_dsindices)
        {
            hipMemcpy(info->asm_self_dsindices, info->asm_self_sindices, sizeof(PetscInt) * info->asm_self_sindices_len, hipMemcpyHostToDevice);
        }
        if (info->asm_self_drindices)
        {
            hipMemcpy(info->asm_self_drindices, info->asm_self_rindices, sizeof(PetscInt) * info->asm_self_rindices_len, hipMemcpyHostToDevice);
        }
    }

    if (!info->init_gpu_called)
    {
        for (i = 0; i < 3; i++)
        {
            hipStreamCreate(&info->mystream[i]);
        }
        info->cublas_stat = hipblasCreate(&info->cublas_handle);
        if (info->cublas_stat != HIPBLAS_STATUS_SUCCESS)
        {
            printf("ERROR in hipblasCreate in rank=%d\n", info->rank);
        }
        info->cusparse_stat = hipsparseCreate(&info->cusparse_handle);
        if (info->cusparse_stat != HIPSPARSE_STATUS_SUCCESS)
        {
            printf("ERROR in hipsparseCreate in rank=%d\n", info->rank);
        }
        info->cusparse_stat = hipsparseCreateMatDescr(&info->descr);
        if (info->cusparse_stat != HIPSPARSE_STATUS_SUCCESS)
        {
            printf("ERROR in hipsparseCreateMatDescr in rank=%d\n", info->rank);
        }
        hipsparseSetMatIndexBase(info->descr, HIPSPARSE_INDEX_BASE_ZERO);
        hipsparseSetMatType(info->descr, HIPSPARSE_MATRIX_TYPE_GENERAL);
        info->dir = HIPSPARSE_DIRECTION_COLUMN;
        info->trans = HIPSPARSE_OPERATION_NON_TRANSPOSE;
    }
    // here maybe we need to create different streams in order to achieve better performance
    if (!info->dv_tmp)
    {
        hipMalloc((void **)&info->dv_tmp, sizeof(PetscReal) * info->vsz);
    }

    info->init_gpu_called = 1;
}

void DestroyGMRES_INFO()
{
    // we only deallocate memory on GPUs, and the memory on CPU are just managed by PETSc.
    PetscInt i;
    if (info->dMainRowPtr)
    {
        hipFree(info->dMainRowPtr);
        info->dMainRowPtr = NULL;
    }
    if (info->dMainColVal)
    {
        hipFree(info->dMainColVal);
        info->dMainColVal = NULL;
    }
    if (info->dMainBlkVal)
    {
        hipFree(info->dMainBlkVal);
        info->dMainBlkVal = NULL;
    }
    if (info->dOffRowPtr)
    {
        hipFree(info->dOffRowPtr);
        info->dOffRowPtr = NULL;
    }
    if (info->dOffColVal)
    {
        hipFree(info->dOffColVal);
        info->dOffColVal = NULL;
    }
    if (info->dOffBlkVal)
    {
        hipFree(info->dOffBlkVal);
        info->dOffBlkVal = NULL;
    }
    //   FREE MEMROY FOR　ILU(k) factorization
    // L and U on host
    if (info->hLRowPtr)
    {
        free(info->hLRowPtr);
        info->hLRowPtr = NULL;
    }
    if (info->hLColVal)
    {
        free(info->hLColVal);
        info->hLColVal = NULL;
    }
    if (info->hLBlkVal)
    {
        free(info->hLBlkVal);
        info->hLBlkVal = NULL;
    }
    if (info->hURowPtr)
    {
        free(info->hURowPtr);
        info->hURowPtr = NULL;
    }
    if (info->hUColVal)
    {
        free(info->hUColVal);
        info->hUColVal = NULL;
    }
    if (info->hUBlkVal)
    {
        free(info->hUBlkVal);
        info->hUBlkVal = NULL;
    }
    // L and U on Device
    if (info->dLRowPtr)
    {
        hipFree(info->dLRowPtr);
        info->dLRowPtr = NULL;
    }
    if (info->dLColVal)
    {
        hipFree(info->dLColVal);
        info->dLColVal = NULL;
    }
    if (info->dLBlkVal)
    {
        hipFree(info->dLBlkVal);
        info->dLBlkVal = NULL;
    }
    if (info->dURowPtr)
    {
        hipFree(info->dURowPtr);
        info->dURowPtr = NULL;
    }
    if (info->dUColVal)
    {
        hipFree(info->dUColVal);
        info->dUColVal = NULL;
    }
    if (info->dUBlkVal)
    {
        hipFree(info->dUBlkVal);
        info->dUBlkVal = NULL;
    }

    if (info->drhs)
    {
        hipFree(info->drhs);
        info->drhs = NULL;
    }
    if (info->dsol)
    {
        hipFree(info->dsol);
        info->dsol = NULL;
    }
    if (info->dsindices)
    {
        hipFree(info->dsindices);
        info->dsindices = NULL;
    }
    if (info->dsend_buf)
    {
        hipFree(info->dsend_buf);
        info->dsend_buf = NULL;
    }
    if (info->dlvec)
    {
        hipFree(info->dlvec);
        info->dlvec = NULL;
    }
    if (info->hsend_buf)
    {
        free(info->hsend_buf);
        info->hsend_buf = NULL;
    }
    if (info->hlvec)
    {
        free(info->hlvec);
        info->hlvec = NULL;
    }
    if (info->dvv)
    {
        for (i = 0; i < info->vvdim; i++)
        {
            hipFree(info->dvv[i]);
        }
        free(info->dvv);
        info->dvv = NULL;
    }
    if (info->dlhh)
    {
        hipFree(info->dlhh);
        info->dlhh = NULL;
    }
    if (info->hdotres)
    {
        hipFree(info->hdotres[0]);
        hipFree(info->hdotres[1]);
    }
    if (info->ddotres)
    {
        hipFree(info->ddotres[0]);
        hipFree(info->ddotres[1]);
    }
    for (i = 0; i < 3; i++)
    {
        if (info->mystream[i])
        {
            hipStreamDestroy(info->mystream[i]);
        }
    }
    if (info->descr)
    {
        hipsparseDestroyMatDescr(info->descr);
    }
    if (info->cublas_handle)
    {
        hipblasDestroy(info->cublas_handle);
    }
    if (info->cusparse_handle)
    {
        hipsparseDestroy(info->cusparse_handle);
    }

    // for bsrilu from cusparse
    if (info->dv_tmp)
    {
        hipFree(info->dv_tmp);
        info->dv_tmp = NULL;
    }
    // hipDeviceReset();
    if (info->sprocs)
    {
        free(info->sprocs);
        info->sprocs = NULL;
    }
    if (info->rprocs)
    {
        free(info->rprocs);
        info->rprocs = NULL;
    }
    if (info->sstarts)
    {
        free(info->sstarts);
        info->sstarts = NULL;
    }
    if (info->rstarts)
    {
        free(info->rstarts);
        info->rstarts = NULL;
    }
    if (info->sindices)
    {
        free(info->sindices);
        info->sindices = NULL;
    }
    if (info->rindices)
    {
        free(info->rindices);
        info->rindices = NULL;
    }
    if (info->swaits)
    {
        free(info->swaits);
        info->swaits = NULL;
    }
    if (info->rwaits)
    {
        free(info->rwaits);
        info->rwaits = NULL;
    }

    // free memoeries for ASM
    // if(USE_ASM)
    if (info->use_asm)
    {
        if (info->asm_sprocs)
        {
            free(info->asm_sprocs);
            info->asm_sprocs = NULL;
        }
        if (info->asm_rprocs)
        {
            free(info->asm_rprocs);
            info->asm_rprocs = NULL;
        }
        if (info->asm_sstarts)
        {
            free(info->asm_sstarts);
            info->asm_sstarts = NULL;
        }
        if (info->asm_rstarts)
        {
            free(info->asm_rstarts);
            info->asm_rstarts = NULL;
        }
        if (info->asm_swaits)
        {
            free(info->asm_swaits);
            info->asm_swaits = NULL;
        }
        if (info->asm_rwaits)
        {
            free(info->asm_rwaits);
            info->asm_rwaits = NULL;
        }
        if (info->asm_sindices)
        {
            free(info->asm_sindices);
            info->asm_sindices = NULL;
        }
        if (info->asm_rindices)
        {
            free(info->asm_rindices);
            info->asm_rindices = NULL;
        }
        if (info->asm_self_sindices)
        {
            free(info->asm_self_sindices);
            info->asm_self_sindices = NULL;
        }
        if (info->asm_self_rindices)
        {
            free(info->asm_self_rindices);
            info->asm_self_rindices = NULL;
        }

        if (!info->asm_vecx)
        {
            free(info->asm_vecx);
            info->asm_vecx = NULL;
        }
        if (!info->asm_vecy)
        {
            free(info->asm_vecy);
            info->asm_vecy = NULL;
        }
        if (!info->asm_send_buf)
        {
            free(info->asm_send_buf);
            info->asm_send_buf = NULL;
        }
        if (!info->asm_recv_buf)
        {
            free(info->asm_recv_buf);
            info->asm_recv_buf = NULL;
        }
        if (!info->asm_lx)
        {
            free(info->asm_lx);
            info->asm_lx = NULL;
        }
        if (!info->asm_ly)
        {
            free(info->asm_ly);
            info->asm_ly = NULL;
        }

        if (info->asm_dsindices)
        {
            hipFree(info->asm_dsindices);
            info->asm_dsindices = NULL;
        }
        if (info->asm_drindices)
        {
            hipFree(info->asm_drindices);
            info->asm_drindices = NULL;
        }
        if (info->asm_self_dsindices)
        {
            hipFree(info->asm_self_dsindices);
            info->asm_self_dsindices = NULL;
        }
        if (info->asm_self_drindices)
        {
            hipFree(info->asm_self_drindices);
            info->asm_self_drindices = NULL;
        }
        if (info->asm_dsend_buf)
        {
            hipFree(info->asm_dsend_buf);
            info->asm_dsend_buf = NULL;
        }
        if (info->asm_drecv_buf)
        {
            hipFree(info->asm_drecv_buf);
            info->asm_drecv_buf = NULL;
        }
        if (info->asm_dlx)
        {
            hipFree(info->asm_dlx);
            info->asm_dlx = NULL;
        }
        if (info->asm_dly)
        {
            hipFree(info->asm_dly);
            info->asm_dly = NULL;
        }
        if (info->asm_dltmp)
        {
            hipFree(info->asm_dltmp);
            info->asm_dltmp = NULL;
        }
    }

    free(info);
    // printf("DestroyGMRES_INFO succefully\n");
}

PetscErrorCode InitialGMRES_INFO_CPU(KSP ksp)
{
    PetscErrorCode ierr;
    KSP_GMRES *gmres = (KSP_GMRES *)ksp->data;
    Mat Amat, Pmat;
    // Mat		subAmat,subPmat;  // depending on preconditioners (ASM or BJacobi)
    Mat_MPIBAIJ *mpibaij;
    Mat_SeqBAIJ *mainbaij, *offbaij;
    // Mat_SeqBAIJ	*factbaij;
    //  get rank and idev
    char hostname[MPI_MAX_PROCESSOR_NAME];
    PetscInt namelen;
    MPI_Comm sub_comm;
    //  Debugging variables
    PetscInt rank;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    info->rank = rank;
    PetscInt dbpt = 0; // debugprinting
                       //  Debugging variables
    if (!rank && dbpt)
    {
        printf("rank=%d,entering InitialGMRES_CPU\n", rank);
    }
    PetscFunctionBegin;

    ierr = KSPGetOperators(ksp, &Amat, &Pmat);
    CHKERRQ(ierr);
    info->bs = Amat->rmap->bs;
    if (!rank && dbpt)
    {
        printf("rank=%d,info->bs=%d\n", rank, info->bs);
    } // debugging

    mpibaij = (Mat_MPIBAIJ *)Amat->data;
    mainbaij = (Mat_SeqBAIJ *)mpibaij->A->data;
    offbaij = (Mat_SeqBAIJ *)mpibaij->B->data;
    info->main_nrows = mainbaij->mbs;
    if (!rank && dbpt)
    {
        printf("rank=%d,info->main_nrows=%d\n", rank, info->main_nrows);
    } // debugging
    info->main_ncols = mainbaij->nbs;
    if (!rank && dbpt)
    {
        printf("rank=%d,info->main_ncols=%d\n", rank, info->main_ncols);
    } // debugging
    info->off_nrows = offbaij->mbs;
    if (!rank && dbpt)
    {
        printf("rank=%d,info->off_nrows=%d\n", rank, info->off_nrows);
    } // debugging
    info->off_ncols = offbaij->nbs;
    if (!rank && dbpt)
    {
        printf("rank=%d,info->off_ncols=%d\n", rank, info->off_ncols);
    } // debugging

    info->hMainRowPtr = mainbaij->i;
    info->hMainColVal = mainbaij->j;
    info->hMainBlkVal = mainbaij->a;
    info->hOffRowPtr = offbaij->i;
    info->hOffColVal = offbaij->j;
    info->hOffBlkVal = offbaij->a;

    info->main_nnz = info->hMainRowPtr[info->main_nrows];
    info->off_nnz = info->hOffRowPtr[info->off_nrows];
    if (!rank && dbpt)
    {
        printf("rank=%d,info->main_nnz=%d\n", rank, info->main_nnz);
    } // debugging
    if (!rank && dbpt)
    {
        printf("rank=%d,info->off_nnz=%d\n", rank, info->off_nnz);
    } // debugging

    // Get Subpc fact_nrows depends on the used preconditioners
    pre_ilu(ksp);

    // SPMV
    info->Mvctx = mpibaij->Mvctx;
    info->lvec = mpibaij->lvec;
    ierr = VecGetLocalSize(info->lvec, &info->lvsz);
    CHKERRQ(ierr);

    // printf("rank=%d,lvsz=%d\n",info->rank,info->lvsz);
    if (!rank && dbpt)
    {
        printf("rank=%d,info->lvsz=%d\n", rank, info->lvsz);
    } // debugging

    VecScatter_MPI_General *to = (VecScatter_MPI_General *)info->Mvctx->todata;
    VecScatter_MPI_General *from = (VecScatter_MPI_General *)info->Mvctx->fromdata;

    info->snp = to->n;
    info->rnp = from->n;
    // info->sprocs=to->procs;  info->rprocs=from->procs;

    if (!info->sprocs)
    {
        info->sprocs = (PetscMPIInt *)malloc(info->snp * sizeof(PetscMPIInt));
        memcpy(info->sprocs, to->procs, info->snp * sizeof(PetscMPIInt));
    }
    if (!info->rprocs)
    {
        info->rprocs = (PetscMPIInt *)malloc(info->rnp * sizeof(PetscMPIInt));
        memcpy(info->rprocs, from->procs, info->rnp * sizeof(PetscMPIInt));
    }

    // info->sstarts=to->starts; info->rstarts=from->starts;
    if (!info->sstarts)
    {
        info->sstarts = (PetscInt *)malloc((info->snp + 1) * sizeof(PetscInt));
        memcpy(info->sstarts, to->starts, (info->snp + 1) * sizeof(PetscInt));
    }
    if (!info->rstarts)
    {
        info->rstarts = (PetscInt *)malloc((info->rnp + 1) * sizeof(PetscInt));
        memcpy(info->rstarts, from->starts, (info->rnp + 1) * sizeof(PetscInt));
    }

    info->indices_len = info->sstarts[info->snp];
    info->buf_len = info->indices_len * info->bs;

    // info->sindices=to->indices; info->rindices=from->indices;
    if (!info->sindices)
    {
        info->sindices = (PetscInt *)malloc(info->indices_len * sizeof(PetscInt));
        memcpy(info->sindices, to->indices, info->indices_len * sizeof(PetscInt));
    }
    if (!info->rindices)
    {
        info->rindices = (PetscInt *)malloc(info->rstarts[info->rnp] * sizeof(PetscInt));
        memcpy(info->rindices, from->indices, info->rstarts[info->rnp] * sizeof(PetscInt));
    }
    // info->swaits=to->requests; info->rwaits=from->requests;
    if (!info->swaits)
    {
        info->swaits = (MPI_Request *)malloc(info->snp * sizeof(MPI_Request));
    }
    if (!info->rwaits)
    {
        info->rwaits = (MPI_Request *)malloc(info->rnp * sizeof(MPI_Request));
    }
    if (!rank && dbpt)
    {
        printf("rank=%d,info->indices=%d,info->buf_len=%d\n", rank, info->indices_len, info->buf_len);
    } // debugging

    if (!info->hsend_buf)
    {
        info->hsend_buf = (PetscReal *)malloc(sizeof(PetscReal) * info->buf_len);
    }
    if (!info->hlvec)
    {
        info->hlvec = (PetscReal *)malloc(sizeof(PetscReal) * info->lvsz);
    }
    //
    info->rhs = ksp->vec_rhs;
    info->sol = ksp->vec_sol;
    ierr = VecGetLocalSize(info->rhs, &info->vsz);
    CHKERRQ(ierr);
    if (!rank && dbpt)
    {
        printf("rank=%d,info->vsz=%d\n", rank, info->vsz);
    } // debugging
    // ASM preconditioner:

    if (info->use_asm)
    {
        if (!rank && dbpt)
        {
            printf("rank=%d,info->use_asm=%d\n", rank, info->use_asm);
        }
        PC pc;
        KSPGetPC(ksp, &pc);
        PC_ASM *osm = (PC_ASM *)pc->data;
        info->asm_restriction = osm->restriction;
        ierr = VecGetLocalSize(osm->lx, &info->asm_lxsz);
        CHKERRQ(ierr);
        VecScatter_MPI_General *asm_to = (VecScatter_MPI_General *)info->asm_restriction->todata;
        VecScatter_MPI_General *asm_from = (VecScatter_MPI_General *)info->asm_restriction->fromdata;
        info->asm_snp = asm_to->n;
        info->asm_rnp = asm_from->n;
        if (!info->asm_sprocs)
        {
            info->asm_sprocs = (PetscMPIInt *)malloc(info->asm_snp * sizeof(PetscMPIInt));
            memcpy(info->asm_sprocs, asm_to->procs, info->asm_snp * sizeof(PetscMPIInt));
        }
        if (!info->asm_rprocs)
        {
            info->asm_rprocs = (PetscMPIInt *)malloc(info->asm_rnp * sizeof(PetscMPIInt));
            memcpy(info->asm_rprocs, asm_from->procs, info->asm_rnp * sizeof(PetscMPIInt));
        }
        if (!info->asm_sstarts)
        {
            info->asm_sstarts = (PetscInt *)malloc((info->asm_snp + 1) * sizeof(PetscInt));
            memcpy(info->asm_sstarts, asm_to->starts, (info->asm_snp + 1) * sizeof(PetscInt));
        }
        if (!info->asm_rstarts)
        {
            info->asm_rstarts = (PetscInt *)malloc((info->asm_rnp + 1) * sizeof(PetscInt));
            memcpy(info->asm_rstarts, asm_from->starts, (info->asm_rnp + 1) * sizeof(PetscInt));
        }
        info->asm_sindices_len = info->asm_sstarts[info->asm_snp];
        info->asm_rindices_len = info->asm_rstarts[info->asm_rnp];
        info->asm_sendbuf_len = info->asm_sindices_len; // different from SPMV no need to * info->bs
        info->asm_recvbuf_len = info->asm_rindices_len;
        if (!info->asm_sindices)
        {
            info->asm_sindices = (PetscInt *)malloc(info->asm_sindices_len * sizeof(PetscInt));
            memcpy(info->asm_sindices, asm_to->indices, info->asm_sindices_len * sizeof(PetscInt));
        }
        if (!info->asm_rindices)
        {
            info->asm_rindices = (PetscInt *)malloc(info->asm_rindices_len * sizeof(PetscInt));
            memcpy(info->asm_rindices, asm_from->indices, info->asm_rindices_len * sizeof(PetscInt));
        }
        info->asm_self_sindices_len = asm_to->local.n;
        info->asm_self_rindices_len = asm_to->local.n; // they are the same
        if (!info->asm_self_sindices)
        {
            info->asm_self_sindices = (PetscInt *)malloc(info->asm_self_sindices_len * sizeof(PetscInt));
            memcpy(info->asm_self_sindices, asm_to->local.vslots, info->asm_self_sindices_len * sizeof(PetscInt));
        }
        if (!info->asm_self_rindices)
        {
            info->asm_self_rindices = (PetscInt *)malloc(info->asm_self_rindices_len * sizeof(PetscInt));
            memcpy(info->asm_self_rindices, asm_from->local.vslots, info->asm_self_rindices_len * sizeof(PetscInt));
        }
        if (!info->asm_swaits)
        {
            info->asm_swaits = (MPI_Request *)malloc(info->asm_snp * sizeof(MPI_Request));
        }
        if (!info->asm_rwaits)
        {
            info->asm_rwaits = (MPI_Request *)malloc(info->asm_rnp * sizeof(MPI_Request));
        }

        if (dbpt)
        {
            printf("rank=%d,asm_lxsz=%d,vsz=%d,asm_sendbuf_len=%d\n", info->rank, info->asm_lxsz, info->vsz, info->asm_sendbuf_len);
        }
        if (!info->asm_vecx)
        {
            info->asm_vecx = (PetscReal *)malloc(info->vsz * sizeof(PetscReal));
        }
        if (!info->asm_vecy)
        {
            info->asm_vecy = (PetscReal *)malloc(info->vsz * sizeof(PetscReal));
        }
        if (!info->asm_send_buf)
        {
            info->asm_send_buf = (PetscReal *)malloc(sizeof(PetscReal) * info->asm_sendbuf_len);
        }
        if (!info->asm_recv_buf)
        {
            info->asm_recv_buf = (PetscReal *)malloc(sizeof(PetscReal) * info->asm_recvbuf_len);
        }
        if (!info->asm_lx)
        {
            info->asm_lx = (PetscReal *)malloc(sizeof(PetscReal) * info->asm_lxsz);
        }
        if (!info->asm_ly)
        {
            info->asm_ly = (PetscReal *)malloc(sizeof(PetscReal) * info->asm_lxsz);
        }
    }
    //
    info->vecs = gmres->vecs;
    info->vvdim = gmres->max_k + 4 + gmres->nextra_vecs;
    if (!rank && dbpt)
    {
        printf("rank=%d,info->vvdim=%d,gmres->nextra_vecs=%d\n", rank, info->vvdim, gmres->nextra_vecs);
    } // debugging

    // MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Get_processor_name(hostname, &namelen);
    // g3501, g3502, g3503, g3504........... for "YUAN" supercomputer
    // strncpy(dname,hostname+1,strlen(hostname));
    // color=atoi(dname);
    // MPI_Comm_split(MPI_COMM_WORLD,color,rank,&new_comm);
    // grouping processes on the same host in a sub_comm, so we can use its local index to determine its GPU index
    MPI_Comm_split_type(MPI_COMM_WORLD, OMPI_COMM_TYPE_HOST, rank, MPI_INFO_NULL, &sub_comm);
    MPI_Comm_rank(sub_comm, &info->idev);
    if (dbpt)
    {
        printf("rank=%d,info->idev=%d,hostname=%s\n", rank, info->idev, hostname);
    }
    MPI_Comm_free(&sub_comm);

    PetscFunctionReturn(0);
    //	return 0;
}

#define KSP_MONITOR_GMRES

PetscErrorCode KSPSolve_GMRES_GPU(KSP ksp)
{
    // printf("rank=%d,entering KSPSolve_GMRES_GPU\n", info->rank);
    PetscErrorCode ierr;
    PetscInt its, itcount;
    KSP_GMRES *gmres = (KSP_GMRES *)ksp->data;
    PetscBool guess_zero = ksp->guess_zero;

    Mat Amat, Pmat;
    PetscReal res_norm, res, hapbnd;
    double tt;
    PetscBool hapend;
    PetscInt it; // it and its in inner cycle
    PetscInt max_k = gmres->max_k;
    PetscInt j;
    PetscScalar *hh, *hes, *lhh;

    PetscInt rank;
    ///////////////////////////////////
    PetscReal *hsol, *hlvec; // PetscReal *hveci; PetscReal *hvecj;
    //  PetscReal   	*hvecs;
    double alpha, beta;
    double local_res_norm, tmp;
    PetscInt ii, k;
    PetscInt itm; // it-1 (it minus 1)
    PetscReal *nrs;
    /////////////////////////////////////
    int nblocks;
    //  Ax_nblocks;
    // int bilu_nblocks,csc_to_csr_nblocks;
    // int invDmU_nblocks;
    // int invUDiag_nblocks;
    // int isweep;
    struct timeval tstart, tend;
    double telapse = 0.0;

    struct timeval Ax_tstart, Ax_tend;
    double Ax_t = 0.0; // main matrix vector multi time
    struct timeval Ox_tstart, Ox_tend;
    double Ox_t = 0.0; // off diag matrix vector multi time
    struct timeval c_tstart, c_tend;
    double c_t = 0.0; // matrix vector multi communication time
    struct timeval vd_tstart, vd_tend;
    double vd_t = 0.0; // vector dot time
    struct timeval vr_tstart, vr_tend;
    double vr_t = 0.0; // mpi reduce time in vector dot
    struct timeval vx_tstart, vx_tend;
    double vx_t = 0.0; //  Maxpy time
    struct timeval btrisv_tstart, btrisv_tend;
    double btrisv_t = 0.0; // triangluar solve time
    struct timeval outAx_tstart, outAx_tend;
    double outAx_t = 0.0; // matrix vector multi :out of restart
    struct timeval outOx_tstart, outOx_tend;
    double outOx_t = 0.0; // matrix vector multi :out of restart
    struct timeval outc_tstart, outc_tend;
    double outc_t = 0.0; // matrix vector multi communication time: out of restart
    struct timeval outvx_tstart, outvx_tend;
    double outvx_t = 0.0; // out of restart Maxpy
    struct timeval total_tstart, total_tend;
    double total_t = 0.0; // out of restart Maxpy
    struct timeval all_tstart, all_tend;
    double all_t = 0.0; // all
    struct timeval UpdateHessenberg_tstart, UpdateHessenberg_tend;
    double UpdateHessenberg_t = 0.0; // UpdateHessenberg
    struct timeval host_tstart, host_tend;
    double host_t = 0.0; // host
    struct timeval norm_tstart, norm_tend;
    double norm_t = 0.0; // norm
    struct timeval setbuf_tstart, setbuf_tend;
    double setbuf_t = 0.0; // setbuf

    InitialGMRES_INFO_CPU(ksp);
    InitialGMRES_INFO_GPU();
    MPI_Barrier(MPI_COMM_WORLD);
    // printf("rank=%d,afterGMRES_INFO\n", info->rank);

    hipStream_t Axstream;
    hipStreamCreate(&Axstream);
    hipsparseSetStream(info->cusparse_handle, Axstream);
    nblocks = (info->buf_len - 1) / THREADS_PER_BLOCK + 1;

    PBILU_Factorization();
    // printf("rank=%d,after PBILU_Factorization()\n", info->rank);

    ////////////////////////////////////////////////////////////////////////

    PetscFunctionBegin;
    if (ksp->calc_sings && !gmres->Rsvd)
        SETERRQ(PetscObjectComm((PetscObject)ksp), PETSC_ERR_ORDER, "Must call KSPSetComputeSingularValues() before KSPSetUp() is called");

    ierr = PetscObjectSAWsTakeAccess((PetscObject)ksp);
    CHKERRQ(ierr);
    ksp->its = 0;
    ierr = PetscObjectSAWsGrantAccess((PetscObject)ksp);
    CHKERRQ(ierr);

    // itcount = 0;
    // gmres->fullcycle = 0;
    // ksp->reason = KSP_CONVERGED_ITERATING;
    ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    CHKERRQ(ierr);

    // run 10 times to record the time
    int run_times = 1;
    for (int i = 0; i < run_times; i++)
    {

        // reset the info->dsol to zero
        // hipMemset(info->dsol, 0, info->vsz * sizeof(PetscReal));
        ksp->its = 0;
        ksp->rnorm = 0.0;
        ksp->guess_zero = guess_zero;
        hipMemset(info->dsol, 0, info->vsz * sizeof(PetscReal));
        itcount = 0;
        gmres->fullcycle = 0;
        ksp->reason = KSP_CONVERGED_ITERATING;

        ////////////////////////////////////////
        MPI_Barrier(PETSC_COMM_WORLD);
        gettimeofday(&all_tstart, NULL);
        while (!ksp->reason)
        {
            //******************************S1: calculating: b-Ax:**************************//
            // VEC_TEMP=A*vec_sol, VEC_TEMP_MATOP=vec_rhs, VEC_TMP_MATOP=VEC_TMP_MATOP-VEC_TEMP
            if (!ksp->pc)
            {
                KSPGetPC(ksp, &ksp->pc);
            }
            PCGetOperators(ksp->pc, &Amat, &Pmat);
            // if(itcount)
            {
                MPI_Barrier(MPI_COMM_WORLD);
                gettimeofday(&total_tstart, NULL);
                // CPU:MatMult(Amat,ksp->vec_sol,VEC_TEMP);
                // CPU:VecCopy(ksp->vec_rhs,VEC_TEMP_MATOP);VecAXPY(VEC_TEMP_MATOP,-1.0,VEC_TEMP);
                // CPU:VecCopy(VEC_TEMP_MATOP,VEC_VV(0));

                // GPU: copy dsol to vec_sol
                VecGetArray(info->sol, &hsol);
                gettimeofday(&host_tstart, NULL);
                hipMemcpy(hsol, info->dsol, sizeof(PetscReal) * info->vsz, hipMemcpyDeviceToHost);
                VecRestoreArray(info->sol, &hsol);
                gettimeofday(&host_tend, NULL);
                host_t += ((double)((host_tend.tv_sec * 1000000.0 + host_tend.tv_usec) - (host_tstart.tv_sec * 1000000.0 + host_tstart.tv_usec))) / 1000.0;
                // MPI_Barrier(MPI_COMM_WORLD);

                // GPU: we do Amat->A * dsol = VEC_TEMP  alpha=1.0, beta=0.0
                alpha = 1.0;
                beta = 0.0;
                gettimeofday(&Ax_tstart, NULL);
                hipsparseDbsrmv(info->cusparse_handle, info->dir, info->trans, info->main_nrows, info->main_ncols,
                                info->main_nnz, &alpha, info->descr, info->dMainBlkVal, info->dMainRowPtr, info->dMainColVal, info->bs,
                                info->dsol, &beta, info->dvv[0]);
                hipDeviceSynchronize();
                gettimeofday(&Ax_tend, NULL);
                Ax_t += ((double)((Ax_tend.tv_sec * 1000000.0 + Ax_tend.tv_usec) - (Ax_tstart.tv_sec * 1000000.0 + Ax_tstart.tv_usec))) / 1000.0;
                // printf("rank=%d,after main Ax\n",info->rank);
                // GPU: make sure dsol has been copied to host memory
                // GPU: MPI communication
                gettimeofday(&c_tstart, NULL);
                VecScatterBegin(info->Mvctx, info->sol, info->lvec, INSERT_VALUES, SCATTER_FORWARD);
                VecScatterEnd(info->Mvctx, info->sol, info->lvec, INSERT_VALUES, SCATTER_FORWARD);
                VecGetArray(info->lvec, &hlvec);
                gettimeofday(&host_tstart, NULL);
                hipMemcpy(info->dlvec, hlvec, sizeof(PetscReal) * info->lvsz, hipMemcpyHostToDevice);
                VecRestoreArray(info->lvec, &hlvec);
                gettimeofday(&host_tend, NULL);
                host_t += ((double)((host_tend.tv_sec * 1000000.0 + host_tend.tv_usec) - (host_tstart.tv_sec * 1000000.0 + host_tstart.tv_usec))) / 1000.0;
                // GPU DIRECT:
                // AxStartComm();
                // AxEndComm();
                gettimeofday(&c_tend, NULL);
                c_t += ((double)((c_tend.tv_sec * 1000000.0 + c_tend.tv_usec) - (c_tstart.tv_sec * 1000000.0 + c_tstart.tv_usec))) / 1000.0;
                // printf("After communication, rank=%d\n",info->rank);
                // GPU: do Amat->B * lvec accumulate to info->vv[0]  alpha =1.0, beta=1.0
                alpha = 1.0;
                beta = 1.0;
                gettimeofday(&Ax_tstart, NULL);
                hipsparseDbsrmv(info->cusparse_handle, info->dir, info->trans, info->off_nrows, info->off_ncols,
                                info->off_nnz, &alpha, info->descr, info->dOffBlkVal, info->dOffRowPtr, info->dOffColVal, info->bs,
                                info->dlvec, &beta, info->dvv[0]);
                hipDeviceSynchronize();
                gettimeofday(&Ax_tend, NULL);
                Ax_t += ((double)((Ax_tend.tv_sec * 1000000.0 + Ax_tend.tv_usec) - (Ax_tstart.tv_sec * 1000000.0 + Ax_tstart.tv_usec))) / 1000.0;
                // printf("rank=%d,after off diag matmult\n",info->rank);
                // GPU: copy rhs vv[1]
                hipblasDcopy(info->cublas_handle, info->vsz, info->drhs, 1, info->dvv[1], 1);
                hipDeviceSynchronize();
                // GPU: dvv[1]= dvv[1]-1.0*dvv[0]
                alpha = -1.0;
                gettimeofday(&vx_tstart, NULL);
                hipblasDaxpy(info->cublas_handle, info->vsz, &alpha, info->dvv[0], 1, info->dvv[1], 1);
                // hipDeviceSynchronize();
                // GPU: copy dvv[1] to dvv[2]
                hipblasDcopy(info->cublas_handle, info->vsz, info->dvv[1], 1, info->dvv[2], 1);
                hipDeviceSynchronize();
                gettimeofday(&vx_tend, NULL);
                vx_t += ((double)((vx_tend.tv_sec * 1000000.0 + vx_tend.tv_usec) - (vx_tstart.tv_sec * 1000000.0 + vx_tstart.tv_usec))) / 1000.0;
                 
                // printf("rank=%d,after cublas copy and axpy\n",info->rank);
            }
            // else{VecCopy(ksp->vec_rhs,VEC_TEMP_MATOP);VecCopy(ksp->vec_rhs,VEC_VV(0));}

            //******************************      q END OF S1       ************************//

            //**************S2:Check norm AND enter KSPGMRESCycle(&its,ksp); *******************//
            //******************************S2-1: CHECK NORM       *****************************//
            it = 0;
            its = 0;
            hapend = PETSC_FALSE;
            // CPU:ierr=VecNormalize(VEC_VV(0),&res_norm);CHKERRQ(ierr);

            // GPU: compute local norm2 on GPU
            hipblasDdot(info->cublas_handle, info->vsz, info->dvv[2], 1, info->dvv[2], 1, &local_res_norm);
            //	printf("rank=%d,local_res_norm=%lf\n",info->rank,local_res_norm);
            hipDeviceSynchronize();
            gettimeofday(&vr_tstart, NULL);
            // GPU: MPI communication res_norm need to be initialized to 0.0?
            MPI_Allreduce(&local_res_norm, &res_norm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            gettimeofday(&vr_tend, NULL);
            vr_t += ((double)((vr_tend.tv_sec * 1000000.0 + vr_tend.tv_usec) - (vr_tstart.tv_sec * 1000000.0 + vr_tstart.tv_usec))) / 1000.0;
            //	printf("rank=%d,res_norm=%20.15lf\n",info->rank,res_norm);
            res_norm = PetscSqrtReal(res_norm);
            tmp = 1.0 / res_norm;
            gettimeofday(&vx_tstart, NULL);
            hipblasDscal(info->cublas_handle, info->vsz, &tmp, info->dvv[2], 1);
            hipDeviceSynchronize();
            gettimeofday(&vx_tend, NULL);
            vx_t += ((double)((vx_tend.tv_sec * 1000000.0 + vx_tend.tv_usec) - (vx_tstart.tv_sec * 1000000.0 + vx_tstart.tv_usec))) / 1000.0;

            KSPCheckNorm(ksp, res_norm);
            res = res_norm;
            *GRS(0) = res_norm;
            /* check for the convergence */
            ierr = PetscObjectSAWsTakeAccess((PetscObject)ksp);
            CHKERRQ(ierr);
            ksp->rnorm = res;
            ierr = PetscObjectSAWsGrantAccess((PetscObject)ksp);
            CHKERRQ(ierr);
            gmres->it = (it - 1);
#ifdef KSP_MONITOR_GMRES
            ierr = KSPLogResidualHistory(ksp, res);
            CHKERRQ(ierr);
            ierr = KSPMonitor(ksp, ksp->its, res);
            CHKERRQ(ierr);
#endif
            if (!res)
            {
                ksp->reason = KSP_CONVERGED_ATOL;
                ierr = PetscInfo(ksp, "Converged due to zero residual norm on entry\n");
                CHKERRQ(ierr);
                PetscFunctionReturn(0);
            }
            ierr = (*ksp->converged)(ksp, ksp->its, res, &ksp->reason, ksp->cnvP);
            CHKERRQ(ierr);

            //******************************S2-2:  ENTER CYCLE ******************************//
            MPI_Barrier(MPI_COMM_WORLD);
            gettimeofday(&tstart, NULL);
            while (!ksp->reason && it < max_k && ksp->its < ksp->max_it)
            {
                if (it)
                {
#ifdef KSP_MONITOR_GMRES
                    ierr = KSPLogResidualHistory(ksp, res);
                    CHKERRQ(ierr);
                    ierr = KSPMonitor(ksp, ksp->its, res);
                    CHKERRQ(ierr);
#endif
                    // if(!rank){printf("it=%d,res=%22.15e-in\n",it,res);}
                }
                gmres->it = (it - 1);

                // MPI_Barrier(PETSC_COMM_WORLD);
                gettimeofday(&btrisv_tstart, NULL);
                PB_Preconditioning(ksp->pc, VEC_VV(it), VEC_TEMP_MATOP, info->dvv[2 + it], info->dvv[1], info->dv_tmp, info->vsz);
                gettimeofday(&btrisv_tend, NULL);
                btrisv_t += ((double)((btrisv_tend.tv_sec * 1000000.0 + btrisv_tend.tv_usec) - (btrisv_tstart.tv_sec * 1000000.0 + btrisv_tstart.tv_usec))) / 1000.0;
                //

                // now we get z stored in dvv[1], then we need to do matrix-vector multiplication, dvv[2+it+1]=A * dvv[1]
                // we just copy the way we did in none preconditioner case
                gettimeofday(&setbuf_tstart, NULL);
                hipLaunchKernelGGL(SetSendbuffer, nblocks, THREADS_PER_BLOCK, 0, 0, info->dvv[1], info->vsz, info->dsindices,
                                   info->dsend_buf, info->buf_len, info->bs);
                hipDeviceSynchronize();
                gettimeofday(&setbuf_tend, NULL);
                setbuf_t += ((double)((setbuf_tend.tv_sec * 1000000.0 + setbuf_tend.tv_usec) - (setbuf_tstart.tv_sec * 1000000.0 + setbuf_tstart.tv_usec))) / 1000.0;
                // VecGetArray(info->vecs[1],&hvecs);
                // hipMemcpy(hvecs,info->dvv[1],sizeof(PetscReal)*info->vsz,hipMemcpyDeviceToHost);
                // VecRestoreArray(info->vecs[1],&hvecs);
                gettimeofday(&host_tstart, NULL);
                hipMemcpy(info->hsend_buf, info->dsend_buf, sizeof(PetscReal) * info->buf_len, hipMemcpyDeviceToHost);
                gettimeofday(&host_tend, NULL);
                host_t += ((double)((host_tend.tv_sec * 1000000.0 + host_tend.tv_usec) - (host_tstart.tv_sec * 1000000.0 + host_tstart.tv_usec))) / 1000.0;

                // MPI_Barrier(PETSC_COMM_WORLD);
                alpha = 1.0;
                beta = 0.0;
                gettimeofday(&Ax_tstart, NULL);
                // VecScatterBegin(info->Mvctx,info->vecs[1],info->lvec,INSERT_VALUES,SCATTER_FORWARD);

                hipsparseDbsrmv(info->cusparse_handle, info->dir, info->trans, info->main_nrows, info->main_nrows,
                                info->main_nnz, &alpha, info->descr, info->dMainBlkVal, info->dMainRowPtr, info->dMainColVal, info->bs,
                                info->dvv[1], &beta, info->dvv[2 + it + 1]);

                hipDeviceSynchronize();
                gettimeofday(&Ax_tend, NULL);
                Ax_t += ((double)((Ax_tend.tv_sec * 1000000.0 + Ax_tend.tv_usec) - (Ax_tstart.tv_sec * 1000000.0 + Ax_tstart.tv_usec))) / 1000.0;
               

                gettimeofday(&c_tstart, NULL);
                // VecScatterBegin(info->Mvctx,info->vecs[1],info->lvec,INSERT_VALUES,SCATTER_FORWARD);
                // VecScatterEnd(info->Mvctx,info->vecs[1],info->lvec,INSERT_VALUES,SCATTER_FORWARD);

                AxStartComm();
                AxEndComm();
                // GPU Direct
                // AxStartComm();
                // AxEndComm();
                gettimeofday(&c_tend, NULL);
                c_t += ((double)((c_tend.tv_sec * 1000000.0 + c_tend.tv_usec) - (c_tstart.tv_sec * 1000000.0 + c_tstart.tv_usec))) / 1000.0;

                 gettimeofday(&host_tstart, NULL);
                // VecGetArray(info->lvec,&hlvec);
                hipMemcpy(info->dlvec, info->hlvec, sizeof(PetscReal) * info->lvsz, hipMemcpyHostToDevice);
                // VecRestoreArray(info->lvec,&hlvec);
                gettimeofday(&host_tend, NULL);
                host_t += ((double)((host_tend.tv_sec * 1000000.0 + host_tend.tv_usec) - (host_tstart.tv_sec * 1000000.0 + host_tstart.tv_usec))) / 1000.0;
                alpha = 1.0;
                beta = 1.0;
                gettimeofday(&Ax_tstart, NULL);
                hipsparseDbsrmv(info->cusparse_handle, info->dir, info->trans, info->off_nrows, info->off_ncols,
                                info->off_nnz, &alpha, info->descr, info->dOffBlkVal, info->dOffRowPtr, info->dOffColVal, info->bs,
                                info->dlvec, &beta, info->dvv[2 + it + 1]);
                hipDeviceSynchronize();
                gettimeofday(&Ax_tend, NULL);
                Ax_t += ((double)((Ax_tend.tv_sec * 1000000.0 + Ax_tend.tv_usec) - (Ax_tstart.tv_sec * 1000000.0 + Ax_tstart.tv_usec))) / 1000.0;

                // do gramschmidtOrthogonalization
                if (!gmres->orthogwork)
                {
                    PetscMalloc1(gmres->max_k + 2, &gmres->orthogwork);
                }
                lhh = gmres->orthogwork;
                /* update Hessenberg matrix and do unmodified Gram-Schmidt */
                hh = HH(0, it);
                hes = HES(0, it);
                /* Clear hh and hes since we will accumulate values into them */
                for (j = 0; j <= it; j++)
                {
                    hh[j] = 0.0;
                    hes[j] = 0.0;
                }
                // GPU: j=0:it do dvv[2+it+1] * dvv[2+j]
                // hipblasSetPointerMode(info->cublas_handle,HIPBLAS_POINTER_MODE_DEVICE);
                // MPI_Barrier(PETSC_COMM_WORLD);
                gettimeofday(&vd_tstart, NULL);

                VecMdot_GPU(info->dvv[2 + it + 1], info->vsz, it + 1, info->dvv + 2, info->mystream, info->ddotres, info->hdotres, lhh);
                //	printf("rank=%d,it=%d,VecMdot_GPU\n",info->rank,it);
                gettimeofday(&vd_tend, NULL);
                vd_t += ((double)((vd_tend.tv_sec * 1000000.0 + vd_tend.tv_usec) - (vd_tstart.tv_sec * 1000000.0 + vd_tstart.tv_usec))) / 1000.0;

                // MPI_Barrier(PETSC_COMM_WORLD);
                gettimeofday(&vr_tstart, NULL);
                MPI_Allreduce(MPI_IN_PLACE, lhh, it + 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
                // MPI_Allreduce(MPI_IN_PLACE,info->dlhh,it+1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
                gettimeofday(&vr_tend, NULL);
                vr_t += ((double)((vr_tend.tv_sec * 1000000.0 + vr_tend.tv_usec) - (vr_tstart.tv_sec * 1000000.0 + vr_tstart.tv_usec))) / 1000.0;

                for (j = 0; j <= it; j++)
                {
                    lhh[j] = -lhh[j];
                }

                // GPU: j=0:it do dvv[2+it+1] + = alpha * dvv[2+j]
                // MPI_Barrier(PETSC_COMM_WORLD);
                gettimeofday(&vx_tstart, NULL);

                VecMAXPY_GPU(info->dvv[2 + it + 1], info->vsz, lhh, info->dvv + 2, it + 1);
                gettimeofday(&vx_tend, NULL);
                vx_t += ((double)((vx_tend.tv_sec * 1000000.0 + vx_tend.tv_usec) - (vx_tstart.tv_sec * 1000000.0 + vx_tstart.tv_usec))) / 1000.0;
                // note lhh[j] is -<v,vnew> , hence the subtraction
                for (j = 0; j <= it; j++)
                {
                    hh[j] -= lhh[j];
                    hes[j] -= lhh[j];
                }
                if (ksp->reason)
                {
                    break;
                }
                // vv(i+1) . vv(i+1)
                // VecNormalize(VEC_VV(it+1),&tt);
                // GPU: compute local norm2
                gettimeofday(&norm_tstart, NULL);
                info->cublas_stat = hipblasDnrm2(info->cublas_handle, info->vsz, info->dvv[2 + it + 1], 1, &local_res_norm);
                hipDeviceSynchronize();
                gettimeofday(&norm_tend, NULL);
                norm_t += ((double)((norm_tend.tv_sec * 1000000.0 + norm_tend.tv_usec) - (norm_tstart.tv_sec * 1000000.0 + norm_tstart.tv_usec))) / 1000.0;
                if (info->cublas_stat != HIPBLAS_STATUS_SUCCESS)
                {
                    printf("ERROR in hipblasDnrm2 in rank=%d\n", info->rank);
                }
                local_res_norm = local_res_norm * local_res_norm;
                // GPU: MPI communication
                gettimeofday(&vr_tstart, NULL);
                MPI_Allreduce(&local_res_norm, &tt, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
                gettimeofday(&vr_tend, NULL);
                vr_t += ((double)((vr_tend.tv_sec * 1000000.0 + vr_tend.tv_usec) - (vr_tstart.tv_sec * 1000000.0 + vr_tstart.tv_usec))) / 1000.0;
                tt = PetscSqrtReal(tt);
                tmp = 1.0 / tt;
                gettimeofday(&vx_tstart, NULL);
                info->cublas_stat = hipblasDscal(info->cublas_handle, info->vsz, &tmp, info->dvv[2 + it + 1], 1);
                hipDeviceSynchronize();
                gettimeofday(&vx_tend, NULL);
                vx_t += ((double)((vx_tend.tv_sec * 1000000.0 + vx_tend.tv_usec) - (vx_tstart.tv_sec * 1000000.0 + vx_tstart.tv_usec))) / 1000.0;
                if (info->cublas_stat != HIPBLAS_STATUS_SUCCESS)
                {
                    printf("ERROR in hipblasDscal in rank=%d\n", info->rank);
                }
                // if(!info->rank){printf("it=%d,tt=%22.15lf\n",it,tt);}
                KSPCheckNorm(ksp, tt);
                // save the magnitude
                *HH(it + 1, it) = tt;
                *HES(it + 1, it) = tt;
                // check for the happy breakdown
                hapbnd = PetscAbsScalar(tt / *GRS(it));
                if (hapbnd > gmres->haptol)
                {
                    hapbnd = gmres->haptol;
                }
                if (tt < hapbnd)
                {
                    PetscInfo2(ksp, "Detected happy breakdown, current hapbnd = %14.12e tt = %14.12e\n", (double)hapbnd, (double)tt);
                    hapend = PETSC_TRUE;
                }
                
                KSPGMRESUpdateHessenberg_MARK_GPU(ksp, it, hapend, &res);
                
                it++;
                gmres->it = (it - 1);
                ksp->its++;
                ksp->rnorm = res;
                if (ksp->reason)
                {
                    break;
                }
                (*ksp->converged)(ksp, ksp->its, res, &ksp->reason, ksp->cnvP);

                /* Catch error in happy breakdown and signal convergence and break from loop */
                // if (hapend)
                // {
                //     if (ksp->normtype == KSP_NORM_NONE)
                //     {
                //         ksp->reason = KSP_CONVERGED_HAPPY_BREAKDOWN;
                //     }
                //     else if (!ksp->reason)
                //     {
                //         if (ksp->errorifnotconverged)
                //         {
                //             SETERRQ1(PetscObjectComm((PetscObject)ksp), PETSC_ERR_NOT_CONVERGED, "You reached the happy break down, but convergence was not indicated. Residual norm = %g", (double)res);
                //         }
                //         else
                //         {
                //             ksp->reason = KSP_DIVERGED_BREAKDOWN;
                //             break;
                //         }
                //     }
                // }
            }
            //************************END OF S2***********************************
            gettimeofday(&tend, NULL);
            telapse += (double)((tend.tv_sec * 1000000.0 + tend.tv_usec) - (tstart.tv_sec * 1000000.0 + tstart.tv_usec)) / 1000.0;

            //*************************S3: We need to build solution *************
            /* Monitor if we know that we will not return for a restart */
            if (it && (ksp->reason || ksp->its >= ksp->max_it))
            {
#ifdef KSP_MONITOR_GMRES
                KSPLogResidualHistory(ksp, res);
                KSPMonitor(ksp, ksp->its, res);
#endif
            }
            its = it;

            // build solution
            // CPU:KSPGMRESBuildSoln_MARK_GPU(GRS(0),ksp->vec_sol,ksp->vec_sol,ksp,it-1);
            //  GPU:
            itm = it - 1;
            nrs = GRS(0);
            // If it is <0, no gmres steps have been performed
            if (itm < 0)
            {
                //        PetscFunctionReturn(0);
                goto endbuild;
            }
            if (*HH(itm, itm) != 0.0)
            {
                nrs[itm] = *GRS(itm) / *HH(itm, itm);
            }
            else
            {
                ksp->reason = KSP_DIVERGED_BREAKDOWN;
                ierr = PetscInfo2(ksp, "Likely your matrix or preconditioner is singular. HH(itm,itm) is identically zero; it = %D GRS(itm) = %g\n", itm, (double)PetscAbsScalar(*GRS(itm)));
                CHKERRQ(ierr);
                //	   PetscFunctionReturn(0);
                goto endbuild;
            }
            for (ii = 1; ii <= itm; ii++)
            {
                k = itm - ii;
                tt = *GRS(k);
                for (j = k + 1; j <= itm; j++)
                {
                    tt = tt - *HH(k, j) * nrs[j];
                }
                if (*HH(k, k) == 0.0)
                {
                    ksp->reason = KSP_DIVERGED_BREAKDOWN;
                    ierr = PetscInfo1(ksp, "Likely your matrix or preconditioner is singular. HH(k,k) is identically zero; k = %D\n", k);
                    CHKERRQ(ierr);
                    // PetscFunctionReturn(0);
                    goto endbuild;
                }
                nrs[k] = tt / *HH(k, k);
            }

            hipMemset(info->dvv[0], 0, sizeof(PetscReal) * info->vsz);
            MPI_Barrier(PETSC_COMM_WORLD);
            gettimeofday(&vx_tstart, NULL);

            VecMAXPY_GPU(info->dvv[0], info->vsz, nrs, info->dvv + 2, itm + 1);
            gettimeofday(&vx_tend, NULL);
            vx_t += ((double)((vx_tend.tv_sec * 1000000.0 + vx_tend.tv_usec) - (vx_tstart.tv_sec * 1000000.0 + vx_tstart.tv_usec))) / 1000.0;
            // with right preconditioning: x_m = x_0 + inv(M) * V_m * y_m
            // V_m * y_m = dvv[0];   inv(M) * dvv[0] = t;  M * t = dvv[0];  LU * t = dvv[0];
            //			L* dvv[1] = dvv[0];  U * t= dvv[1]; we store t in dvv[0] finally
            gettimeofday(&btrisv_tstart, NULL);
            PB_Preconditioning(ksp->pc, VEC_TEMP, VEC_TEMP_MATOP, info->dvv[0], info->dvv[1], info->dv_tmp, info->vsz);
            gettimeofday(&btrisv_tend, NULL);
            btrisv_t += ((double)((btrisv_tend.tv_sec * 1000000.0 + btrisv_tend.tv_usec) - (btrisv_tstart.tv_sec * 1000000.0 + btrisv_tstart.tv_usec))) / 1000.0;
            gettimeofday(&host_tstart, NULL);
            hipMemcpy(info->dvv[0], info->dvv[1], sizeof(PetscReal) * info->vsz, hipMemcpyDeviceToDevice);
            gettimeofday(&host_tend, NULL);
            host_t += ((double)((host_tend.tv_sec * 1000000.0 + host_tend.tv_usec) - (host_tstart.tv_sec * 1000000.0 + host_tstart.tv_usec))) / 1000.0;

            // without precondition: x_m = x_0+ V_m * y_m
            alpha = 1.0;
            gettimeofday(&vx_tstart, NULL);
            hipblasDaxpy(info->cublas_handle, info->vsz, &alpha, info->dvv[0], 1, info->dsol, 1);
            gettimeofday(&vx_tend, NULL);
            vx_t += ((double)((vx_tend.tv_sec * 1000000.0 + vx_tend.tv_usec) - (vx_tstart.tv_sec * 1000000.0 + vx_tstart.tv_usec))) / 1000.0;
            hipDeviceSynchronize();

            gettimeofday(&total_tend, NULL);
            total_t += ((double)((total_tend.tv_sec * 1000000.0 + total_tend.tv_usec) - (total_tstart.tv_sec * 1000000.0 + total_tstart.tv_usec))) / 1000.0;

        endbuild:

            if (its == gmres->max_k)
            {
                gmres->fullcycle++;
            }
            itcount += its;
            if (itcount >= ksp->max_it)
            {
                if (!ksp->reason)
                {
                    ksp->reason = KSP_DIVERGED_ITS;
                }
                break;
            }

            // every future call to KSPInitialResidual() will have nonzero guess
            ksp->guess_zero = PETSC_FALSE;
        }
        gettimeofday(&all_tend, NULL);
        all_t += ((double)((all_tend.tv_sec * 1000000.0 + all_tend.tv_usec) - (all_tstart.tv_sec * 1000000.0 + all_tstart.tv_usec))) / 1000.0;
    }

    if (info->rank == 0)
    {
        printf("rank=%d,all_t=%12.8lf, btrisv_t=%12.8lf, Ax_t=%12.8lf, c_t=%12.8lf, vd_t=%12.8lf, vr_t=%12.8lf, vx_t=%12.8lf, host_t=%12.8lf, norm_t=%12.8lf, setbuf_t=%12.8lf\n", info->rank, all_t / run_times, btrisv_t / run_times, Ax_t / run_times, c_t / run_times, vd_t / run_times, vr_t / run_times, vx_t / run_times,  host_t / run_times, norm_t / run_times, setbuf_t / run_times);
        printf("pack_time=%12.8lf, unpack_time=%12.8lf,  comm_time=%12.8lf, ASMLvecToVec_time=%12.8lf\n",
               pack_time / run_times, unpack_time / run_times, comm_time / run_times, ASMLvecToVec_time / run_times);
        printf("solve_time in fast_solve=%12.8lf, gemv_time in fast_solve=%12.8lf, construct_tEFG_time=%12.8lf, collect_boundary_values_time=%12.8lf\n", solve_time / run_times, gemv_time / run_times, construct_tEFG_time / run_times, collect_boundary_values_time / run_times);
    }

    // we need to copy dsol to hsol
    VecGetArray(info->sol, &hsol);
    hipMemcpy(hsol, info->dsol, sizeof(PetscReal) * info->vsz, hipMemcpyDeviceToHost);
    VecRestoreArray(info->sol, &hsol);
    ksp->guess_zero = guess_zero; // restore if user provided nonzero initial guess

    hipStreamDestroy(Axstream);
    PetscFunctionReturn(0);
}

PetscErrorCode KSPGMRESUpdateHessenberg_MARK_GPU(KSP ksp, PetscInt it, PetscBool hapend, PetscReal *res)
{
    PetscScalar *hh, *cc, *ss, tt;
    PetscInt j;
    KSP_GMRES *gmres = (KSP_GMRES *)(ksp->data);

    PetscFunctionBegin;
    hh = HH(0, it);
    cc = CC(0);
    ss = SS(0);

    /* Apply all the previously computed plane rotations to the new column
       of the Hessenberg matrix */
    for (j = 1; j <= it; j++)
    {
        tt = *hh;
        *hh = PetscConj(*cc) * tt + *ss * *(hh + 1);
        hh++;
        *hh = *cc++ * *hh - (*ss++ * tt);
    }

    /*
      compute the new plane rotation, and apply it to:
       1) the right-hand-side of the Hessenberg system
       2) the new column of the Hessenberg matrix
      thus obtaining the updated value of the residual
    */
    if (!hapend)
    {
        tt = PetscSqrtScalar(PetscConj(*hh) * *hh + PetscConj(*(hh + 1)) * *(hh + 1));
        if (tt == 0.0)
        {
            ksp->reason = KSP_DIVERGED_NULL;
            PetscFunctionReturn(0);
        }
        *cc = *hh / tt;
        *ss = *(hh + 1) / tt;
        *GRS(it + 1) = -(*ss * *GRS(it));
        *GRS(it) = PetscConj(*cc) * *GRS(it);
        *hh = PetscConj(*cc) * *hh + *ss * *(hh + 1);
        *res = PetscAbsScalar(*GRS(it + 1));
    }
    else
    {
        /* happy breakdown: HH(it+1, it) = 0, therfore we don't need to apply
                another rotation matrix (so RH doesn't change).  The new residual is
                always the new sine term times the residual from last time (GRS(it)),
                but now the new sine rotation would be zero...so the residual should
                be zero...so we will multiply "zero" by the last residual.  This might
                not be exactly what we want to do here -could just return "zero". */

        *res = 0.0;
    }
    PetscFunctionReturn(0);
}

#define KSP_MONITOR

PetscErrorCode KSPSolve_BiCGSTAB_GPU(KSP ksp)
{

    PetscErrorCode ierr;
    PetscInt its;
    PetscBool guess_zero = ksp->guess_zero;
    Mat Amat, Pmat;
    PetscReal res_norm;
    PetscInt rank;

    PetscReal *hsol, *hlvec;
    double alpha, beta, one = 1.0, neg_one = -1.0;
    double local_res_norm, global_res_norm;
    double rho, rho_old, alpha_scalar, omega;
    double local_rho, global_rho;
    PetscInt nblocks;

    struct timeval tstart, tend, total_tstart, total_tend;
    double total_t = 0.0;
    double telapse = 0.0;
    struct timeval Ax_tstart, Ax_tend;
    double Ax_t = 0.0;
    struct timeval c_tstart, c_tend;
    double c_t = 0.0;
    struct timeval vd_tstart, vd_tend;
    double vd_t = 0.0;
    struct timeval vr_tstart, vr_tend;
    double vr_t = 0.0;
    struct timeval vx_tstart, vx_tend;
    double vx_t = 0.0;
    struct timeval btrisv_tstart, btrisv_tend;
    double btrisv_t = 0.0;
    struct timeval host_tstart, host_tend;
    double host_t = 0.0; // Host-side vector ops
    struct timeval cpu_tstart, cpu_tend;
    double cpu_t = 0.0; // CPU-side computations
    struct timeval log_tstart, log_tend;
    double log_t = 0.0; // Logging and convergence checks

    hipStream_t Axstream;
    hipError_t hip_err;

    // 临时 PETSc 向量用于通信
    Vec p_tilde_vec, s_tilde_vec;

    PetscFunctionBegin;

    // 初始化 KSP 和 MPI
    ierr = PetscObjectSAWsTakeAccess((PetscObject)ksp);
    CHKERRQ(ierr);
    ksp->its = 0;
    ksp->reason = KSP_CONVERGED_ITERATING;
    ierr = PetscObjectSAWsGrantAccess((PetscObject)ksp);
    CHKERRQ(ierr);
    ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    CHKERRQ(ierr);

    // 创建临时向量用于 p_tilde 和 s_tilde 的通信
    ierr = VecDuplicate(ksp->vec_sol, &p_tilde_vec);
    CHKERRQ(ierr);
    ierr = VecDuplicate(ksp->vec_sol, &s_tilde_vec);
    CHKERRQ(ierr);

    // 创建 HIP 流
    hip_err = hipStreamCreate(&Axstream);
    if (hip_err != hipSuccess)
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, "Failed to create HIP stream");
    hipsparseSetStream(info->cusparse_handle, Axstream);
    nblocks = (info->buf_len - 1) / THREADS_PER_BLOCK + 1;

    // 初始化 GPU 数据
    double t = 0.0;

    InitialGMRES_INFO_CPU(ksp);
    InitialGMRES_INFO_GPU();
    PBILU_Factorization();

    // 获取矩阵和预条件器
    if (!ksp->pc)
    {
        KSPGetPC(ksp, &ksp->pc);
    }
    PCGetOperators(ksp->pc, &Amat, &Pmat);

    // 初始化解向量 info->dsol
    VecGetArray(info->sol, &hsol);
    hipMemcpy(info->dsol, hsol, sizeof(PetscReal) * info->vsz, hipMemcpyHostToDevice);
    VecRestoreArray(info->sol, &hsol);

    // warm up GPU and record time
    // warmup_rocblas_dgemm();

    // run 10 times to record the time
    int run_times = 1;
    for (int i = 0; i < run_times; i++)
    {

        // 重置所有 BiCGSTAB 向量
        hipMemset(info->dvv[1], 0, info->vsz * sizeof(PetscReal)); // r
        hipMemset(info->dvv[2], 0, info->vsz * sizeof(PetscReal)); // rhat
        hipMemset(info->dvv[3], 0, info->vsz * sizeof(PetscReal)); // p
        hipMemset(info->dvv[4], 0, info->vsz * sizeof(PetscReal)); // v
        hipMemset(info->dvv[5], 0, info->vsz * sizeof(PetscReal)); // s
        hipMemset(info->dvv[6], 0, info->vsz * sizeof(PetscReal)); // t
        hipMemset(info->dvv[7], 0, info->vsz * sizeof(PetscReal)); // p_tilde
        hipMemset(info->dvv[8], 0, info->vsz * sizeof(PetscReal)); // s_tilde

        // reset the info->dsol to zero
        hipMemset(info->dsol, 0, info->vsz * sizeof(PetscReal));

        MPI_Barrier(PETSC_COMM_WORLD);
        hipDeviceSynchronize();
        gettimeofday(&total_tstart, NULL);

        // 计算初始残差 r0 = b - A * x0
        if (guess_zero)
        {
            hipblasDcopy(info->cublas_handle, info->vsz, info->drhs, 1, info->dvv[1], 1); // r0 = b
            hipDeviceSynchronize();
        }
        else
        {
            alpha = 1.0;
            beta = 0.0;
            gettimeofday(&Ax_tstart, NULL);
            hipsparseDbsrmv(info->cusparse_handle, info->dir, info->trans, info->main_nrows, info->main_ncols,
                            info->main_nnz, &alpha, info->descr, info->dMainBlkVal, info->dMainRowPtr, info->dMainColVal, info->bs,
                            info->dsol, &beta, info->dvv[4]); // v = A * x0
            hipDeviceSynchronize();
            gettimeofday(&Ax_tend, NULL);
            Ax_t += ((double)((Ax_tend.tv_sec * 1000000.0 + Ax_tend.tv_usec) - (Ax_tstart.tv_sec * 1000000.0 + Ax_tstart.tv_usec))) / 1000.0;

            gettimeofday(&c_tstart, NULL);
            VecScatterBegin(info->Mvctx, info->sol, info->lvec, INSERT_VALUES, SCATTER_FORWARD);
            VecScatterEnd(info->Mvctx, info->sol, info->lvec, INSERT_VALUES, SCATTER_FORWARD);
            VecGetArray(info->lvec, &hlvec);
            hipMemcpy(info->dlvec, hlvec, sizeof(PetscReal) * info->lvsz, hipMemcpyHostToDevice);
            VecRestoreArray(info->lvec, &hlvec);
            gettimeofday(&c_tend, NULL);
            c_t += ((double)((c_tend.tv_sec * 1000000.0 + c_tend.tv_usec) - (c_tstart.tv_sec * 1000000.0 + c_tstart.tv_usec))) / 1000.0;

            alpha = 1.0;
            beta = 1.0;
            hipsparseDbsrmv(info->cusparse_handle, info->dir, info->trans, info->off_nrows, info->off_ncols,
                            info->off_nnz, &alpha, info->descr, info->dOffBlkVal, info->dOffRowPtr, info->dOffColVal, info->bs,
                            info->dlvec, &beta, info->dvv[4]);
            hipDeviceSynchronize();

            hipblasDcopy(info->cublas_handle, info->vsz, info->drhs, 1, info->dvv[1], 1);
            hipblasDaxpy(info->cublas_handle, info->vsz, &neg_one, info->dvv[4], 1, info->dvv[1], 1); // r0 = b - A * x0
            hipDeviceSynchronize();
        }

        // if(rank==0){
        // 	printf("rank=%d, after hipblasDcopy in KSPSolve_BiCGSTAB_GPU\n", info->rank);
        // }

        // rhat = r0
        hipblasDcopy(info->cublas_handle, info->vsz, info->dvv[1], 1, info->dvv[2], 1);
        hipDeviceSynchronize();

        // 计算初始残差范数
        gettimeofday(&vd_tstart, NULL);
        hipblasDdot(info->cublas_handle, info->vsz, info->dvv[1], 1, info->dvv[1], 1, &local_res_norm);
        hipDeviceSynchronize();
        gettimeofday(&vd_tend, NULL);
        vd_t += ((double)((vd_tend.tv_sec * 1000000.0 + vd_tend.tv_usec) - (vd_tstart.tv_sec * 1000000.0 + vd_tstart.tv_usec))) / 1000.0;

        gettimeofday(&vr_tstart, NULL);
        MPI_Allreduce(&local_res_norm, &global_res_norm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        gettimeofday(&vr_tend, NULL);
        vr_t += ((double)((vr_tend.tv_sec * 1000000.0 + vr_tend.tv_usec) - (vr_tstart.tv_sec * 1000000.0 + vr_tstart.tv_usec))) / 1000.0;

        res_norm = PetscSqrtReal(global_res_norm);
        ksp->rnorm = res_norm;
#ifdef KSP_MONITOR
        KSPLogResidualHistory(ksp, res_norm);
        KSPMonitor(ksp, ksp->its, res_norm);
#endif
        if (res_norm < ksp->abstol)
        {
            ksp->reason = KSP_CONVERGED_ATOL;
            ierr = PetscInfo(ksp, "Converged due to residual norm below absolute tolerance on entry\n");
            CHKERRQ(ierr);
            goto cleanup;
        }
        (*ksp->converged)(ksp, ksp->its, res_norm, &ksp->reason, ksp->cnvP);

        // 初始化 BiCGSTAB 参数
        rho = 1.0;
        rho_old = 1.0;
        alpha_scalar = 1.0;
        omega = 1.0;
        hipMemset(info->dvv[3], 0, info->vsz * sizeof(PetscReal)); // p = 0

        MPI_Barrier(PETSC_COMM_WORLD);
        gettimeofday(&tstart, NULL);
        ksp->its = 1; // 从第 1 步开始，避免第 0 步重复

        while (!ksp->reason && ksp->its < ksp->max_it)
        {

            // rho_new = rhat^T * r
            gettimeofday(&vd_tstart, NULL);
            hipblasDdot(info->cublas_handle, info->vsz, info->dvv[2], 1, info->dvv[1], 1, &local_rho);
            hipDeviceSynchronize();
            gettimeofday(&vd_tend, NULL);
            vd_t += ((double)((vd_tend.tv_sec * 1000000.0 + vd_tend.tv_usec) - (vd_tstart.tv_sec * 1000000.0 + vd_tstart.tv_usec))) / 1000.0;

            gettimeofday(&vr_tstart, NULL);
            MPI_Allreduce(&local_rho, &rho, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            gettimeofday(&vr_tend, NULL);
            vr_t += ((double)((vr_tend.tv_sec * 1000000.0 + vr_tend.tv_usec) - (vr_tstart.tv_sec * 1000000.0 + vr_tstart.tv_usec))) / 1000.0;

            // if (fabs(rho) < 1e-30)
            // {
            //     ksp->reason = KSP_DIVERGED_BREAKDOWN;
            //     ierr = PetscInfo(ksp, "BiCGSTAB breakdown: rho = 0\n");
            //     CHKERRQ(ierr);
            //     break;
            // }

            // beta = (rho / rho_old) * (alpha / omega)
            beta = (rho / rho_old) * (alpha_scalar / omega);

            // p = r + beta * (p - omega * v)
            double neg_omega = -omega;
            gettimeofday(&vx_tstart, NULL);
            hipblasDcopy(info->cublas_handle, info->vsz, info->dvv[3], 1, info->dvv[7], 1);             // tmp = p
            hipblasDaxpy(info->cublas_handle, info->vsz, &neg_omega, info->dvv[4], 1, info->dvv[7], 1); // tmp = p - omega * v
            hipblasDscal(info->cublas_handle, info->vsz, &beta, info->dvv[7], 1);                       // tmp = beta * (p - omega * v)
            hipblasDaxpy(info->cublas_handle, info->vsz, &one, info->dvv[1], 1, info->dvv[7], 1);       // p = r + beta * (p - omega * v)
            hipMemcpy(info->dvv[3], info->dvv[7], info->vsz * sizeof(PetscReal), hipMemcpyDeviceToDevice);
            hipDeviceSynchronize();
            gettimeofday(&vx_tend, NULL);
            vx_t += ((double)((vx_tend.tv_sec * 1000000.0 + vx_tend.tv_usec) - (vx_tstart.tv_sec * 1000000.0 + vx_tstart.tv_usec))) / 1000.0;

            // p_tilde = M^-1 * p
            hipDeviceSynchronize();
            gettimeofday(&btrisv_tstart, NULL);
            PB_Preconditioning(ksp->pc, NULL, NULL, info->dvv[3], info->dvv[7], info->dv_tmp, info->vsz);
            hipDeviceSynchronize();
            gettimeofday(&btrisv_tend, NULL);
            btrisv_t += ((double)((btrisv_tend.tv_sec * 1000000.0 + btrisv_tend.tv_usec) - (btrisv_tstart.tv_sec * 1000000.0 + btrisv_tstart.tv_usec))) / 1000.0;

            // v = A * p_tilde
            alpha = 1.0;
            beta = 0.0;
            gettimeofday(&Ax_tstart, NULL);
            hipsparseDbsrmv(info->cusparse_handle, info->dir, info->trans, info->main_nrows, info->main_ncols,
                            info->main_nnz, &alpha, info->descr, info->dMainBlkVal, info->dMainRowPtr, info->dMainColVal, info->bs,
                            info->dvv[7], &beta, info->dvv[4]); // v = A_main * p_tilde
            hipDeviceSynchronize();
            gettimeofday(&Ax_tend, NULL);
            Ax_t += ((double)((Ax_tend.tv_sec * 1000000.0 + Ax_tend.tv_usec) - (Ax_tstart.tv_sec * 1000000.0 + Ax_tstart.tv_usec))) / 1000.0;

            // 将 p_tilde 从 GPU 复制到主机
            PetscReal *h_p_tilde;
            // record the host time
            gettimeofday(&host_tstart, NULL);
            VecGetArray(p_tilde_vec, &h_p_tilde);
            hipMemcpy(h_p_tilde, info->dvv[7], sizeof(PetscReal) * info->vsz, hipMemcpyDeviceToHost);
            VecRestoreArray(p_tilde_vec, &h_p_tilde);
            gettimeofday(&host_tend, NULL);
            host_t += ((double)((host_tend.tv_sec * 1000000.0 + host_tend.tv_usec) - (host_tstart.tv_sec * 1000000.0 + host_tstart.tv_usec))) / 1000.0;

            // 通信 p_tilde 的 halo 部分
            gettimeofday(&c_tstart, NULL);
            VecScatterBegin(info->Mvctx, p_tilde_vec, info->lvec, INSERT_VALUES, SCATTER_FORWARD);
            VecScatterEnd(info->Mvctx, p_tilde_vec, info->lvec, INSERT_VALUES, SCATTER_FORWARD);
            VecGetArray(info->lvec, &hlvec);
            hipMemcpy(info->dlvec, hlvec, sizeof(PetscReal) * info->lvsz, hipMemcpyHostToDevice);
            VecRestoreArray(info->lvec, &hlvec);
            gettimeofday(&c_tend, NULL);
            c_t += ((double)((c_tend.tv_sec * 1000000.0 + c_tend.tv_usec) - (c_tstart.tv_sec * 1000000.0 + c_tstart.tv_usec))) / 1000.0;

            alpha = 1.0;
            beta = 1.0;
            // record the Ax_tstart
            gettimeofday(&Ax_tstart, NULL);
            hipsparseDbsrmv(info->cusparse_handle, info->dir, info->trans, info->off_nrows, info->off_ncols,
                            info->off_nnz, &alpha, info->descr, info->dOffBlkVal, info->dOffRowPtr, info->dOffColVal, info->bs,
                            info->dlvec, &beta, info->dvv[4]); // v += A_off * p_tilde_halo
            hipDeviceSynchronize();
            // record the Ax_tend
            gettimeofday(&Ax_tend, NULL);
            Ax_t += ((double)((Ax_tend.tv_sec * 1000000.0 + Ax_tend.tv_usec) - (Ax_tstart.tv_sec * 1000000.0 + Ax_tstart.tv_usec))) / 1000.0;

            // alpha = rho / (rhat^T * v)
            double local_rhatv, global_rhatv;
            gettimeofday(&vd_tstart, NULL);
            hipblasDdot(info->cublas_handle, info->vsz, info->dvv[2], 1, info->dvv[4], 1, &local_rhatv);
            hipDeviceSynchronize();
            gettimeofday(&vd_tend, NULL);
            vd_t += ((double)((vd_tend.tv_sec * 1000000.0 + vd_tend.tv_usec) - (vd_tstart.tv_sec * 1000000.0 + vd_tstart.tv_usec))) / 1000.0;

            gettimeofday(&vr_tstart, NULL);
            MPI_Allreduce(&local_rhatv, &global_rhatv, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            gettimeofday(&vr_tend, NULL);
            vr_t += ((double)((vr_tend.tv_sec * 1000000.0 + vr_tend.tv_usec) - (vr_tstart.tv_sec * 1000000.0 + vr_tstart.tv_usec))) / 1000.0;

            // if (fabs(global_rhatv) < 1e-30)
            // {
            //     ksp->reason = KSP_DIVERGED_BREAKDOWN;
            //     ierr = PetscInfo(ksp, "BiCGSTAB breakdown: rhat^T v = 0\n");
            //     CHKERRQ(ierr);
            //     break;
            // }
            alpha_scalar = rho / global_rhatv;

            // s = r - alpha * v
            double neg_alpha = -alpha_scalar;
            gettimeofday(&vx_tstart, NULL);
            hipblasDcopy(info->cublas_handle, info->vsz, info->dvv[1], 1, info->dvv[5], 1);
            hipblasDaxpy(info->cublas_handle, info->vsz, &neg_alpha, info->dvv[4], 1, info->dvv[5], 1);
            hipDeviceSynchronize();
            gettimeofday(&vx_tend, NULL);
            vx_t += ((double)((vx_tend.tv_sec * 1000000.0 + vx_tend.tv_usec) - (vx_tstart.tv_sec * 1000000.0 + vx_tstart.tv_usec))) / 1000.0;

            // s_tilde = M^-1 * s

            gettimeofday(&btrisv_tstart, NULL);
            PB_Preconditioning(ksp->pc, NULL, NULL, info->dvv[5], info->dvv[8], info->dv_tmp, info->vsz);

            gettimeofday(&btrisv_tend, NULL);
            btrisv_t += ((double)((btrisv_tend.tv_sec * 1000000.0 + btrisv_tend.tv_usec) - (btrisv_tstart.tv_sec * 1000000.0 + btrisv_tstart.tv_usec))) / 1000.0;

            // t = A * s_tilde
            alpha = 1.0;
            beta = 0.0;
            gettimeofday(&Ax_tstart, NULL);
            hipsparseDbsrmv(info->cusparse_handle, info->dir, info->trans, info->main_nrows, info->main_ncols,
                            info->main_nnz, &alpha, info->descr, info->dMainBlkVal, info->dMainRowPtr, info->dMainColVal, info->bs,
                            info->dvv[8], &beta, info->dvv[6]); // t = A_main * s_tilde
            hipDeviceSynchronize();
            gettimeofday(&Ax_tend, NULL);
            Ax_t += ((double)((Ax_tend.tv_sec * 1000000.0 + Ax_tend.tv_usec) - (Ax_tstart.tv_sec * 1000000.0 + Ax_tstart.tv_usec))) / 1000.0;

            // 将 s_tilde 从 GPU 复制到主机
            PetscReal *h_s_tilde;
            // record the host time
            gettimeofday(&host_tstart, NULL);
            VecGetArray(s_tilde_vec, &h_s_tilde);
            hipMemcpy(h_s_tilde, info->dvv[8], sizeof(PetscReal) * info->vsz, hipMemcpyDeviceToHost);
            VecRestoreArray(s_tilde_vec, &h_s_tilde);
            gettimeofday(&host_tend, NULL);
            host_t += ((double)((host_tend.tv_sec * 1000000.0 + host_tend.tv_usec) - (host_tstart.tv_sec * 1000000.0 + host_tstart.tv_usec))) / 1000.0;

            // 通信 s_tilde 的 halo 部分
            gettimeofday(&c_tstart, NULL);
            VecScatterBegin(info->Mvctx, s_tilde_vec, info->lvec, INSERT_VALUES, SCATTER_FORWARD);
            VecScatterEnd(info->Mvctx, s_tilde_vec, info->lvec, INSERT_VALUES, SCATTER_FORWARD);
            VecGetArray(info->lvec, &hlvec);
            hipMemcpy(info->dlvec, hlvec, sizeof(PetscReal) * info->lvsz, hipMemcpyHostToDevice);
            VecRestoreArray(info->lvec, &hlvec);
            gettimeofday(&c_tend, NULL);
            c_t += ((double)((c_tend.tv_sec * 1000000.0 + c_tend.tv_usec) - (c_tstart.tv_sec * 1000000.0 + c_tstart.tv_usec))) / 1000.0;

            alpha = 1.0;
            beta = 1.0;
            // record the Ax_tstart
            gettimeofday(&Ax_tstart, NULL);
            hipsparseDbsrmv(info->cusparse_handle, info->dir, info->trans, info->off_nrows, info->off_ncols,
                            info->off_nnz, &alpha, info->descr, info->dOffBlkVal, info->dOffRowPtr, info->dOffColVal, info->bs,
                            info->dlvec, &beta, info->dvv[6]); // t += A_off * s_tilde_halo
            hipDeviceSynchronize();
            // record the Ax_tend
            gettimeofday(&Ax_tend, NULL);
            Ax_t += ((double)((Ax_tend.tv_sec * 1000000.0 + Ax_tend.tv_usec) - (Ax_tstart.tv_sec * 1000000.0 + Ax_tstart.tv_usec))) / 1000.0;

            // omega = (t^T * s) / (t^T * t)
            double local_ts, global_ts, local_tt, global_tt;
            gettimeofday(&vd_tstart, NULL);
            hipblasDdot(info->cublas_handle, info->vsz, info->dvv[6], 1, info->dvv[5], 1, &local_ts);
            hipblasDdot(info->cublas_handle, info->vsz, info->dvv[6], 1, info->dvv[6], 1, &local_tt);
            hipDeviceSynchronize();
            gettimeofday(&vd_tend, NULL);
            vd_t += ((double)((vd_tend.tv_sec * 1000000.0 + vd_tend.tv_usec) - (vd_tstart.tv_sec * 1000000.0 + vd_tstart.tv_usec))) / 1000.0;

            gettimeofday(&vr_tstart, NULL);
            MPI_Allreduce(&local_ts, &global_ts, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(&local_tt, &global_tt, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

            gettimeofday(&vr_tend, NULL);
            vr_t += ((double)((vr_tend.tv_sec * 1000000.0 + vr_tend.tv_usec) - (vr_tstart.tv_sec * 1000000.0 + vr_tstart.tv_usec))) / 1000.0;

            // double vr_tstart_d = MPI_Wtime();
            // MPI_Allreduce(&local_ts, &global_ts, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            // MPI_Allreduce(&local_tt, &global_tt, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            // double vr_tend_d = MPI_Wtime();
            // vr_t += (vr_tend_d - vr_tstart_d) * 1000.0; // 转换为毫秒

            // if (fabs(global_tt) < 1e-30)
            // {
            //     ksp->reason = KSP_DIVERGED_BREAKDOWN;
            //     ierr = PetscInfo(ksp, "BiCGSTAB breakdown: t^T t = 0\n");
            //     CHKERRQ(ierr);
            //     break;
            // }
            omega = global_ts / global_tt;

            // x = x + alpha * p_tilde + omega * s_tilde (直接更新 info->dsol)
            gettimeofday(&vx_tstart, NULL);
            hipblasDaxpy(info->cublas_handle, info->vsz, &alpha_scalar, info->dvv[7], 1, info->dsol, 1); // x += alpha * p_tilde
            hipblasDaxpy(info->cublas_handle, info->vsz, &omega, info->dvv[8], 1, info->dsol, 1);        // x += omega * s_tilde
            hipDeviceSynchronize();
            gettimeofday(&vx_tend, NULL);
            vx_t += ((double)((vx_tend.tv_sec * 1000000.0 + vx_tend.tv_usec) - (vx_tstart.tv_sec * 1000000.0 + vx_tstart.tv_usec))) / 1000.0;

            // r = s - omega * t
            neg_omega = -omega;
            // record the vx_tstart
            gettimeofday(&vx_tstart, NULL);
            hipblasDcopy(info->cublas_handle, info->vsz, info->dvv[5], 1, info->dvv[1], 1);
            hipblasDaxpy(info->cublas_handle, info->vsz, &neg_omega, info->dvv[6], 1, info->dvv[1], 1);
            hipDeviceSynchronize();
            // record the vx_tend
            gettimeofday(&vx_tend, NULL);
            vx_t += ((double)((vx_tend.tv_sec * 1000000.0 + vx_tend.tv_usec) - (vx_tstart.tv_sec * 1000000.0 + vx_tstart.tv_usec))) / 1000.0;

            // 更新残差范数
            gettimeofday(&vd_tstart, NULL);
            hipblasDdot(info->cublas_handle, info->vsz, info->dvv[1], 1, info->dvv[1], 1, &local_res_norm);
            hipDeviceSynchronize();
            gettimeofday(&vd_tend, NULL);
            vd_t += ((double)((vd_tend.tv_sec * 1000000.0 + vd_tend.tv_usec) - (vd_tstart.tv_sec * 1000000.0 + vd_tstart.tv_usec))) / 1000.0;

            gettimeofday(&vr_tstart, NULL);
            MPI_Allreduce(&local_res_norm, &global_res_norm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            gettimeofday(&vr_tend, NULL);
            vr_t += ((double)((vr_tend.tv_sec * 1000000.0 + vr_tend.tv_usec) - (vr_tstart.tv_sec * 1000000.0 + vr_tstart.tv_usec))) / 1000.0;

            // ksp->its++;

            res_norm = PetscSqrtReal(global_res_norm);
            ksp->rnorm = res_norm;
#ifdef KSP_MONITOR
            KSPLogResidualHistory(ksp, res_norm);
            KSPMonitor(ksp, ksp->its, res_norm);
#endif
            (*ksp->converged)(ksp, ksp->its, res_norm, &ksp->reason, ksp->cnvP);
            rho_old = rho;
            ksp->its++;
        }

        gettimeofday(&tend, NULL);
        telapse += ((double)((tend.tv_sec * 1000000.0 + tend.tv_usec) - (tstart.tv_sec * 1000000.0 + tstart.tv_usec))) / 1000.0;

        if (ksp->its >= ksp->max_it && !ksp->reason)
        {
            ksp->reason = KSP_DIVERGED_ITS;
            ierr = PetscInfo(ksp, "BiCGSTAB diverged: maximum iterations reached\n");
            CHKERRQ(ierr);
        }
        hipDeviceSynchronize();
        gettimeofday(&total_tend, NULL);
        total_t += ((double)((total_tend.tv_sec * 1000000.0 + total_tend.tv_usec) - (total_tstart.tv_sec * 1000000.0 + total_tstart.tv_usec))) / 1000.0;
    }

    if (rank == 0)
    {
        // 打印MILU性能统计
        // CudaTimer::printStats(result_file);
        // // 重置计时统计，为下一次GMRES迭代做准备
        // CudaTimer::resetStats();
        printf("rank=%d, total_t=%12.8lf, Ax_t=%12.8lf, c_t=%12.8lf, vd_t=%12.8lf, vr_t=%12.8lf, vx_t=%12.8lf, host_t=%12.8lf, btrisv_t=%12.8lf, telapse=%12.8lf\n",
               info->rank, total_t / run_times, Ax_t / run_times, c_t / run_times, vd_t / run_times, vr_t / run_times, vx_t / run_times, host_t / run_times, btrisv_t / run_times, telapse / run_times);
        printf("pack_time=%12.8lf, unpack_time=%12.8lf,  comm_time=%12.8lf, ASMLvecToVec_time=%12.8lf\n",
               pack_time / run_times, unpack_time / run_times, comm_time / run_times, ASMLvecToVec_time / run_times);
        printf("solve_time in fast_solve=%12.8lf, gemv_time in fast_solve=%12.8lf, construct_tEFG_time=%12.8lf, collect_boundary_values_time=%12.8lf\n", solve_time / run_times, gemv_time / run_times, construct_tEFG_time / run_times, collect_boundary_values_time / run_times);
    }

    // 将最终解从 info->dsol 同步到主机
    VecGetArray(info->sol, &hsol);
    hipMemcpy(hsol, info->dsol, sizeof(PetscReal) * info->vsz, hipMemcpyDeviceToHost);
    VecRestoreArray(info->sol, &hsol);

cleanup:
    ksp->guess_zero = guess_zero;
    hipStreamDestroy(Axstream);
    VecDestroy(&p_tilde_vec); // 释放临时向量
    VecDestroy(&s_tilde_vec);

    if (ksp->reason == KSP_CONVERGED_ITERATING)
    {
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "BiCGSTAB internal error: exiting with unconverged reason");
    }

    PetscFunctionReturn(0);
}

PetscErrorCode KSPSolve_CG_GPU(KSP ksp)
{
    printf("KSPSolve_CG_GPU\n");
    PetscErrorCode ierr;
    PetscInt its;
    PetscBool guess_zero = ksp->guess_zero;
    Mat Amat, Pmat;
    PetscReal res_norm, res_norm_old;
    PetscInt rank;

    PetscReal *hsol, *hlvec;
    double alpha, beta, one = 1.0, neg_one = -1.0;
    double local_res_norm, global_res_norm;
    double local_rtr, global_rtr, local_pAp, global_pAp;
    PetscInt nblocks;

    // Timing variables (for loop only)
    struct timeval tstart, tend, total_tstart, total_tend;
    double telapse = 0.0, total_t = 0.0;
    struct timeval Ax_tstart, Ax_tend;
    double Ax_t = 0.0; // Matrix-vector multiplication
    struct timeval c_tstart, c_tend;
    double c_t = 0.0; // Communication (scatter + memcpy)
    struct timeval vd_tstart, vd_tend;
    double vd_t = 0.0; // Vector dot products
    struct timeval vr_tstart, vr_tend;
    double vr_t = 0.0; // MPI reductions
    struct timeval vx_tstart, vx_tend;
    double vx_t = 0.0; // Vector updates (axpy, scal)
    struct timeval host_tstart, host_tend;
    double host_t = 0.0; // Host-side vector ops
    struct timeval cpu_tstart, cpu_tend;
    double cpu_t = 0.0; // CPU-side computations
    struct timeval log_tstart, log_tend;
    double log_t = 0.0; // Logging and convergence checks

    hipStream_t Axstream;
    hipError_t hip_err;

    Vec p_vec; // Temporary vector for p on host

    PetscFunctionBegin;
    printf("KSPSolve_CG_GPU 1\n");

    // Initialization (no timing, as requested)
    ierr = PetscObjectSAWsTakeAccess((PetscObject)ksp);
    CHKERRQ(ierr);
    ksp->its = 0;
    ksp->reason = KSP_CONVERGED_ITERATING;
    ierr = PetscObjectSAWsGrantAccess((PetscObject)ksp);
    CHKERRQ(ierr);
    ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    CHKERRQ(ierr);
    printf("KSPSolve_CG_GPU 2\n");

    ierr = VecDuplicate(ksp->vec_sol, &p_vec);
    CHKERRQ(ierr);
    printf("KSPSolve_CG_GPU 3\n");

    hip_err = hipStreamCreate(&Axstream);
    if (hip_err != hipSuccess)
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, "Failed to create HIP stream");
    hipsparseSetStream(info->cusparse_handle, Axstream);
    nblocks = (info->buf_len - 1) / THREADS_PER_BLOCK + 1;
    printf("KSPSolve_CG_GPU 4\n");

    InitialGMRES_INFO_CPU(ksp);
    InitialGMRES_INFO_GPU();
    printf("KSPSolve_CG_GPU 5\n");

    if (!ksp->pc)
    {
        KSPGetPC(ksp, &ksp->pc);
    }
    PCGetOperators(ksp->pc, &Amat, &Pmat);
    printf("KSPSolve_CG_GPU 6\n");

    VecGetArray(info->sol, &hsol);
    hipMemcpy(info->dsol, hsol, sizeof(PetscReal) * info->vsz, hipMemcpyHostToDevice);
    VecRestoreArray(info->sol, &hsol);
    printf("KSPSolve_CG_GPU 7\n");

    MPI_Barrier(PETSC_COMM_WORLD);
    if (guess_zero)
    {
        hipblasDcopy(info->cublas_handle, info->vsz, info->drhs, 1, info->dvv[1], 1); // r0 = b
        hipDeviceSynchronize();
    }
    else
    {
        alpha = 1.0;
        beta = 0.0;
        hipsparseDbsrmv(info->cusparse_handle, info->dir, info->trans, info->main_nrows, info->main_ncols,
                        info->main_nnz, &alpha, info->descr, info->dMainBlkVal, info->dMainRowPtr, info->dMainColVal, info->bs,
                        info->dsol, &beta, info->dvv[0]); // temp = A * x0
        hipDeviceSynchronize();

        VecScatterBegin(info->Mvctx, info->sol, info->lvec, INSERT_VALUES, SCATTER_FORWARD);
        VecScatterEnd(info->Mvctx, info->sol, info->lvec, INSERT_VALUES, SCATTER_FORWARD);
        VecGetArray(info->lvec, &hlvec);
        hipMemcpy(info->dlvec, hlvec, sizeof(PetscReal) * info->lvsz, hipMemcpyHostToDevice);
        VecRestoreArray(info->lvec, &hlvec);

        alpha = 1.0;
        beta = 1.0;
        hipsparseDbsrmv(info->cusparse_handle, info->dir, info->trans, info->off_nrows, info->off_ncols,
                        info->off_nnz, &alpha, info->descr, info->dOffBlkVal, info->dOffRowPtr, info->dOffColVal, info->bs,
                        info->dlvec, &beta, info->dvv[0]);
        hipDeviceSynchronize();

        hipblasDcopy(info->cublas_handle, info->vsz, info->drhs, 1, info->dvv[1], 1);
        hipblasDaxpy(info->cublas_handle, info->vsz, &neg_one, info->dvv[0], 1, info->dvv[1], 1); // r0 = b - A * x0
        hipDeviceSynchronize();
    }

    hipblasDcopy(info->cublas_handle, info->vsz, info->dvv[1], 1, info->dvv[2], 1); // p = r
    hipDeviceSynchronize();

    hipblasDdot(info->cublas_handle, info->vsz, info->dvv[1], 1, info->dvv[1], 1, &local_res_norm);
    hipDeviceSynchronize();
    MPI_Allreduce(&local_res_norm, &global_res_norm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    res_norm = PetscSqrtReal(global_res_norm);
    ksp->rnorm = res_norm;
    KSPLogResidualHistory(ksp, res_norm);
    KSPMonitor(ksp, 0, res_norm);
    res_norm_old = res_norm;

    if (res_norm < ksp->abstol)
    {
        ksp->reason = KSP_CONVERGED_ATOL;
        ierr = PetscInfo(ksp, "Converged due to residual norm below absolute tolerance on entry\n");
        CHKERRQ(ierr);
        goto cleanup;
    }
    (*ksp->converged)(ksp, ksp->its, res_norm, &ksp->reason, ksp->cnvP);

    // CG iteration loop (total_t starts here)
    MPI_Barrier(PETSC_COMM_WORLD);
    gettimeofday(&total_tstart, NULL); // Start total timing at loop entry
    gettimeofday(&tstart, NULL);
    ksp->its = 1;
    while (!ksp->reason && ksp->its < ksp->max_it)
    {
        // v = A * p
        alpha = 1.0;
        beta = 0.0;
        gettimeofday(&Ax_tstart, NULL);
        hipsparseDbsrmv(info->cusparse_handle, info->dir, info->trans, info->main_nrows, info->main_ncols,
                        info->main_nnz, &alpha, info->descr, info->dMainBlkVal, info->dMainRowPtr, info->dMainColVal, info->bs,
                        info->dvv[2], &beta, info->dvv[0]); // v = A_main * p
        hipDeviceSynchronize();
        gettimeofday(&Ax_tend, NULL);
        Ax_t += ((double)((Ax_tend.tv_sec * 1000000.0 + Ax_tend.tv_usec) - (Ax_tstart.tv_sec * 1000000.0 + Ax_tstart.tv_usec))) / 1000.0;

        // Sync p to host-side p_vec
        gettimeofday(&host_tstart, NULL);
        VecGetArray(p_vec, &hsol);
        hipMemcpy(hsol, info->dvv[2], sizeof(PetscReal) * info->vsz, hipMemcpyDeviceToHost);
        VecRestoreArray(p_vec, &hsol);
        gettimeofday(&host_tend, NULL);
        host_t += ((double)((host_tend.tv_sec * 1000000.0 + host_tend.tv_usec) - (host_tstart.tv_sec * 1000000.0 + host_tstart.tv_usec))) / 1000.0;

        // Communication for halo
        gettimeofday(&c_tstart, NULL);
        VecScatterBegin(info->Mvctx, p_vec, info->lvec, INSERT_VALUES, SCATTER_FORWARD);
        VecScatterEnd(info->Mvctx, p_vec, info->lvec, INSERT_VALUES, SCATTER_FORWARD);
        VecGetArray(info->lvec, &hlvec);
        hipMemcpy(info->dlvec, hlvec, sizeof(PetscReal) * info->lvsz, hipMemcpyHostToDevice);
        VecRestoreArray(info->lvec, &hlvec);
        gettimeofday(&c_tend, NULL);
        c_t += ((double)((c_tend.tv_sec * 1000000.0 + c_tend.tv_usec) - (c_tstart.tv_sec * 1000000.0 + c_tstart.tv_usec))) / 1000.0;

        alpha = 1.0;
        beta = 1.0;
        gettimeofday(&Ax_tstart, NULL);
        hipsparseDbsrmv(info->cusparse_handle, info->dir, info->trans, info->off_nrows, info->off_ncols,
                        info->off_nnz, &alpha, info->descr, info->dOffBlkVal, info->dOffRowPtr, info->dOffColVal, info->bs,
                        info->dlvec, &beta, info->dvv[0]); // v += A_off * p_halo
        hipDeviceSynchronize();
        gettimeofday(&Ax_tend, NULL);
        Ax_t += ((double)((Ax_tend.tv_sec * 1000000.0 + Ax_tend.tv_usec) - (Ax_tstart.tv_sec * 1000000.0 + Ax_tstart.tv_usec))) / 1000.0;

        // alpha = (r^T r) / (p^T A p)
        gettimeofday(&vd_tstart, NULL);
        hipblasDdot(info->cublas_handle, info->vsz, info->dvv[2], 1, info->dvv[0], 1, &local_pAp); // p^T A p
        hipblasDdot(info->cublas_handle, info->vsz, info->dvv[1], 1, info->dvv[1], 1, &local_rtr); // r^T r
        hipDeviceSynchronize();
        gettimeofday(&vd_tend, NULL);
        vd_t += ((double)((vd_tend.tv_sec * 1000000.0 + vd_tend.tv_usec) - (vd_tstart.tv_sec * 1000000.0 + vd_tstart.tv_usec))) / 1000.0;

        gettimeofday(&vr_tstart, NULL);
        MPI_Allreduce(&local_pAp, &global_pAp, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&local_rtr, &global_rtr, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        gettimeofday(&vr_tend, NULL);
        vr_t += ((double)((vr_tend.tv_sec * 1000000.0 + vr_tend.tv_usec) - (vr_tstart.tv_sec * 1000000.0 + vr_tstart.tv_usec))) / 1000.0;

        gettimeofday(&cpu_tstart, NULL);
        if (fabs(global_pAp) < 1e-30)
        {
            ksp->reason = KSP_DIVERGED_BREAKDOWN;
            ierr = PetscInfo(ksp, "CG breakdown: p^T A p = 0\n");
            CHKERRQ(ierr);
            break;
        }
        alpha = global_rtr / global_pAp;
        gettimeofday(&cpu_tend, NULL);
        cpu_t += ((double)((cpu_tend.tv_sec * 1000000.0 + cpu_tend.tv_usec) - (cpu_tstart.tv_sec * 1000000.0 + cpu_tstart.tv_usec))) / 1000.0;

        // x = x + alpha * p
        gettimeofday(&vx_tstart, NULL);
        hipblasDaxpy(info->cublas_handle, info->vsz, &alpha, info->dvv[2], 1, info->dsol, 1);
        hipDeviceSynchronize();
        gettimeofday(&vx_tend, NULL);
        vx_t += ((double)((vx_tend.tv_sec * 1000000.0 + vx_tend.tv_usec) - (vx_tstart.tv_sec * 1000000.0 + vx_tstart.tv_usec))) / 1000.0;

        // r = r - alpha * A p
        double neg_alpha = -alpha;
        gettimeofday(&vx_tstart, NULL);
        hipblasDaxpy(info->cublas_handle, info->vsz, &neg_alpha, info->dvv[0], 1, info->dvv[1], 1);
        hipDeviceSynchronize();
        gettimeofday(&vx_tend, NULL);
        vx_t += ((double)((vx_tend.tv_sec * 1000000.0 + vx_tend.tv_usec) - (vx_tstart.tv_sec * 1000000.0 + vx_tstart.tv_usec))) / 1000.0;

        // Compute new residual norm
        gettimeofday(&vd_tstart, NULL);
        hipblasDdot(info->cublas_handle, info->vsz, info->dvv[1], 1, info->dvv[1], 1, &local_res_norm);
        hipDeviceSynchronize();
        gettimeofday(&vd_tend, NULL);
        vd_t += ((double)((vd_tend.tv_sec * 1000000.0 + vd_tend.tv_usec) - (vd_tstart.tv_sec * 1000000.0 + vd_tstart.tv_usec))) / 1000.0;

        gettimeofday(&vr_tstart, NULL);
        MPI_Allreduce(&local_res_norm, &global_res_norm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        gettimeofday(&vr_tend, NULL);
        vr_t += ((double)((vr_tend.tv_sec * 1000000.0 + vr_tend.tv_usec) - (vr_tstart.tv_sec * 1000000.0 + vr_tstart.tv_usec))) / 1000.0;

        gettimeofday(&cpu_tstart, NULL);
        res_norm = PetscSqrtReal(global_res_norm);
        beta = res_norm * res_norm / (res_norm_old * res_norm_old);
        res_norm_old = res_norm;
        gettimeofday(&cpu_tend, NULL);
        cpu_t += ((double)((cpu_tend.tv_sec * 1000000.0 + cpu_tend.tv_usec) - (cpu_tstart.tv_sec * 1000000.0 + cpu_tstart.tv_usec))) / 1000.0;

        gettimeofday(&log_tstart, NULL);
        ksp->rnorm = res_norm;
        KSPLogResidualHistory(ksp, res_norm);
        KSPMonitor(ksp, ksp->its, res_norm);
        (*ksp->converged)(ksp, ksp->its, res_norm, &ksp->reason, ksp->cnvP);
        gettimeofday(&log_tend, NULL);
        log_t += ((double)((log_tend.tv_sec * 1000000.0 + log_tend.tv_usec) - (log_tstart.tv_sec * 1000000.0 + log_tstart.tv_usec))) / 1000.0;

        // p = r + beta * p
        gettimeofday(&vx_tstart, NULL);
        hipblasDscal(info->cublas_handle, info->vsz, &beta, info->dvv[2], 1);                 // p = beta * p
        hipblasDaxpy(info->cublas_handle, info->vsz, &one, info->dvv[1], 1, info->dvv[2], 1); // p = r + beta * p
        hipDeviceSynchronize();
        gettimeofday(&vx_tend, NULL);
        vx_t += ((double)((vx_tend.tv_sec * 1000000.0 + vx_tend.tv_usec) - (vx_tstart.tv_sec * 1000000.0 + vx_tstart.tv_usec))) / 1000.0;

        ksp->its++;
    }

    gettimeofday(&tend, NULL);
    telapse = ((double)((tend.tv_sec * 1000000.0 + tend.tv_usec) - (tstart.tv_sec * 1000000.0 + tstart.tv_usec))) / 1000.0;

    if (ksp->its >= ksp->max_it && !ksp->reason)
    {
        ksp->reason = KSP_DIVERGED_ITS;
        ierr = PetscInfo(ksp, "CG diverged: maximum iterations reached\n");
        CHKERRQ(ierr);
    }

    // End total timing at loop exit
    gettimeofday(&total_tend, NULL);
    total_t = ((double)((total_tend.tv_sec * 1000000.0 + total_tend.tv_usec) - (total_tstart.tv_sec * 1000000.0 + total_tstart.tv_usec))) / 1000.0;

    // Print detailed timing (loop only)
    if (rank == 0)
    {
        printf("rank=%d, total_t=%12.8lf, Ax_t=%12.8lf, c_t=%12.8lf, vd_t=%12.8lf, vr_t=%12.8lf, vx_t=%12.8lf, host_t=%12.8lf, cpu_t=%12.8lf, log_t=%12.8lf, telapse=%12.8lf, iterations=%d\n",
               rank, total_t, Ax_t, c_t, vd_t, vr_t, vx_t, host_t, cpu_t, log_t, telapse, ksp->its);
    }

    // Cleanup (no timing)
    VecGetArray(info->sol, &hsol);
    hipMemcpy(hsol, info->dsol, sizeof(PetscReal) * info->vsz, hipMemcpyDeviceToHost);
    VecRestoreArray(info->sol, &hsol);

    ksp->guess_zero = guess_zero;
    hipStreamDestroy(Axstream);
    ierr = VecDestroy(&p_vec);
    CHKERRQ(ierr);

cleanup:
    if (ksp->reason == KSP_CONVERGED_ITERATING)
    {
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "CG internal error: exiting with unconverged reason");
    }

    PetscFunctionReturn(0);
}
