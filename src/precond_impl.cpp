#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include "KSPSolve_GMRES_GPU.h"
#include <hipblas.h>
#include <hip/hip_runtime.h>
#include <hipsparse.h>
#include "CudaTimer.h"
#include "fast_solve.h"


#define threads_per_block_fast_solve 256
#define blocks_fast_solve_pn (PN + threads_per_block_fast_solve - 1) / threads_per_block_fast_solve
#define blocks_fast_solve_num_boundary (NUM_BOUNDARY + threads_per_block_fast_solve - 1) / threads_per_block_fast_solve
 





#define CHECK_HIP(func)                                                    \
    do {                                                                   \
        hipError_t status = (func);                                        \
        if (status != hipSuccess) {                                        \
            printf("HIP API 错误 at line %d in file %s: %s (%d)\n",        \
                   __LINE__, __FILE__, hipGetErrorString(status), status); \
            exit(1);                                                       \
        }                                                                  \
    } while (0)

extern GMRES_INFO *info;
__global__ void ASMVecToSendbuffer(double *vec,int vec_len, 
				   double *asm_dsend_buf, int *asm_dsindices, int asm_sendbuf_len)
{
	int tid=threadIdx.x+blockDim.x*blockIdx.x;
	int vecidx;
	if(tid<asm_sendbuf_len)
	{
		vecidx=asm_dsindices[tid];
		asm_dsend_buf[tid]=vec[vecidx];
	}
	
}
//   x + recvbuf -> lx
__global__ void ASMVecRecvbufferToLvec(double *asm_drecv_buf, int *asm_drindices, int asm_recvbuf_len,
				    double *vec, int *asm_self_dsindices, int vec_len,
				    double *asm_lvec, int *asm_self_drindices, int asm_lvec_len)
{
	int tid=threadIdx.x+blockDim.x*blockIdx.x;
	int lvecidx;
	int vecidx;
	//1. vec->lvec
	if(tid<vec_len)
	{
		vecidx=asm_self_dsindices[tid];
		lvecidx=asm_self_drindices[tid];
		asm_lvec[lvecidx]=vec[vecidx];	
	}
	if(tid<asm_recvbuf_len)
	{
		lvecidx=asm_drindices[tid];
		asm_lvec[lvecidx]=asm_drecv_buf[tid];
	}
}
// ly -> recvbuffer
__global__ void ASMLvecToRecvbuffer(double *asm_lvec, 
				    double *asm_drecv_buf, int  *asm_drindices, int asm_recvbuf_len) 
{
	int tid=threadIdx.x + blockDim.x * blockIdx.x;
	int lvecidx;
	if(tid<asm_recvbuf_len)
	{
		lvecidx=asm_drindices[tid];
		asm_drecv_buf[tid]=asm_lvec[lvecidx];
	}	
}
// after recvbuffer -> sendbuffer,  ly + send_buffer -> y
__global__ void ASMLvecToVec(  double *asm_lvec, int *asm_self_drindices, int asm_lvec_len,
			       double *vec,      int *asm_self_dsindices, int vec_len )
{
	int tid=threadIdx.x + blockDim.x * blockIdx.x;
	int vecidx;
	int lvecidx;
	if(tid<vec_len)
	{
		vecidx=asm_self_dsindices[tid];
		lvecidx=asm_self_drindices[tid];
		vec[vecidx]=asm_lvec[lvecidx];
	}	
}
__global__ void ASMSendbufferAddtoVec(double *asm_dsend_buf, int *asm_dsindices, int asm_sendbuf_len,
				     double *vec, int vec_len)
{
	// we don't use atomic operation, so we need to do processors one by one 
	// the asm_dsend_buf, asm_dsindices, asm_sendbuf_len is decalred by processor,
	// and this function is going to be called many times to accumulate values to vec
	int tid=threadIdx.x + blockDim.x * blockIdx.x;
	int vecidx;
	if(tid<asm_sendbuf_len)
	{
		vecidx=asm_dsindices[tid];
		vec[vecidx]+=asm_dsend_buf[tid];
	}

}
//ASM restriction communication
void ASMStartRestrictComm()
{
	int i;
	int asm_snp=info->asm_snp;
	int asm_rnp=info->asm_rnp;
	//int bs=info->bs;
	int asm_tag=info->rank;
	for(i=0;i<asm_snp;i++)
	{
		MPI_Isend(info->asm_dsend_buf+info->asm_sstarts[i],(info->asm_sstarts[i+1]-info->asm_sstarts[i]),
		MPI_DOUBLE,info->asm_sprocs[i],asm_tag,MPI_COMM_WORLD,info->asm_swaits+i);
	}
	for(i=0;i<asm_rnp;i++)
	{
		MPI_Irecv(info->asm_drecv_buf+info->asm_rstarts[i],(info->asm_rstarts[i+1]-info->asm_rstarts[i]),
		MPI_DOUBLE,info->asm_rprocs[i],info->asm_rprocs[i],MPI_COMM_WORLD,info->asm_rwaits+i);
	}
	 
}
void ASMEndRestrictComm()
{
	MPI_Status stat;
	int i;
	for(i=0;i<info->asm_rnp;i++)
	{
		MPI_Wait(info->asm_rwaits+i,&stat);
		//printf("rank %d wait successfully\n",info->rank);
	}
}
void ASMStartProlongationComm()
{
	
	int i;
	int asm_snp=info->asm_snp;
	int asm_rnp=info->asm_rnp;
	int asm_tag=info->rank;
	for(i=0;i<asm_rnp;i++)
	{
		MPI_Isend(info->asm_drecv_buf+info->asm_rstarts[i],(info->asm_rstarts[i+1]-info->asm_rstarts[i]),
		MPI_DOUBLE,info->asm_rprocs[i],asm_tag,MPI_COMM_WORLD,info->asm_rwaits+i);
	}
	for(i=0;i<asm_snp;i++)
	{
		MPI_Irecv(info->asm_dsend_buf+info->asm_sstarts[i],(info->asm_sstarts[i+1]-info->asm_sstarts[i]),
		MPI_DOUBLE,info->asm_sprocs[i],info->asm_sprocs[i],MPI_COMM_WORLD,info->asm_swaits+i);
	}
}
void ASMEndProlongationComm()
{
	MPI_Status stat;
	int i;
	for(i=0;i<info->asm_snp;i++)
	{
		MPI_Wait(info->asm_swaits+i,&stat);
		//printf("rank %d wait successfully\n",info->rank);
	}
	
}

// suppose we already got a point-block ILU(k) factorization : L and U 
// we need to get the incomplete inverse of L and U, denoted as InvL and InvU  
extern GMRES_INFO *info;
__global__ void ISAI_kernel_L_V2(int *dInvLColPtr,
								 int *dInvLRowVal,
								 double *dInvLBlkVal,
								 int *dLRowPtr,
								 int *dLColVal,
								 double *dLBlkVal,
								 double *dRhsLBlkVal,
								 int fact_n)
{
	__shared__ double s[WARPS_PER_BLOCK][WARP_SIZE];
	__shared__ double inv[WARPS_PER_BLOCK][NVARS][NVARS];
	__shared__ double ssol[WARPS_PER_BLOCK][NVARS][NVARS];
	int tid=threadIdx.x+blockIdx.x*blockDim.x+blockIdx.y*gridDim.x*blockDim.x;
	int wpid=threadIdx.x/WARP_SIZE;
	int gwpid=tid/WARP_SIZE;
	int lane=threadIdx.x%WARP_SIZE;
	int r,c,irow,myrow;
	int InvStIdx,InvEdIdx,idx,tmpidx,myidx;
	int range=WARP_SIZE/(NVARS*NVARS)*(NVARS*NVARS);
	if(gwpid < fact_n)
	{
		if(lane < range)
		{
			r=lane%NVARS;
			c=(lane/NVARS)%NVARS;
			InvStIdx=dInvLColPtr[gwpid];
			InvEdIdx=dInvLColPtr[gwpid+1];
			for(idx=InvStIdx;idx<InvEdIdx;idx++)
			{
				irow=dInvLRowVal[idx];
				if(lane < NVARS*NVARS)
				{
					tmpidx=dLRowPtr[irow+1]-1;
					inv[wpid][r][c]=dLBlkVal[tmpidx*NVARS*NVARS+lane];
					s[wpid][lane]=dRhsLBlkVal[idx*NVARS*NVARS+lane];	
					if(NVARS==4)
					{
					ssol[wpid][r][c]=inv[wpid][r][0]*s[wpid][c*NVARS+0]
									+inv[wpid][r][1]*s[wpid][c*NVARS+1]
									+inv[wpid][r][2]*s[wpid][c*NVARS+2]
									+inv[wpid][r][3]*s[wpid][c*NVARS+3]; // for NVARS=4
					}
					if(NVARS==3)
					{
					ssol[wpid][r][c]=inv[wpid][r][0]*s[wpid][c*NVARS+0]
									+inv[wpid][r][1]*s[wpid][c*NVARS+1]
									+inv[wpid][r][2]*s[wpid][c*NVARS+2];
					}
					if(NVARS==5)
					{
					ssol[wpid][r][c]=inv[wpid][r][0]*s[wpid][c*NVARS+0]
								+inv[wpid][r][1]*s[wpid][c*NVARS+1]
								+inv[wpid][r][2]*s[wpid][c*NVARS+2]
								+inv[wpid][r][3]*s[wpid][c*NVARS+3]
								+inv[wpid][r][4]*s[wpid][c*NVARS+4];

					}
					dInvLBlkVal[idx*NVARS*NVARS+lane]=ssol[wpid][r][c];
				}
				myidx=idx+1+lane/(NVARS*NVARS);
				while(myidx < InvEdIdx)
				{
					s[wpid][lane]=0.0;
					myrow=dInvLRowVal[myidx];
					tmpidx=dLRowPtr[myrow+1]-1;
					while(tmpidx >= dLRowPtr[myrow] && dLColVal[tmpidx] > irow){tmpidx--;}
					if(tmpidx >= dLRowPtr[myrow] && dLColVal[tmpidx] == irow)
					{
						//load into s
						s[wpid][lane]=dLBlkVal[tmpidx*NVARS*NVARS+lane%(NVARS*NVARS)];	
					}
					if(NVARS==4)
					{
						dRhsLBlkVal[myidx*NVARS*NVARS+lane%(NVARS*NVARS)]-=s[wpid][lane/(NVARS*NVARS)*(NVARS*NVARS)+0*NVARS+r]*ssol[wpid][0][c]
								   +s[wpid][lane/(NVARS*NVARS)*(NVARS*NVARS)+1*NVARS+r]*ssol[wpid][1][c]
								   +s[wpid][lane/(NVARS*NVARS)*(NVARS*NVARS)+2*NVARS+r]*ssol[wpid][2][c]
								   +s[wpid][lane/(NVARS*NVARS)*(NVARS*NVARS)+3*NVARS+r]*ssol[wpid][3][c]; // for NVARS=4
					}
					if(NVARS==3)
					{
						 dRhsLBlkVal[myidx*NVARS*NVARS+lane%(NVARS*NVARS)]-=s[wpid][lane/(NVARS*NVARS)*(NVARS*NVARS)+0*NVARS+r]*ssol[wpid][0][c]
								   +s[wpid][lane/(NVARS*NVARS)*(NVARS*NVARS)+1*NVARS+r]*ssol[wpid][1][c]
								   +s[wpid][lane/(NVARS*NVARS)*(NVARS*NVARS)+2*NVARS+r]*ssol[wpid][2][c];
					}
					if(NVARS==5)
					{
					dRhsLBlkVal[myidx*NVARS*NVARS+lane%(NVARS*NVARS)]-=s[wpid][lane/(NVARS*NVARS)*(NVARS*NVARS)+0*NVARS+r]*ssol[wpid][0][c]
								   +s[wpid][lane/(NVARS*NVARS)*(NVARS*NVARS)+1*NVARS+r]*ssol[wpid][1][c]
								   +s[wpid][lane/(NVARS*NVARS)*(NVARS*NVARS)+2*NVARS+r]*ssol[wpid][2][c]
								   +s[wpid][lane/(NVARS*NVARS)*(NVARS*NVARS)+3*NVARS+r]*ssol[wpid][3][c]
								   +s[wpid][lane/(NVARS*NVARS)*(NVARS*NVARS)+4*NVARS+r]*ssol[wpid][4][c];
					}

					myidx+=WARP_SIZE/(NVARS*NVARS);
				}
			}

		}
	}
	
}

__global__ void ISAI_kernel_U_V2(int *dInvUColPtr,
								 int *dInvURowVal,
								 double *dInvUBlkVal,
								 int *dURowPtr,
								 int *dUColVal,
								 double *dUBlkVal,
								 double *dRhsUBlkVal,
								 int fact_n)
{
	//__shared__ double a[WARPS_PER_BLOCK][NVARS][NVARS];
	__shared__ double s[WARPS_PER_BLOCK][WARP_SIZE];
	__shared__ double inv[WARPS_PER_BLOCK][NVARS][NVARS];
	__shared__ double ssol[WARPS_PER_BLOCK][NVARS][NVARS];

	int tid=threadIdx.x+blockIdx.x*blockDim.x+blockIdx.y*gridDim.x*blockDim.x;
	int wpid=threadIdx.x/WARP_SIZE;
	int gwpid=tid/WARP_SIZE;  // one warp for one block column
	int lane=threadIdx.x%WARP_SIZE;
	int r,c,irow,myrow;
	int InvStIdx,InvEdIdx,idx,myidx,tmpidx;
	int range=WARP_SIZE/(NVARS*NVARS)*(NVARS*NVARS);
	//double det,res;
	//int sign;
	if(gwpid < fact_n)
	{
		if(lane < range)
		{
			InvStIdx=dInvUColPtr[gwpid];
			InvEdIdx=dInvUColPtr[gwpid+1]-1;
			r=lane%NVARS;
			c=(lane/NVARS)%NVARS;
			for(idx=InvEdIdx;idx>=InvStIdx;idx--)
			{
				irow=dInvURowVal[idx];
				if(lane < NVARS*NVARS)
				{
					//load 
					tmpidx=dURowPtr[irow];
					inv[wpid][r][c]=dUBlkVal[tmpidx*NVARS*NVARS+lane];
					//a[wpid][r][c]=inv[wpid][r][c];
					//det=0.0;
					s[wpid][lane]=dRhsUBlkVal[idx*NVARS*NVARS+lane];
					if(NVARS==4)
					{
					ssol[wpid][r][c]=inv[wpid][r][0]*s[wpid][c*NVARS+0]
								+inv[wpid][r][1]*s[wpid][c*NVARS+1]
								+inv[wpid][r][2]*s[wpid][c*NVARS+2]
								+inv[wpid][r][3]*s[wpid][c*NVARS+3];// for NVARS=4
					}
					if(NVARS==3)
					{
				ssol[wpid][r][c]=inv[wpid][r][0]*s[wpid][c*NVARS+0]
								+inv[wpid][r][1]*s[wpid][c*NVARS+1]
								+inv[wpid][r][2]*s[wpid][c*NVARS+2];
					}
					if(NVARS==5)
					{
				ssol[wpid][r][c]=inv[wpid][r][0]*s[wpid][c*NVARS+0]
								+inv[wpid][r][1]*s[wpid][c*NVARS+1]
								+inv[wpid][r][2]*s[wpid][c*NVARS+2]
								+inv[wpid][r][3]*s[wpid][c*NVARS+3]
								+inv[wpid][r][4]*s[wpid][c*NVARS+4];
					}
				// write the solution to global memory
					dInvUBlkVal[idx*NVARS*NVARS+lane]=ssol[wpid][r][c];
				}
				myidx=InvStIdx+lane/(NVARS*NVARS);
				while(myidx < idx)
				{
					s[wpid][lane]=0.0;
					myrow=dInvURowVal[myidx];
					tmpidx=dURowPtr[myrow];
					while(tmpidx < dURowPtr[myrow+1] && dUColVal[tmpidx] < irow ){tmpidx++;}
					if(tmpidx < dURowPtr[myrow+1] && dUColVal[tmpidx] == irow)
					{
						s[wpid][lane]=dUBlkVal[tmpidx*NVARS*NVARS+lane%(NVARS*NVARS)];
					}
					if(NVARS==4)
					{
					dRhsUBlkVal[myidx*NVARS*NVARS+lane%(NVARS*NVARS)]-=s[wpid][lane/(NVARS*NVARS)*(NVARS*NVARS)+0*NVARS+r]*ssol[wpid][0][c]
								   +s[wpid][lane/(NVARS*NVARS)*(NVARS*NVARS)+1*NVARS+r]*ssol[wpid][1][c]
								   +s[wpid][lane/(NVARS*NVARS)*(NVARS*NVARS)+2*NVARS+r]*ssol[wpid][2][c]
								   +s[wpid][lane/(NVARS*NVARS)*(NVARS*NVARS)+3*NVARS+r]*ssol[wpid][3][c]; // for NVARS=4
					}
					if(NVARS==3)
					{
					dRhsUBlkVal[myidx*NVARS*NVARS+lane%(NVARS*NVARS)]-=s[wpid][lane/(NVARS*NVARS)*(NVARS*NVARS)+0*NVARS+r]*ssol[wpid][0][c]
								   +s[wpid][lane/(NVARS*NVARS)*(NVARS*NVARS)+1*NVARS+r]*ssol[wpid][1][c]
								   +s[wpid][lane/(NVARS*NVARS)*(NVARS*NVARS)+2*NVARS+r]*ssol[wpid][2][c];
					}
					if(NVARS==5)
					{
					dRhsUBlkVal[myidx*NVARS*NVARS+lane%(NVARS*NVARS)]-=s[wpid][lane/(NVARS*NVARS)*(NVARS*NVARS)+0*NVARS+r]*ssol[wpid][0][c]
								   +s[wpid][lane/(NVARS*NVARS)*(NVARS*NVARS)+1*NVARS+r]*ssol[wpid][1][c]
								   +s[wpid][lane/(NVARS*NVARS)*(NVARS*NVARS)+2*NVARS+r]*ssol[wpid][2][c]
								   +s[wpid][lane/(NVARS*NVARS)*(NVARS*NVARS)+3*NVARS+r]*ssol[wpid][3][c]
								   +s[wpid][lane/(NVARS*NVARS)*(NVARS*NVARS)+4*NVARS+r]*ssol[wpid][4][c];
					}
					myidx+=WARP_SIZE/(NVARS*NVARS);
				}
			}
		}
	}
}

void estimate_bisai_numeric()
{
	struct timeval Inv_tstart, Inv_tend; double Inv_t=0.0;

	DS_PRECOND_BISAI * ds = (DS_PRECOND_BISAI *)info->DS_PRECOND;

	int block_size=128;
	//int nblocks=info->fact_n*WARP_SIZE/block_size+1;
	int total_threads=info->fact_n*WARP_SIZE;
	int gridx,gridy;
	if(total_threads/block_size > 65535)
	{
		gridx=65535;
		gridy=total_threads/block_size/gridx+1;
	}
	else
	{
		gridx=total_threads/block_size+1;
		gridy=1;
	}
	//gridx=256;
	//gridy=total_threads/block_size/gridx+1;
	printf("gridx=%d,gridy=%d\n",gridx,gridy);

	dim3 nblocks(gridx,gridy);

	gettimeofday(&Inv_tstart,NULL);
	hipLaunchKernelGGL(ISAI_kernel_L_V2, nblocks, block_size, 0, 0, ds->dcsc_InvLColPtr,ds->dcsc_InvLRowVal,ds->dcsc_InvLBlkVal,
						info->dLRowPtr,info->dLColVal,info->dLBlkVal,ds->dLRhs,info->fact_n);
	hipDeviceSynchronize();

	hipLaunchKernelGGL(ISAI_kernel_U_V2, nblocks, block_size, 0, 0, ds->dcsc_InvUColPtr,ds->dcsc_InvURowVal,ds->dcsc_InvUBlkVal,
						info->dURowPtr,info->dUColVal,info->dUBlkVal,ds->dURhs,info->fact_n);
	hipDeviceSynchronize();
	gettimeofday(&Inv_tend,NULL);
    	Inv_t=((double) ((Inv_tend.tv_sec*1000000.0 + Inv_tend.tv_usec)-(Inv_tstart.tv_sec*1000000.0+Inv_tstart.tv_usec)))/1000.0;
	printf("time for compute_InvL_InvU_GPU is:=%12.8lf\n",Inv_t);
	
}



void compute_InvL_InvU_GPU()
{
	struct timeval Inv_tstart, Inv_tend; double Inv_t=0.0;

	DS_PRECOND_BISAI * ds = (DS_PRECOND_BISAI *)info->DS_PRECOND;

	int block_size=128;
	//int nblocks=info->fact_n*WARP_SIZE/block_size+1;
	int total_threads=info->fact_n*WARP_SIZE;
	int gridx,gridy;
	if(total_threads/block_size > 65535)
	{
		gridx=256;
		gridy=total_threads/block_size/gridx+1;
	}
	else
	{
		gridx=total_threads/block_size+1;
		gridy=1;
	}
	//gridx=256;
	//gridy=total_threads/block_size/gridx+1;
	printf("gridx=%d,gridy=%d\n",gridx,gridy);

	dim3 nblocks(gridx,gridy);

	gettimeofday(&Inv_tstart,NULL);
	hipLaunchKernelGGL(ISAI_kernel_L_V2, nblocks, block_size, 0, 0, ds->dcsc_InvLColPtr,ds->dcsc_InvLRowVal,ds->dcsc_InvLBlkVal,
											info->dLRowPtr,info->dLColVal,info->dLBlkVal,ds->dLRhs,info->fact_n);
	hipDeviceSynchronize();

	hipLaunchKernelGGL(ISAI_kernel_U_V2, nblocks, block_size, 0, 0, ds->dcsc_InvUColPtr,ds->dcsc_InvURowVal,ds->dcsc_InvUBlkVal,
											 info->dURowPtr,info->dUColVal,info->dUBlkVal,ds->dURhs,info->fact_n);
	hipDeviceSynchronize();
	gettimeofday(&Inv_tend,NULL);
    Inv_t=((double) ((Inv_tend.tv_sec*1000000.0 + Inv_tend.tv_usec)-(Inv_tstart.tv_sec*1000000.0+Inv_tstart.tv_usec)))/1000.0;
	printf("time for compute_InvL_InvU_GPU is:=%12.8lf\n",Inv_t);

	//we copy dcsc_InvLBlkVal to hcsc_InvLBlkVal
	PetscInt bs=info->bs;
	hipMemcpy(ds->hcsc_InvLBlkVal,ds->dcsc_InvLBlkVal,sizeof(PetscReal)*ds->InvLnnz*bs*bs,hipMemcpyDeviceToHost);

	hipDeviceSynchronize();
	hipMemcpy(ds->hcsc_InvUBlkVal,ds->dcsc_InvUBlkVal,sizeof(PetscReal)*ds->InvUnnz*bs*bs,hipMemcpyDeviceToHost);
	hipDeviceSynchronize();
/*
	if(FIRSTONE)
	{
		FIRSTONE=0;
		char filename[50];
		// before we print, we copy info->dLBlkVal
		hipMemcpy(info->hLBlkVal,info->dLBlkVal,sizeof(double)*info->Lnnz*info->bs*info->bs,hipMemcpyDeviceToHost);
		sprintf(filename,"V2debugfile_%d",info->rank);
	//	sprintf(filename,"debugfile_%d",info->rank);
		FILE *dbf=fopen(filename,"w");
		int itmp;
		int jtmp;
		int sidx,eidx,idx;
		// print L 
		fprintf(dbf,"the following info is L in CSR\n");
		for(itmp=0;itmp<info->fact_n;itmp++)
		{
			sidx=info->hLRowPtr[itmp];
			eidx=info->hLRowPtr[itmp+1];
			fprintf(dbf,"%d row:",itmp);
			for(idx=sidx;idx<eidx;idx++)
			{
				fprintf(dbf,"%d,",info->hLColVal[idx]);
				if(itmp==2 || itmp ==4)
				{
					if(info->hLColVal[idx] == 1)
					{
						for(jtmp=0;jtmp<9;jtmp++){fprintf(dbf,"%12.8f,",info->hLBlkVal[idx*9+jtmp]);}
					}
				}
			}
			fprintf(dbf,"\n");
		}
		fprintf(dbf,"the following info is InvL in CSC\n");
		for(itmp=0;itmp<info->fact_n;itmp++)
		{
			sidx=info->hcsc_InvLColPtr[itmp];
			eidx=info->hcsc_InvLColPtr[itmp+1];
			fprintf(dbf,"%d column:",itmp);
			for(idx=sidx;idx<eidx;idx++)
			{
				fprintf(dbf,"%d,",info->hcsc_InvLRowVal[idx]);
			}
			fprintf(dbf,"\n");
		}



		for(itmp=0;itmp<info->InvLnnz*bs*bs;itmp++)
		{
			fprintf(dbf,"csc_InvLBlkVal[%d]=%22.15e \t cscInvUBlkVal[%d]=%22.15e\n",itmp,info->hcsc_InvLBlkVal[itmp],itmp,info->hcsc_InvUBlkVal[itmp]);
		}
		fclose(dbf);
	}
*/
	// we do a transformation from csc to csr, we don't need to transform ptr and col/row val, we just need to transform blkval
	PetscInt *tmpcounter=(PetscInt *)malloc(sizeof(PetscInt)*info->fact_n);
	memset(tmpcounter,0,sizeof(PetscInt)*info->fact_n);
	PetscInt icol, row, stidx,edidx,idx,itmp,pos;
	for(icol=0;icol<info->fact_n;icol++)
	{
		stidx=ds->hcsc_InvLColPtr[icol];
		edidx=ds->hcsc_InvLColPtr[icol+1];
		for(idx=stidx;idx<edidx;idx++)
		{
			row=ds->hcsc_InvLRowVal[idx];
			pos=ds->hInvLRowPtr[row]+tmpcounter[row];
			for(itmp=0;itmp<bs*bs;itmp++){ds->hInvLBlkVal[pos*bs*bs+itmp]=ds->hcsc_InvLBlkVal[idx*bs*bs+itmp];}
			tmpcounter[row]++;

		}
	}

	memset(tmpcounter,0,sizeof(PetscInt)*info->fact_n);
	for(icol=0;icol<info->fact_n;icol++)
	{
		stidx=ds->hcsc_InvUColPtr[icol];
		edidx=ds->hcsc_InvUColPtr[icol+1];
		for(idx=stidx;idx<edidx;idx++)
		{
			row=ds->hcsc_InvURowVal[idx];
			pos=ds->hInvURowPtr[row]+tmpcounter[row];
			for(itmp=0;itmp<bs*bs;itmp++){ds->hInvUBlkVal[pos*bs*bs+itmp]=ds->hcsc_InvUBlkVal[idx*bs*bs+itmp];}
			tmpcounter[row]++;

		}
	}
	free(tmpcounter);

	// now we got the CSR format of InvL and InvU
	hipMemcpy(ds->dInvLBlkVal,ds->hInvLBlkVal,sizeof(PetscReal)*ds->InvLnnz*bs*bs,hipMemcpyHostToDevice);
	hipMemcpy(ds->dInvUBlkVal,ds->hInvUBlkVal,sizeof(PetscReal)*ds->InvUnnz*bs*bs,hipMemcpyHostToDevice);
	
}
void bisai_precond(PetscReal *in_x, PetscReal *out_y, PetscReal *mid_tmp, PetscInt in_vsz)
{
	DS_PRECOND_BISAI * ds = (DS_PRECOND_BISAI *)info->DS_PRECOND;
	if(in_vsz != info->fact_n*info->bs){printf("ERROR: the preconditioning vector size in_vsz != matrix size info->fact_n\n");}
	PetscReal alpha=1.0;
	PetscReal beta=0.0;
	// fact_n == vsz TRUE  for both Block-Jacobi and ASM 
	info->cusparse_stat =hipsparseDbsrmv(info->cusparse_handle,info->dir,info->trans,info->fact_n,info->fact_n,
		ds->InvLnnz,&alpha,info->descr,ds->dInvLBlkVal,ds->dInvLRowPtr,ds->dInvLColVal,info->bs,
		in_x,&beta,mid_tmp);
	hipDeviceSynchronize();
	if(info->cusparse_stat != HIPSPARSE_STATUS_SUCCESS){printf("ERROR in hipsparseDbsrmv for bisai_precond in rank=%d\n",info->rank);}

	alpha=1.0; beta=0.0;
	info->cusparse_stat =hipsparseDbsrmv(info->cusparse_handle,info->dir,info->trans,info->fact_n,info->fact_n,
		ds->InvUnnz,&alpha,info->descr,ds->dInvUBlkVal,ds->dInvURowPtr,ds->dInvUColVal,info->bs,
		mid_tmp,&beta,out_y);
	hipDeviceSynchronize();
	if(info->cusparse_stat != HIPSPARSE_STATUS_SUCCESS){printf("ERROR in hipsparseDbsrmv for bisai_precond in rank=%d\n",info->rank);}

}





// we need to call this function two times, one is for InvL, the other is for InvU
void estimate_bisai_pattern(PetscInt *csrRowPtr,PetscInt *csrColInd, PetscInt nnz,// input L or U
		PetscInt **pInvCsrRowPtr, PetscInt **pInvCsrColInd, PetscReal **pInvCsrBlkVal,PetscInt *pnnz) // output InvL or InvU
{
	DS_PRECOND_BISAI * ds=(DS_PRECOND_BISAI *)info->DS_PRECOND;
	PetscInt level = ds->bisai_dense_level;
	PetscInt n=info->fact_n;
	PetscInt bs=info->bs;

	if(level ==1)
	{
		*pnnz=nnz;
		if(!(*pInvCsrRowPtr)){hipMalloc((void **)pInvCsrRowPtr, sizeof(PetscInt)*(n+1));}
		if(!(*pInvCsrColInd)){hipMalloc((void **)pInvCsrColInd,sizeof(PetscInt)*nnz);}
		if(!(*pInvCsrBlkVal)){hipMalloc((void **)pInvCsrBlkVal,sizeof(PetscReal)*nnz*bs*bs);}
		// memory copy
		if(*pInvCsrRowPtr){hipMemcpy(*pInvCsrRowPtr,csrRowPtr,sizeof(PetscInt)*(n+1),hipMemcpyHostToDevice);}
		if(*pInvCsrColInd){hipMemcpy(*pInvCsrColInd,csrColInd,sizeof(PetscInt)*nnz,hipMemcpyHostToDevice);}
		if(*pInvCsrBlkVal){hipMemset(*pInvCsrBlkVal,0,sizeof(PetscReal)*nnz*bs*bs);}
		return;
	}
	// otherwise level > 1 
	// if level == 2 or level == 3

	// Step1: we need to estimate the non-zero patterns for the inverses of L and U
	hipsparseHandle_t hd;
	hipsparseMatDescr_t descrA;
	hipsparseMatDescr_t descrB;
	hipsparseMatDescr_t descrC;
	hipsparseMatDescr_t descrD;

	hipsparseCreate(&hd);
	info->cusparse_stat=hipsparseCreateMatDescr(&descrA);
	if(info->cusparse_stat != HIPSPARSE_STATUS_SUCCESS){printf("ERROR in hipsparseCreateMatDescr in rank=%d\n",info->rank);}
	hipsparseSetMatIndexBase(descrA,HIPSPARSE_INDEX_BASE_ZERO);
	hipsparseSetMatType(descrA,HIPSPARSE_MATRIX_TYPE_GENERAL);

	info->cusparse_stat=hipsparseCreateMatDescr(&descrB);
	if(info->cusparse_stat != HIPSPARSE_STATUS_SUCCESS){printf("ERROR in hipsparseCreateMatDescr in rank=%d\n",info->rank);}
	hipsparseSetMatIndexBase(descrB,HIPSPARSE_INDEX_BASE_ZERO);
	hipsparseSetMatType(descrB,HIPSPARSE_MATRIX_TYPE_GENERAL);

	info->cusparse_stat=hipsparseCreateMatDescr(&descrC);
	if(info->cusparse_stat != HIPSPARSE_STATUS_SUCCESS){printf("ERROR in hipsparseCreateMatDescr in rank=%d\n",info->rank);}
	hipsparseSetMatIndexBase(descrC,HIPSPARSE_INDEX_BASE_ZERO);
	hipsparseSetMatType(descrC,HIPSPARSE_MATRIX_TYPE_GENERAL);

	if(level > 2)
	{
		info->cusparse_stat=hipsparseCreateMatDescr(&descrD);
		if(info->cusparse_stat != HIPSPARSE_STATUS_SUCCESS){printf("ERROR in hipsparseCreateMatDescr in rank=%d\n",info->rank);}
		hipsparseSetMatIndexBase(descrD,HIPSPARSE_INDEX_BASE_ZERO);
		hipsparseSetMatType(descrD,HIPSPARSE_MATRIX_TYPE_GENERAL);
	}
	//use cusparse to do sparse matrix-matrix multiplication
	// allocate A and B using the lower triangular factor L
	PetscInt nnzA=nnz;
	PetscInt *csrRowPtrA=NULL;
	PetscInt *csrColIndA=NULL;
	PetscReal *csrValA=NULL;

	PetscInt nnzB=nnzA;
	PetscInt *csrRowPtrB=NULL;
	PetscInt *csrColIndB=NULL;
	PetscReal *csrValB=NULL;

	printf("rank=%d, before nnzA=%d, nnzB=%d\n", info->rank, nnzA, nnzB);
	PetscInt nnzC=0;
	PetscInt *csrRowPtrC=NULL;
	PetscInt *csrColIndC=NULL;
	PetscReal *csrValC=NULL;

	PetscInt nnzD=0;
	PetscInt *csrRowPtrD=NULL;
	PetscInt *csrColIndD=NULL;
	PetscReal *csrValD=NULL;

  	struct timeval calsp_tstart,calsp_tend; double calsp_t=0.0;  // main matrix vector multiple times
	gettimeofday(&calsp_tstart,NULL);

	hipMalloc((void **)&csrRowPtrA,sizeof(PetscInt)*(n+1));
	hipMalloc((void **)&csrColIndA,sizeof(PetscInt)*nnzA);
	hipMalloc((void **)&csrValA,sizeof(PetscReal)*nnzA);

	hipMalloc((void **)&csrRowPtrB,sizeof(PetscInt)*(n+1));
	hipMalloc((void **)&csrColIndB,sizeof(PetscInt)*nnzB);
	hipMalloc((void **)&csrValB,sizeof(PetscReal)*nnzB);

	hipMalloc((void **)&csrRowPtrC,sizeof(PetscInt)*(n+1));

	if(level > 2) // == 3 we need A,B,C,D -> C=AB, D=AC
	{
		hipMalloc((void **)&csrRowPtrD,sizeof(PetscInt)*(n+1));
	}
//	copy memory from host to device

	hipMemcpy(csrRowPtrA,csrRowPtr,sizeof(PetscInt)*(n+1),hipMemcpyHostToDevice);
	hipMemcpy(csrColIndA,csrColInd,sizeof(PetscInt)*nnzA,hipMemcpyHostToDevice);
	hipMemset(csrValA,0,sizeof(PetscReal)*nnzA);

	hipMemcpy(csrRowPtrB,csrRowPtr,sizeof(PetscInt)*(n+1),hipMemcpyHostToDevice);
	hipMemcpy(csrColIndB,csrColInd,sizeof(PetscInt)*nnzB,hipMemcpyHostToDevice);
	hipMemset(csrValB,0,sizeof(PetscReal)*nnzB);

	//AB=C
	info->cusparse_stat=hipsparseXcsrgemmNnz(hd,HIPSPARSE_OPERATION_NON_TRANSPOSE,HIPSPARSE_OPERATION_NON_TRANSPOSE,n,n,n,
	descrA,nnzA,csrRowPtrA,csrColIndA,descrB,nnzB,csrRowPtrB,csrColIndB,descrC,csrRowPtrC,&nnzC);
	hipDeviceSynchronize();
	if(info->cusparse_stat != HIPSPARSE_STATUS_SUCCESS){printf("ERROR in hipsparseXcsrgemmNnz in rank=%d\n",info->rank);}
	printf("nnzA=%d,nnzB=%d,nnzC=%d\n",nnzA,nnzB,nnzC);
	hipMalloc((void **)&csrColIndC,sizeof(PetscInt)*nnzC);
	hipMalloc((void **)&csrValC,sizeof(PetscReal)*nnzC);

	info->cusparse_stat=hipsparseDcsrgemm(hd,HIPSPARSE_OPERATION_NON_TRANSPOSE,HIPSPARSE_OPERATION_NON_TRANSPOSE,n,n,n,
	descrA,nnzA,csrValA,csrRowPtrA,csrColIndA,descrB,nnzB,csrValB,csrRowPtrB,csrColIndB,descrC,csrValC,csrRowPtrC,csrColIndC);
	hipDeviceSynchronize();
	if(info->cusparse_stat != HIPSPARSE_STATUS_SUCCESS){printf("ERROR in hipsparseDcsrgemm in rank=%d\n",info->rank);}
	gettimeofday(&calsp_tend,NULL);
    	calsp_t=((double) ((calsp_tend.tv_sec*1000000.0 + calsp_tend.tv_usec)-(calsp_tstart.tv_sec*1000000.0+calsp_tstart.tv_usec)))/1000.0;
	printf("time for cal_sparsitypattern of C=AB on_GPU is:=%12.8lf\n",calsp_t);
/////////////////////////////debug/////////////////////////

//	int *tmp=(int *)malloc(sizeof(int)*(n+1));
//	hipMemcpy(tmp,csrRowPtrC, sizeof(int)*(n+1),hipMemcpyDeviceToHost);
//	hipDeviceSynchronize();
//	printf("rank=%d.tmp[0]=%d,tmp[1]=%d,tmp[2]=%d,tmp[3]=%d\n",info->rank,tmp[0],tmp[1],tmp[2],tmp[3]);
//	free(tmp);
	


///////////////////////////debug///////////////////////////
	// AC=D
	gettimeofday(&calsp_tstart,NULL);
	if(level > 2)
	{
		hipsparseXcsrgemmNnz(hd,HIPSPARSE_OPERATION_NON_TRANSPOSE,HIPSPARSE_OPERATION_NON_TRANSPOSE,n,n,n,
		descrA,nnzA,csrRowPtrA,csrColIndA,descrC,nnzC,csrRowPtrC,csrColIndC,descrD,csrRowPtrD,&nnzD);
		hipDeviceSynchronize();
		printf("nnzA=%d,nnzB=%d,nnzC=%d,nnzD=%d\n",nnzA,nnzB,nnzC,nnzD);
		hipMalloc((void **)&csrColIndD,sizeof(PetscInt)*nnzD);
		hipMalloc((void **)&csrValD,sizeof(PetscReal)*nnzD);

		hipsparseDcsrgemm(hd,HIPSPARSE_OPERATION_NON_TRANSPOSE,HIPSPARSE_OPERATION_NON_TRANSPOSE,n,n,n,
		descrA,nnzA,csrValA,csrRowPtrA,csrColIndA,descrC,nnzC,csrValC,csrRowPtrC,csrColIndC,descrD,csrValD,csrRowPtrD,csrColIndD);
		hipDeviceSynchronize();
	}
	gettimeofday(&calsp_tend,NULL);
    	calsp_t=((double) ((calsp_tend.tv_sec*1000000.0 + calsp_tend.tv_usec)-(calsp_tstart.tv_sec*1000000.0+calsp_tstart.tv_usec)))/1000.0;
	printf("time for cal_sparsitypattern D=AC on_GPU is:=%12.8lf\n",calsp_t);



////////////////////////////////////////////////////////////////////////////////////////////////
	// set the output InvL or InvU
	*pnnz=level>2? nnzD: nnzC;
	PetscInt *out1=level>2?  csrRowPtrD : csrRowPtrC;
	PetscInt *out2=level>2?  csrColIndD : csrColIndC;
	//printf("rank=%d,pnnz=%d,level=%d,out1=%d,csrRowPtrC=%d\n",info->rank,*pnnz,level,*out1,*csrRowPtrC);
	if(!(*pInvCsrRowPtr)){hipMalloc((void **)pInvCsrRowPtr, sizeof(PetscInt)*(n+1));}
	if(!(*pInvCsrColInd)){hipMalloc((void **)pInvCsrColInd, sizeof(PetscInt)*(*pnnz));}	
	if(!(*pInvCsrBlkVal)){hipMalloc((void **)pInvCsrBlkVal, sizeof(PetscReal)*(*pnnz)*bs*bs);}
	// memory copy
	if(*pInvCsrRowPtr){hipMemcpy(*pInvCsrRowPtr,out1,sizeof(PetscInt)*(n+1),hipMemcpyDeviceToDevice);}
	if(*pInvCsrColInd){hipMemcpy(*pInvCsrColInd,out2,sizeof(PetscInt)*(*pnnz),hipMemcpyDeviceToDevice);}
	if(*pInvCsrBlkVal){hipMemset(*pInvCsrBlkVal,0,sizeof(PetscReal)*(*pnnz)*bs*bs);}		
/////////////////////////////////////////////////////////////////////////////////////////////////

	hipDeviceSynchronize();
	// free memory	
	if(csrRowPtr){hipFree(csrRowPtrA);}
	if(csrColIndA){hipFree(csrColIndA);}
	if(csrValA){ hipFree(csrValA);}
	hipsparseDestroyMatDescr(descrA);

	if(csrRowPtrB){hipFree(csrRowPtrB);} 
	if(csrColIndB){ hipFree(csrColIndB);}
	if(csrValB){ hipFree(csrValB);}
	hipsparseDestroyMatDescr(descrB);

	if(csrRowPtrC){hipFree(csrRowPtrC);}
	if(csrColIndC){ hipFree(csrColIndC);}
	if(csrValC){hipFree(csrValC);}
	hipsparseDestroyMatDescr(descrC);


	if(level > 2)
	{
		if(csrRowPtrD){hipFree(csrRowPtrD);}
		if(csrColIndD){ hipFree(csrColIndD);}
		if(csrValD){hipFree(csrValD);}

		hipsparseDestroyMatDescr(descrD);
	}
	hipsparseDestroy(hd);
	
		
}
// do block csr to csc (csc to csr)
void cusparse_bsr2bsc(PetscInt *inRowPtr, PetscInt *inColInd, PetscReal *inVal,
		      PetscInt *outColPtr, PetscInt *outRowInd, PetscReal *outVal,
		      PetscInt n, PetscInt nnz, PetscInt bs)
{
	hipsparseHandle_t hd;
	hipsparseStatus_t stat;
	size_t buffersize;
	void *pBuffer=NULL;

	hipsparseCreate(&hd);
	
	stat = hipsparseDgebsr2gebsc_bufferSize(hd,n,n,nnz,inVal,inRowPtr,inColInd,bs,bs,&buffersize);
	hipDeviceSynchronize();
	printf("rank=%d,buffersize=%d\n",info->rank,buffersize);
	if(stat != HIPSPARSE_STATUS_SUCCESS){printf("ERROR in hipsparseDgebsr2gebsc_bufferSize in rank=%d\n",info->rank);}

	hipMalloc((void **)&pBuffer, buffersize);

	stat = hipsparseDgebsr2gebsc(hd,n,n,nnz,inVal,inRowPtr,inColInd,bs,bs,outVal,outRowInd,outColPtr,
			HIPSPARSE_ACTION_NUMERIC,HIPSPARSE_INDEX_BASE_ZERO,pBuffer);
	hipDeviceSynchronize();
	if(stat != HIPSPARSE_STATUS_SUCCESS){printf("ERROR in hipsparseDgebsr2gebsc in rank=%d\n",info->rank);}

	hipFree(pBuffer);

	hipsparseDestroy(hd);
}
	
// inv(M) x= y -> My=x,
void cusparse_pbilu_cusparse_precond(PetscReal *in_x, PetscReal *out_y, PetscReal *mid_tmp, PetscInt in_vsz)
{
		// FOR BLOCK-Jacobi
		DS_PBILU_PRECOND_CUSPARSE *ds = (DS_PBILU_PRECOND_CUSPARSE *)info->DS_PBILU;
		if(in_vsz != info->bs*info->fact_n){printf("ERROR: in cusparse_precond: the input vector size in_vsz != matrix size info->fact_n\n");}
		double alpha=1.0; 
		//L(dv_tmp)=x
		ds->cup_stat=hipsparseDbsrsv2_solve(ds->handle_L,ds->dir_L,ds->trans_L,info->fact_n,
		info->Factnnz,&alpha,ds->descr_L,ds->dFactBlkVal,ds->dFactRowPtr,ds->dFactColVal,info->bs,ds->info_L,
		in_x, mid_tmp, ds->policy_L,ds->pLBuffer);
		hipDeviceSynchronize();
		if(ds->cup_stat != HIPSPARSE_STATUS_SUCCESS){printf("ERROR in L hipsparseDbsrsv2_solve\n");}
		// U(y)=dv_tmp
		alpha=1.0;
		ds->cup_stat=hipsparseDbsrsv2_solve(ds->handle_L,ds->dir_U,ds->trans_U,info->fact_n,
		info->Factnnz,&alpha,ds->descr_U,ds->dFactBlkVal,ds->dFactRowPtr,ds->dFactColVal,info->bs,ds->info_U,
		mid_tmp,out_y, ds->policy_U,ds->pUBuffer);  // 2021.9.9 change info->dv_tmp into dv_tmp
		hipDeviceSynchronize();
		if(ds->cup_stat != HIPSPARSE_STATUS_SUCCESS){printf("ERROR in U hipsparseDbsrsv2_solve\n");}

		// FOR ASM
}
void petsc_precond(PC pc,Vec px, Vec py, PetscReal *vecx, PetscReal *vecy, PetscReal *vectmp, PetscInt vsz)
{
	// we don't have to distingulish ASM and Blcok-Jacobi becasue PCApply does everything
	PetscReal *hveci=NULL;
	PetscReal *hvecj=NULL;
	VecGetArray(px,&hveci);
	hipMemcpy(hveci,vecx,sizeof(PetscReal)*vsz,hipMemcpyDeviceToHost);
	VecRestoreArray(px,&hveci);

	PCApply(pc,px,py);
		
	VecGetArray(py,&hvecj);
	hipMemcpy(vecy,hvecj,sizeof(PetscReal)*vsz,hipMemcpyHostToDevice);
	VecRestoreArray(py,&hvecj);
	hipDeviceSynchronize();      
}
	

void iterative_precond(PetscReal *in_x, PetscReal *out_y, PetscReal *mid_tmp, PetscInt in_vsz)
{
	PetscInt memsize=sizeof(PetscReal)*in_vsz;
	PetscInt iter;
	PetscReal alpha, beta;
	PetscInt NUM_ITER_PRECOND=5;

	DS_PRECOND_ITERATIVE * ds =(DS_PRECOND_ITERATIVE *)info->DS_PRECOND;




	// Step1: solve (Lhat + LD)mid_tmp = in_x
	// 1.1 initialize solution (mid_tmp) with rhs(in_x)
	hipMemcpy(mid_tmp,in_x,memsize,hipMemcpyDeviceToDevice);
	for(iter=1;iter<=NUM_ITER_PRECOND;iter++)
	{
		alpha=-1.0;
		beta=1.0;	
		hipMemcpy(ds->diter_tmp1,in_x,memsize,hipMemcpyDeviceToDevice);
		hipDeviceSynchronize();
		hipsparseDbsrmv(ds->cusparse_handle,ds->dir,ds->trans,info->fact_n,info->fact_n,
		info->Lnnz,&alpha,ds->descr,info->dLBlkVal,info->dLRowPtr,info->dLColVal,info->bs,
		mid_tmp,&beta,ds->diter_tmp1);
		hipDeviceSynchronize();
		hipMemcpy(mid_tmp,ds->diter_tmp1,memsize,hipMemcpyDeviceToDevice);
		hipDeviceSynchronize();
	}

	// Step2: solve (uhat + UD) out_y = mid_tmp
	alpha=1.0; beta=0.0;
	hipsparseDbsrmv(ds->cusparse_handle,ds->dir,ds->trans,info->fact_n,info->fact_n,
		info->fact_n,&alpha,ds->descr,ds->dUdiagBlkVal,ds->dUdiagRowPtr,ds->dUdiagColVal,info->bs,
		mid_tmp,&beta,ds->diter_tmp1);
	hipDeviceSynchronize();
	
	hipMemcpy(out_y,ds->diter_tmp1,memsize,hipMemcpyDeviceToDevice);
	
	for(iter=1;iter<=NUM_ITER_PRECOND;iter++)
	{
		alpha=1.0; beta=0.0;
		hipsparseDbsrmv(ds->cusparse_handle,ds->dir,ds->trans,info->fact_n,info->fact_n,
			info->Unnz,&alpha,ds->descr,info->dUBlkVal,info->dURowPtr,info->dUColVal,info->bs,
			out_y,&beta,ds->diter_tmp2);
		hipDeviceSynchronize();
		hipMemcpy(out_y,ds->diter_tmp1,memsize,hipMemcpyDeviceToDevice);
		hipDeviceSynchronize();
		alpha=-1.0; beta=1.0;
		hipsparseDbsrmv(ds->cusparse_handle,ds->dir,ds->trans,info->fact_n,info->fact_n,
			info->fact_n,&alpha,ds->descr,ds->dUdiagBlkVal,ds->dUdiagRowPtr,ds->dUdiagColVal,info->bs,
			ds->diter_tmp2,&beta,out_y);
		hipDeviceSynchronize();
	}
	
	
}
void asynchronous_pbilu_cusparse_precond(PetscReal *in_x,PetscReal *out_y,PetscReal *mid_tmp,PetscInt in_vsz)
{
    	DS_PBILU_ASYN * ds=(DS_PBILU_ASYN *)info->DS_PBILU; 
	if(!ds->pLBuffer || !ds->pUBuffer)
	{
		//S1: bufferSize, run only once for both linear and non-linear problems
		ds->cup_stat=hipsparseDbsrsv2_bufferSize(ds->handle_L,ds->dir_L,ds->trans_L,
		info->fact_n,info->Lnnz,ds->descr_L,info->dLBlkVal,info->dLRowPtr,info->dLColVal,
		info->bs,ds->info_L,&ds->pLBufferSize);
		hipDeviceSynchronize();
		if(ds->cup_stat != HIPSPARSE_STATUS_SUCCESS){printf("ERROR in L hipsparseDbsrsv2_bufferSize\n");}

		ds->cup_stat=hipsparseDbsrsv2_bufferSize(ds->handle_L,ds->dir_U,ds->trans_U,
		info->fact_n,info->Unnz,ds->descr_U,ds->dUStarBlkVal,info->dURowPtr,info->dUColVal,
		info->bs,ds->info_U,&ds->pUBufferSize);
		hipDeviceSynchronize();
		if(ds->cup_stat != HIPSPARSE_STATUS_SUCCESS){printf("ERROR in U hipsparseDbsrsv2_bufferSize\n");}

		//allocate memory
		hipMalloc((void **)&ds->pLBuffer,ds->pLBufferSize);
		hipMalloc((void **)&ds->pUBuffer,ds->pUBufferSize);

		//S2: Analysis
		ds->cup_stat=hipsparseDbsrsv2_analysis(ds->handle_L,ds->dir_L,ds->trans_L,info->fact_n,
		info->Lnnz,ds->descr_L,info->dLBlkVal,info->dLRowPtr,info->dLColVal,info->bs,ds->info_L,
		ds->policy_L,ds->pLBuffer);
		hipDeviceSynchronize();
		if(ds->cup_stat != HIPSPARSE_STATUS_SUCCESS){printf("ERROR in L hipsparseDbsrsv2_analysis\n");}

		ds->cup_stat=hipsparseDbsrsv2_analysis(ds->handle_L,ds->dir_U,ds->trans_U,info->fact_n,
		info->Unnz,ds->descr_U,ds->dUStarBlkVal,info->dURowPtr,info->dUColVal,info->bs,ds->info_U,
		ds->policy_U,ds->pUBuffer);
		hipDeviceSynchronize();
		if(ds->cup_stat != HIPSPARSE_STATUS_SUCCESS){printf("ERROR in U hipsparseDbsrsv2_analysis\n");}

    	}
		double alpha=1.0; double beta=0.0;
		// z=(LU)^-1 VV(it):   LUz= VV(it) -> L(tmp1)=VV(it) 
		ds->cup_stat=hipsparseDbsrsv2_solve(ds->handle_L,ds->dir_L,ds->trans_L,info->fact_n,
		info->Lnnz,&alpha,ds->descr_L,info->dLBlkVal,info->dLRowPtr,info->dLColVal,info->bs,ds->info_L,
		in_x, mid_tmp, ds->policy_L,ds->pLBuffer);
		hipDeviceSynchronize();
		if(ds->cup_stat != HIPSPARSE_STATUS_SUCCESS){printf("ERROR in L hipsparseDbsrsv2_solve\n");}


		// DUstar ()=tmp1 ->  Ustar  () = inv(D)*tmp1= tmp0   2020.1.22
        	alpha=1.0; beta=0.0;
		hipsparseDbsrmv(ds->handle_L,HIPSPARSE_DIRECTION_COLUMN,ds->trans_U,info->fact_n,info->fact_n,
		info->fact_n,&alpha,ds->descr_U,ds->dUInvDiagVal,ds->dUDiagRowPtr,ds->dUDiagColVal,info->bs,
		mid_tmp,&beta,ds->tmp); // need to be reviewed if error occurs, because descr_U is set to upper   
		hipDeviceSynchronize();

        	alpha=1.0;
		// Ustar( ) =ds->tmp 
		ds->cup_stat=hipsparseDbsrsv2_solve(ds->handle_L,ds->dir_U,ds->trans_U,info->fact_n,
		info->Unnz,&alpha,ds->descr_U,ds->dUStarBlkVal,info->dURowPtr,info->dUColVal,info->bs,ds->info_U,
		ds->tmp, out_y, ds->policy_U,ds->pUBuffer);
		hipDeviceSynchronize();
		if(ds->cup_stat != HIPSPARSE_STATUS_SUCCESS){printf("ERROR in U hipsparseDbsrsv2_solve\n");}

}



const char* hipsparseStatusGetString(hipsparseStatus_t status) {
    switch (status) {
        case HIPSPARSE_STATUS_SUCCESS:
            return "HIPSPARSE_STATUS_SUCCESS";
        case HIPSPARSE_STATUS_NOT_INITIALIZED:
            return "HIPSPARSE_STATUS_NOT_INITIALIZED";
        case HIPSPARSE_STATUS_ALLOC_FAILED:
            return "HIPSPARSE_STATUS_ALLOC_FAILED";
        case HIPSPARSE_STATUS_INVALID_VALUE:
            return "HIPSPARSE_STATUS_INVALID_VALUE";
        case HIPSPARSE_STATUS_ARCH_MISMATCH:
            return "HIPSPARSE_STATUS_ARCH_MISMATCH";
        case HIPSPARSE_STATUS_MAPPING_ERROR:
            return "HIPSPARSE_STATUS_MAPPING_ERROR";
        case HIPSPARSE_STATUS_EXECUTION_FAILED:
            return "HIPSPARSE_STATUS_EXECUTION_FAILED";
        case HIPSPARSE_STATUS_INTERNAL_ERROR:
            return "HIPSPARSE_STATUS_INTERNAL_ERROR";
        case HIPSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
            return "HIPSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
        case HIPSPARSE_STATUS_ZERO_PIVOT:
            return "HIPSPARSE_STATUS_ZERO_PIVOT";
        default:
            return "HIPSPARSE_STATUS_UNKNOWN";
    }
}



__global__ void init_vectors_kernel(
    const int *d_p, const double *d_rowscal, const double *d_b, int offset,
    double *d_b_y1, double *d_y2, int nB, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n)
    {
        // 初始化 b_y1
        if (i < nB)
        {
            int p_idx = d_p[i] - 1;
            d_b_y1[i] = d_rowscal[p_idx] * d_b[p_idx + offset];
        }

        // 初始化 y2
        if (i >= nB && i < n)
        {
            int idx = i - nB;
            int p_idx = d_p[i] - 1;
            d_y2[idx] = d_rowscal[p_idx] * d_b[p_idx + offset];
        }
    }
}

// CUDA kernel 函数：复制 b_y1 到 b
__global__ void copy_b_y1_to_b_kernel(const double *b_y1, double *b, int offset, int nB)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < nB)
    {
        b[offset + i] = b_y1[i];
    }
}

// CUDA kernel 函数：复制 y2 到 b
__global__ void copy_y2_to_b_kernel(const double *y2, double *b, int offset, int nB, int b_n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < b_n)
    {
        b[(offset + nB) + i] = y2[i];
    }
}

// CUDA kernel 函数：复制 b 到 b_y1 和 y2
__global__ void copy_b_to_vectors_kernel(const double *b, double *b_y1, double *y2,
                                         int offset, int nB, int b_n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // 复制到 b_y1
    if (i < nB)
    {
        b_y1[i] = b[offset + i];
    }

    // 复制到 y2
    if (i < b_n)
    {
        y2[i] = b[(offset + nB) + i];
    }
}

// CUDA kernel 函数：重排解向量

__global__ void reorder_solution_kernel(
    const double *d_b_y1, const double *d_y2, double *d_b,
    const int *d_q, const double *d_colscal, int offset, int nB, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n)
    {
        int q_idx = d_q[i] - 1;
        double value = (i < nB) ? d_b_y1[i] : d_y2[i - nB];
        d_b[q_idx + offset] = value * d_colscal[q_idx];
    }
}


// 稀疏矩阵向量乘法 - GPU 版本（直接处理 GPU 上的数据）
    void cusparsespmv_gpu(const int m, const int n, const int nnz,
                          const int *d_csrRowPtrA, const int *d_csrColIndA, const double *d_csrValA,
                          const double *d_x, double *d_y, const double alpha, const double beta)
    {
        // 初始化 cuSPARSE
        //cusparseHandle_t handle;
        //CHECK_CUSPARSE(cusparseCreate(&handle));
        DS_PRECOND_MULTILEVEL *ds = (DS_PRECOND_MULTILEVEL *)info->DS_PRECOND;

        // 创建矩阵描述符
        hipsparseMatDescr_t descr;
        CHECK_HIPSPARSE(hipsparseCreateMatDescr(&descr));
        hipsparseSetMatType(descr, HIPSPARSE_MATRIX_TYPE_GENERAL);
        hipsparseSetMatIndexBase(descr, HIPSPARSE_INDEX_BASE_ZERO);

        // 创建并设置矩阵和向量 - 需要类型转换以匹配 API
        hipsparseSpMatDescr_t matA;
        hipsparseDnVecDescr_t vecX, vecY;
        CHECK_HIPSPARSE(hipsparseCreateCsr(&matA, m, n, nnz,
                                         (void *)d_csrRowPtrA, (void *)d_csrColIndA, (void *)d_csrValA,
                                         HIPSPARSE_INDEX_32I, HIPSPARSE_INDEX_32I, HIPSPARSE_INDEX_BASE_ZERO, HIP_R_64F));
        CHECK_HIPSPARSE(hipsparseCreateDnVec(&vecX, n, (void *)d_x, HIP_R_64F));
        CHECK_HIPSPARSE(hipsparseCreateDnVec(&vecY, m, d_y, HIP_R_64F));

        // 分配工作区缓冲区
        size_t bufferSize;
        CHECK_HIPSPARSE(hipsparseSpMV_bufferSize(ds->handle, HIPSPARSE_OPERATION_NON_TRANSPOSE,
                                               &alpha, matA, vecX, &beta, vecY, HIP_R_64F,
                                               HIPSPARSE_SPMV_CSR_ALG2, &bufferSize));
        void *dBuffer;
         hipMalloc(&dBuffer, bufferSize);

        // 执行稀疏矩阵向量乘法
        CHECK_HIPSPARSE(hipsparseSpMV(ds->handle, HIPSPARSE_OPERATION_NON_TRANSPOSE,
                                    &alpha, matA, vecX, &beta, vecY, HIP_R_64F,
                                    HIPSPARSE_SPMV_CSR_ALG2, dBuffer));

        // 清理资源
        CHECK_HIPSPARSE(hipsparseDestroySpMat(matA));
        CHECK_HIPSPARSE(hipsparseDestroyDnVec(vecX));
        CHECK_HIPSPARSE(hipsparseDestroyDnVec(vecY));
         hipFree(dBuffer);
        CHECK_HIPSPARSE(hipsparseDestroyMatDescr(descr));
        // CHECK_CUSPARSE(cusparseDestroy(handle));
    }
 

static void solve_milu_ginkgo(const emxArray_struct0_T *M_inv, int lvl, emxArray_real_T *b,
                              int offset, emxArray_real_T *b_y1, emxArray_real_T *y2)
{
    int nB = M_inv->data[lvl - 1].L.nrows <= 0 ? 1 : M_inv->data[lvl - 1].L.nrows;
    int n = M_inv->data[lvl - 1].negE.nrows < 0 ? nB : (M_inv->data[lvl - 1].L.nrows + M_inv->data[lvl - 1].negE.nrows);

    double *mid;
    hipMalloc(&mid, nB * sizeof(double));
    hipMemset(mid, 0, nB * sizeof(double));

    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    init_vectors_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        M_inv->data[lvl - 1].p->data, M_inv->data[lvl - 1].rowscal->data, b->data, offset,
        b_y1->data, y2->data, nB, n);

    hipDeviceSynchronize();

    if (n > M_inv->data[lvl - 1].L.nrows)
    {
        blocksPerGrid = (nB + threadsPerBlock - 1) / threadsPerBlock;
        copy_b_y1_to_b_kernel<<<blocksPerGrid, threadsPerBlock>>>(b_y1->data, b->data, offset, nB);
    }

    if ((M_inv->data[lvl - 1].L.val->size[0] == 0) && (M_inv->data[lvl - 1].U.val->size[0] == n * n))
    {
        CudaTimer &timer = CudaTimer::getInstance();
        timer.setContext(lvl, CudaTimer::DENSE_SOLVE);
        timer.start();

        DS_PRECOND_MULTILEVEL *ds = (DS_PRECOND_MULTILEVEL *)info->DS_PRECOND;
        const double alpha = 1.0;
        const double beta = 0.0;

        double *d_temp;
        hipMalloc(&d_temp, nB * sizeof(double));
        hipMemset(d_temp, 0, nB * sizeof(double));

        hipblasDgemv(ds->handle_hipblas, HIPBLAS_OP_N, nB, nB, &alpha, M_inv->data[lvl - 1].U.val->data, nB, b_y1->data, 1, &beta, d_temp, 1);
        hipblasDcopy(ds->handle_hipblas, nB, d_temp, 1, b_y1->data, 1);

        hipFree(d_temp);
        timer.stop();
    }
    else if (M_inv->data[lvl - 1].L.val->size[0] == -1)
    {
        CudaTimer &timer = CudaTimer::getInstance();
        timer.setContext(lvl, CudaTimer::TRIANGULAR_SOLVE);
        timer.start();

        DS_PRECOND_MULTILEVEL *ds = (DS_PRECOND_MULTILEVEL *)info->DS_PRECOND;
        const double alpha = 1.0;
        const double beta = 0.0;

        double *d_temp;
        hipMalloc(&d_temp, nB * sizeof(double));
        hipMemset(d_temp, 0, nB * sizeof(double));

        hipblasDgemv(ds->handle_hipblas, HIPBLAS_OP_N, nB, nB, &alpha, M_inv->data[lvl - 1].U.val->data, nB, b_y1->data, 1, &beta, d_temp, 1);
        hipblasDcopy(ds->handle_hipblas, nB, d_temp, 1, b_y1->data, 1);

        hipFree(d_temp);
        timer.stop();
    }
    else
    {
        CudaTimer &timer = CudaTimer::getInstance();
        timer.setContext(lvl, CudaTimer::TRIANGULAR_SOLVE);
        timer.start();

        cusparsespmv_gpu(M_inv->data[lvl - 1].L.nrows, M_inv->data[lvl - 1].L.ncols, M_inv->data[lvl - 1].L.val->size[0],
                         M_inv->data[lvl - 1].L.col_ptr->data, M_inv->data[lvl - 1].L.row_ind->data, M_inv->data[lvl - 1].L.val->data,
                         b_y1->data, mid, 1.0, 0.0);
        cusparsespmv_gpu(M_inv->data[lvl - 1].U.nrows, M_inv->data[lvl - 1].U.ncols, M_inv->data[lvl - 1].U.val->size[0],
                         M_inv->data[lvl - 1].U.col_ptr->data, M_inv->data[lvl - 1].U.row_ind->data, M_inv->data[lvl - 1].U.val->data,
                         mid, b_y1->data, 1.0, 0.0);

        timer.stop();
    }

    if (n > M_inv->data[lvl - 1].L.nrows)
    {
        double alpha = 1.0, beta = 1.0;
        int b_n = n - M_inv->data[lvl - 1].L.nrows;

        CudaTimer &timer = CudaTimer::getInstance();
        timer.setContext(lvl, CudaTimer::SPMV_SCHUR);
        timer.start();
        cusparsespmv_gpu(M_inv->data[lvl - 1].negE.nrows, M_inv->data[lvl - 1].negE.ncols, M_inv->data[lvl - 1].negE.val->size[0],
                         M_inv->data[lvl - 1].negE.row_ptr->data, M_inv->data[lvl - 1].negE.col_ind->data, M_inv->data[lvl - 1].negE.val->data,
                         b_y1->data, y2->data, alpha, beta);
        timer.stop();

        blocksPerGrid = (b_n + threadsPerBlock - 1) / threadsPerBlock;
        copy_y2_to_b_kernel<<<blocksPerGrid, threadsPerBlock>>>(y2->data, b->data, offset, nB, b_n);

        solve_milu_ginkgo(M_inv, lvl + 1, b, offset + M_inv->data[lvl - 1].L.nrows, b_y1, y2);

        blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
        copy_b_to_vectors_kernel<<<blocksPerGrid, threadsPerBlock>>>(b->data, b_y1->data, y2->data, offset, nB, b_n);

        timer.setContext(lvl, CudaTimer::SPMV_SCHUR);
        timer.start();
        cusparsespmv_gpu(M_inv->data[lvl - 1].negF.nrows, M_inv->data[lvl - 1].negF.ncols, M_inv->data[lvl - 1].negF.val->size[0],
                         M_inv->data[lvl - 1].negF.row_ptr->data, M_inv->data[lvl - 1].negF.col_ind->data, M_inv->data[lvl - 1].negF.val->data,
                         y2->data, b_y1->data, alpha, beta);
        timer.stop();

        if (M_inv->data[lvl - 1].L.val->size[0] != -1)
        {
            CudaTimer &timer2 = CudaTimer::getInstance();
            timer2.setContext(lvl, CudaTimer::TRIANGULAR_SOLVE);
            timer2.start();

            cusparsespmv_gpu(M_inv->data[lvl - 1].L.nrows, M_inv->data[lvl - 1].L.ncols, M_inv->data[lvl - 1].L.val->size[0],
                             M_inv->data[lvl - 1].L.col_ptr->data, M_inv->data[lvl - 1].L.row_ind->data, M_inv->data[lvl - 1].L.val->data,
                             b_y1->data, mid, 1.0, 0.0);
            cusparsespmv_gpu(M_inv->data[lvl - 1].U.nrows, M_inv->data[lvl - 1].U.ncols, M_inv->data[lvl - 1].U.val->size[0],
                             M_inv->data[lvl - 1].U.col_ptr->data, M_inv->data[lvl - 1].U.row_ind->data, M_inv->data[lvl - 1].U.val->data,
                             mid, b_y1->data, 1.0, 0.0);

            timer2.stop();
        }
        else
        {
            CudaTimer &timer = CudaTimer::getInstance();
            timer.setContext(lvl, CudaTimer::TRIANGULAR_SOLVE);
            timer.start();

            DS_PRECOND_MULTILEVEL *ds = (DS_PRECOND_MULTILEVEL *)info->DS_PRECOND;
            const double alpha = 1.0;
            const double beta = 0.0;

            double *d_temp;
            hipMalloc(&d_temp, nB * sizeof(double));
            hipMemset(d_temp, 0, nB * sizeof(double));

            hipblasDgemv(ds->handle_hipblas, HIPBLAS_OP_N, nB, nB, &alpha, M_inv->data[lvl - 1].U.val->data, nB, b_y1->data, 1, &beta, d_temp, 1);
            hipblasDcopy(ds->handle_hipblas, nB, d_temp, 1, b_y1->data, 1);

            hipFree(d_temp);
            timer.stop();
        }
    }

    blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    reorder_solution_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        b_y1->data, y2->data, b->data, M_inv->data[lvl - 1].q->data, M_inv->data[lvl - 1].colscal->data, offset, nB, n);

    hipFree(mid);
}



void MILUsolve_2args_ginkgo(const emxArray_struct0_T *M_inv, emxArray_real_T *b)
{
    int u0 = M_inv->data[0].L.nrows;
    int u1 = M_inv->data[0].negE.nrows;

    if (u0 <= 0)
        u0 = 1;
    if (u1 <= 0)
        u1 = 1;

    int max_size = (u0 > u1) ? u0 : u1;

    if (max_size > 1000000)
        max_size = 1000000;

    emxArray_real_T *b_y1 = (emxArray_real_T *)malloc(sizeof(emxArray_real_T));
    CHECK_HIP(hipMalloc((void **)&b_y1->data, max_size * sizeof(double)));
    CHECK_HIP(hipMemset(b_y1->data, 0, max_size * sizeof(double)));
    b_y1->size = (int *)malloc(sizeof(int));
    b_y1->size[0] = max_size;

    emxArray_real_T *y2 = (emxArray_real_T *)malloc(sizeof(emxArray_real_T));
    CHECK_HIP(hipMalloc((void **)&y2->data, max_size * sizeof(double)));
    CHECK_HIP(hipMemset(y2->data, 0, max_size * sizeof(double)));
    y2->size = (int *)malloc(sizeof(int));
    y2->size[0] = M_inv->data[0].negE.nrows > 0 ? M_inv->data[0].negE.nrows : 1;

    solve_milu_ginkgo(M_inv, 1, b, 0, b_y1, y2);

    CHECK_HIP(hipFree(b_y1->data));
    free(b_y1->size);
    free(b_y1);

    CHECK_HIP(hipFree(y2->data));
    free(y2->size);
    free(y2);
}



void constructb_gpu(emxArray_real_T *b, double *rhs, int length)
{
    size_t free_before, total;
    CHECK_HIP(hipMemGetInfo(&free_before, &total));

    CHECK_HIP(hipMalloc((void **)&b->data, length * sizeof(double)));

    size_t free_after;
    CHECK_HIP(hipMemGetInfo(&free_after, &total));
    
	// printf("b->rhs = %f, rhs = %f, length = %d\n",b->data[0],rhs[0],length);
    CHECK_HIP(hipMemcpy(b->data, rhs, (size_t)length * sizeof(double), hipMemcpyDeviceToDevice));

    b->size = (int *)malloc(sizeof(int));
    b->size[0] = length;
}


PetscErrorCode Apply_MILU_ginkgo(double *rhs, double *sol)
{
    PC_MILU *milu = (PC_MILU *)info->milu_data;

    emxArray_real_T *b = (emxArray_real_T *)malloc(sizeof(emxArray_real_T));

	// printf("rhs = %f, sol = %f, milu->A_pack.nr = %d\n",rhs[0],sol[0],milu->A_pack.nr);
    constructb_gpu(b, rhs, milu->A_pack.nr);

    MILUsolve_2args_ginkgo(milu->M_pack_inv, b);

    CHECK_HIP(hipMemcpy(sol, b->data, milu->A_pack.nr * sizeof(double), hipMemcpyDeviceToDevice));

    CHECK_HIP(hipFree(b->data));
    free(b->size);
    free(b);

    return 0;
}



void multilevel_ginkgo(PetscReal *in_x, PetscReal *out_y, PetscInt in_vsz)
{
//#ifdef ginkgo_DEBUG
	// printf("I am in multilevel_ginkgo preconditioning\n");
	// printf("in_vsz = %d\n", in_vsz);
	// printf("info->fact_n = %d\n", info->fact_n);
	// printf("info->bs = %d\n", info->bs);
	// printf("in_vsz = %d\n", in_vsz);
	// if (in_vsz != info->fact_n * info->bs)
	// {
	// 	printf("ERROR: the preconditioning vector size in_vsz != info->fact_n * info->bs\n");
	// 	return;
	// }
//#endif
	Apply_MILU_ginkgo(in_x, out_y);
}


// void PB_Preconditioning(PC pc, Vec px, Vec py,								 // Petsc vector
// 						PetscReal *vecx, PetscReal *vecy, PetscReal *vectmp, // dtmp is optional,for block-jacobi preconditioner
// 						PetscInt vsz)
// {    
// 	// using no preconditioning , just copy vecx to vecy
// 	hipMemcpy(vecy,vecx, sizeof(PetscReal) * vsz, hipMemcpyDeviceToDevice);
// }


 

// void PB_Preconditioning(PC pc, Vec px, Vec py,								 // Petsc vector
// 						PetscReal *vecx, PetscReal *vecy, PetscReal *vectmp, // dtmp is optional,for block-jacobi preconditioner
// 						PetscInt vsz)
// { 
	 
// 	// printf("I am in PB_Preconditioning with use_asm = %d\n", info->use_asm);

// 	PetscReal *in_x = vecx;
// 	PetscReal *out_y = vecy;
// 	// without ASM, we use vectmp as a temporary vector,
// 	// with ASM, we use info->asm_dltmp as a temporary vector, the sizes are different
// 	PetscReal *mid_tmp = vectmp;
// 	PetscInt in_vsz = vsz;

// 	if (info->use_asm)
// 	{
// 		int i;
// 		MPI_Status stat;
// 		int threads_per_block = 256;
// 		int ASM_nblocks_sendbuf = info->asm_sendbuf_len / threads_per_block + 1;
// 		int ASM_nblocks_recvbuf = info->asm_recvbuf_len / threads_per_block + 1;
// 		int ASM_nblocks_vec = vsz / threads_per_block + 1;
// 		// we have to enlarge vecx as the input of preconditiong
// 		// step1:  vecx -> lx
// 		hipDeviceSynchronize();
// 		gettimeofday(&pack_tstart, NULL);
// 		hipLaunchKernelGGL(ASMVecToSendbuffer, ASM_nblocks_sendbuf, threads_per_block, 0, 0, vecx, vsz,
// 						   info->asm_dsend_buf, info->asm_dsindices, info->asm_sendbuf_len);
// 		hipDeviceSynchronize();
// 		 // step2:  MPI communication: sendbuffer -> recvbuffer
	 
// 		hipMemcpy(info->asm_send_buf, info->asm_dsend_buf, sizeof(double) * info->asm_sendbuf_len, hipMemcpyDeviceToHost);
// 		hipDeviceSynchronize();
// 		gettimeofday(&pack_tend, NULL);
// 		pack_time += ((double)((pack_tend.tv_sec * 1000000.0 + pack_tend.tv_usec) - (pack_tstart.tv_sec * 1000000.0 + pack_tstart.tv_usec))) / 1000.0;
		
// 		int msg_tag = info->rank;
// 		hipDeviceSynchronize();
// 		gettimeofday(&comm_tstart, NULL);
// 		for (i = 0; i < info->asm_snp; i++)
// 		{
// 			MPI_Isend(info->asm_send_buf + info->asm_sstarts[i], (info->asm_sstarts[i + 1] - info->asm_sstarts[i]), MPI_DOUBLE, info->asm_sprocs[i], msg_tag, MPI_COMM_WORLD, info->asm_swaits + i);
// 		}
// 		for (i = 0; i < info->asm_rnp; i++)
// 		{
// 			MPI_Irecv(info->asm_recv_buf + info->asm_rstarts[i], (info->asm_rstarts[i + 1] - info->asm_rstarts[i]), MPI_DOUBLE, info->asm_rprocs[i], info->asm_rprocs[i], MPI_COMM_WORLD, info->asm_rwaits + i);
// 		}

// 		for (i = 0; i < info->asm_rnp; i++)
// 		{
// 			MPI_Wait(info->asm_rwaits + i, &stat);
// 		}

// 		gettimeofday(&comm_tend, NULL);
// 		comm_time += ((double)((comm_tend.tv_sec * 1000000.0 + comm_tend.tv_usec) - (comm_tstart.tv_sec * 1000000.0 + comm_tstart.tv_usec))) / 1000.0;
		
		
// 		gettimeofday(&unpack_tstart, NULL);
// 		hipMemcpy(info->asm_drecv_buf, info->asm_recv_buf, sizeof(double) * info->asm_recvbuf_len, hipMemcpyHostToDevice);
// 		hipDeviceSynchronize();
// 		// step3: vecx + recvbuffer -> lx
		 
// 		hipLaunchKernelGGL(ASMVecRecvbufferToLvec, ASM_nblocks_vec, threads_per_block, 0, 0, info->asm_drecv_buf, info->asm_drindices, info->asm_recvbuf_len, vecx, info->asm_self_dsindices, vsz, info->asm_dlx, info->asm_self_drindices, info->asm_lxsz);
// 		hipDeviceSynchronize();
// 		gettimeofday(&unpack_tend, NULL);
// 		unpack_time += ((double)((unpack_tend.tv_sec * 1000000.0 + unpack_tend.tv_usec) - (unpack_tstart.tv_sec * 1000000.0 + unpack_tstart.tv_usec))) / 1000.0;
// 		in_x = info->asm_dlx;
// 		out_y = info->asm_dly;
// 		mid_tmp = info->asm_dltmp;
// 		in_vsz = info->asm_lxsz;
// 	}
     
     
// 		// using no preconditioning , just copy vecx to vecy
// 	 hipMemcpy(out_y,in_x, sizeof(PetscReal) * in_vsz, hipMemcpyDeviceToDevice);

// 	// info->multilevel_ginkgo = PETSC_TRUE;
// 	// // 1. multilevel_ginkgo
// 	// if (info->multilevel_ginkgo)
// 	// {
// 	// 	// printf("I am in multilevel_ginkgo preconditioning\n");
// 	// 	// printf("in_x = %f, out_y = %f, vsz = %d, in_vsz = %d\n", in_x[0], out_y[0], vsz, in_vsz);
// 	// 	multilevel_ginkgo(in_x, out_y, in_vsz);
 
// 	// }

 

// 	// after preconditioning: we only need to keep local part: i.e. out_y -> vecy
// 	if (info->use_asm)
// 	{
// 		int threads_per_block = 256;
// 		int ASM_nblocks_vec = vsz / threads_per_block + 1;
// 		hipDeviceSynchronize();
// 		gettimeofday(&ASMLvecToVec_tstart, NULL);
// 		hipLaunchKernelGGL(ASMLvecToVec, ASM_nblocks_vec, threads_per_block, 0, 0, out_y, info->asm_self_drindices, info->asm_lxsz, vecy, info->asm_self_dsindices, vsz);
// 		hipDeviceSynchronize();
// 		gettimeofday(&ASMLvecToVec_tend, NULL);
// 		ASMLvecToVec_time += ((double)((ASMLvecToVec_tend.tv_sec * 1000000.0 + ASMLvecToVec_tend.tv_usec) - (ASMLvecToVec_tstart.tv_sec * 1000000.0 + ASMLvecToVec_tstart.tv_usec))) / 1000.0;
// 	}
// }

// Error checking macros
#define HIP_CHECK(err) do { \
    if (err != hipSuccess) { \
        fprintf(stderr, "HIP error: %s at line %d\n", hipGetErrorString(err), __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

#define HIPBLAS_CHECK(status) do { \
    if (status != HIPBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "hipBLAS error: %d at line %d\n", status, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} while(0)


// construct the global variables to record the time of pack and unpack and the time of ASM communication and ASMLvecToVec
double pack_time = 0.0;
double unpack_time = 0.0;
double ASMLvecToVec_time = 0.0;
double comm_time = 0.0;
struct timeval pack_tstart, pack_tend;
struct timeval unpack_tstart, unpack_tend;
struct timeval comm_tstart, comm_tend;
struct timeval ASMLvecToVec_tstart, ASMLvecToVec_tend;


// construct the global variables to record the time of solve() and gemv() in fast_solve  
double solve_time = 0.0;
struct timeval solve_tstart, solve_tend;
double gemv_time = 0.0;
struct timeval gemv_tstart, gemv_tend;

double construct_tEFG_time = 0.0;
struct timeval construct_tEFG_tstart, construct_tEFG_tend;

double collect_boundary_values_time = 0.0;
struct timeval collect_boundary_values_tstart, collect_boundary_values_tend;



void fast_solve(PetscReal *in_x, PetscReal *out_y, PetscInt in_vsz)
{
	PreprocessedData* preprocessed_ptr = (PreprocessedData*)info->preprocessed_ptr;
	int pn = preprocessed_ptr->pn;
	int num_boundary = preprocessed_ptr->num_boundary;
	rocblas_handle handle = preprocessed_ptr->handle;


	// Allocate GPU memory for intermediate results
    double *d_tEFG = preprocessed_ptr->d_tEFG;
	double *d_boundary_values = preprocessed_ptr->d_boundary_values;
	double *d_corrected_values = preprocessed_ptr->d_corrected_values;
	double *d_tEFG_new = preprocessed_ptr->d_tEFG_new;


	double* d_E3 = d_tEFG;
	double* d_F3 = d_tEFG + pn;
	double* d_G3 = d_tEFG + 2 * pn;
   
    HIP_CHECK(hipMemset(d_tEFG_new, 0, 3 * pn * sizeof(double)));


    // Extract on GPU with in_x as input
	double* d_extracted_x = preprocessed_ptr->d_extracted_x;
	double* d_rE = d_extracted_x;
	double* d_rF = d_extracted_x + pn;
	double* d_rG = d_extracted_x + 2 * pn;
	extract_kernel<<<blocks_fast_solve_pn, threads_per_block_fast_solve>>>(in_x, d_rE, d_rF, d_rG, pn);
	

	// Record time for first solve
	HIP_CHECK(hipDeviceSynchronize());
	gettimeofday(&solve_tstart, NULL);

    // First Solve
    Solve(d_rE, d_rF, d_rG, preprocessed_ptr, N, 1.0, d_E3, d_F3, d_G3);
    HIP_CHECK(hipDeviceSynchronize());
    gettimeofday(&solve_tend, NULL);
    solve_time += ((double)((solve_tend.tv_sec * 1000000.0 + solve_tend.tv_usec) - (solve_tstart.tv_sec * 1000000.0 + solve_tstart.tv_usec))) / 1000.0;
    

    // GPU operations between first and second Solve
    HIP_CHECK(hipDeviceSynchronize());
	gettimeofday(&collect_boundary_values_tstart, NULL);
    collect_boundary_values_kernel<<<blocks_fast_solve_num_boundary, threads_per_block_fast_solve>>>(d_tEFG, preprocessed_ptr->d_boundary_indices, d_boundary_values, num_boundary);
    HIP_CHECK(hipGetLastError());
	HIP_CHECK(hipDeviceSynchronize());
	gettimeofday(&collect_boundary_values_tend, NULL);
	collect_boundary_values_time += ((double)((collect_boundary_values_tend.tv_sec * 1000000.0 + collect_boundary_values_tend.tv_usec) - (collect_boundary_values_tstart.tv_sec * 1000000.0 + collect_boundary_values_tstart.tv_usec))) / 1000.0;

    // Matrix-vector multiplication using hipBLAS
    double gemv_alpha = 1.0;
    double gemv_beta = 0.0;
    
	HIP_CHECK(hipDeviceSynchronize());
    gettimeofday(&gemv_tstart, NULL);
    //HIPBLAS_CHECK(hipblasDgemv(handle, HIPBLAS_OP_N, num_boundary, num_boundary, &gemv_alpha, preprocessed_ptr->d_MEFG, num_boundary, d_boundary_values, 1, &gemv_beta, d_corrected_values, 1));
    HIPBLAS_CHECK(hipblasDsymv(handle, HIPBLAS_FILL_MODE_UPPER, num_boundary, &gemv_alpha, preprocessed_ptr->d_MEFG, num_boundary, d_boundary_values, 1, &gemv_beta, d_corrected_values, 1));
    //HIPBLAS_CHECK(hipblasDsymv(handle, HIPBLAS_FILL_MODE_UPPER, n, &alpha, d_A, lda, d_x, incx, &beta, d_y, incy));
	HIP_CHECK(hipDeviceSynchronize());
    gettimeofday(&gemv_tend, NULL);
    gemv_time += ((double)((gemv_tend.tv_sec * 1000000.0 + gemv_tend.tv_usec) - (gemv_tstart.tv_sec * 1000000.0 + gemv_tstart.tv_usec))) / 1000.0;
    


    // the block of scatter is same as the block of collect_boundary_values
    scatter_corrected_values_kernel<<<blocks_fast_solve_num_boundary, threads_per_block_fast_solve>>>(d_tEFG_new, preprocessed_ptr->d_boundary_indices, d_corrected_values, num_boundary, 3 * pn);
    HIP_CHECK(hipGetLastError());

    // Reshape tEFG_new to gtE, gtF, gtG
    double* d_gtE = d_tEFG_new;
    double* d_gtF = d_tEFG_new + pn;
    double* d_gtG = d_tEFG_new + 2 * pn;

    // Allocate GPU memory for second Solve outputs
    double *d_tE = preprocessed_ptr->d_tE;
	double *d_tF = preprocessed_ptr->d_tF;
	double *d_tG = preprocessed_ptr->d_tG;
 

    // Second Solve
	HIP_CHECK(hipDeviceSynchronize());
	gettimeofday(&solve_tstart, NULL);
    Solve(d_gtE, d_gtF, d_gtG, preprocessed_ptr, N, 1.0, d_tE, d_tF, d_tG);
	HIP_CHECK(hipDeviceSynchronize());
	gettimeofday(&solve_tend, NULL);
	solve_time += ((double)((solve_tend.tv_sec * 1000000.0 + solve_tend.tv_usec) - (solve_tstart.tv_sec * 1000000.0 + solve_tstart.tv_usec))) / 1000.0;

 
    // E3 -= tE, F3 -= tF, G3 -= tG
    vector_subtraction_kernel<<<blocks_fast_solve_pn, threads_per_block_fast_solve>>>(d_E3, d_E3, d_tE, d_F3, d_F3, d_tF, d_G3, d_G3, d_tG, pn);
    HIP_CHECK(hipGetLastError());
   
	//  d_E3, d_F3, d_G3  Interleave on GPU to out_y	
	interleave_kernel<<<blocks_fast_solve_pn, threads_per_block_fast_solve>>>(d_E3, d_F3, d_G3, out_y, pn);
	HIP_CHECK(hipGetLastError());
	HIP_CHECK(hipDeviceSynchronize());

}


void PB_Preconditioning(PC pc, Vec px, Vec py,								 // Petsc vector
						PetscReal *vecx, PetscReal *vecy, PetscReal *vectmp, // dtmp is optional,for block-jacobi preconditioner
						PetscInt vsz)
{ 
	 
	// printf("I am in PB_Preconditioning with use_asm = %d\n", info->use_asm);

	PetscReal *in_x = vecx;
	PetscReal *out_y = vecy;
	// without ASM, we use vectmp as a temporary vector,
	// with ASM, we use info->asm_dltmp as a temporary vector, the sizes are different
	PetscReal *mid_tmp = vectmp;
	PetscInt in_vsz = vsz;

	if (info->use_asm)
	{
		int i;
		MPI_Status stat;
		int threads_per_block = 256;
		int ASM_nblocks_sendbuf = info->asm_sendbuf_len / threads_per_block + 1;
		int ASM_nblocks_recvbuf = info->asm_recvbuf_len / threads_per_block + 1;
		int ASM_nblocks_vec = vsz / threads_per_block + 1;
		// we have to enlarge vecx as the input of preconditiong
		// step1:  vecx -> lx
		hipDeviceSynchronize();
		gettimeofday(&pack_tstart, NULL);
		hipLaunchKernelGGL(ASMVecToSendbuffer, ASM_nblocks_sendbuf, threads_per_block, 0, 0, vecx, vsz,
						   info->asm_dsend_buf, info->asm_dsindices, info->asm_sendbuf_len);
		hipDeviceSynchronize();
		 // step2:  MPI communication: sendbuffer -> recvbuffer
	 
		hipMemcpy(info->asm_send_buf, info->asm_dsend_buf, sizeof(double) * info->asm_sendbuf_len, hipMemcpyDeviceToHost);
		hipDeviceSynchronize();
		gettimeofday(&pack_tend, NULL);
		pack_time += ((double)((pack_tend.tv_sec * 1000000.0 + pack_tend.tv_usec) - (pack_tstart.tv_sec * 1000000.0 + pack_tstart.tv_usec))) / 1000.0;
		
		int msg_tag = info->rank;
		hipDeviceSynchronize();
		gettimeofday(&comm_tstart, NULL);
		for (i = 0; i < info->asm_snp; i++)
		{
			MPI_Isend(info->asm_send_buf + info->asm_sstarts[i], (info->asm_sstarts[i + 1] - info->asm_sstarts[i]), MPI_DOUBLE, info->asm_sprocs[i], msg_tag, MPI_COMM_WORLD, info->asm_swaits + i);
		}
		for (i = 0; i < info->asm_rnp; i++)
		{
			MPI_Irecv(info->asm_recv_buf + info->asm_rstarts[i], (info->asm_rstarts[i + 1] - info->asm_rstarts[i]), MPI_DOUBLE, info->asm_rprocs[i], info->asm_rprocs[i], MPI_COMM_WORLD, info->asm_rwaits + i);
		}

		for (i = 0; i < info->asm_rnp; i++)
		{
			MPI_Wait(info->asm_rwaits + i, &stat);
		}

		gettimeofday(&comm_tend, NULL);
		comm_time += ((double)((comm_tend.tv_sec * 1000000.0 + comm_tend.tv_usec) - (comm_tstart.tv_sec * 1000000.0 + comm_tstart.tv_usec))) / 1000.0;
		
		
		gettimeofday(&unpack_tstart, NULL);
		hipMemcpy(info->asm_drecv_buf, info->asm_recv_buf, sizeof(double) * info->asm_recvbuf_len, hipMemcpyHostToDevice);
		hipDeviceSynchronize();
		// step3: vecx + recvbuffer -> lx
		 
		hipLaunchKernelGGL(ASMVecRecvbufferToLvec, ASM_nblocks_vec, threads_per_block, 0, 0, info->asm_drecv_buf, info->asm_drindices, info->asm_recvbuf_len, vecx, info->asm_self_dsindices, vsz, info->asm_dlx, info->asm_self_drindices, info->asm_lxsz);
		hipDeviceSynchronize();
		gettimeofday(&unpack_tend, NULL);
		unpack_time += ((double)((unpack_tend.tv_sec * 1000000.0 + unpack_tend.tv_usec) - (unpack_tstart.tv_sec * 1000000.0 + unpack_tstart.tv_usec))) / 1000.0;
		in_x = info->asm_dlx;
		out_y = info->asm_dly;
		mid_tmp = info->asm_dltmp;
		in_vsz = info->asm_lxsz;
	}
     
     
		// using no preconditioning , just copy vecx to vecy
	//  hipMemcpy(out_y,in_x, sizeof(PetscReal) * in_vsz, hipMemcpyDeviceToDevice);

	//using fast_solve
    // printf("I am in fast_solve, in_vsz = %d\n", in_vsz);
	fast_solve(in_x, out_y, in_vsz);



	// after preconditioning: we only need to keep local part: i.e. out_y -> vecy
	if (info->use_asm)
	{
		int threads_per_block = 256;
		int ASM_nblocks_vec = vsz / threads_per_block + 1;
		hipDeviceSynchronize();
		gettimeofday(&ASMLvecToVec_tstart, NULL);
		hipLaunchKernelGGL(ASMLvecToVec, ASM_nblocks_vec, threads_per_block, 0, 0, out_y, info->asm_self_drindices, info->asm_lxsz, vecy, info->asm_self_dsindices, vsz);
		hipDeviceSynchronize();
		gettimeofday(&ASMLvecToVec_tend, NULL);
		ASMLvecToVec_time += ((double)((ASMLvecToVec_tend.tv_sec * 1000000.0 + ASMLvecToVec_tend.tv_usec) - (ASMLvecToVec_tstart.tv_sec * 1000000.0 + ASMLvecToVec_tstart.tv_usec))) / 1000.0;
	}
}



 
 