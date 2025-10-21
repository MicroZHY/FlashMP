#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include "KSPSolve_GMRES_GPU.h"
#include "../milu_inc/ilupack.h"
#include "../inc/MILUsolve.h"
#include <hipsparse.h>
#include <sys/stat.h>
#include "fast_solve.h"
 



const char* hipsparseStatusGetString(hipsparseStatus_t status);
 char filename[256];
char result_file[256];

void show_M_pack(const emxArray_struct0_T *M_pack, const char *output_filename)
{
    if (!M_pack || !M_pack->size)
    {
        printf("Error: Invalid M_pack structure\n");
        return;
    }

    FILE *file = fopen(output_filename, "a");
    if (file == NULL)
    {
        printf("Error: Unable to open the output file.\n");
        return;
    }

    printf("\n=== Factorization Info: %s ===\n", filename); // 注意：这里用了 output_filename，因为 filename 未定义
    fprintf(file, "\n=== Factorization Info: %s ===\n", filename);

    printf("\n=== M_pack Structure ===\n");
    printf("Total Levels: %d\n", M_pack->size[0]);
    printf("=======================\n");

    fprintf(file, "\n=== M_pack Structure ===\n");
    fprintf(file, "Total Levels: %d\n", M_pack->size[0]);
    fprintf(file, "=======================\n");

    for (int lvl = 0; lvl < M_pack->size[0]; lvl++)
    {
        printf("\nLevel %d:\n", lvl);
        printf("--------\n");

        fprintf(file, "\nLevel %d:\n", lvl);
        fprintf(file, "--------\n");

        if (lvl < M_pack->size[0] - 1)
        {
            // 非最后一层：显示 L, U, negE, negF

            // L 矩阵（CSC 格式）
            int L_nnz = M_pack->data[lvl].L.val->size[0];
            long long L_size = (long long)M_pack->data[lvl].L.nrows * M_pack->data[lvl].L.ncols; // 使用 long long 防止溢出
            double L_sparsity = ((double)(L_size - L_nnz)) / L_size;
            printf("  L: Size %d x %d, NNZ %d, Sparsity (%.2f%%)\n",
                   M_pack->data[lvl].L.nrows, M_pack->data[lvl].L.ncols, L_nnz, L_sparsity * 100);
            fprintf(file, "  L: Size %d x %d, NNZ %d, Sparsity (%.2f%%)\n",
                    M_pack->data[lvl].L.nrows, M_pack->data[lvl].L.ncols, L_nnz, L_sparsity * 100);

            // U 矩阵（CSC 格式）
            int U_nnz = M_pack->data[lvl].U.val->size[0];
            long long U_size = (long long)M_pack->data[lvl].U.nrows * M_pack->data[lvl].U.ncols;
            double U_sparsity = ((double)(U_size - U_nnz)) / U_size;
            printf("  U: Size %d x %d, NNZ %d, Sparsity (%.2f%%)\n",
                   M_pack->data[lvl].U.nrows, M_pack->data[lvl].U.ncols, U_nnz, U_sparsity * 100);
            fprintf(file, "  U: Size %d x %d, NNZ %d, Sparsity (%.2f%%)\n",
                    M_pack->data[lvl].U.nrows, M_pack->data[lvl].U.ncols, U_nnz, U_sparsity * 100);

            // negE 矩阵（CSR 格式）
            int negE_nnz = (M_pack->data[lvl].negE.nrows > 0) ? M_pack->data[lvl].negE.row_ptr->data[M_pack->data[lvl].negE.nrows] : 0;
            long long negE_size = (long long)M_pack->data[lvl].negE.nrows * M_pack->data[lvl].negE.ncols;
            double negE_sparsity = (M_pack->data[lvl].negE.nrows > 0) ? ((double)(negE_size - negE_nnz)) / negE_size : 1.0;
            printf("  negE: Size %d x %d, NNZ %d, Sparsity  (%.2f%%)\n",
                   M_pack->data[lvl].negE.nrows, M_pack->data[lvl].negE.ncols, negE_nnz, negE_sparsity * 100);
            fprintf(file, "  negE: Size %d x %d, NNZ %d, Sparsity  (%.2f%%)\n",
                    M_pack->data[lvl].negE.nrows, M_pack->data[lvl].negE.ncols, negE_nnz, negE_sparsity * 100);

            // negF 矩阵（CSR 格式）
            int negF_nnz = (M_pack->data[lvl].negF.nrows > 0) ? M_pack->data[lvl].negF.row_ptr->data[M_pack->data[lvl].negF.nrows] : 0;
            long long negF_size = (long long)M_pack->data[lvl].negF.nrows * M_pack->data[lvl].negF.ncols;
            double negF_sparsity = (M_pack->data[lvl].negF.nrows > 0) ? ((double)(negF_size - negF_nnz)) / negF_size : 1.0;
            printf("  negF: Size %d x %d, NNZ %d, Sparsity  (%.2f%%)\n",
                   M_pack->data[lvl].negF.nrows, M_pack->data[lvl].negF.ncols, negF_nnz, negF_sparsity * 100);
            fprintf(file, "  negF: Size %d x %d, NNZ %d, Sparsity  (%.2f%%)\n",
                    M_pack->data[lvl].negF.nrows, M_pack->data[lvl].negF.ncols, negF_nnz, negF_sparsity * 100);
        }
        else
        {
            // 最后一层：检查是密集还是稀疏
            int is_dense = (M_pack->data[lvl].L.val->size[0] == 0 &&
                            M_pack->data[lvl].U.val->size[0] == M_pack->data[lvl].U.nrows * M_pack->data[lvl].U.ncols);

            if (is_dense)
            {
                printf("  Last Level (Dense):\n");
                printf("    U: Size %d x %d\n",
                       M_pack->data[lvl].U.nrows, M_pack->data[lvl].U.ncols);
                fprintf(file, "  Last Level (Dense):\n");
                fprintf(file, "    U: Size %d x %d\n",
                        M_pack->data[lvl].U.nrows, M_pack->data[lvl].U.ncols);
            }
            else
            {
                // 稀疏情况：显示 L 和 U
                int L_nnz = M_pack->data[lvl].L.val->size[0];
                long long L_size = (long long)M_pack->data[lvl].L.nrows * M_pack->data[lvl].L.ncols;
                double L_sparsity = ((double)(L_size - L_nnz)) / L_size;
                printf("  L: Size %d x %d, NNZ %d, Sparsity  (%.2f%%)\n",
                       M_pack->data[lvl].L.nrows, M_pack->data[lvl].L.ncols, L_nnz, L_sparsity * 100);
                fprintf(file, "  L: Size %d x %d, NNZ %d, Sparsity  (%.2f%%)\n",
                        M_pack->data[lvl].L.nrows, M_pack->data[lvl].L.ncols, L_nnz, L_sparsity * 100);

                int U_nnz = M_pack->data[lvl].U.val->size[0];
                long long U_size = (long long)M_pack->data[lvl].U.nrows * M_pack->data[lvl].U.ncols;
                double U_sparsity = ((double)(U_size - U_nnz)) / U_size;
                printf("  U: Size %d x %d, NNZ %d, Sparsity  (%.2f%%)\n",
                       M_pack->data[lvl].U.nrows, M_pack->data[lvl].U.ncols, U_nnz, U_sparsity * 100);
                fprintf(file, "  U: Size %d x %d, NNZ %d, Sparsity  (%.2f%%)\n",
                        M_pack->data[lvl].U.nrows, M_pack->data[lvl].U.ncols, U_nnz, U_sparsity * 100);
            }
        }
    }

    printf("=======================\n");
    fprintf(file, "=======================\n");

    fclose(file);
}



extern GMRES_INFO * info;
__global__ void invDmU(double *dUInvDiagVal, int *dURowPtr, int *dUColVal, double *dUBlkVal, double *dUStarBlkVal,  int size)
{
	//int tid=blockDim.x*blockIdx.x+threadIdx.x;
	int tid=blockIdx.y*gridDim.x*blockDim.x + blockDim.x*blockIdx.x + threadIdx.x;
	__shared__ double s[WARPS_PER_BLOCK][THREE*THREE];
	int irow=tid/WARP_SIZE;
	int st_idx=dURowPtr[irow];
	int ed_idx=dURowPtr[irow+1];
	int lane=threadIdx.x%WARP_SIZE;
	int warp_id=threadIdx.x/WARP_SIZE;
	int offset=st_idx*THREE*THREE+lane;
	//int r=(lane/THREE)%THREE;
	//int c=lane%THREE;
	int c=(lane/THREE)%THREE;
	int r=lane%THREE;
	int range=(WARP_SIZE/(THREE*THREE))*(THREE*THREE);
	int idx;
	double rc_val;
	if(irow < size)
	{
		if(lane < range)
		{
			if(lane < (THREE*THREE))
			{s[warp_id][lane]=dUInvDiagVal[irow*THREE*THREE+lane];}
			while(offset < ed_idx*THREE*THREE)
			{
				idx=offset/(THREE*THREE);
				// r row of UInvDiagVal c column of UBlkVal
				//rc_val=s[warp_id][r*THREE+0]*dUBlkVal[idx*THREE*THREE+0*THREE+c]
				  //     + s[warp_id][r*THREE+1]*dUBlkVal[idx*THREE*THREE+1*THREE+c]
				    //   + s[warp_id][r*THREE+2]*dUBlkVal[idx*THREE*THREE+2*THREE+c];
				//dUStarBlkVal[idx*THREE*THREE+r*THREE+c]=rc_val;
				rc_val=s[warp_id][r+0*THREE]*dUBlkVal[idx*THREE*THREE+0+c*THREE]
				       + s[warp_id][r+1*THREE]*dUBlkVal[idx*THREE*THREE+1+c*THREE]
				       + s[warp_id][r+2*THREE]*dUBlkVal[idx*THREE*THREE+2+c*THREE];
				dUStarBlkVal[idx*THREE*THREE+r+c*THREE]=rc_val;
				offset+=range;
			}

		}
	}
}


__global__
void GPU_BILU_SWEEP_new(int *A_row_ptr,int *A_col_val,double *A_blk_val,
			int *L_row_ptr,int *L_col_val,double *L_blk_val,
			int *U_col_ptr,int *U_row_val,double *U_blk_val,int size,
			double *U_diag_val)
{
	__shared__ double s[WARPS_PER_BLOCK][WARP_SIZE];
	//int tid=blockDim.x*blockIdx.x+threadIdx.x;
	int tid=blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x;
	int irow=tid/WARP_SIZE;
	int st_idx=A_row_ptr[irow];
	int ed_idx=A_row_ptr[irow+1];
	int lane=threadIdx.x%WARP_SIZE;
	int warp_id=threadIdx.x/WARP_SIZE;
	int offset=st_idx*THREE*THREE+lane;
//	int r=(lane/THREE)%THREE;   FOR ROW-MAJOR FORMAT
//	int c=lane%THREE;
	int r=lane%THREE;
	int c=(lane/THREE)%THREE;
	int range=(WARP_SIZE/(THREE*THREE))*(THREE*THREE);
	int jcol,lidx,uidx,kcol,krow,kmax;
	int l_ed_idx,u_ed_idx;
	int itmp,idx;
	double det;

	if(irow < size)
	{
		if(lane < range)
		{	while(offset < ed_idx*THREE*THREE){
			idx=offset/(THREE*THREE);
			jcol=A_col_val[idx];
			lidx=L_row_ptr[irow];
		    	l_ed_idx=L_row_ptr[irow+1];	
			uidx=U_col_ptr[jcol];
			u_ed_idx=U_col_ptr[jcol+1];
			s[warp_id][lane]=0.0;	
			kmax=irow > jcol ? jcol:irow;
			while(lidx < l_ed_idx && uidx < u_ed_idx)
			{
				kcol=L_col_val[lidx];
				krow=U_row_val[uidx];
				if(kcol == krow)
				{
					if(kcol < kmax){
			//s[warp_id][lane]+=L_blk_val[lidx*THREE*THREE+r*THREE+0]*U_blk_val[uidx*THREE*THREE+c]
			//		+L_blk_val[lidx*THREE*THREE+r*THREE+1]*U_blk_val[uidx*THREE*THREE+THREE+c]
			//		+L_blk_val[lidx*THREE*THREE+r*THREE+2]*U_blk_val[uidx*THREE*THREE+2*THREE+c];
			s[warp_id][lane]+=L_blk_val[lidx*THREE*THREE+0*THREE+r]*U_blk_val[uidx*THREE*THREE+c*THREE+0]
					+L_blk_val[lidx*THREE*THREE+1*THREE+r]*U_blk_val[uidx*THREE*THREE+c*THREE+1]
					+L_blk_val[lidx*THREE*THREE+2*THREE+r]*U_blk_val[uidx*THREE*THREE+c*THREE+2];
						}
		
				}
				if(kcol <= krow){lidx++;}
				if(kcol >= krow){uidx++;}
			}
			s[warp_id][lane]=A_blk_val[offset]-s[warp_id][lane];
			if(irow > jcol) // update L_ij
			{//         we need to known c value  (0,c) (1,c) (2,c) -> (c,0) (c,1) (c,2)
				itmp=(uidx-1)*THREE*THREE;	
				det=U_blk_val[itmp+0]*U_blk_val[itmp+4]*U_blk_val[itmp+8]
				-U_blk_val[itmp+0]*U_blk_val[itmp+5]*U_blk_val[itmp+7]
				-U_blk_val[itmp+1]*U_blk_val[itmp+3]*U_blk_val[itmp+8]
				+U_blk_val[itmp+1]*U_blk_val[itmp+5]*U_blk_val[itmp+6]
				+U_blk_val[itmp+2]*U_blk_val[itmp+3]*U_blk_val[itmp+7]
				-U_blk_val[itmp+2]*U_blk_val[itmp+4]*U_blk_val[itmp+6];
				det=1.0/det;
				
		     //det=det * (s[warp_id][lane/THREE*THREE]*(U_blk_val[itmp+((c+1)%3)*3+1]*U_blk_val[itmp+((c+2)%3)*3+2]-
  		//					      U_blk_val[itmp+((c+1)%3)*3+2]*U_blk_val[itmp+((c+2)%3)*3+1])
		//	    +s[warp_id][lane/THREE*THREE+1]*(U_blk_val[itmp+((c+1)%3)*3+2]*U_blk_val[itmp+((c+2)%3)*3+0]-
		//					      U_blk_val[itmp+((c+1)%3)*3+0]*U_blk_val[itmp+((c+2)%3)*3+2])
		//	    +s[warp_id][lane/THREE*THREE+2]*(U_blk_val[itmp+((c+1)%3)*3+0]*U_blk_val[itmp+((c+2)%3)*3+1]-
		//					      U_blk_val[itmp+((c+1)%3)*3+1]*U_blk_val[itmp+((c+2)%3)*3+0]) );
		//		L_blk_val[(lidx-1)*THREE*THREE+r*THREE+c]=det;
		     det=det * (s[warp_id][lane/9*9+0*THREE+r]*(U_blk_val[itmp+((c+1)%3)+1*3]*U_blk_val[itmp+((c+2)%3)+2*3]-
  							      U_blk_val[itmp+((c+1)%3)+2*3]*U_blk_val[itmp+((c+2)%3)+1*3])
			    +s[warp_id][lane/9*9+1*THREE+r]*(U_blk_val[itmp+((c+1)%3)+2*3]*U_blk_val[itmp+((c+2)%3)+0*3]-
							      U_blk_val[itmp+((c+1)%3)+0*3]*U_blk_val[itmp+((c+2)%3)+2*3])
			    +s[warp_id][lane/9*9+2*THREE+r]*(U_blk_val[itmp+((c+1)%3)+0*3]*U_blk_val[itmp+((c+2)%3)+1*3]-
							      U_blk_val[itmp+((c+1)%3)+1*3]*U_blk_val[itmp+((c+2)%3)+0*3]) );
				L_blk_val[(lidx-1)*THREE*THREE+c*THREE+r]=det;

			}
			else
			{
				//U_blk_val[(uidx-1)*THREE*THREE+r*THREE+c]=s[warp_id][lane];
				U_blk_val[(uidx-1)*THREE*THREE+c*THREE+r]=s[warp_id][lane];
				// if irow == jcol , 2020.1.20 output diagonal of U for inverse computation
				if(irow == jcol)
				{
					//U_diag_val[irow*THREE*THREE+r*THREE+c]=s[warp_id][lane];
					U_diag_val[irow*THREE*THREE+c*THREE+r]=s[warp_id][lane];
				}
			}
			offset+=range;
			}
		}
	}

	
}
__global__ void invUdiag_Setzero(int *dLRowPtr, double *dLBlkVal, int *dURowPtr, double *dUBlkVal,
				double *dUDiagVal, double * dUInvDiagVal, int nrows)
{
	int tid=blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x;
	int warp_id=tid/WARP_SIZE;
	int local_id=tid%WARP_SIZE;
	int r=local_id%THREE;   // COLUMN-MAJOR
	int c=(local_id/THREE)%THREE;
	int irow=warp_id*(WARP_SIZE/(THREE*THREE))+local_id/(THREE*THREE);
	int range=WARP_SIZE/(THREE*THREE)*(THREE*THREE);
	double det;
	int itmp;
	int lidx, uidx;
	if(irow < nrows)
	{
		if(local_id < range)
		{
			// set the diag for dLBlkVal and dUBlkVal to zeros, and the L -> L_hat, U->U_hat
			lidx= (dLRowPtr[irow+1]-1)*THREE*THREE; // The last element-block at each row in CSR format
			dLBlkVal[lidx+c*THREE+r]=0.0;
			uidx = (dURowPtr[irow])*THREE*THREE; // The first element-block at each row in CSR format
			dUBlkVal[uidx+c*THREE+r]=0.0;	

			// read dUDiagVal, and write into dUInvDiagVal
			itmp=irow*THREE*THREE;			
			det=dUDiagVal[itmp+0]*dUDiagVal[itmp+4]*dUDiagVal[itmp+8]
			-dUDiagVal[itmp+0]*dUDiagVal[itmp+5]*dUDiagVal[itmp+7]
			-dUDiagVal[itmp+1]*dUDiagVal[itmp+3]*dUDiagVal[itmp+8]
			+dUDiagVal[itmp+1]*dUDiagVal[itmp+5]*dUDiagVal[itmp+6]
			+dUDiagVal[itmp+2]*dUDiagVal[itmp+3]*dUDiagVal[itmp+7]
			-dUDiagVal[itmp+2]*dUDiagVal[itmp+4]*dUDiagVal[itmp+6];
			det=1.0/det;
			dUInvDiagVal[itmp+c*THREE+r]=det*(dUDiagVal[itmp+(c+1)%3+(r+1)%3*THREE]*dUDiagVal[itmp+(c+2)%3+(r+2)%3*THREE]
						         -dUDiagVal[itmp+(c+1)%3+(r+2)%3*THREE]*dUDiagVal[itmp+(c+2)%3+(r+1)%3*THREE]);
		}	
	}
	
}
// extract the diagonal part from U no matter they are inversed or not. dUdiagVal is output
__global__ void extractUdiag(int *dURowPtr,double *dUBlkVal,double *dUDiagBlkVal, int nrows)
{
	int tid=blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x;
	int warp_id=tid/WARP_SIZE;
	int local_id=tid%WARP_SIZE;
	int r=local_id%THREE;   // COLUMN-MAJOR
	int c=(local_id/THREE)%THREE;
	int irow=warp_id*(WARP_SIZE/(THREE*THREE))+local_id/(THREE*THREE);
	int range=WARP_SIZE/(THREE*THREE)*(THREE*THREE);
	int itmp;
	int uidx;
	if(irow < nrows)
	{
		if(local_id < range)
		{
			uidx = (dURowPtr[irow])*THREE*THREE; // The first element-block at each row in CSR format
			// read dUDiagVal, and write into dUInvDiagVal
			itmp=irow*THREE*THREE;			
			dUDiagBlkVal[itmp+c*THREE+r]=dUBlkVal[uidx+c*THREE+r];
		}
	}	
	
}
__global__ void invUdiag_Inplace(int *dURowPtr,  double *dUBlkVal, int nrows)
{
	__shared__ double s[WARPS_PER_BLOCK][WARP_SIZE];
	int tid=blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x;
	int warp_id=tid/WARP_SIZE;
	int lwpid=threadIdx.x/WARP_SIZE;
	int local_id=tid%WARP_SIZE;
	int c=(local_id/THREE)%THREE; // we use column major format within blocks here, IMPORTANT
	int r=local_id%THREE;
	int irow=warp_id*(WARP_SIZE/(THREE*THREE))+local_id/(THREE*THREE);
	int range=WARP_SIZE/(THREE*THREE)*(THREE*THREE);
	int stidx;
	double det;
	int itmp;
	int pos;
	if(irow < nrows)
	{
		if(local_id < range)
		{
			stidx=dURowPtr[irow];
			itmp=stidx*THREE*THREE;			
			// load the diagonal part to shared memory
			s[lwpid][local_id]=dUBlkVal[itmp+c*THREE+r];  
			pos=local_id/(THREE*THREE)*(THREE*THREE);
			det=s[lwpid][pos+0]*s[lwpid][pos+4]*s[lwpid][pos+8]
			-s[lwpid][pos+0]*s[lwpid][pos+5]*s[lwpid][pos+7]
			-s[lwpid][pos+1]*s[lwpid][pos+3]*s[lwpid][pos+8]
			+s[lwpid][pos+1]*s[lwpid][pos+5]*s[lwpid][pos+6]
			+s[lwpid][pos+2]*s[lwpid][pos+3]*s[lwpid][pos+7]
			-s[lwpid][pos+2]*s[lwpid][pos+4]*s[lwpid][pos+6];
			det=1.0/det;
			dUBlkVal[itmp+c*THREE+r]=det*(s[lwpid][pos+(c+1)%3+((r+1)%3)*THREE]*s[lwpid][pos+(c+2)%3+((r+2)%3)*THREE]
						     -s[lwpid][pos+(c+1)%3+((r+2)%3)*THREE]*s[lwpid][pos+(c+2)%3+((r+1)%3)*THREE]);
		}	
	}
}
__global__ void invUDiag(double *dUDiagVal, double *dUInvDiagVal, int nrows)
{
	// a warp 32 threads -> 3 block rows =27 threads 
	//int tid=blockDim.x*blockIdx.x+threadIdx.x;
	int tid=blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x;
	int warp_id=tid/WARP_SIZE;
	int local_id=tid%WARP_SIZE;
	//int r=(local_id/THREE)%THREE;
	//int c=local_id%THREE;
	int r=local_id%THREE;
	int c=(local_id/THREE)%THREE;
	int irow=warp_id*(WARP_SIZE/(THREE*THREE))+local_id/(THREE*THREE);
	int range=WARP_SIZE/(THREE*THREE)*(THREE*THREE);
	double det;
	int itmp;
	if(irow < nrows)
	{
		if(local_id < range)
		{
			itmp=irow*THREE*THREE;			
			det=dUDiagVal[itmp+0]*dUDiagVal[itmp+4]*dUDiagVal[itmp+8]
			-dUDiagVal[itmp+0]*dUDiagVal[itmp+5]*dUDiagVal[itmp+7]
			-dUDiagVal[itmp+1]*dUDiagVal[itmp+3]*dUDiagVal[itmp+8]
			+dUDiagVal[itmp+1]*dUDiagVal[itmp+5]*dUDiagVal[itmp+6]
			+dUDiagVal[itmp+2]*dUDiagVal[itmp+3]*dUDiagVal[itmp+7]
			-dUDiagVal[itmp+2]*dUDiagVal[itmp+4]*dUDiagVal[itmp+6];
			det=1.0/det;
			//dUInvDiagVal[itmp+r*THREE+c]=det*(dUDiagVal[itmp+(c+1)%3*THREE+(r+1)%3]*dUDiagVal[itmp+(c+2)%3*THREE+(r+2)%3]
			//			         -dUDiagVal[itmp+(c+1)%3*THREE+(r+2)%3]*dUDiagVal[itmp+(c+2)%3*THREE+(r+1)%3]);
			dUInvDiagVal[itmp+c*THREE+r]=det*(dUDiagVal[itmp+(c+1)%3+(r+1)%3*THREE]*dUDiagVal[itmp+(c+2)%3+(r+2)%3*THREE]
						         -dUDiagVal[itmp+(c+1)%3+(r+2)%3*THREE]*dUDiagVal[itmp+(c+2)%3+(r+1)%3*THREE]);
		}	
	}
}


__global__ void csc_to_csr(double *dCSC_UBlkVal,double *dUBlkVal,int *dcsc_to_csr_map, 
			  int Unnz, int bs)
{
	int tid=blockIdx.x*blockDim.x+threadIdx.x;
	int val_len=Unnz*bs*bs;
	int bz=bs*bs;
	int csc_pos;
	int offset;
	int csr_pos;

	while(tid < val_len)
	{
		csc_pos=tid/bz;
		offset=tid%bz;
		csr_pos=dcsc_to_csr_map[csc_pos];
		dUBlkVal[csr_pos*bs*bs+offset]=dCSC_UBlkVal[tid];	
		tid+=gridDim.x*blockDim.x;
	}
}
// for ILU factorization and triangluar solves by using cuSPARSE totally
void cusparse_bsrilu()
{
  struct timeval prebsrsv_tstart, prebsrsv_tend; double prebsrsv_t=0.0; // all
  // get the DS_PBILU_PRECOND_CUSPARSE 
  DS_PBILU_PRECOND_CUSPARSE * ds=(DS_PBILU_PRECOND_CUSPARSE *)info->DS_PBILU;

	if(!ds->pBuffer )
	{
		// we need to do buffersize and analysis only once
		ds->cup_stat=hipsparseDbsrilu02_bufferSize(ds->handle_L,ds->dir_A,info->fact_n,info->Factnnz,ds->descr_A,
			ds->dFactBlkVal,ds->dFactRowPtr,ds->dFactColVal,info->bs,ds->info_A,&ds->pBufferSize_A);
		hipDeviceSynchronize();
		if(ds->cup_stat != HIPSPARSE_STATUS_SUCCESS){printf("ERROR in A hipsparseDbsrilu02_bufferSize\n");}
		
		// allocate memory
		hipMalloc((void **)&ds->pBuffer,ds->pBufferSize_A);

		ds->cup_stat=hipsparseDbsrilu02_analysis(ds->handle_L,ds->dir_A,info->fact_n,info->Factnnz,ds->descr_A,
			ds->dFactBlkVal,ds->dFactRowPtr,ds->dFactColVal,info->bs,ds->info_A,ds->policy_A,ds->pBuffer);
		hipDeviceSynchronize();
		if(ds->cup_stat != HIPSPARSE_STATUS_SUCCESS){printf("ERROR in A hipsparseDbsrilu02_analysis\n");}

		
	}
  	gettimeofday(&prebsrsv_tstart,NULL);
	if(!ds->pLBuffer || !ds->pUBuffer)
	{
		//S1: bufferSize, run only once for both linear and non-linear problems
		ds->cup_stat=hipsparseDbsrsv2_bufferSize(ds->handle_L,ds->dir_L,ds->trans_L,
		info->fact_n,info->Factnnz,ds->descr_L,ds->dFactBlkVal,ds->dFactRowPtr,ds->dFactColVal,
		info->bs,ds->info_L,&ds->pLBufferSize);
		hipDeviceSynchronize();
		if(ds->cup_stat != HIPSPARSE_STATUS_SUCCESS){printf("ERROR in L hipsparseDbsrsv2_bufferSize\n");}

		//info->cup_stat=hipsparseDbsrsv2_bufferSize(info->handle_U,info->dir_U,info->trans_U,
		ds->cup_stat=hipsparseDbsrsv2_bufferSize(ds->handle_L,ds->dir_U,ds->trans_U,
		info->fact_n,info->Factnnz,ds->descr_U,ds->dFactBlkVal,ds->dFactRowPtr,ds->dFactColVal,
		info->bs,ds->info_U,&ds->pUBufferSize);
		hipDeviceSynchronize();
		if(ds->cup_stat != HIPSPARSE_STATUS_SUCCESS){printf("ERROR in U hipsparseDbsrsv2_bufferSize\n");}

		//allocate memory
		hipMalloc((void **)&ds->pLBuffer,ds->pLBufferSize);
		hipMalloc((void **)&ds->pUBuffer,ds->pUBufferSize);

		//S2: Analysis
		ds->cup_stat=hipsparseDbsrsv2_analysis(ds->handle_L,ds->dir_L,ds->trans_L,info->fact_n,
		info->Factnnz,ds->descr_L,ds->dFactBlkVal,ds->dFactRowPtr,ds->dFactColVal,info->bs,ds->info_L,
		ds->policy_L,ds->pLBuffer);
		hipDeviceSynchronize();
		if(ds->cup_stat != HIPSPARSE_STATUS_SUCCESS){printf("ERROR in L hipsparseDbsrsv2_analysis\n");}
		
		//info->cup_stat=hipsparseDbsrsv2_analysis(info->handle_U,info->dir_U,info->trans_U,info->fact_n,
		ds->cup_stat=hipsparseDbsrsv2_analysis(ds->handle_L,ds->dir_U,ds->trans_U,info->fact_n,
		info->Factnnz,ds->descr_U,ds->dFactBlkVal,ds->dFactRowPtr,ds->dFactColVal,info->bs,ds->info_U,
		ds->policy_U,ds->pUBuffer);
		hipDeviceSynchronize();
		if(ds->cup_stat != HIPSPARSE_STATUS_SUCCESS){printf("ERROR in U hipsparseDbsrsv2_analysis\n");}

		
	} 
  	gettimeofday(&prebsrsv_tend,NULL);
        prebsrsv_t=((double) ((prebsrsv_tend.tv_sec*1000000.0 + prebsrsv_tend.tv_usec)-(prebsrsv_tstart.tv_sec*1000000.0+prebsrsv_tstart.tv_usec)))/1000.0;
        printf("rank=%d,prebsr_t=%12.8lf\n",info->rank,prebsrsv_t);

	// do bsr ilu factorization
        ds->cup_stat=hipsparseDbsrilu02(ds->handle_L,ds->dir_A,info->fact_n,info->Factnnz,ds->descr_A,
		ds->dFactBlkVal,ds->dFactRowPtr,ds->dFactColVal,info->bs,ds->info_A,ds->policy_A,ds->pBuffer);
        hipDeviceSynchronize();
	if(ds->cup_stat != HIPSPARSE_STATUS_SUCCESS){printf("ERROR in A hipsparseDbsrilu02\n");}
	
	
}





void asynchronous_pbilu()
{
	// NOTES: GPU_BILU_SWEEP_NEW is row-major (NOT column major ) with blocks
	DS_PBILU_ASYN *ds=(DS_PBILU_ASYN *)info->DS_PBILU;
	PetscInt bs=info->bs;
	PetscInt bsbs=bs*bs;
	PetscInt isweep;
	PetscInt block_size,total_threads;
	PetscInt gridx, gridy;
	PetscInt nblks_per_warp;
	// the initial value of L(k) and U(k) are initialized using the operating matrix A_i in each subdomain
	//but we need the CSC format for asynchronous point-block factorization 
	if(ds->asyn_use_exactilu_asfirst && !ds->asyn_use_exactilu_called)
	{
		// just set the values of L and U on GPU as the values of exactL and exactU
		// currently, the values in dLBlkVal and dUBlkVal are copied and initialized in InitialGMRES_GPU 
		// NOTE that the dLBlkVal and dUBlkVal are overwritten by hExactLBlkVal, and hExactUBlkVal if 
		// asyn_use_exactilu_asfirst is used during a non-linear newton process
		hipMemcpy(info->dLBlkVal,ds->hExactLBlkVal,sizeof(PetscReal)*info->Lnnz*bsbs,hipMemcpyHostToDevice);
		hipMemcpy(info->dUBlkVal,ds->hExactUBlkVal,sizeof(PetscReal)*info->Unnz*bsbs,hipMemcpyHostToDevice);
		hipDeviceSynchronize();
		// currently dUBlkVal's diagonal is already inversed,we need to inverse back
		block_size=128;
		nblks_per_warp=WARP_SIZE/bsbs;
		total_threads=(info->fact_n/nblks_per_warp+1)*WARP_SIZE;
		if(total_threads/block_size +1 > 65535)
		{  gridx=65535; gridy=total_threads/(gridx*block_size)+1;}
		else
		{ gridx=total_threads/block_size+1;  gridy=1;}
		dim3 invUdiag_Inplace_nblocks(gridx,gridy);
		hipLaunchKernelGGL(invUdiag_Inplace, invUdiag_Inplace_nblocks, block_size, 0, 0, info->dURowPtr,info->dUBlkVal,info->fact_n);
		hipDeviceSynchronize();
		// then we obtain the normal L and U (diagonal is not inversed)
		dim3 extractUdiag_nblocks(gridx,gridy);
		hipLaunchKernelGGL(extractUdiag, extractUdiag_nblocks, block_size, 0, 0, info->dURowPtr, info->dUBlkVal, ds->dUDiagVal, info->fact_n);
		hipDeviceSynchronize();

		ds->asyn_use_exactilu_called=PETSC_TRUE;
	}
	else
	{  
		//if(ds->asyn_use_exactilu_asfrist) // means the first step uses exact ILU, here is the 
		// use asynchronous ILU factorization, and use the exact L and U factors as initial guess  
		// 2021.10.24 
		cusparse_bsr2bsc(info->dURowPtr, info->dUColVal, info->dUBlkVal,
				ds->dCSC_UColPtr, ds->dCSC_URowVal, ds->dCSC_UBlkVal,info->fact_n, info->Unnz, info->bs);

		total_threads=info->fact_n*WARP_SIZE;
		block_size=128;
		if(total_threads/block_size+1 > 65535)
		{
			gridx=65535;
			gridy=total_threads/(gridx*block_size)+1;
		}
		else
		{
			gridx=total_threads/block_size+1;
			gridy=1;
		}
		dim3 nblocks(gridx,gridy);

		for(isweep=1;isweep<=ds->sweep_num;isweep++)
		{
			hipLaunchKernelGGL(GPU_BILU_SWEEP_new, nblocks, block_size, 0, 0, ds->dFactRowPtr,ds->dFactColVal,ds->dFactBlkVal,
			info->dLRowPtr,info->dLColVal,info->dLBlkVal,ds->dCSC_UColPtr,ds->dCSC_URowVal,ds->dCSC_UBlkVal,
					info->fact_n,ds->dUDiagVal);
			hipDeviceSynchronize();
		}	

		// we need to do and csc to csr transformation  
	//	hipLaunchKernelGGL(csc_to_csr, 256, 256, 0, 0, ds->dCSC_UBlkVal,info->dUBlkVal,ds->dcsc_to_csr_map,info->Unnz,info->bs);
	//	hipDeviceSynchronize();
		// 2021.10.25. perhaps we don't have to write our own bsr2bsr subroutine, just call from cuSPARSE 
		cusparse_bsr2bsc(ds->dCSC_UColPtr,ds->dCSC_URowVal,ds->dCSC_UBlkVal,
				 info->dURowPtr, info->dUColVal, info->dUBlkVal, info->fact_n, info->Unnz, info->bs);
	}
	// then we obtain the L and U factors via asynchronous point-block ILU factorization   
	
	// in GPU_BILU_SWEEP_new outputs dUdiagVal 
	if(info->cusparse_precond)
	{
		// we need to compute the inverse of the diagonal blocks
		block_size=128;
		nblks_per_warp=WARP_SIZE/bsbs;
		total_threads=(info->fact_n/nblks_per_warp+1)*WARP_SIZE;
		if(total_threads/block_size +1 > 65535)
		{
			gridx=65535;
			gridy=total_threads/(gridx*block_size)+1;
		}
		else
		{
			gridx=total_threads/block_size+1;
            		gridy=1;
		}
        	dim3 invUdiag_nblocks(gridx,gridy);
		hipLaunchKernelGGL(invUDiag, invUdiag_nblocks, block_size, 0, 0, ds->dUDiagVal,ds->dUInvDiagVal, info->fact_n);
		 
		hipDeviceSynchronize();
		// once the inverse of the diagonal blocks are obtained
		// compute Ustar = inv(D)*U, one warp for one block row 
		block_size=128;
		total_threads=info->fact_n*WARP_SIZE;
		if(total_threads/block_size +1 > 65535)
		{
		    gridx=65535;
		    gridy=total_threads/(gridx*block_size)+1;
		}
		else
		{
		    gridx=total_threads/block_size+1;
		    gridy=1;
		}
		// compute Ustar
		dim3 invDmU_nblocks(gridx,gridy);
		hipLaunchKernelGGL(invDmU, invDmU_nblocks, block_size, 0, 0, ds->dUInvDiagVal,info->dURowPtr,info->dUColVal,info->dUBlkVal,ds->dUStarBlkVal,info->fact_n);
		hipDeviceSynchronize();
	}	
	if(info->bisai_precond)
	{
		DS_PRECOND_BISAI *ds_bisai = (DS_PRECOND_BISAI *)info->DS_PRECOND;
		// right now, we have L and U factors on GPU, and we have InvL and InvU 's pattern.	
		// Firstly, compute the inverse of the diagonal blocks
		block_size=128;
		nblks_per_warp=WARP_SIZE/bsbs;
		total_threads=(info->fact_n/nblks_per_warp+1)*WARP_SIZE;
		if(total_threads/block_size +1 > 65535)
		{
			gridx=65535;
			gridy=total_threads/(gridx*block_size)+1;
		}
		else
		{
			gridx=total_threads/block_size+1;
            		gridy=1;
		}
		dim3 invUdiag_Inplace_nblocks(gridx,gridy);
		hipLaunchKernelGGL(invUdiag_Inplace, invUdiag_Inplace_nblocks, block_size, 0, 0, info->dURowPtr,info->dUBlkVal,info->fact_n);
		hipDeviceSynchronize();
		
		// then we need to obtain the values for  InvL and InvU 
		estimate_bisai_numeric();
		// currently, InvL and InvU is expressed in CSC, we need a CSR version for Sparse Matrix-vector Multiplication
		cusparse_bsr2bsc(ds_bisai->dcsc_InvLColPtr, ds_bisai->dcsc_InvLRowVal, ds_bisai->dcsc_InvLBlkVal,
			ds_bisai->dInvLRowPtr, ds_bisai->dInvLColVal, ds_bisai->dInvLBlkVal,info->fact_n, ds_bisai->InvLnnz, info->bs);

		cusparse_bsr2bsc(ds_bisai->dcsc_InvUColPtr, ds_bisai->dcsc_InvURowVal, ds_bisai->dcsc_InvUBlkVal,
			ds_bisai->dInvURowPtr, ds_bisai->dInvUColVal, ds_bisai->dInvUBlkVal,info->fact_n, ds_bisai->InvUnnz, info->bs);

	}
	if(info->iterative_precond)
	{
		DS_PRECOND_ITERATIVE *ds_iter=(DS_PRECOND_ITERATIVE *)info->DS_PRECOND;
		block_size=128;
		nblks_per_warp=WARP_SIZE/bsbs;
		total_threads=(info->fact_n/nblks_per_warp+1)*WARP_SIZE;
		if(total_threads/block_size + 1 > 65535)
		{
			gridx=65535;
			gridy=total_threads/(gridx*block_size)+1;
		}
		else
		{
			gridx=total_threads/block_size+1;
			gridy=1;
		}
		dim3 invUDiag_Setzero_nblocks(gridx,gridy);
		// inverse the diagonal of U in dUdiagBlkVal, at the same time ,set the diaongal of 
		// L and U factors to zeros in order to form L_hat and U_hat in the iterative method
		hipLaunchKernelGGL(invUdiag_Setzero, invUDiag_Setzero_nblocks, block_size, 0, 0, info->dLRowPtr,info->dLBlkVal,info->dURowPtr,info->dUBlkVal,
							   ds->dUDiagVal,ds_iter->dUdiagBlkVal,info->fact_n); // dUdiagBlkVal is for inverse
		hipDeviceSynchronize();
	}
	
}






// 检查文件是否存在
int file_exists_inv(const char *filename)
{
    struct stat buffer;
    return (stat(filename, &buffer) == 0);
}

// 生成文件名 (基于矩阵大小和参数)  using ngridx droptol condest
void generate_filename_inv(char *filename, int ngridx, double droptol, double condest, double deltat)
{
    sprintf(filename, "M_pack_inv_ngrid%d_deltat%.1f_droptol%.8f_condest%.1f.bin", ngridx, deltat, droptol, condest);
}

// 初始化层级结构
static void initialize_level_structure(struct0_T *level)
{
    // 初始化 L 矩阵结构
    level->L.col_ptr = (emxArray_int32_T *)malloc(sizeof(emxArray_int32_T));
    level->L.col_ptr->data = NULL;
    level->L.col_ptr->size = (int *)malloc(sizeof(int));
    level->L.col_ptr->size[0] = 0;
    level->L.col_ptr->numDimensions = 1;
    level->L.col_ptr->allocatedSize = 0;
    level->L.col_ptr->canFreeData = true;

    level->L.row_ind = (emxArray_int32_T *)malloc(sizeof(emxArray_int32_T));
    level->L.row_ind->data = NULL;
    level->L.row_ind->size = (int *)malloc(sizeof(int));
    level->L.row_ind->size[0] = 0;
    level->L.row_ind->numDimensions = 1;
    level->L.row_ind->allocatedSize = 0;
    level->L.row_ind->canFreeData = true;

    level->L.val = (emxArray_real_T *)malloc(sizeof(emxArray_real_T));
    level->L.val->data = NULL;
    level->L.val->size = (int *)malloc(sizeof(int));
    level->L.val->size[0] = 0;
    level->L.val->numDimensions = 1;
    level->L.val->allocatedSize = 0;
    level->L.val->canFreeData = true;

    // 初始化 U 矩阵结构
    level->U.col_ptr = (emxArray_int32_T *)malloc(sizeof(emxArray_int32_T));
    level->U.col_ptr->data = NULL;
    level->U.col_ptr->size = (int *)malloc(sizeof(int));
    level->U.col_ptr->size[0] = 0;
    level->U.col_ptr->numDimensions = 1;
    level->U.col_ptr->allocatedSize = 0;
    level->U.col_ptr->canFreeData = true;

    level->U.row_ind = (emxArray_int32_T *)malloc(sizeof(emxArray_int32_T));
    level->U.row_ind->data = NULL;
    level->U.row_ind->size = (int *)malloc(sizeof(int));
    level->U.row_ind->size[0] = 0;
    level->U.row_ind->numDimensions = 1;
    level->U.row_ind->allocatedSize = 0;
    level->U.row_ind->canFreeData = true;

    level->U.val = (emxArray_real_T *)malloc(sizeof(emxArray_real_T));
    level->U.val->data = NULL;
    level->U.val->size = (int *)malloc(sizeof(int));
    level->U.val->size[0] = 0;
    level->U.val->numDimensions = 1;
    level->U.val->allocatedSize = 0;
    level->U.val->canFreeData = true;

    // 初始化 negE 矩阵结构
    level->negE.row_ptr = (emxArray_int32_T *)malloc(sizeof(emxArray_int32_T));
    level->negE.row_ptr->data = NULL;
    level->negE.row_ptr->size = (int *)malloc(sizeof(int));
    level->negE.row_ptr->size[0] = 0;
    level->negE.row_ptr->numDimensions = 1;
    level->negE.row_ptr->allocatedSize = 0;
    level->negE.row_ptr->canFreeData = true;

    level->negE.col_ind = (emxArray_int32_T *)malloc(sizeof(emxArray_int32_T));
    level->negE.col_ind->data = NULL;
    level->negE.col_ind->size = (int *)malloc(sizeof(int));
    level->negE.col_ind->size[0] = 0;
    level->negE.col_ind->numDimensions = 1;
    level->negE.col_ind->allocatedSize = 0;
    level->negE.col_ind->canFreeData = true;

    level->negE.val = (emxArray_real_T *)malloc(sizeof(emxArray_real_T));
    level->negE.val->data = NULL;
    level->negE.val->size = (int *)malloc(sizeof(int));
    level->negE.val->size[0] = 0;
    level->negE.val->numDimensions = 1;
    level->negE.val->allocatedSize = 0;
    level->negE.val->canFreeData = true;

    // 初始化 negF 矩阵结构
    level->negF.row_ptr = (emxArray_int32_T *)malloc(sizeof(emxArray_int32_T));
    level->negF.row_ptr->data = NULL;
    level->negF.row_ptr->size = (int *)malloc(sizeof(int));
    level->negF.row_ptr->size[0] = 0;
    level->negF.row_ptr->numDimensions = 1;
    level->negF.row_ptr->allocatedSize = 0;
    level->negF.row_ptr->canFreeData = true;

    level->negF.col_ind = (emxArray_int32_T *)malloc(sizeof(emxArray_int32_T));
    level->negF.col_ind->data = NULL;
    level->negF.col_ind->size = (int *)malloc(sizeof(int));
    level->negF.col_ind->size[0] = 0;
    level->negF.col_ind->numDimensions = 1;
    level->negF.col_ind->allocatedSize = 0;
    level->negF.col_ind->canFreeData = true;

    level->negF.val = (emxArray_real_T *)malloc(sizeof(emxArray_real_T));
    level->negF.val->data = NULL;
    level->negF.val->size = (int *)malloc(sizeof(int));
    level->negF.val->size[0] = 0;
    level->negF.val->numDimensions = 1;
    level->negF.val->allocatedSize = 0;
    level->negF.val->canFreeData = true;

    // 初始化向量结构
    level->p = (emxArray_int32_T *)malloc(sizeof(emxArray_int32_T));
    level->p->data = NULL;
    level->p->size = (int *)malloc(sizeof(int));
    level->p->size[0] = 0;
    level->p->numDimensions = 1;
    level->p->allocatedSize = 0;
    level->p->canFreeData = true;

    level->q = (emxArray_int32_T *)malloc(sizeof(emxArray_int32_T));
    level->q->data = NULL;
    level->q->size = (int *)malloc(sizeof(int));
    level->q->size[0] = 0;
    level->q->numDimensions = 1;
    level->q->allocatedSize = 0;
    level->q->canFreeData = true;

    level->d = (emxArray_real_T *)malloc(sizeof(emxArray_real_T));
    level->d->data = NULL;
    level->d->size = (int *)malloc(sizeof(int));
    level->d->size[0] = 0;
    level->d->numDimensions = 1;
    level->d->allocatedSize = 0;
    level->d->canFreeData = true;

    level->rowscal = (emxArray_real_T *)malloc(sizeof(emxArray_real_T));
    level->rowscal->data = NULL;
    level->rowscal->size = (int *)malloc(sizeof(int));
    level->rowscal->size[0] = 0;
    level->rowscal->numDimensions = 1;
    level->rowscal->allocatedSize = 0;
    level->rowscal->canFreeData = true;

    level->colscal = (emxArray_real_T *)malloc(sizeof(emxArray_real_T));
    level->colscal->data = NULL;
    level->colscal->size = (int *)malloc(sizeof(int));
    level->colscal->size[0] = 0;
    level->colscal->numDimensions = 1;
    level->colscal->allocatedSize = 0;
    level->colscal->canFreeData = true;
}


// 从二进制文件加载 M_pack_inv 结构
int loadM_pack_inv_binary(emxArray_struct0_T *M_pack_inv, const char *filename)
{
    FILE *fp = fopen(filename, "rb");
    if (!fp)
    {
        printf("Error: Cannot open file %s\n", filename);
        return -1;
    }

    // 读取总层数
    int nlev;
    if (fread(&nlev, sizeof(int), 1, fp) != 1)
    {
        printf("Error: Failed to read nlev\n");
        fclose(fp);
        return -1;
    }

    printf("Debug: Loading M_pack_inv with %d levels\n", nlev);

    // 初始化 M_pack_inv 结构
    M_pack_inv->size = (int *)malloc(sizeof(int));
    M_pack_inv->size[0] = nlev;
    M_pack_inv->numDimensions = 1;
    M_pack_inv->allocatedSize = nlev;
    M_pack_inv->canFreeData = true;
    M_pack_inv->data = (struct0_T *)calloc(nlev, sizeof(struct0_T));

    // 遍历每一层
    for (int lvl = 0; lvl < nlev; lvl++)
    {
        struct0_T *current_level = &M_pack_inv->data[lvl];
        initialize_level_structure(current_level);

        // 判断是否为最后一层
        if (lvl < nlev - 1)
        {
            // 读取稠密/稀疏标志
            int is_dense;
            if (fread(&is_dense, sizeof(int), 1, fp) != 1)
            {
                printf("Error: Failed to read is_dense flag at level %d\n", lvl);
                fclose(fp);
                return -1;
            }

            // 读取 L 矩阵基本信息
            if (fread(&current_level->L.nrows, sizeof(int), 1, fp) != 1 ||
                fread(&current_level->L.ncols, sizeof(int), 1, fp) != 1)
            {
                printf("Error: Failed to read L matrix dimensions at level %d\n", lvl);
                fclose(fp);
                return -1;
            }

            if (is_dense)
            {
                // 稠密矩阵情况
                int val_size;
                if (fread(&val_size, sizeof(int), 1, fp) != 1 || val_size != -1)
                {
                    printf("Error: Failed to read or invalid L val_size at level %d\n", lvl);
                    fclose(fp);
                    return -1;
                }
                current_level->L.val->size[0] = -1;
                current_level->L.val->data = NULL;

                // 读取 U 矩阵
                if (fread(&current_level->U.nrows, sizeof(int), 1, fp) != 1 ||
                    fread(&current_level->U.ncols, sizeof(int), 1, fp) != 1)
                {
                    printf("Error: Failed to read U matrix dimensions at level %d\n", lvl);
                    fclose(fp);
                    return -1;
                }

                if (fread(&val_size, sizeof(int), 1, fp) != 1)
                {
                    printf("Error: Failed to read U val_size at level %d\n", lvl);
                    fclose(fp);
                    return -1;
                }

                current_level->U.val->size[0] = val_size;
                double *h_U_val = (double *)malloc(val_size * sizeof(double));
                if (fread(h_U_val, sizeof(double), val_size, fp) != val_size)
                {
                    printf("Error: Failed to read U matrix values at level %d\n", lvl);
                    free(h_U_val);
                    fclose(fp);
                    return -1;
                }
                hipMalloc((void **)&current_level->U.val->data, val_size * sizeof(double));
                hipMemcpy(current_level->U.val->data, h_U_val, val_size * sizeof(double), hipMemcpyHostToDevice);
                free(h_U_val);
            }
            else
            {
                // 稀疏矩阵情况
                int col_ptr_size;
                if (fread(&col_ptr_size, sizeof(int), 1, fp) != 1)
                {
                    printf("Error: Failed to read L col_ptr_size at level %d\n", lvl);
                    fclose(fp);
                    return -1;
                }
                current_level->L.col_ptr->size[0] = col_ptr_size;
                int *h_L_col_ptr = (int *)malloc(col_ptr_size * sizeof(int));
                if (fread(h_L_col_ptr, sizeof(int), col_ptr_size, fp) != col_ptr_size)
                {
                    printf("Error: Failed to read L col_ptr at level %d\n", lvl);
                    free(h_L_col_ptr);
                    fclose(fp);
                    return -1;
                }
                hipMalloc((void **)&current_level->L.col_ptr->data, col_ptr_size * sizeof(int));
                hipMemcpy(current_level->L.col_ptr->data, h_L_col_ptr, col_ptr_size * sizeof(int), hipMemcpyHostToDevice);
                free(h_L_col_ptr);

                int row_ind_size;
                if (fread(&row_ind_size, sizeof(int), 1, fp) != 1)
                {
                    printf("Error: Failed to read L row_ind_size at level %d\n", lvl);
                    fclose(fp);
                    return -1;
                }
                current_level->L.row_ind->size[0] = row_ind_size;
                int *h_L_row_ind = (int *)malloc(row_ind_size * sizeof(int));
                if (fread(h_L_row_ind, sizeof(int), row_ind_size, fp) != row_ind_size)
                {
                    printf("Error: Failed to read L row_ind at level %d\n", lvl);
                    free(h_L_row_ind);
                    fclose(fp);
                    return -1;
                }
                hipMalloc((void **)&current_level->L.row_ind->data, row_ind_size * sizeof(int));
                hipMemcpy(current_level->L.row_ind->data, h_L_row_ind, row_ind_size * sizeof(int), hipMemcpyHostToDevice);
                free(h_L_row_ind);

                int val_size;
                if (fread(&val_size, sizeof(int), 1, fp) != 1)
                {
                    printf("Error: Failed to read L val_size at level %d\n", lvl);
                    fclose(fp);
                    return -1;
                }
                current_level->L.val->size[0] = val_size;
                double *h_L_val = (double *)malloc(val_size * sizeof(double));
                if (fread(h_L_val, sizeof(double), val_size, fp) != val_size)
                {
                    printf("Error: Failed to read L values at level %d\n", lvl);
                    free(h_L_val);
                    fclose(fp);
                    return -1;
                }
                hipMalloc((void **)&current_level->L.val->data, val_size * sizeof(double));
                hipMemcpy(current_level->L.val->data, h_L_val, val_size * sizeof(double), hipMemcpyHostToDevice);
                free(h_L_val);

                // 读取 U 矩阵
                if (fread(&current_level->U.nrows, sizeof(int), 1, fp) != 1 ||
                    fread(&current_level->U.ncols, sizeof(int), 1, fp) != 1)
                {
                    printf("Error: Failed to read U matrix dimensions at level %d\n", lvl);
                    fclose(fp);
                    return -1;
                }

                if (fread(&col_ptr_size, sizeof(int), 1, fp) != 1)
                {
                    printf("Error: Failed to read U col_ptr_size at level %d\n", lvl);
                    fclose(fp);
                    return -1;
                }
                current_level->U.col_ptr->size[0] = col_ptr_size;
                int *h_U_col_ptr = (int *)malloc(col_ptr_size * sizeof(int));
                if (fread(h_U_col_ptr, sizeof(int), col_ptr_size, fp) != col_ptr_size)
                {
                    printf("Error: Failed to read U col_ptr at level %d\n", lvl);
                    free(h_U_col_ptr);
                    fclose(fp);
                    return -1;
                }
                hipMalloc((void **)&current_level->U.col_ptr->data, col_ptr_size * sizeof(int));
                hipMemcpy(current_level->U.col_ptr->data, h_U_col_ptr, col_ptr_size * sizeof(int), hipMemcpyHostToDevice);
                free(h_U_col_ptr);

                if (fread(&row_ind_size, sizeof(int), 1, fp) != 1)
                {
                    printf("Error: Failed to read U row_ind_size at level %d\n", lvl);
                    fclose(fp);
                    return -1;
                }
                current_level->U.row_ind->size[0] = row_ind_size;
                int *h_U_row_ind = (int *)malloc(row_ind_size * sizeof(int));
                if (fread(h_U_row_ind, sizeof(int), row_ind_size, fp) != row_ind_size)
                {
                    printf("Error: Failed to read U row_ind at level %d\n", lvl);
                    free(h_U_row_ind);
                    fclose(fp);
                    return -1;
                }
                hipMalloc((void **)&current_level->U.row_ind->data, row_ind_size * sizeof(int));
                hipMemcpy(current_level->U.row_ind->data, h_U_row_ind, row_ind_size * sizeof(int), hipMemcpyHostToDevice);
                free(h_U_row_ind);

                if (fread(&val_size, sizeof(int), 1, fp) != 1)
                {
                    printf("Error: Failed to read U val_size at level %d\n", lvl);
                    fclose(fp);
                    return -1;
                }
                current_level->U.val->size[0] = val_size;
                double *h_U_val = (double *)malloc(val_size * sizeof(double));
                if (fread(h_U_val, sizeof(double), val_size, fp) != val_size)
                {
                    printf("Error: Failed to read U values at level %d\n", lvl);
                    free(h_U_val);
                    fclose(fp);
                    return -1;
                }
                hipMalloc((void **)&current_level->U.val->data, val_size * sizeof(double));
                hipMemcpy(current_level->U.val->data, h_U_val, val_size * sizeof(double), hipMemcpyHostToDevice);
                free(h_U_val);
            }

            // 读取 negE 矩阵
            if (fread(&current_level->negE.nrows, sizeof(int), 1, fp) != 1 ||
                fread(&current_level->negE.ncols, sizeof(int), 1, fp) != 1)
            {
                printf("Error: Failed to read negE dimensions at level %d\n", lvl);
                fclose(fp);
                return -1;
            }

            int row_ptr_size;
            if (fread(&row_ptr_size, sizeof(int), 1, fp) != 1)
            {
                printf("Error: Failed to read negE row_ptr_size at level %d\n", lvl);
                fclose(fp);
                return -1;
            }
            current_level->negE.row_ptr->size[0] = row_ptr_size;
            int *h_negE_row_ptr = (int *)malloc(row_ptr_size * sizeof(int));
            if (fread(h_negE_row_ptr, sizeof(int), row_ptr_size, fp) != row_ptr_size)
            {
                printf("Error: Failed to read negE row_ptr at level %d\n", lvl);
                free(h_negE_row_ptr);
                fclose(fp);
                return -1;
            }
            hipMalloc((void **)&current_level->negE.row_ptr->data, row_ptr_size * sizeof(int));
            hipMemcpy(current_level->negE.row_ptr->data, h_negE_row_ptr, row_ptr_size * sizeof(int), hipMemcpyHostToDevice);
            free(h_negE_row_ptr);

            int col_ind_size;
            if (fread(&col_ind_size, sizeof(int), 1, fp) != 1)
            {
                printf("Error: Failed to read negE col_ind_size at level %d\n", lvl);
                fclose(fp);
                return -1;
            }
            current_level->negE.col_ind->size[0] = col_ind_size;
            int *h_negE_col_ind = (int *)malloc(col_ind_size * sizeof(int));
            if (fread(h_negE_col_ind, sizeof(int), col_ind_size, fp) != col_ind_size)
            {
                printf("Error: Failed to read negE col_ind at level %d\n", lvl);
                free(h_negE_col_ind);
                fclose(fp);
                return -1;
            }
            hipMalloc((void **)&current_level->negE.col_ind->data, col_ind_size * sizeof(int));
            hipMemcpy(current_level->negE.col_ind->data, h_negE_col_ind, col_ind_size * sizeof(int), hipMemcpyHostToDevice);
            free(h_negE_col_ind);

            int val_size;
            if (fread(&val_size, sizeof(int), 1, fp) != 1)
            {
                printf("Error: Failed to read negE val_size at level %d\n", lvl);
                fclose(fp);
                return -1;
            }
            current_level->negE.val->size[0] = val_size;
            double *h_negE_val = (double *)malloc(val_size * sizeof(double));
            if (fread(h_negE_val, sizeof(double), val_size, fp) != val_size)
            {
                printf("Error: Failed to read negE values at level %d\n", lvl);
                free(h_negE_val);
                fclose(fp);
                return -1;
            }
            hipMalloc((void **)&current_level->negE.val->data, val_size * sizeof(double));
            hipMemcpy(current_level->negE.val->data, h_negE_val, val_size * sizeof(double), hipMemcpyHostToDevice);
            free(h_negE_val);

            // 读取 negF 矩阵
            if (fread(&current_level->negF.nrows, sizeof(int), 1, fp) != 1 ||
                fread(&current_level->negF.ncols, sizeof(int), 1, fp) != 1)
            {
                printf("Error: Failed to read negF dimensions at level %d\n", lvl);
                fclose(fp);
                return -1;
            }

            if (fread(&row_ptr_size, sizeof(int), 1, fp) != 1)
            {
                printf("Error: Failed to read negF row_ptr_size at level %d\n", lvl);
                fclose(fp);
                return -1;
            }
            current_level->negF.row_ptr->size[0] = row_ptr_size;
            int *h_negF_row_ptr = (int *)malloc(row_ptr_size * sizeof(int));
            if (fread(h_negF_row_ptr, sizeof(int), row_ptr_size, fp) != row_ptr_size)
            {
                printf("Error: Failed to read negF row_ptr at level %d\n", lvl);
                free(h_negF_row_ptr);
                fclose(fp);
                return -1;
            }
            hipMalloc((void **)&current_level->negF.row_ptr->data, row_ptr_size * sizeof(int));
            hipMemcpy(current_level->negF.row_ptr->data, h_negF_row_ptr, row_ptr_size * sizeof(int), hipMemcpyHostToDevice);
            free(h_negF_row_ptr);

            if (fread(&col_ind_size, sizeof(int), 1, fp) != 1)
            {
                printf("Error: Failed to read negF col_ind_size at level %d\n", lvl);
                fclose(fp);
                return -1;
            }
            current_level->negF.col_ind->size[0] = col_ind_size;
            int *h_negF_col_ind = (int *)malloc(col_ind_size * sizeof(int));
            if (fread(h_negF_col_ind, sizeof(int), col_ind_size, fp) != col_ind_size)
            {
                printf("Error: Failed to read negF col_ind at level %d\n", lvl);
                free(h_negF_col_ind);
                fclose(fp);
                return -1;
            }
            hipMalloc((void **)&current_level->negF.col_ind->data, col_ind_size * sizeof(int));
            hipMemcpy(current_level->negF.col_ind->data, h_negF_col_ind, col_ind_size * sizeof(int), hipMemcpyHostToDevice);
            free(h_negF_col_ind);

            if (fread(&val_size, sizeof(int), 1, fp) != 1)
            {
                printf("Error: Failed to read negF val_size at level %d\n", lvl);
                fclose(fp);
                return -1;
            }
            current_level->negF.val->size[0] = val_size;
            double *h_negF_val = (double *)malloc(val_size * sizeof(double));
            if (fread(h_negF_val, sizeof(double), val_size, fp) != val_size)
            {
                printf("Error: Failed to read negF values at level %d\n", lvl);
                free(h_negF_val);
                fclose(fp);
                return -1;
            }
            hipMalloc((void **)&current_level->negF.val->data, val_size * sizeof(double));
            hipMemcpy(current_level->negF.val->data, h_negF_val, val_size * sizeof(double), hipMemcpyHostToDevice);
            free(h_negF_val);

            // 读取 d 向量
            int d_size;
            if (fread(&d_size, sizeof(int), 1, fp) != 1)
            {
                printf("Error: Failed to read d size at level %d\n", lvl);
                fclose(fp);
                return -1;
            }
            current_level->d->size[0] = d_size;
            if (d_size > 0)
            {
                double *h_d = (double *)malloc(d_size * sizeof(double));
                if (fread(h_d, sizeof(double), d_size, fp) != d_size)
                {
                    printf("Error: Failed to read d values at level %d\n", lvl);
                    free(h_d);
                    fclose(fp);
                    return -1;
                }
                hipMalloc((void **)&current_level->d->data, d_size * sizeof(double));
                hipMemcpy(current_level->d->data, h_d, d_size * sizeof(double), hipMemcpyHostToDevice);
                free(h_d);
            }
        }
        else
        {
            // 最后一层的处理
            int is_last_level, is_dense;
            if (fread(&is_last_level, sizeof(int), 1, fp) != 1 ||
                fread(&is_dense, sizeof(int), 1, fp) != 1)
            {
                printf("Error: Failed to read flags at level %d\n", lvl);
                fclose(fp);
                return -1;
            }

            if (is_dense)
            {   
                printf("I am in the last level in loadM_pack_inv_binary, is_dense: %d\n", is_dense);
                // 稠密矩阵情况
                if (fread(&current_level->L.nrows, sizeof(int), 1, fp) != 1 ||
                    fread(&current_level->L.ncols, sizeof(int), 1, fp) != 1 ||
                    fread(&current_level->U.nrows, sizeof(int), 1, fp) != 1 ||
                    fread(&current_level->U.ncols, sizeof(int), 1, fp) != 1)
                {
                    printf("Error: Failed to read matrix dimensions at level %d\n", lvl);
                    fclose(fp);
                    return -1;
                }

                int U_val_size;
                if (fread(&U_val_size, sizeof(int), 1, fp) != 1)
                {
                    printf("Error: Failed to read U val_size at level %d\n", lvl);
                    fclose(fp);
                    return -1;
                }
                current_level->L.val->size[0] = -1; // 表示稠密情况下 L 为空
                current_level->U.val->size[0] = U_val_size;
                double *h_U_val = (double *)malloc(U_val_size * sizeof(double));
                if (fread(h_U_val, sizeof(double), U_val_size, fp) != U_val_size)
                {
                    printf("Error: Failed to read U values at level %d\n", lvl);
                    free(h_U_val);
                    fclose(fp);
                    return -1;
                }
                hipMalloc((void **)&current_level->U.val->data, U_val_size * sizeof(double));
                hipMemcpy(current_level->U.val->data, h_U_val, U_val_size * sizeof(double), hipMemcpyHostToDevice);
                free(h_U_val);
            }
            else
            {
                // 稀疏矩阵情况
                if (fread(&current_level->L.nrows, sizeof(int), 1, fp) != 1 ||
                    fread(&current_level->L.ncols, sizeof(int), 1, fp) != 1)
                {
                    printf("Error: Failed to read L dimensions at level %d\n", lvl);
                    fclose(fp);
                    return -1;
                }

                int col_ptr_size;
                if (fread(&col_ptr_size, sizeof(int), 1, fp) != 1)
                {
                    printf("Error: Failed to read L col_ptr_size at level %d\n", lvl);
                    fclose(fp);
                    return -1;
                }
                current_level->L.col_ptr->size[0] = col_ptr_size;
                int *h_L_col_ptr = (int *)malloc(col_ptr_size * sizeof(int));
                if (fread(h_L_col_ptr, sizeof(int), col_ptr_size, fp) != col_ptr_size)
                {
                    printf("Error: Failed to read L col_ptr at level %d\n", lvl);
                    free(h_L_col_ptr);
                    fclose(fp);
                    return -1;
                }
                hipMalloc((void **)&current_level->L.col_ptr->data, col_ptr_size * sizeof(int));
                hipMemcpy(current_level->L.col_ptr->data, h_L_col_ptr, col_ptr_size * sizeof(int), hipMemcpyHostToDevice);
                free(h_L_col_ptr);

                int row_ind_size;
                if (fread(&row_ind_size, sizeof(int), 1, fp) != 1)
                {
                    printf("Error: Failed to read L row_ind_size at level %d\n", lvl);
                    fclose(fp);
                    return -1;
                }
                current_level->L.row_ind->size[0] = row_ind_size;
                int *h_L_row_ind = (int *)malloc(row_ind_size * sizeof(int));
                if (fread(h_L_row_ind, sizeof(int), row_ind_size, fp) != row_ind_size)
                {
                    printf("Error: Failed to read L row_ind at level %d\n", lvl);
                    free(h_L_row_ind);
                    fclose(fp);
                    return -1;
                }
                hipMalloc((void **)&current_level->L.row_ind->data, row_ind_size * sizeof(int));
                hipMemcpy(current_level->L.row_ind->data, h_L_row_ind, row_ind_size * sizeof(int), hipMemcpyHostToDevice);
                free(h_L_row_ind);

                int val_size;
                if (fread(&val_size, sizeof(int), 1, fp) != 1)
                {
                    printf("Error: Failed to read L val_size at level %d\n", lvl);
                    fclose(fp);
                    return -1;
                }
                current_level->L.val->size[0] = val_size;
                double *h_L_val = (double *)malloc(val_size * sizeof(double));
                if (fread(h_L_val, sizeof(double), val_size, fp) != val_size)
                {
                    printf("Error: Failed to read L values at level %d\n", lvl);
                    free(h_L_val);
                    fclose(fp);
                    return -1;
                }
                hipMalloc((void **)&current_level->L.val->data, val_size * sizeof(double));
                hipMemcpy(current_level->L.val->data, h_L_val, val_size * sizeof(double), hipMemcpyHostToDevice);
                free(h_L_val);

                // 读取 U 矩阵
                if (fread(&current_level->U.nrows, sizeof(int), 1, fp) != 1 ||
                    fread(&current_level->U.ncols, sizeof(int), 1, fp) != 1)
                {
                    printf("Error: Failed to read U dimensions at level %d\n", lvl);
                    fclose(fp);
                    return -1;
                }

                if (fread(&col_ptr_size, sizeof(int), 1, fp) != 1)
                {
                    printf("Error: Failed to read U col_ptr_size at level %d\n", lvl);
                    fclose(fp);
                    return -1;
                }
                current_level->U.col_ptr->size[0] = col_ptr_size;
                int *h_U_col_ptr = (int *)malloc(col_ptr_size * sizeof(int));
                if (fread(h_U_col_ptr, sizeof(int), col_ptr_size, fp) != col_ptr_size)
                {
                    printf("Error: Failed to read U col_ptr at level %d\n", lvl);
                    free(h_U_col_ptr);
                    fclose(fp);
                    return -1;
                }
                hipMalloc((void **)&current_level->U.col_ptr->data, col_ptr_size * sizeof(int));
                hipMemcpy(current_level->U.col_ptr->data, h_U_col_ptr, col_ptr_size * sizeof(int), hipMemcpyHostToDevice);
                free(h_U_col_ptr);

                if (fread(&row_ind_size, sizeof(int), 1, fp) != 1)
                {
                    printf("Error: Failed to read U row_ind_size at level %d\n", lvl);
                    fclose(fp);
                    return -1;
                }
                current_level->U.row_ind->size[0] = row_ind_size;
                int *h_U_row_ind = (int *)malloc(row_ind_size * sizeof(int));
                if (fread(h_U_row_ind, sizeof(int), row_ind_size, fp) != row_ind_size)
                {
                    printf("Error: Failed to read U row_ind at level %d\n", lvl);
                    free(h_U_row_ind);
                    fclose(fp);
                    return -1;
                }
                hipMalloc((void **)&current_level->U.row_ind->data, row_ind_size * sizeof(int));
                hipMemcpy(current_level->U.row_ind->data, h_U_row_ind, row_ind_size * sizeof(int), hipMemcpyHostToDevice);
                free(h_U_row_ind);

                if (fread(&val_size, sizeof(int), 1, fp) != 1)
                {
                    printf("Error: Failed to read U val_size at level %d\n", lvl);
                    fclose(fp);
                    return -1;
                }
                current_level->U.val->size[0] = val_size;
                double *h_U_val = (double *)malloc(val_size * sizeof(double));
                if (fread(h_U_val, sizeof(double), val_size, fp) != val_size)
                {
                    printf("Error: Failed to read U values at level %d\n", lvl);
                    free(h_U_val);
                    fclose(fp);
                    return -1;
                }
                hipMalloc((void **)&current_level->U.val->data, val_size * sizeof(double));
                hipMemcpy(current_level->U.val->data, h_U_val, val_size * sizeof(double), hipMemcpyHostToDevice);
                free(h_U_val);
                printf("L.val->size[0] = %d, U.val->size[0]: %d\n", current_level->L.val->size[0], current_level->U.val->size[0]);
            }

            // 读取 negE 矩阵
            if (fread(&current_level->negE.nrows, sizeof(int), 1, fp) != 1 ||
                fread(&current_level->negE.ncols, sizeof(int), 1, fp) != 1)
            {
                printf("Error: Failed to read negE dimensions at level %d\n", lvl);
                fclose(fp);
                return -1;
            }

            int row_ptr_size;
            if (fread(&row_ptr_size, sizeof(int), 1, fp) != 1)
            {
                printf("Error: Failed to read negE row_ptr_size at level %d\n", lvl);
                fclose(fp);
                return -1;
            }
            current_level->negE.row_ptr->size[0] = row_ptr_size;
            int *h_negE_row_ptr = (int *)malloc(row_ptr_size * sizeof(int));
            if (fread(h_negE_row_ptr, sizeof(int), row_ptr_size, fp) != row_ptr_size)
            {
                printf("Error: Failed to read negE row_ptr at level %d\n", lvl);
                free(h_negE_row_ptr);
                fclose(fp);
                return -1;
            }
            hipMalloc((void **)&current_level->negE.row_ptr->data, row_ptr_size * sizeof(int));
            hipMemcpy(current_level->negE.row_ptr->data, h_negE_row_ptr, row_ptr_size * sizeof(int), hipMemcpyHostToDevice);
            free(h_negE_row_ptr);

            int col_ind_size;
            if (fread(&col_ind_size, sizeof(int), 1, fp) != 1)
            {
                printf("Error: Failed to read negE col_ind_size at level %d\n", lvl);
                fclose(fp);
                return -1;
            }
            current_level->negE.col_ind->size[0] = col_ind_size;
            int *h_negE_col_ind = (int *)malloc(col_ind_size * sizeof(int));
            if (fread(h_negE_col_ind, sizeof(int), col_ind_size, fp) != col_ind_size)
            {
                printf("Error: Failed to read negE col_ind at level %d\n", lvl);
                free(h_negE_col_ind);
                fclose(fp);
                return -1;
            }
            if (col_ind_size > 0)
            {
                hipMalloc((void **)&current_level->negE.col_ind->data, col_ind_size * sizeof(int));
                hipMemcpy(current_level->negE.col_ind->data, h_negE_col_ind, col_ind_size * sizeof(int), hipMemcpyHostToDevice);
            }
            free(h_negE_col_ind);

            int val_size;
            if (fread(&val_size, sizeof(int), 1, fp) != 1)
            {
                printf("Error: Failed to read negE val_size at level %d\n", lvl);
                fclose(fp);
                return -1;
            }
            current_level->negE.val->size[0] = val_size;
            double *h_negE_val = (double *)malloc(val_size * sizeof(double));
            if (fread(h_negE_val, sizeof(double), val_size, fp) != val_size)
            {
                printf("Error: Failed to read negE values at level %d\n", lvl);
                free(h_negE_val);
                fclose(fp);
                return -1;
            }
            if (val_size > 0)
            {
                hipMalloc((void **)&current_level->negE.val->data, val_size * sizeof(double));
                hipMemcpy(current_level->negE.val->data, h_negE_val, val_size * sizeof(double), hipMemcpyHostToDevice);
            }
            free(h_negE_val);

            // 读取 negF 矩阵
            if (fread(&current_level->negF.nrows, sizeof(int), 1, fp) != 1 ||
                fread(&current_level->negF.ncols, sizeof(int), 1, fp) != 1)
            {
                printf("Error: Failed to read negF dimensions at level %d\n", lvl);
                fclose(fp);
                return -1;
            }

            if (fread(&row_ptr_size, sizeof(int), 1, fp) != 1)
            {
                printf("Error: Failed to read negF row_ptr_size at level %d\n", lvl);
                fclose(fp);
                return -1;
            }
            current_level->negF.row_ptr->size[0] = row_ptr_size;
            int *h_negF_row_ptr = (int *)malloc(row_ptr_size * sizeof(int));
            if (fread(h_negF_row_ptr, sizeof(int), row_ptr_size, fp) != row_ptr_size)
            {
                printf("Error: Failed to read negF row_ptr at level %d\n", lvl);
                free(h_negF_row_ptr);
                fclose(fp);
                return -1;
            }
            hipMalloc((void **)&current_level->negF.row_ptr->data, row_ptr_size * sizeof(int));
            hipMemcpy(current_level->negF.row_ptr->data, h_negF_row_ptr, row_ptr_size * sizeof(int), hipMemcpyHostToDevice);
            free(h_negF_row_ptr);

            if (fread(&col_ind_size, sizeof(int), 1, fp) != 1)
            {
                printf("Error: Failed to read negF col_ind_size at level %d\n", lvl);
                fclose(fp);
                return -1;
            }
            current_level->negF.col_ind->size[0] = col_ind_size;
            int *h_negF_col_ind = (int *)malloc(col_ind_size * sizeof(int));
            if (fread(h_negF_col_ind, sizeof(int), col_ind_size, fp) != col_ind_size)
            {
                printf("Error: Failed to read negF col_ind at level %d\n", lvl);
                free(h_negF_col_ind);
                fclose(fp);
                return -1;
            }
            if (col_ind_size > 0)
            {
                hipMalloc((void **)&current_level->negF.col_ind->data, col_ind_size * sizeof(int));
                hipMemcpy(current_level->negF.col_ind->data, h_negF_col_ind, col_ind_size * sizeof(int), hipMemcpyHostToDevice);
            }
            free(h_negF_col_ind);

            if (fread(&val_size, sizeof(int), 1, fp) != 1)
            {
                printf("Error: Failed to read negF val_size at level %d\n", lvl);
                fclose(fp);
                return -1;
            }
            current_level->negF.val->size[0] = val_size;
            double *h_negF_val = (double *)malloc(val_size * sizeof(double));
            if (fread(h_negF_val, sizeof(double), val_size, fp) != val_size)
            {
                printf("Error: Failed to read negF values at level %d\n", lvl);
                free(h_negF_val);
                fclose(fp);
                return -1;
            }
            if (val_size > 0)
            {
                hipMalloc((void **)&current_level->negF.val->data, val_size * sizeof(double));
                hipMemcpy(current_level->negF.val->data, h_negF_val, val_size * sizeof(double), hipMemcpyHostToDevice);
            }
            free(h_negF_val);

            // 读取 d 向量
            int d_size;
            if (fread(&d_size, sizeof(int), 1, fp) != 1)
            {
                printf("Error: Failed to read d size at level %d\n", lvl);
                fclose(fp);
                return -1;
            }
            current_level->d->size[0] = d_size;
            if (d_size > 0)
            {
                double *h_d = (double *)malloc(d_size * sizeof(double));
                if (fread(h_d, sizeof(double), d_size, fp) != d_size)
                {
                    printf("Error: Failed to read d values at level %d\n", lvl);
                    free(h_d);
                    fclose(fp);
                    return -1;
                }
                hipMalloc((void **)&current_level->d->data, d_size * sizeof(double));
                hipMemcpy(current_level->d->data, h_d, d_size * sizeof(double), hipMemcpyHostToDevice);
                free(h_d);
            }
        }

        // 读取 p 向量
        int p_size;
        if (fread(&p_size, sizeof(int), 1, fp) != 1)
        {
            printf("Error: Failed to read p size at level %d\n", lvl);
            fclose(fp);
            return -1;
        }
        current_level->p->size[0] = p_size;
        int *h_p = (int *)malloc(p_size * sizeof(int));
        if (fread(h_p, sizeof(int), p_size, fp) != p_size)
        {
            printf("Error: Failed to read p values at level %d\n", lvl);
            free(h_p);
            fclose(fp);
            return -1;
        }
        hipMalloc((void **)&current_level->p->data, p_size * sizeof(int));
        hipMemcpy(current_level->p->data, h_p, p_size * sizeof(int), hipMemcpyHostToDevice);
        free(h_p);

        // 读取 q 向量
        int q_size;
        if (fread(&q_size, sizeof(int), 1, fp) != 1)
        {
            printf("Error: Failed to read q size at level %d\n", lvl);
            fclose(fp);
            return -1;
        }
        current_level->q->size[0] = q_size;
        int *h_q = (int *)malloc(q_size * sizeof(int));
        if (fread(h_q, sizeof(int), q_size, fp) != q_size)
        {
            printf("Error: Failed to read q values at level %d\n", lvl);
            free(h_q);
            fclose(fp);
            return -1;
        }
        hipMalloc((void **)&current_level->q->data, q_size * sizeof(int));
        hipMemcpy(current_level->q->data, h_q, q_size * sizeof(int), hipMemcpyHostToDevice);
        free(h_q);

        // 读取 rowscal 向量
        int rowscal_size;
        if (fread(&rowscal_size, sizeof(int), 1, fp) != 1)
        {
            printf("Error: Failed to read rowscal size at level %d\n", lvl);
            fclose(fp);
            return -1;
        }
        current_level->rowscal->size[0] = rowscal_size;
        double *h_rowscal = (double *)malloc(rowscal_size * sizeof(double));
        if (fread(h_rowscal, sizeof(double), rowscal_size, fp) != rowscal_size)
        {
            printf("Error: Failed to read rowscal values at level %d\n", lvl);
            free(h_rowscal);
            fclose(fp);
            return -1;
        }
        hipMalloc((void **)&current_level->rowscal->data, rowscal_size * sizeof(double));
        hipMemcpy(current_level->rowscal->data, h_rowscal, rowscal_size * sizeof(double), hipMemcpyHostToDevice);
        free(h_rowscal);

        // 读取 colscal 向量
        int colscal_size;
        if (fread(&colscal_size, sizeof(int), 1, fp) != 1)
        {
            printf("Error: Failed to read colscal size at level %d\n", lvl);
            fclose(fp);
            return -1;
        }
        current_level->colscal->size[0] = colscal_size;
        double *h_colscal = (double *)malloc(colscal_size * sizeof(double));
        if (fread(h_colscal, sizeof(double), colscal_size, fp) != colscal_size)
        {
            printf("Error: Failed to read colscal values at level %d\n", lvl);
            free(h_colscal);
            fclose(fp);
            return -1;
        }
        hipMalloc((void **)&current_level->colscal->data, colscal_size * sizeof(double));
        hipMemcpy(current_level->colscal->data, h_colscal, colscal_size * sizeof(double), hipMemcpyHostToDevice);
        free(h_colscal);
    }

    fclose(fp);
    return 0;
}



void multilevel_pbilu()
{
	// printf("multilevel_pbilu is used\n");
	 
	PC_MILU *milu;
    PetscErrorCode ierr;
    // 分配内存并初始化自定义预处理器上下文
    ierr = PetscNew(&milu);
    info->milu_data = (void *)milu;
 // create a cusparse handle
    DS_PRECOND_MULTILEVEL *ds;
    info->DS_PRECOND = malloc(sizeof(DS_PRECOND_MULTILEVEL));
    memset(info->DS_PRECOND, 0, sizeof(DS_PRECOND_MULTILEVEL));
    ds = (DS_PRECOND_MULTILEVEL *)info->DS_PRECOND;

	CHECK_HIPSPARSE(hipsparseCreate(&ds->handle));
	hipblasCreate(&ds->handle_hipblas);

    info->multilevel_ginkgo = PETSC_TRUE;

	if (info->multilevel_ginkgo)
	{ // record the time
		{
			// char filename[256];

			printf("generate_filename_inv is called\n");
			// print the parameters
			// printf("A_pack.nr: %d\n", milu->A_pack.nr);
			// printf("param->droptol: %f\n", milu->param->droptol);
			// printf("param->condest: %f\n", milu->param->condest);
			// printf("ds->deltat: %f\n", ds->deltat);
            // milu->param = (DILUPACKparam *)MAlloc((size_t)sizeof(DILUPACKparam),
            //                               "DGNLilupackfactor:param");
			
            //milu->A_pack.nr = 98304; // overlap 0
            //milu->A_pack.nr = 117912; // overlap 1
            //milu->A_pack.nr = 139968; // overlap 2
            milu->A_pack.nr = 164616; // overlap 3


            // milu->param->droptol = 0.00125;
            // milu->param->condest = 1.0;
            ds->deltat = 16.0;
			// generate_filename_inv(filename, 117912, 0.00125, 1.0, 16.0);
			generate_filename_inv(filename, milu->A_pack.nr, 0.00125, 3.0, ds->deltat);
			printf("generate_filename_inv: %s is used\n", filename);
			
			char temp[100];			   // Make sure the buffer is large enough to hold the concatenated string
			strcpy(temp, "inv_data/"); // Copy "data/" into temp
			strcat(temp, filename);	   // Concatenate filename to "data/"
			strcpy(filename, temp);

			printf("ready to load M_pack_inv from cache: %s\n", filename);

			// 加载示例
			if (file_exists_inv(filename))
			{    
				milu->M_pack_inv = (emxArray_struct0_T *)malloc(sizeof(emxArray_struct0_T));
				if (loadM_pack_inv_binary(milu->M_pack_inv, filename) == 0)
				{
					printf("Successfully loaded M_pack_inv from cache\n");
				}
				else
				{
					printf("Error: Failed to load M_pack_inv from cache\n");
				}
			}
			else
			{
				printf("cache is not found,can not use multilevel_ginkgo\n");
			}
		}
		return;
	}
}

 
void PBILU_Factorization()
{
// 	multilevel_pbilu(); 
   
//    // show the M_pack_inv
     
// 	if(info->rank == 0)
// 	{   
// 		strcpy(result_file, "size32_result.txt");
// 		PC_MILU *milu = (PC_MILU *)info->milu_data;
// 		show_M_pack(milu->M_pack_inv, result_file);
// 	}

// fast solve load 

    int pn = PN;
    double dt = 16;
    double alpha = (dt/2) * (dt/2);
    int num_boundary = NUM_BOUNDARY;

    // Read input data
    size_t wrE_size = PN;
    size_t wrF_size = PN;
    size_t wrG_size = PN;
    size_t ds_size = N;
    size_t U_size = N * N;
    size_t V_size = N * N;
    size_t MEFG_size = num_boundary * num_boundary;
    size_t boundary_indices_size = num_boundary;
    size_t E_size = PN;
    size_t F_size = PN;
    size_t G_size = PN;


    double* ds = read_double_bin_file("ds.bin", ds_size);
    // print_double_array("ds", ds, ds_size);

    double* U = read_double_bin_file("U.bin", U_size);
    // print_double_array("U", U, U_size);

    double* V = read_double_bin_file("V.bin", V_size);
    // print_double_array("V", V, V_size);

    double* MEFG = read_double_bin_file("MEFG.bin", MEFG_size);
    // print_double_array("MEFG", MEFG, MEFG_size);

    int* boundary_indices = read_int_bin_file("boundary_indices.bin", boundary_indices_size);
    // print_int_array("boundary_indices", boundary_indices, boundary_indices_size);


    // Transpose U, V
    double* U_t = allocate_double_array(N * N);
    double* V_t = allocate_double_array(N * N);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            U_t[j * N + i] = U[i * N + j];
            V_t[j * N + i] = V[i * N + j];
        }
    }

    PreprocessedData *preprocessed_ptr;
    PetscErrorCode ierr;
    // 分配内存并初始化自定义预处理器上下文
    ierr = PetscNew(&preprocessed_ptr);
    info->preprocessed_ptr = (void *)preprocessed_ptr;

    // Preprocess static data    the sentence is memory leak
    preprocess_data(preprocessed_ptr, U, V, U_t, V_t, ds, MEFG, boundary_indices, N, alpha);

    // Cleanup preprocessed data
    // cleanup_preprocessed_data(preprocessed_ptr);

    free(ds);
    free(U);
    free(V);
    free(MEFG);
    free(boundary_indices);
    free(U_t);
    free(V_t);


}


// void PBILU_Factorization()
// {
// 	//1. petsc_pbilu
// 	if(info->petsc_pbilu)
// 	{
// 		if(info->bisai_precond)// we don't need to cal LU facatorization but we do need to compute InvL and InvU before precodnitioning 
// 		{
// 			DS_PRECOND_BISAI * ds_bisai=(DS_PRECOND_BISAI *)info->DS_PRECOND;
// 			if(ds_bisai->bisai_estpattern_CPU)
// 			{
// 				// When InvL and InvU are obtained, copy them from the device to host, csc->csr, and copy the csr formats back to device
// 				// all above operations are performed in compute_invL_InvU_GPU(); 
// 				compute_InvL_InvU_GPU();
// 			}
// 			else  // but when it comes to estimating sparsity patterns on GPUs, the operations are not included in a single function 
// 			{	
// 				// then we need to obtain the values for  InvL and InvU 
// 				estimate_bisai_numeric();
// 				// currently, InvL and InvU is expressed in CSC, we need a CSR version for Sparse Matrix-vector Multiplication
// 				cusparse_bsr2bsc(ds_bisai->dcsc_InvLColPtr, ds_bisai->dcsc_InvLRowVal, ds_bisai->dcsc_InvLBlkVal,
// 					ds_bisai->dInvLRowPtr, ds_bisai->dInvLColVal, ds_bisai->dInvLBlkVal,info->fact_n, ds_bisai->InvLnnz, info->bs);

// 				cusparse_bsr2bsc(ds_bisai->dcsc_InvUColPtr, ds_bisai->dcsc_InvURowVal, ds_bisai->dcsc_InvUBlkVal,
// 					ds_bisai->dInvURowPtr, ds_bisai->dInvUColVal, ds_bisai->dInvUBlkVal,info->fact_n, ds_bisai->InvUnnz, info->bs);
// 			}
// 			return;
// 		}
// 	} // do not need to to anything}
// 	//2. cusparse_pbilu
// 	if(info->cusparse_pbilu)
// 	{
// 		cusparse_bsrilu();
// 	}
// 	//3. asynchronous_pbilu
// 	if(info->asynchronous_pbilu)
// 	{
// 		asynchronous_pbilu();
// 	} 
	
// }
