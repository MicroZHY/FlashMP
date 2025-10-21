#ifndef __KSPSOLVE_GMRES_GPU__
#define __KSPSOLVE_GMRES_GPU__
#include <petscksp.h>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include "hipblas.h"
#include <hipsparse.h>
#include <mpi.h>

#include "MILUsolve_types.h"
#include "MILUsolve.h"
#include "../milu_inc/ilupack.h"
#ifdef real
#define real_saved real
#undef real
#endif
#include <rocblas.h>
#ifdef real_saved
#define real real_saved
#undef real_saved
#endif

#define CHECK_HIPSPARSE(func)                                              \
	{                                                                      \
		hipsparseStatus_t status = (func);                                 \
		if (status != HIPSPARSE_STATUS_SUCCESS)                            \
		{                                                                  \
			printf("HIPSPARSE API failed at line %d with error: %s (%d)\n",\
				   __LINE__, hipsparseStatusGetString(status), status);    \
			return;                                                        \
		}                                                                  \
	}

// 
//#define NVARS 3

#ifdef AMD_PLATFORM   // warp_size = 64 for DCU 
        #define WARP_SIZE 64
        #define WARPS_PER_BLOCK 2    // we usually use 128 threads in a thread_block      
#else
        #define WARP_SIZE  32
        #define WARPS_PER_BLOCK  4
#endif

#define THREE 	3
#define FOUR 	4
#define FIVE 	5
#define USE_ITS_SELF 0
#define SQUARE3   0

// the following macros are not used in current version 
#define THREADS_PER_BLOCK 256
#define THREADS_PER_BLOCK_AX 256
#define THREADS_PER_BLOCK_VX 256
#define THREADS_PER_BLOCK_DOT 128
#define BLOCKS_PER_GRID_DOT  128

//        FOR GPU_BILU_SWEEP_new
#define THREADS_PER_BLOCK_BILU  128
#define THREADS_PER_BLOCK_CSC_TO_CSR 256
#define THREADS_PER_BLOCK_INVDMU 128
#define THREADS_PER_BLOCK_INVUDIAG 128
#define SWEEP_NUM 100
//////////////////////////////////////////////////////

#ifdef __cplusplus
extern "C"
{
#endif


// Preprocessing structure to hold static GPU pointers
typedef struct {
    double* d_U;
    double* d_V;
    double* d_U_t;
    double* d_V_t;
    int* d_row_ptr;
    int* d_col_ind;
    double* d_val;
    double* d_MEFG;
    int* d_boundary_indices;
	rocblas_handle handle;
	int pn;
	int num_boundary;
	hipsparseHandle_t sparse_handle;
	hipsparseMatDescr_t descr;
	double* d_E3;
	double* d_F3;
	double* d_G3;
	double* d_E4;
	double* d_F4;
	double* d_G4;
	double* d_tEFG;
	double* d_boundary_values;
	double* d_corrected_values;
	double* d_tEFG_new;
	double* d_tE;
	double* d_tF;
	double* d_tG;
	double* d_extracted_x;
    // temporary arrays in  solve function
	double *d_Et, *d_Ft, *d_Gt, *d_Ep, *d_Fp, *d_Gp, *d_E1p, *d_F1p, *d_G1p, *d_E2, *d_F2, *d_G2;
	double *d_r, *d_c;

} PreprocessedData;

typedef struct
{
	// cusparse for upper and lower triangular solve 
	PetscInt	*dFactRowPtr;	
	PetscInt 	*dFactColVal;	
	PetscReal	*dFactBlkVal;	 // device
	PetscInt	*hFactRowPtr;
	PetscInt	*hFactColVal; 
	PetscReal	*hFactBlkVal;    // host
	// cusparse for block ilu factorization 
	hipsparseMatDescr_t descr_A;
	bsrilu02Info_t info_A;
	int pBufferSize_A;
	void * pBuffer;
	hipsparseSolvePolicy_t policy_A;
	hipsparseDirection_t dir_A;

	hipsparseStatus_t	cup_stat;
	hipsparseHandle_t	handle_L;
	hipsparseHandle_t	handle_U; 
	bsrsv2Info_t	info_L;
	bsrsv2Info_t	info_U;
	hipsparseMatDescr_t descr_L;
	hipsparseMatDescr_t descr_U;
	hipsparseSolvePolicy_t	policy_L;
	hipsparseSolvePolicy_t	policy_U;
	hipsparseDirection_t dir_L;
	hipsparseDirection_t dir_U;
	hipsparseOperation_t	trans_L;
	hipsparseOperation_t	trans_U;
	void *pLBuffer;
	void *pUBuffer;
	int pLBufferSize;
	int pUBufferSize;
	
	
}DS_PBILU_PRECOND_CUSPARSE; // data structure for point-block preconditioning using cusparse

typedef struct
{
	PetscInt 	sweep_num;
	PetscBool	asyn_use_exactilu_asfirst;  // in Newton's non-linear solver, use exact ilu in the first newton step 
	PetscBool	asyn_use_exactilu_called; // TRUE if the first newton step is already  
	PetscInt	*dFactRowPtr;	
	PetscInt 	*dFactColVal;	
	PetscReal	*dFactBlkVal;	
	PetscInt	*hFactRowPtr;
	PetscInt	*hFactColVal; 
	PetscReal	*hFactBlkVal;

	PetscInt	*dCSC_UColPtr;	
	PetscInt	*dCSC_URowVal;	
	PetscReal	*dCSC_UBlkVal;	
	PetscInt	*hCSC_UColPtr;  // the  host pointer is discarded in the newest version, because we have done CSR->CSC on GPU using cuSPARSE  
	PetscInt	*hCSC_URowVal;
	PetscReal	*hCSC_UBlkVal;

	PetscInt	*hcsc_to_csr_map;
	PetscInt 	*dcsc_to_csr_map;

	PetscInt 	*dUDiagRowPtr;	PetscInt 	*hUDiagRowPtr;
	PetscInt	*dUDiagColVal;	PetscInt	*hUDiagColVal;
	PetscReal	*dUDiagVal;	PetscReal	*hUDiagVal;
	PetscReal	*hExactLBlkVal; // use with the varialble asyn_use_exactilu_asfirst
	PetscReal	*hExactUBlkVal;
	// FOR CUSPARSE PRECOND
	PetscReal 	*tmp; // temporary array for preconditioning
	PetscReal	*dUInvDiagVal;  PetscReal	*hUInvDiagVal;
	PetscReal	*dUStarBlkVal;  PetscReal	*hUStarBlkVal;
	hipsparseStatus_t	cup_stat;
	hipsparseHandle_t	handle_L; 
	bsrsv2Info_t	info_L;
	bsrsv2Info_t	info_U;
	hipsparseMatDescr_t descr_L;
	hipsparseMatDescr_t descr_U;
	hipsparseSolvePolicy_t	policy_L;
	hipsparseSolvePolicy_t	policy_U;
	hipsparseDirection_t dir_L;
	hipsparseDirection_t dir_U;
	hipsparseOperation_t	trans_L;
	hipsparseOperation_t	trans_U;
	void *pLBuffer;
	void *pUBuffer;
	int pLBufferSize;
	int pUBufferSize;
	
}DS_PBILU_ASYN;  // data structure for asynchronous point-block ILU factorization

typedef struct
{
	// CSR format for InvL 
	PetscInt        *hInvLRowPtr;           
	PetscInt        *hInvLColVal;           
	PetscReal       *hInvLBlkVal;           
	PetscInt 	*dInvLRowPtr;         
	PetscReal 	*dInvLBlkVal;        
	PetscInt 	*dInvLColVal;         
	// CSC format for InvL
	PetscInt 	*hcsc_InvLColPtr;      
	PetscInt 	*hcsc_InvLRowVal;       
	PetscReal 	*hcsc_InvLBlkVal;      
	PetscInt        *dcsc_InvLColPtr;
	PetscInt        *dcsc_InvLRowVal;
	PetscReal       *dcsc_InvLBlkVal;
	// CSR format for InvU
	PetscInt        *hInvURowPtr;          
	PetscInt        *hInvUColVal;           
	PetscReal       *hInvUBlkVal;            
	PetscInt 	*dInvURowPtr;          
	PetscInt 	*dInvUColVal;          
	PetscReal 	*dInvUBlkVal;         
	// CSC format for InvU 
	PetscInt 	*hcsc_InvUColPtr;       
	PetscInt 	*hcsc_InvURowVal;       
	PetscReal 	*hcsc_InvUBlkVal;      
	PetscInt        *dcsc_InvUColPtr; 
	PetscInt        *dcsc_InvURowVal;
	PetscReal       *dcsc_InvUBlkVal;
	//
	PetscInt        InvLnnz;
	PetscInt        InvUnnz;
	PetscReal       *hLRhs;                                 
	PetscReal       *hURhs;                                
	PetscReal       *dLRhs;
	PetscReal       *dURhs;

	PetscBool 	bisai_estpattern_CPU; // FALSE by default, that means we estimate sparisty pattern for InvL and InvU on GPU 
	PetscInt 	bisai_dense_level;
	hipsparseStatus_t	cup_stat;    // not used right now
	hipsparseHandle_t	handle;   // not used right now
}DS_PRECOND_BISAI;



	typedef struct
	{
		double *a;
		int *ia;
		int *ja;
		int nnz;
		int nr;
		int bs;
		hipsparseHandle_t handle;
		hipblasHandle_t handle_hipblas;
		PetscReal deltat;
		PetscReal condest;
		PetscInt ngridx;
		PetscReal droptol;
		// void *gpu_M_pack; // GPU版本的M_pack
	} DS_PRECOND_MULTILEVEL;


typedef struct
{
// for iterative precond
	PetscInt num_iterations;
        PetscInt *hUdiagRowPtr;
	PetscInt *hUdiagColVal;
	PetscReal *hUdiagBlkVal; 
	PetscInt *dUdiagRowPtr;
	PetscInt *dUdiagColVal;
	PetscReal *dUdiagBlkVal;
	PetscReal *diter_tmp1;
	PetscReal *diter_tmp2;
	// handles for SPMV
	hipsparseHandle_t	cusparse_handle;
	hipsparseStatus_t	cusparse_stat;
	hipsparseMatDescr_t descr;
	hipsparseDirection_t dir;
	hipsparseOperation_t	trans;
}DS_PRECOND_ITERATIVE;

typedef struct
{
	// ILU type and preconditioning type
	PetscBool petsc_pbilu; PetscBool cusparse_pbilu; PetscBool asynchronous_pbilu;
	PetscBool petsc_precond; PetscBool iterative_precond; PetscBool bisai_precond; PetscBool cusparse_precond;
	PetscBool use_asm;
		PetscBool multilevel_cusparse;
		PetscBool multilevel_ginkgo;
		PetscBool coarse_inverse;
	//
	PetscInt	rank;
	PetscInt 	idev;// one host node may consist of more than one GPUs
	PetscInt	init_gpu_called;
	PetscInt	bs;	
	PetscInt	main_nrows, off_nrows; //A->A  A->B  seq A->A is square but B may not
	PetscInt	main_ncols, off_ncols; 
	PetscInt	main_nnz,   off_nnz;
//	PetscInt	fact_nrows;  // seq: square
	// GPU pointers for A and B(off diag)
	PetscInt	*dMainRowPtr; 	PetscInt	*hMainRowPtr;
	PetscInt	*dMainColVal;	PetscInt	*hMainColVal;
	PetscReal	*dMainBlkVal;	PetscReal	*hMainBlkVal;
	// Petsc stores point-block matrix in column major format
	PetscInt	*dOffRowPtr;	PetscInt	*hOffRowPtr;
	PetscInt	*dOffColVal;	PetscInt	*hOffColVal;
	PetscReal	*dOffBlkVal; 	PetscReal	*hOffBlkVal;

////////////////////////////////BEGIN of ILU(k) factorization + preconditioning/////////////////////////////
// GENERAL STRUCTURE
	// integer information
	PetscInt 	fact_n;
	PetscInt 	Factnnz;
	PetscInt 	Lnnz;
	PetscInt	Unnz;
	//GPU pointers for L and U factors in CSR format, and U in CSC format
	//L in CSR
	PetscInt	*dLRowPtr;      PetscInt 	*hLRowPtr;
	PetscInt	*dLColVal;      PetscInt	*hLColVal;
	PetscReal	*dLBlkVal;	PetscReal	*hLBlkVal;
	// U in CSR
	PetscInt	*dURowPtr;	PetscInt	*hURowPtr;
	PetscInt	*dUColVal;	PetscInt	*hUColVal;
	PetscReal	*dUBlkVal;	PetscReal	*hUBlkVal;
// END GENERAL STRUCUTRE
//      
	void * DS_PBILU; // specific data structure (DS) for the method of Point-block ILU factorization  

	void * DS_PRECOND; // specific data structure (DS) for the preconditioning method

//////////////////////////////END of ILU(k) factorization+ preconditioning///////////////////////////////
	// righ hand side and solution
	PetscInt	vsz;	
	Vec		rhs;	PetscReal	*drhs; 	
	Vec		sol;	PetscReal	*dsol;

	// MPI communication for SPMV (Sparse matrix-vector multiplication)
	VecScatter 	Mvctx;
	PetscInt	snp;      PetscInt   rnp;   // number of sending or reciving processors
	PetscMPIInt	*sprocs;  PetscMPIInt   *rprocs; // sending and receiving processors array
	PetscInt	*sstarts; PetscInt   *rstarts;
	PetscInt	*sindices; PetscInt   *rindices;
	MPI_Request	*swaits;  MPI_Request *rwaits;
	PetscInt	*dsindices; PetscInt indices_len;
        PetscReal	*dsend_buf;PetscInt buf_len;
	PetscReal	*hsend_buf;
	Vec 		lvec;			// used for matrix vector multiplication
	PetscReal	*dlvec;
	PetscReal	*hlvec;
	PetscInt	lvsz;

	//MPI communication for ASM preconditioner M^(-1) v = z 
	VecScatter 	asm_restriction;
	PetscInt	asm_snp;   	PetscInt asm_rnp; // number of sending or recving processors
	PetscMPIInt	*asm_sprocs; 	PetscMPIInt *asm_rprocs;
	PetscInt	*asm_sstarts;	PetscInt *asm_rstarts;
	PetscInt	*asm_sindices;	PetscInt *asm_rindices;
	MPI_Request	*asm_swaits;	MPI_Request *asm_rwaits;
	PetscInt	*asm_self_sindices; 
	PetscInt	*asm_self_rindices;	

	PetscReal 	*asm_send_buf;
	PetscReal	*asm_recv_buf;
	PetscReal	*asm_lx;
	PetscReal	*asm_ly;
	PetscReal	*asm_vecx;
	PetscReal	*asm_vecy;

	PetscInt	*asm_dsindices; PetscInt asm_sindices_len;
	PetscInt	*asm_drindices; PetscInt asm_rindices_len;
	PetscInt	*asm_self_dsindices; PetscInt asm_self_sindices_len;
	PetscInt	*asm_self_drindices; PetscInt asm_self_rindices_len;
	PetscReal	*asm_dsend_buf; PetscInt asm_sendbuf_len;
	PetscReal	*asm_drecv_buf; PetscInt asm_recvbuf_len;	
	PetscReal	*asm_dlx;
	PetscReal	*asm_dly;
	PetscReal	*asm_dltmp;
	PetscInt	asm_lxsz;
	
	


	// memory for pure GPU GMRES
	PetscReal	*dlhh;
	PetscReal 	*ddotres[2];
	PetscReal	*hdotres[2];
	// gmres->vecs
	Vec *vecs;
	PetscReal **dvv;
	PetscInt  vvdim;

	// handles and streams for BSR SPMV and vector operations   
	hipblasHandle_t 		cublas_handle; 
	hipsparseHandle_t	cusparse_handle;
	hipblasStatus_t		cublas_stat;
	hipsparseStatus_t	cusparse_stat;
	hipStream_t			cu_stream;    // used for cusparse and cublas computing
	hipStream_t			mem_stream;   // used for memory copy
        hipStream_t            mystream[3];
	hipsparseMatDescr_t descr;
	hipsparseDirection_t dir;
	hipsparseOperation_t	trans;
       void *milu_data;
	PetscReal *dv_tmp;  // used for temporary memory in block-jacobi preconditioning
	void *preprocessed_ptr;

}GMRES_INFO;


	typedef struct
	{
		IS row, col;
		MatFactorInfo info;
		emxArray_struct0_T *M_pack;		// 保存分解后的矩阵
		emxArray_struct0_T *M_pack_gpu; // 保存分解后的gpu矩阵
		emxArray_struct0_T *M_pack_inv; // 保存分解后的gpu矩阵的逆矩阵
		DAMGlevelmat *PRE;
		DILUPACKparam *param;
		Dmat A_pack;
	} PC_MILU;



 void CreateGMRES_INFO();
 void InitialGMRES_INFO_GPU();
	void allocate_gpumem_cusparse();
	void allocate_gpumem_petsc();
	void allocate_gpumem_asynchronous();



 void DestroyGMRES_INFO();
 PetscErrorCode InitialGMRES_INFO_CPU( KSP ksp);
 PetscErrorCode KSPSolve_GMRES_GPU(KSP ksp);
 #ifdef __cplusplus
 extern "C"
 {	
 PetscErrorCode KSPSolve_BiCGSTAB_GPU(KSP ksp);
 PetscErrorCode KSPSolve_CG_GPU(KSP ksp);
 PetscErrorCode KSPSolve_BiCG_GPU(KSP ksp);
 }
 #endif
 PetscErrorCode KSPGMRESUpdateHessenberg_MARK_GPU(KSP ksp,PetscInt it,PetscBool hapend,PetscReal *res);

/////////////////////////pre_ilu ///////////////////
void pre_ilu(KSP ksp); // 
// we have three types of ILU: petsc_pbilu cusparse_pbilu asynchronous_pbilu 
// pre_ilu will call ONLY one of the three types;
	// 1. for petsc_pbilu:  
	void pre_ilu_petsc(PetscInt *facti, PetscInt *factj, PetscInt *factdiag, PetscReal *facta, PetscInt n);
//      1.1 petsc_pbilu + petsc_precond
// 	1.2 petsc_pbilu + bisai_precond
	void pre_ilu_petsc_for_bisai_precond(PetscInt *facti, PetscInt *factj, PetscInt *factdiag, PetscReal *facta, PetscInt n);
		void cal_InvLUSparsityPattern(PetscInt n, PetscInt bs);
		void CSR2CSC(PetscInt *rowptr,  PetscInt *colval,       PetscReal *csrblkval,
	     		PetscInt *colptr,      PetscInt *rowval,       PetscReal *cscblkval,
             		PetscInt n,            PetscInt bs,  PetscInt nnz);
//	1.3 petsc_pbilu + iterative_precond
	void pre_ilu_petsc_for_iterative_precond(PetscInt *facti, PetscInt *factj, PetscInt *factdiag, PetscReal *facta, PetscInt n);



	// 2. for cusparse_pbilu
	void pre_ilu_cusparse( PetscInt *Ai,	PetscInt *Aj, 	PetscInt *Adiag, PetscReal *Aa, 
			PetscInt *facti,PetscInt *factj,PetscInt *factdiag, PetscInt n);



	// 3. for asynchronous_pbilu
	void pre_ilu_asynchronous(PetscInt *Ai, PetscInt *Aj, PetscInt *Adiag, PetscReal *Aa,
			PetscInt *facti, PetscInt *factj, PetscInt *factdiag, PetscReal *facta, PetscInt n);
		void estimate_bisai_pattern(PetscInt *csrRowPtr,PetscInt *csrColInd, PetscInt nnz,// input L or U
		PetscInt **pInvCsrRowPtr, PetscInt **pInvCsrColInd, PetscReal **pInvCsrBlkVal,PetscInt *pnnz); // output InvL or InvU's pattern
		void estimate_bisai_numeric(); // compute InvL and InvU numerically 
		void cusparse_bsr2bsc(PetscInt *inRowPtr, PetscInt *inColInd, PetscReal *inVal,
		      PetscInt *outColPtr, PetscInt *outRowInd, PetscReal *outVal, PetscInt n, PetscInt nnz, PetscInt bs);
///////////////////// pre_ilu//////////////////////


//////////////////////PBILU_Factorization//////////
void PBILU_Factorization();
	void cusparse_bsrilu(); //
	void petsc_pbilu();
	void asynchronous_pbilu();
/////////////////////PBILU_Factorization/////////

////////////////////PB_Preconditioning////////////
void PB_Preconditioning(PC pc, Vec px, Vec py, // we need to do preconditioning on CPUs 
			PetscReal *vecx, PetscReal *vecy, // the corresponding memories on GPUs 
			PetscReal *vectmp, //optional for some algorithms
			PetscInt vsz);
	void petsc_precond(PC pc, Vec px, Vec py, PetscReal *vecx, PetscReal *vecy, PetscReal *vectmp, PetscInt vsz);
	void cusparse_pbilu_cusparse_precond(PetscReal *in_x, PetscReal *out_y, PetscReal *mid_tmp, PetscInt in_vsz);
	void asynchronous_pbilu_cusparse_precond(PetscReal *in_x, PetscReal *out_y, PetscReal *mid_tmp, PetscInt in_vsz);
	void bisai_precond(PetscReal *in_x, PetscReal *out_y, PetscReal *mid_tmp, PetscInt in_vsz);
	void iterative_precond(PetscReal *in_x, PetscReal *out_y, PetscReal *mid_tmp, PetscInt in_vsz);
//////////////////PB_Preconditioning//////////

void compute_InvL_InvU_GPU();



//The following are the API called from the user who use our package 

// void cusparse_btrisv1(int it);
// void cusparse_btrisv2();
//void precondition(int *L_row_ptr, int *L_col_val, double *L_blk_val,
//		  int *U_row_ptr, int *U_col_val, double *U_blk_val,
//		  int size,
//				  double *x,
//				  double *b,
//				  double *y); // y is used for temp memory

// for ilu and triangluar from cusparse library


#ifdef __cplusplus
}
#endif
#endif
