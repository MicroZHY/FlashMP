#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <../src/mat/impls/baij/mpi/mpibaij.h>
#include <../src/mat/impls/baij/seq/baij.h>
#include <../src/ksp/pc/impls/factor/ilu/ilu.h>
#include "KSPSolve_GMRES_GPU.h"

// we have a few combinations of the methods for point-block ILU factorizations
//                           and the methods for point-block preconditioning
// EACH configuration requires different memory sizes and data structures
//                           		pre_ilu_petsc_for_petsc_precond  (acctually do nothing)
//                              	pre_ilu_petsc_for_bisai_precond  
//           pre_ilu_petsc      	pre_ilu_petsc_for_iterative_precond
//
// 
// pre_ilu   pre_ilu_cusparse ONLY FOR  cusparse_precond
//
//
//
//           pre_ilu_asynchronous       pre_ilu_asynchronous_for bisai_precond
//                                      pre_ilu_asynchronous_for_iterative_precond


extern GMRES_INFO *info;


// 保存 Mat_SeqBAIJ 矩阵到文本文件 
// PetscErrorCode SaveMatrixToText(Mat mat, const char *filename, PetscInt rank)
// {
//     PetscErrorCode ierr;
   
//     PetscViewer viewer;
//     ierr = PetscViewerASCIIOpen(PETSC_COMM_SELF, filename, &viewer); CHKERRQ(ierr);
//     ierr = PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_COMMON); CHKERRQ(ierr); // 使用 COMMON 格式，简洁输出
//     ierr = MatView(mat, viewer); CHKERRQ(ierr);
//     ierr = PetscViewerPopFormat(viewer); CHKERRQ(ierr);
//     ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);

//     printf("进程 %d: 矩阵已保存到 %s\n", rank, filename);
//     return 0;
// }


// PetscErrorCode SaveMatrixToText(Mat mat, const char *filename, PetscInt rank)
// {
//     PetscErrorCode ierr;
//     PetscViewer viewer;

//     // 使用 PETSC_VIEWER_ASCII 并明确设置 Matrix Market 格式
//     ierr = PetscViewerCreate(PETSC_COMM_SELF, &viewer); CHKERRQ(ierr);
//     ierr = PetscViewerSetType(viewer, PETSCVIEWERASCII); CHKERRQ(ierr);
//     ierr = PetscViewerFileSetMode(viewer, FILE_MODE_WRITE); CHKERRQ(ierr);
//     ierr = PetscViewerFileSetName(viewer, filename); CHKERRQ(ierr);
//     ierr = PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_MATRIXMARKET); CHKERRQ(ierr);

//     // 保存矩阵
//     ierr = MatView(mat, viewer); CHKERRQ(ierr);

//     // 清理
//     ierr = PetscViewerPopFormat(viewer); CHKERRQ(ierr);
//     ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);

//     printf("进程 %d: 矩阵已保存到 %s（Matrix Market 格式）\n", rank, filename);
//     return 0;
// }


// PetscErrorCode SaveMatrixToText(Mat mat, const char *filename, PetscInt rank)
// {
//     PetscErrorCode ierr;
//     PetscViewer viewer;
//     ierr = PetscViewerBinaryOpen(PETSC_COMM_SELF, filename, FILE_MODE_WRITE, &viewer); CHKERRQ(ierr);
//     ierr = MatView(mat, viewer); CHKERRQ(ierr);
//     ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);
//     printf("进程 %d: 矩阵已保存到 %s（二进制格式）\n", rank, filename);
//     return 0;
// }

PetscErrorCode SaveMatrixToText(Mat mat, const char *filename, PetscInt rank)
{
    PetscErrorCode ierr;
    PetscViewer viewer;
    Mat converted_mat;
    PetscInt rows, cols, nz;
    PetscBool assembled;
    const char *mattype;

    // 验证矩阵
    ierr = MatGetSize(mat, &rows, &cols); CHKERRQ(ierr);
    
    ierr = MatAssembled(mat, &assembled); CHKERRQ(ierr);
    PetscInt bs;
    ierr = MatGetBlockSize(mat, &bs); CHKERRQ(ierr);
    

    // 确保组装
    if (!assembled) {
        printf("进程 %d: 矩阵未组装，尝试组装\n", rank);
        ierr = MatAssemblyBegin(mat, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
        ierr = MatAssemblyEnd(mat, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    }

    // 转换为 MATSEQAIJ
    ierr = MatConvert(mat, MATSEQAIJ, MAT_INITIAL_MATRIX, &converted_mat); CHKERRQ(ierr);
    printf("进程 %d: 转换为 MATSEQAIJ\n", rank);

    // 创建 viewer
    ierr = PetscViewerASCIIOpen(PETSC_COMM_SELF, filename, &viewer); CHKERRQ(ierr);
    
    // 设置 Matrix Market
    ierr = PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_MATRIXMARKET); CHKERRQ(ierr);
    
    // 保存
    printf("进程 %d: 保存到 %s（Matrix Market 格式）\n", rank, filename);
    ierr = MatView(converted_mat, viewer); CHKERRQ(ierr);
    
    // 清理
    ierr = PetscViewerPopFormat(viewer); CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);
    ierr = MatDestroy(&converted_mat); CHKERRQ(ierr);

    printf("进程 %d: 完成保存到 %s\n", rank, filename);
    return 0;
}

 

void pre_ilu(KSP ksp)
{
	KSP *subksp;
	PC pc, subpc;
	PetscInt nlocal, first;
	KSPGetPC(ksp,&pc);
	//call  PCASMGetSubKSP if ASM preconditioner is used.
	//Generally, we use one block for one processor
	PCType pctype;
	info->use_asm=PETSC_FALSE;
	PCGetType(pc,&pctype);
	PetscStrcmp(pctype,PCASM,&info->use_asm);
    // show the type of preconditioner
    // printf("pctype = %s\n",pctype);

	PetscStrcmp(pctype,"geoasm",&info->use_asm);
	if(info->use_asm)
	{   
		// printf("use_asm=PETSC_TRUE\n");
		PCASMGetSubKSP(pc,&nlocal,&first,&subksp);
	}
	else
	{
		// printf("use_asm=PETSC_FALSE\n");
		PCBJacobiGetSubKSP(pc,&nlocal,&first,&subksp);
	}
	KSPGetPC(subksp[0],&subpc);
	// access the symbolic L and U factors for ILU(k)
	PC_ILU *ilu=(PC_ILU *)subpc->data;
	Mat factmat=((PC_Factor *)ilu)->fact;
	Mat_SeqBAIJ	*fact=(Mat_SeqBAIJ *)factmat->data;
	// access the operating matrix corresponding to each sub-domain
	Mat pmat=subpc->pmat;
	Mat_SeqBAIJ	*A=(Mat_SeqBAIJ *)pmat->data;

	PetscInt n=A->mbs;
	PetscInt bs=info->bs;
	PetscInt nz = A->nz; 

	// printf("n = %d, bs = %d, nz = %d in pre_ilu\n",n,bs,nz);
    // char filename[256];
    // snprintf(filename, sizeof(filename), "/public/home/suyuexinghen/zhanghy/FDTD-DMDA/fdtd/app/rank/matrix_A_rank_%d.mtx", info->rank);
    // SaveMatrixToText(pmat, filename, info->rank);

	//2021.9.7 update
	// 3 methods for point-block ILU factorization: petsc_pbilu by default
	info->cusparse_pbilu=PETSC_FALSE; info->asynchronous_pbilu=PETSC_FALSE; info->petsc_pbilu=PETSC_TRUE;
	// 4 preconditioning methods petsc_precond by default 
	info->cusparse_precond=PETSC_FALSE; info->iterative_precond=PETSC_FALSE; info->bisai_precond=PETSC_FALSE; info->petsc_precond=PETSC_TRUE; 
	PetscOptionsGetBool(NULL,NULL,"-cusparse_pbilu",&info->cusparse_pbilu,NULL);
	PetscOptionsGetBool(NULL,NULL,"-asynchronous_pbilu",&info->asynchronous_pbilu,NULL);

	PetscOptionsGetBool(NULL,NULL,"-cusparse_precond",&info->cusparse_precond,NULL);
	PetscOptionsGetBool(NULL,NULL,"-iterative_precond",&info->iterative_precond,NULL);
	PetscOptionsGetBool(NULL,NULL,"-bisai_precond",&info->bisai_precond,NULL);
	// make sure there is only 1 options true from users
	 

	// if(info->rank==0){
	// 	// we do some printing for checking
	// 	printf("petsc_pbilu=%d,cusparse_pbilu=%d,asyc_pbilu=%d,petsc_precond=%d,cusparse_precond=%d,iterative_precond=%d,bisai_precond=%d,use_asm=%d\n",
	// 	info->petsc_pbilu,info->cusparse_pbilu,info->asynchronous_pbilu,
	// 	info->petsc_precond,info->cusparse_precond,info->iterative_precond,info->bisai_precond,info->use_asm);	
	// }

 
}




void pre_ilu_petsc( PetscInt *facti, PetscInt *factj,PetscInt *factdiag,PetscReal *facta,PetscInt n)
{
	if(info->petsc_precond)
	{
		//we donot need to do any work here, because the point-block ilu factorization 
		// has been performed by petsc, already 
		return;
	}
	if(info->bisai_precond)
	{
		pre_ilu_petsc_for_bisai_precond(facti,factj,factdiag,facta,n);
	}
	if(info->iterative_precond)
	{
		pre_ilu_petsc_for_iterative_precond(facti,factj,factdiag,facta,n);	
	}
}

void pre_ilu_petsc_for_bisai_precond(PetscInt *facti, PetscInt *factj, PetscInt *factdiag, PetscReal *facta, PetscInt n)
{
	PetscInt irow,itmp,colcnt,counter;
	PetscInt idx_st,idx_ed,idx;
	PetscInt bs=info->bs; 
	DS_PRECOND_BISAI *ds = (DS_PRECOND_BISAI *)info->DS_PRECOND;	

	// Step1:copy L and U in BCSR from PETSc (fact), every non-linear (Newton )step
	for(irow=0;irow<=n;irow++)
	{
		info->hLRowPtr[irow]=facti[irow]+irow;	
	}
	// set hColVal:
	colcnt=0;
	for(irow=0;irow<n;irow++)
	{
		idx_st=facti[irow];
		idx_ed=facti[irow+1];
		for(idx=idx_st;idx<idx_ed;idx++)
		{
			info->hLColVal[colcnt]=factj[idx];
			// copy fact->a values: column-major within block
			for(itmp=0;itmp<bs*bs;itmp++){info->hLBlkVal[colcnt*bs*bs+itmp]=facta[idx*bs*bs+itmp];}
			colcnt++;
		}
		// add one more element: diagonal element
		info->hLColVal[colcnt]=irow;
		// set values to identities
		for(itmp=0;itmp<bs;itmp++){info->hLBlkVal[colcnt*bs*bs+itmp*bs+itmp]=1.0;}
		colcnt++;
	}

	if(!info->rank){printf("\n");}
	//U: 
	//NOTE: upper triangular matrix contains diagonal elements, but the order PETSc stores
	// upper tri-matrix for each row in factmat is like non-diag0, non-diag1,...., diag_element
	counter=0;
	for(irow=0;irow<n;irow++)
	{	
		idx_st=factdiag[irow+1]+1;
		idx_ed=factdiag[irow];  // it is the diagonal element
		info->hURowPtr[irow]=idx_ed-idx_st+1; // temporarily stores the number of element each row
		// handle diagonal element
		info->hUColVal[counter]=factj[idx_ed];
		for(itmp=0;itmp<bs*bs;itmp++)
		{
			info->hUBlkVal[counter*bs*bs+itmp]=facta[idx_ed*bs*bs+itmp]; // the diagonal is inversed 
		}
		counter++;
		// so we don't need to consider diag element each row 
		for(idx=idx_st;idx<idx_ed;idx++)
		{
			// copy ft in fact to hUColVal
			info->hUColVal[counter]=factj[idx];
			// copy or not copy double values depends
			for(itmp=0;itmp<bs*bs;itmp++)
			{
				info->hUBlkVal[counter*bs*bs+itmp]=facta[idx*bs*bs+itmp];
			}
			counter++;
		}
	}
	// until now, we have filled hURowPtr, hUColVal, and hUBlkVal, but hURowPtr stores 
	// the number of non-zero elements for each row, we just need a transformation
	// ...................................
	for(irow=n-1;irow>=0;irow--)
	{
		info->hURowPtr[irow+1]=info->hURowPtr[irow];
	}
	info->hURowPtr[0]=0;
	for(irow=1;irow<=n;irow++)
	{
		info->hURowPtr[irow]=info->hURowPtr[irow]+info->hURowPtr[irow-1];
	}



	// the following is the estimation of sparisity patterns on CPU (NOT on GPU)
	ds->bisai_dense_level = 2;
	PetscBool is_bisai_dense_level= PETSC_FALSE;
	PetscBool is_bisai_estpattern_CPU=PETSC_FALSE;// default: estimate sparsity on GPU,
	PetscOptionsGetInt(NULL,NULL,"-bisai_dense_level",&ds->bisai_dense_level,&is_bisai_dense_level);
	PetscOptionsGetBool(NULL,NULL,"-bisai_estpattern_CPU",&ds->bisai_estpattern_CPU,&is_bisai_estpattern_CPU);

	if(ds->bisai_estpattern_CPU)
	{	
		struct timeval tstart, tend;
		double pre_t;
		gettimeofday(&tstart,NULL);
		//Step2: estimate the non-zero pattern for invL and invU, we need to do it only once 
		if(!info->init_gpu_called)
		{
			cal_InvLUSparsityPattern(n,bs);
		}

		
		PetscInt icol;

		//Step3: csr -> csc, we already got L ,U, InvL, InvU in CSR
		// csr L -> csc L    csr U -> csc U    csr invL -> csc invL    csr invU -> csc invU
		// L:    CSR								CSC
		//  PetscInt  *hLRowPtr;			PetscInt  *hcsc_LColPtr
		//	PetscInt  *hLColVal;			PetscInt  *hcsc_LRowVal;
		//	PetscReal *hLBlkVal;			PetscReal *hcsc_LBlkVal;
		//
		// U:     CSR								CSC
		//  PetscInt  *hURowPtr;			PetscInt  *hcsc_UColPtr;
		//  PetscInt  *hUColVal;			PetscInt  *hcsc_URowVal;
		//  PetscReal *hUBlkVal;			PetscReal *hcsc_UBlkVal;
		
		// invL   CSR								CSC
		//  PetscInt  *hInvLRowPtr;			PetscInt  *hcsc_InvLColPtr;
		//  PetscInt  *hInvLColVal;			PetscInt  *hcsc_InvLRowVal;
		//  PetscReal *hInvLBlkVal;			PetscReal *hcsc_InvLBlkVal;
		//
		//  InvU     CSR							CSC
		//  PetscInt  *hInvURowPtr;			PetscInt  *hcsc_InvUColPtr;
		//  PetscInt  *hInvUColVal;			PetscInt  *hcsc_InvURowVal;
		//  PetscReal *hInvUBlkVal;			PetscReal *hcsc_InvUBlkVal;
		// allocate memory
		// CSC for L 
		//struct timeval csr2csc_tstart, csr2csc_tend;
		//double csr2csc_t;
		//gettimeofday(&csr2csc_tstart,NULL);
		// why do we need CSC format for L and U
		// CSC for L
		//if(!info->hcsc_LColPtr){info->hcsc_LColPtr=(PetscInt *)malloc(sizeof(PetscInt)*(n+1));}
		//if(!info->hcsc_LRowVal){info->hcsc_LRowVal=(PetscInt *)malloc(sizeof(PetscInt)*Lnnz);}
		//if(!info->hcsc_LBlkVal){info->hcsc_LBlkVal=(PetscReal *)malloc(sizeof(PetscReal)*Lnnz*bs*bs);}
		//CSC for U
		//if(!info->hcsc_UColPtr){info->hcsc_UColPtr=(PetscInt *)malloc(sizeof(PetscInt)*(n+1));}
		//if(!info->hcsc_URowVal){info->hcsc_URowVal=(PetscInt *)malloc(sizeof(PetscInt)*Unnz);}
		//if(!info->hcsc_UBlkVal){info->hcsc_UBlkVal=(PetscReal *)malloc(sizeof(PetscReal)*Unnz*bs*bs);}
		// CSC for InvL
		if(!ds->hcsc_InvLColPtr){ds->hcsc_InvLColPtr=(PetscInt *)malloc(sizeof(PetscInt)*(n+1));}
		if(!ds->hcsc_InvLRowVal){ds->hcsc_InvLRowVal=(PetscInt *)malloc(sizeof(PetscInt)*ds->InvLnnz);}
		if(!ds->hcsc_InvLBlkVal){ds->hcsc_InvLBlkVal=(PetscReal *)malloc(sizeof(PetscReal)*ds->InvLnnz*bs*bs);}
		if(ds->hcsc_InvLBlkVal){memset(ds->hcsc_InvLBlkVal,0,sizeof(PetscReal)*ds->InvLnnz*bs*bs);}
		// CSC for InvU 
		if(!ds->hcsc_InvUColPtr){ds->hcsc_InvUColPtr=(PetscInt *)malloc(sizeof(PetscInt)*(n+1));}
		if(!ds->hcsc_InvURowVal){ds->hcsc_InvURowVal=(PetscInt *)malloc(sizeof(PetscInt)*ds->InvUnnz);}
		if(!ds->hcsc_InvUBlkVal){ds->hcsc_InvUBlkVal=(PetscReal *)malloc(sizeof(PetscReal)*ds->InvUnnz*bs*bs);}
		if(ds->hcsc_InvUBlkVal){memset(ds->hcsc_InvUBlkVal,0,sizeof(PetscReal)*ds->InvUnnz*bs*bs);}

		if(!info->init_gpu_called)
		{
			CSR2CSC(ds->hInvLRowPtr,		ds->hInvLColVal,		ds->hInvLBlkVal,
					ds->hcsc_InvLColPtr,	ds->hcsc_InvLRowVal,	ds->hcsc_InvLBlkVal,n,bs,ds->InvLnnz);
			CSR2CSC(ds->hInvURowPtr,		ds->hInvUColVal,		ds->hInvUBlkVal,
					ds->hcsc_InvUColPtr,	ds->hcsc_InvURowVal,	ds->hcsc_InvUBlkVal,n,bs, ds->InvUnnz);
		}
		//debug
		if(0)
		{
			printf("debug for info->hcsc_L:");
			for(icol=0;icol<n;icol++)
			{
				//idx_st=InvLk_data->i[irow];
				//idx_ed=InvLk_data->i[irow+1];
				idx_st=ds->hcsc_InvLColPtr[icol];
				idx_ed=ds->hcsc_InvLColPtr[icol+1];
				printf("[%d]:",icol);
				for(idx=idx_st;idx<idx_ed;idx++)
				{
					printf("%d,",ds->hcsc_InvLRowVal[idx]);
				}
			}	
			printf("\n");
		}		

		 //S4.3 we create the right hand matrix
		PetscReal *rhs=NULL;
		PetscInt stidx,edidx;
		if(!ds->hLRhs){ds->hLRhs=(PetscReal *)malloc(sizeof(PetscReal)*ds->InvLnnz*bs*bs);}
		if(ds->hLRhs){memset(ds->hLRhs,0,sizeof(PetscReal)*ds->InvLnnz*bs*bs);}
		rhs=ds->hLRhs;
		for(icol=0;icol<n;icol++)
		{
			stidx=ds->hcsc_InvLColPtr[icol];
			for(itmp=0;itmp<bs;itmp++){rhs[stidx*bs*bs+itmp*bs+itmp]=1.0;}
		}
		if(!ds->hURhs){ds->hURhs=(PetscReal *)malloc(sizeof(PetscReal)*ds->InvUnnz*bs*bs);}
		if(ds->hURhs){memset(ds->hURhs,0,sizeof(PetscReal)*ds->InvUnnz*bs*bs);}
		rhs=ds->hURhs;
		for(icol=0;icol<n;icol++)
		{
			edidx=ds->hcsc_InvUColPtr[icol+1]-1;
			for(itmp=0;itmp<bs;itmp++){rhs[edidx*bs*bs+itmp*bs+itmp]=1.0;}
		}


		gettimeofday(&tend,NULL);
    		pre_t=((double) ((tend.tv_sec*1000000.0 + tend.tv_usec)-(tstart.tv_sec*1000000.0+tstart.tv_usec)))/1000.0;
		printf("the time elapse for pre_ISAI on CPU is:%12.8lf\n",pre_t);
	
	
	}

	
	
}


void cal_InvLUSparsityPattern(PetscInt n, PetscInt bs)
{
	struct timeval tstart, tend;
	double cal_sparsepattern_t;
	gettimeofday(&tstart,NULL);

	DS_PRECOND_BISAI *ds = (DS_PRECOND_BISAI *)info->DS_PRECOND;


	PetscInt Lnnz=info->Lnnz;
	PetscInt Unnz=info->Unnz;
	PetscInt InvLnnz,InvUnnz;

	//debugging variables
	PetscInt irow,idx_st,idx_ed,idx;
	PetscInt *hLi2=NULL;
	PetscInt *hLj2=NULL;
	PetscReal *hLa2=NULL;

	// L0 * L1 symbolic multiplication
	PetscInt *hLi0=(PetscInt *)malloc(sizeof(PetscInt)*(n+1));
	PetscInt *hLi1=(PetscInt *)malloc(sizeof(PetscInt)*(n+1));
	PetscInt *hLj0=(PetscInt *)malloc(sizeof(PetscInt)*Lnnz);
	PetscInt *hLj1=(PetscInt *)malloc(sizeof(PetscInt)*Lnnz);
	PetscReal *hLa0=(PetscReal *)malloc(sizeof(PetscReal)*Lnnz);
	PetscReal *hLa1=(PetscReal *)malloc(sizeof(PetscReal)*Lnnz);

	if(SQUARE3)
	{
		hLi2=(PetscInt *)malloc(sizeof(PetscInt)*(n+1));
		hLj2=(PetscInt *)malloc(sizeof(PetscInt)*Lnnz);
		hLa2=(PetscReal *)malloc(sizeof(PetscReal)*Lnnz);
	}
	//
	memcpy(hLi0,info->hLRowPtr,sizeof(PetscInt)*(n+1));
	memcpy(hLi1,info->hLRowPtr,sizeof(PetscInt)*(n+1));
	memcpy(hLj0,info->hLColVal,sizeof(PetscInt)*Lnnz);
	memcpy(hLj1,info->hLColVal,sizeof(PetscInt)*Lnnz);

	if(SQUARE3)
	{
		memcpy(hLi2,info->hLRowPtr,sizeof(PetscInt)*(n+1));
		memcpy(hLj2,info->hLColVal,sizeof(PetscInt)*Lnnz);
	}

	memset(hLa0,0,sizeof(PetscReal)*Lnnz);
	memset(hLa1,0,sizeof(PetscReal)*Lnnz);
	if(SQUARE3)
	{
		memset(hLa2,0,sizeof(PetscReal)*Lnnz);
	}


	Mat L0;
	Mat L1;
	Mat L2;
	Mat InvL_k;
	if(SQUARE3)
	{
		MatCreateSeqAIJWithArrays(PETSC_COMM_SELF,n,n,hLi0,hLj0,hLa0,&L0);
		MatCreateSeqAIJWithArrays(PETSC_COMM_SELF,n,n,hLi1,hLj1,hLa1,&L1);
		MatCreateSeqAIJWithArrays(PETSC_COMM_SELF,n,n,hLi2,hLj2,hLa2,&L2);
		MatMatMatMult(L0,L1,L2,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&InvL_k);
	}
	else
	{
		MatCreateSeqAIJWithArrays(PETSC_COMM_SELF,n,n,hLi0,hLj0,hLa0,&L0);
		MatCreateSeqAIJWithArrays(PETSC_COMM_SELF,n,n,hLi1,hLj1,hLa1,&L1);
		MatMatMult(L0,L1,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&InvL_k);
	}


	Mat_SeqAIJ *InvLk_data=(Mat_SeqAIJ *)InvL_k->data;
	InvLnnz=InvLk_data->i[n];

	if(USE_ITS_SELF)
	{
		ds->InvLnnz=info->Lnnz;
		printf("InvLnnz=%d\n",ds->InvLnnz);
		if(!ds->hInvLRowPtr){ds->hInvLRowPtr=(PetscInt *)malloc(sizeof(PetscInt)*(n+1));}
		if(!ds->hInvLColVal){ds->hInvLColVal=(PetscInt *)malloc(sizeof(PetscInt)*ds->InvLnnz);}
		if(!ds->hInvLBlkVal){	ds->hInvLBlkVal=(PetscReal *)malloc(sizeof(PetscReal)*ds->InvLnnz*bs*bs);}
		memcpy(ds->hInvLRowPtr,info->hLRowPtr,sizeof(PetscInt)*(n+1));
		memcpy(ds->hInvLColVal,info->hLColVal,sizeof(PetscInt)*ds->InvLnnz);
		memset(ds->hInvLBlkVal,0,sizeof(PetscReal)*ds->InvLnnz*bs*bs);
	}
	else
	{
		ds->InvLnnz=InvLnnz;
		printf("InvLnnz=%d\n",InvLnnz);
		//allocate memory for hInvLRowPtr,hInvLColVal, hInvLBlkVal
		if(!ds->hInvLRowPtr){ds->hInvLRowPtr=(PetscInt *)malloc(sizeof(PetscInt)*(n+1));}
		if(!ds->hInvLColVal){ds->hInvLColVal=(PetscInt *)malloc(sizeof(PetscInt)*InvLnnz);}
		if(!ds->hInvLBlkVal){ds->hInvLBlkVal=(PetscReal *)malloc(sizeof(PetscReal)*InvLnnz*bs*bs);}

		memcpy(ds->hInvLRowPtr,InvLk_data->i,sizeof(PetscInt)*(n+1));
		memcpy(ds->hInvLColVal,InvLk_data->j,sizeof(PetscInt)*InvLnnz);
		memset(ds->hInvLBlkVal,0,sizeof(PetscReal)*InvLnnz*bs*bs);
	}
	// debugging
	//	if(!info->rank)
		if(0)
		{
			printf("Lnnz=%d, invLknnz=%d\n", Lnnz,ds->InvLnnz);
			for(irow=0;irow<n;irow++)
			{
				//idx_st=InvLk_data->i[irow];
				//idx_ed=InvLk_data->i[irow+1];
				idx_st=ds->hInvLRowPtr[irow];
				idx_ed=ds->hInvLRowPtr[irow+1];
				printf("[%d]:",irow);
				for(idx=idx_st;idx<idx_ed;idx++)
				{
					//printf("%d,",InvLk_data->j[idx]);
					printf("%d,",ds->hInvLColVal[idx]);
				}
			}	
			printf("\n");
		}		
		
	// free memory
	free(hLi0);free(hLj0);free(hLa0);
	free(hLi1);free(hLj1);free(hLa1);
	if(SQUARE3){free(hLi2);free(hLj2);free(hLa2);}
	MatDestroy(&L0);
	MatDestroy(&L1);
	if(SQUARE3){MatDestroy(&L2);}
	MatDestroy(&InvL_k);


	//U0*U1 symbolic multiplication
	PetscInt *hUi2=NULL;
	PetscInt *hUj2=NULL;
	PetscReal *hUa2=NULL;
	PetscInt *hUi0=(PetscInt *)malloc(sizeof(PetscInt)*(n+1));
	PetscInt *hUi1=(PetscInt *)malloc(sizeof(PetscInt)*(n+1));
	PetscInt *hUj0=(PetscInt *)malloc(sizeof(PetscInt)*Unnz);
	PetscInt *hUj1=(PetscInt *)malloc(sizeof(PetscInt)*Unnz);
	PetscReal *hUa0=(PetscReal *)malloc(sizeof(PetscReal)*Unnz);
	PetscReal *hUa1=(PetscReal *)malloc(sizeof(PetscReal)*Unnz);
	if(SQUARE3)
	{
		hUi2=(PetscInt *)malloc(sizeof(PetscInt)*(n+1));
		hUj2=(PetscInt *)malloc(sizeof(PetscInt)*Unnz);
		hUa2=(PetscReal *)malloc(sizeof(PetscReal)*Unnz);
	}
	
	//
	memcpy(hUi0,info->hURowPtr,sizeof(PetscInt)*(n+1));
	memcpy(hUi1,info->hURowPtr,sizeof(PetscInt)*(n+1));
	memcpy(hUj0,info->hUColVal,sizeof(PetscInt)*Unnz);
	memcpy(hUj1,info->hUColVal,sizeof(PetscInt)*Unnz);
	if(SQUARE3)
	{
		memcpy(hUi2,info->hURowPtr,sizeof(PetscInt)*(n+1));
		memcpy(hUj2,info->hUColVal,sizeof(PetscInt)*Unnz);
	}

	memset(hUa0,0,sizeof(PetscReal)*Unnz);
	memset(hUa1,0,sizeof(PetscReal)*Unnz);
	if(SQUARE3)
	{
		memset(hUa2,0,sizeof(PetscReal)*Unnz);
	}

	Mat U0;
	Mat U1;
	Mat U2;
	Mat InvU_k;
	if(SQUARE3)
	{
		MatCreateSeqAIJWithArrays(PETSC_COMM_SELF,n,n,hUi0,hUj0,hUa0,&U0);
		MatCreateSeqAIJWithArrays(PETSC_COMM_SELF,n,n,hUi1,hUj1,hUa1,&U1);
		MatCreateSeqAIJWithArrays(PETSC_COMM_SELF,n,n,hUi2,hUj2,hUa2,&U2);
		MatMatMatMult(U0,U1,U2,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&InvU_k);
	}
	else
	{
		MatCreateSeqAIJWithArrays(PETSC_COMM_SELF,n,n,hUi0,hUj0,hUa0,&U0);
		MatCreateSeqAIJWithArrays(PETSC_COMM_SELF,n,n,hUi1,hUj1,hUa1,&U1);
		MatMatMult(U0,U1,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&InvU_k);
	}

	Mat_SeqAIJ *InvUk_data=(Mat_SeqAIJ *)InvU_k->data;

	InvUnnz=InvUk_data->i[n];

	if(USE_ITS_SELF)
	{
		ds->InvUnnz=info->Unnz;
		printf("InvUnnz=%d\n",ds->InvUnnz);
		if(!ds->hInvURowPtr){ds->hInvURowPtr=(PetscInt *)malloc(sizeof(PetscInt)*(n+1));}
		if(!ds->hInvUColVal){ds->hInvUColVal=(PetscInt *)malloc(sizeof(PetscInt)*ds->InvUnnz);}
		if(!ds->hInvUBlkVal){ds->hInvUBlkVal=(PetscReal *)malloc(sizeof(PetscReal)*ds->InvUnnz*bs*bs);}
		memcpy(ds->hInvURowPtr,info->hURowPtr,sizeof(PetscInt)*(n+1));
		memcpy(ds->hInvUColVal,info->hUColVal,sizeof(PetscInt)*ds->InvUnnz);
		memset(ds->hInvUBlkVal,0,sizeof(PetscReal)*ds->InvUnnz*bs*bs);

	}
	else
	{
	ds->InvUnnz=InvUnnz;
	printf("InvUnnz=%d\n",ds->InvUnnz);
	
	//allocate memory for hInvURowPtr,hInvUColVal,hInvUBlkVal
	if(!ds->hInvURowPtr){ds->hInvURowPtr=(PetscInt *)malloc(sizeof(PetscInt)*(n+1));}
	if(!ds->hInvUColVal){ds->hInvUColVal=(PetscInt *)malloc(sizeof(PetscInt)*InvUnnz);}
	if(!ds->hInvUBlkVal){ds->hInvUBlkVal=(PetscReal *)malloc(sizeof(PetscReal)*InvUnnz*bs*bs);}


	memcpy(ds->hInvURowPtr,InvUk_data->i,sizeof(PetscInt)*(n+1));
	memcpy(ds->hInvUColVal,InvUk_data->j,sizeof(PetscInt)*InvUnnz);
	memset(ds->hInvUBlkVal,0,sizeof(PetscReal)*InvUnnz*bs*bs);
	}
	//	if(info->rank)
		if(0)
		{
			printf("Unnz=%d, invUknnz=%d\n", Unnz,ds->InvUnnz);
			for(irow=0;irow<n;irow++)
			{
				idx_st=ds->hInvURowPtr[irow];
				idx_ed=ds->hInvURowPtr[irow+1];
				printf("[%d]:",irow);
				for(idx=idx_st;idx<idx_ed;idx++)
				{
					printf("%d,",ds->hInvUColVal[idx]);
				}
			}	
			printf("\n");
		}		
	gettimeofday(&tend,NULL);
    cal_sparsepattern_t=((double) ((tend.tv_sec*1000000.0 + tend.tv_usec)-(tstart.tv_sec*1000000.0+tstart.tv_usec)))/1000.0;
	printf("the time elapse for cal_sparsepattern on CPU is:%12.8lf\n",cal_sparsepattern_t);

	free(hUi0);free(hUj0);free(hUa0);
	free(hUi1);free(hUj1);free(hUa1);
	if(SQUARE3){free(hUi2);free(hUj2);free(hUa2);}
	
	MatDestroy(&U0);
	MatDestroy(&U1);
	if(SQUARE3){MatDestroy(&U2);}
	MatDestroy(&InvU_k);
}

void CSR2CSC(PetscInt *rowptr,	PetscInt *colval,	PetscReal *csrblkval,
			 PetscInt *colptr,	PetscInt *rowval,	PetscReal *cscblkval,
			 PetscInt n,	   	PetscInt bs,  PetscInt nnz)
{
	// we do some checking on rowptr, colval and csrblkval, it should not be null pointers
	if(rowptr==NULL || colval==NULL || csrblkval==NULL)
	{
		printf("one of the CSR formatted arrays is null pointer,please check\n");
	}
	if(colptr==NULL || rowval==NULL || cscblkval==NULL)
	{
		printf("one of the CSC formatted arrays is null pointer, please check\n");
	}
	// we need to make sure the memories for CSC format is zeros
	memset(colptr,0,sizeof(PetscInt)*(n+1));
	memset(rowval,0,sizeof(PetscInt)*nnz);
	memset(cscblkval,0,sizeof(PetscReal)*nnz*bs*bs);
	int irow,icol,col,pos;
	int stidx,edidx,idx;
	int itmp;
	for(irow=0;irow<n;irow++)
	{
		stidx=rowptr[irow];
		edidx=rowptr[irow+1];
		for(idx=stidx;idx<edidx;idx++)
		{
			col=colval[idx];
			colptr[col]++;
		}
	}
	for(icol=n-1;icol>=0;icol--)
	{
		colptr[icol+1]=colptr[icol];
	}
	colptr[0]=0;
	for(icol=1;icol<=n;icol++)
	{
		colptr[icol]=colptr[icol]+colptr[icol-1];
	}

	int *tmpcounter=(int *)malloc(sizeof(int)*n);
	memset(tmpcounter,0,sizeof(int)*n);
	for(irow=0;irow<n;irow++)
	{
		stidx=rowptr[irow];
		edidx=rowptr[irow+1];
		for(idx=stidx;idx<edidx;idx++)
		{
			col=colval[idx];
			pos=colptr[col]+tmpcounter[col];
			rowval[pos]=irow;
			for(itmp=0;itmp<bs*bs;itmp++)
			{
				cscblkval[pos*bs*bs+itmp]=csrblkval[idx*bs*bs+itmp];
			}
			tmpcounter[col]++;
		}
	}
	free(tmpcounter);
}


void pre_ilu_petsc_for_iterative_precond( PetscInt *facti, PetscInt *factj,PetscInt *factdiag,PetscReal *facta,PetscInt n)
{
	PetscInt irow,itmp,colcnt,counter;
	PetscInt idx_st,idx_ed,idx;
	PetscInt bs=info->bs; 
	DS_PRECOND_ITERATIVE * ds = (DS_PRECOND_ITERATIVE *)info->DS_PRECOND;

	PetscBool is_num_iterations=PETSC_FALSE;


	// Step1:copy L and U in BCSR from PETSc (fact)
		for(irow=0;irow<=n;irow++)
		{
			info->hLRowPtr[irow]=facti[irow]+irow;	
		}
		// set hColVal:
		colcnt=0;
		for(irow=0;irow<n;irow++)
		{
			idx_st=facti[irow];
			idx_ed=facti[irow+1];
			for(idx=idx_st;idx<idx_ed;idx++)
			{
				info->hLColVal[colcnt]=factj[idx];
				// copy fact->a values: column-major within block
				for(itmp=0;itmp<bs*bs;itmp++){info->hLBlkVal[colcnt*bs*bs+itmp]=facta[idx*bs*bs+itmp];}
				colcnt++;
			}
			// add one more element: diagonal element
			info->hLColVal[colcnt]=irow;
			// set values to identities
			for(itmp=0;itmp<bs;itmp++){info->hLBlkVal[colcnt*bs*bs+itmp*bs+itmp]=1.0;}
			colcnt++;
		}

	if(!info->rank){printf("\n");}
	//U: 
	//NOTE: upper triangular matrix contains diagonal elements, but the order PETSc stores
	// upper tri-matrix for each row in factmat is like non-diag0, non-diag1,...., diag_element
	counter=0;
	for(irow=0;irow<n;irow++)
	{	
		idx_st=factdiag[irow+1]+1;
		idx_ed=factdiag[irow];  // it is the diagonal element
		info->hURowPtr[irow]=idx_ed-idx_st+1; // temporarily stores the number of element each row
		// handle diagonal element
		info->hUColVal[counter]=factj[idx_ed];
		for(itmp=0;itmp<bs*bs;itmp++)
		{
			info->hUBlkVal[counter*bs*bs+itmp]=facta[idx_ed*bs*bs+itmp]; // the diagonal is inversed 
		}
		counter++;
		// so we don't need to consider diag element each row 
		for(idx=idx_st;idx<idx_ed;idx++)
		{
			// copy ft in fact to hUColVal
			info->hUColVal[counter]=factj[idx];
			// copy or not copy double values depends
			for(itmp=0;itmp<bs*bs;itmp++)
			{
				info->hUBlkVal[counter*bs*bs+itmp]=facta[idx*bs*bs+itmp];
			}
			counter++;
		}
	}
	// until now, we have filled hURowPtr, hUColVal, and hUBlkVal, but hURowPtr stores 
	// the number of non-zero elements for each row, we just need a transformation
	// ...................................
	for(irow=n-1;irow>=0;irow--)
	{
		info->hURowPtr[irow+1]=info->hURowPtr[irow];
	}
	info->hURowPtr[0]=0;
	for(irow=1;irow<=n;irow++)
	{
		info->hURowPtr[irow]=info->hURowPtr[irow]+info->hURowPtr[irow-1];
	}


	//Step2: (Lhat + LD)x=b ->  x(k+1)=inv(LD)b-inv(LD)Lhat x(k) LD is block-identity, so it can be ignored
	//       (Uhat + UD)x=b ->  x(k+1)=inv(UD)b-inv(UD)Uhat x(k) 
	// Since we use Petsc's ILU factorization, the diagonal is already inversed for U
	// reset L's diagonal to zeros then L->Lhat
	for(irow=0;irow<n;irow++)
	{
		idx_st=info->hLRowPtr[irow+1]-1;
		for(itmp=0;itmp<bs*bs;itmp++){info->hLBlkVal[idx_st*bs*bs+itmp]=0.0;}
	}
	// block CSR format for Uhat: it is the block diagonal matrix
	if(!ds->hUdiagRowPtr){ds->hUdiagRowPtr=(PetscInt *)malloc(sizeof(PetscInt)*(n+1));}
	if(!ds->hUdiagColVal){ds->hUdiagColVal=(PetscInt *)malloc(sizeof(PetscInt)*n);}
	if(!ds->hUdiagBlkVal){ds->hUdiagBlkVal=(PetscReal *)malloc(sizeof(PetscReal)*n*bs*bs);}
	for(irow=0;irow<=n;irow++){ds->hUdiagRowPtr[irow]=irow;}
	for(irow=0;irow<n;irow++){ds->hUdiagColVal[irow]=irow;} 
	for(irow=0;irow<n;irow++)
	{
		idx_st=info->hURowPtr[irow];  // the first element is the diagonal
		for(itmp=0;itmp<bs*bs;itmp++)
		{
			ds->hUdiagBlkVal[irow*bs*bs+itmp]=info->hUBlkVal[idx_st*bs*bs+itmp];
		}
		// after copy: reset U's idagonal to zeros and then U -> Uhat
		for(itmp=0;itmp<bs*bs;itmp++)
		{
			info->hUBlkVal[idx_st*bs*bs+itmp]=0.0;
		}
	}
}




// if cusparse ILU , we need the symbolic ILU(k) pattern (fact in petsc) for L(k) and U(k)
// and we need to merge L(k) and U(k) to form the input A of the cusparse function 
// This is to say ,we need ILU pattern of fact, and values of the original A
void pre_ilu_cusparse( 	PetscInt *Ai,	PetscInt *Aj, 	PetscInt *Adiag, PetscReal *Aa, 
			PetscInt *facti,PetscInt *factj,PetscInt *factdiag,  PetscInt n)
{
	PetscInt irow,idx_st,idx_ed,idx, lt, At, ft;
	PetscInt itmp,jtmp;
	PetscInt bs=info->bs;
	PetscInt colcnt,counter;
	PetscInt Lnnz=info->Lnnz;
	PetscInt Unnz=info->Unnz;
	DS_PBILU_PRECOND_CUSPARSE * ds= (DS_PBILU_PRECOND_CUSPARSE *)info->DS_PBILU;
	// L: fact->a[idx] = A->a[idx]
	// NOTE: lower triangular matrix does not contain diagonal elements 
	// every non-linear step, we have to update the non-zero value in L, because A represents
	// Jacobi matrix which changes every non-linear step
	if(!info->init_gpu_called)
	{
		// set hLRowPtr:
		for(irow=0;irow<=n;irow++)
		{
			info->hLRowPtr[irow]=facti[irow]+irow;	
		}
		// set hColVal:
		colcnt=0;
		for(irow=0;irow<n;irow++)
		{
			idx_st=facti[irow];
			idx_ed=facti[irow+1];
			for(idx=idx_st;idx<idx_ed;idx++)
			{
				info->hLColVal[colcnt]=factj[idx];
				colcnt++;
			}
			// add one more element: diagonal element
			info->hLColVal[colcnt]=irow;
			colcnt++;
		}
	}
	for(irow=0;irow<n;irow++)
	{
		idx_st=info->hLRowPtr[irow];
		idx_ed=info->hLRowPtr[irow+1];
		lt=idx_st;
		At=Ai[irow];
		//if(!info->rank){printf("[%d]:",irow);}	
		while(lt < idx_ed && At < (Adiag[irow]+1)) // 2020.1.1
		{
			//if(fact->j[ft] == A->j[At])
			if(info->hLColVal[lt] == Aj[At])   // 2020.1.1
			{
				if(Aj[At] == irow)
				{
					// 2020.1.1  initial with identity matrix
					for(itmp=0;itmp<bs;itmp++)
					{
						info->hLBlkVal[lt*bs*bs+itmp*bs+itmp]=1.0;	
					}		
				}
				else
				{
					for(itmp=0;itmp<bs;itmp++)
					{
						for(jtmp=0;jtmp<bs;jtmp++)
						{
						info->hLBlkVal[lt*bs*bs+itmp*bs+jtmp]=Aa[At*bs*bs+itmp*bs+jtmp];// column-major
					
						}
					}
				}
				At++;
			}
			lt++; //2020.1.1
		}
	}
	if(!info->rank){printf("\n");}
	// we copy fact->i  fact->j  fact->a  to hLRowPtr hLColVal hLBlkVal, we copy only once, because 
	// the symbolic non-zero structure generally remains unchanged during non-linear process
	//U: 
	//NOTE: upper triangular matrix contains diagonal elements, but the order PETSc stores
	// lower tri-matrix for each row in factmat is like non-diag0, non-diag1,...., diag_element
	counter=0;
	for(irow=0;irow<n;irow++)
	{	
		idx_st=factdiag[irow+1]+1;
		idx_ed=factdiag[irow];  // it is the diagonal element
		info->hURowPtr[irow]=idx_ed-idx_st+1; // temporarily stores the number of element each row
		ft=idx_st;
		//At=A->diag[irow]+1; // diagonal element in A ???  need to be reviewed, At=A->diag[irow] ?
		At=Adiag[irow];
		
		// handle diagonal element
		info->hUColVal[counter]=Aj[At];
		for(itmp=0;itmp<bs;itmp++)
		{
			for(jtmp=0;jtmp<bs;jtmp++)
			{
				//info->hUBlkVal[counter*bs*bs+itmp*bs+jtmp]=A->a[At*bs*bs+jtmp*bs+itmp];
				info->hUBlkVal[counter*bs*bs+itmp*bs+jtmp]=Aa[At*bs*bs+itmp*bs+jtmp];//column-major
			}
		}
		counter++;
		At++;   // indicating the first element after diag element 
		// so we don't need to consider diag element each row 
		while(ft<idx_ed) // excluding diag element in fact
		{
			// copy ft in fact to hUColVal
			info->hUColVal[counter]=factj[ft];
			// copy or not copy double values depends
			//if(fact->j[ft] == A->j[At] && At< A->i[irow+1])
			if(At < Ai[irow+1] && factj[ft] == Aj[At])
			{
				for(itmp=0;itmp<bs;itmp++)
				{
					for(jtmp=0;jtmp<bs;jtmp++)
					{
						//info->hUBlkVal[counter*bs*bs+itmp*bs+jtmp]=A->a[At*bs*bs+jtmp*bs+itmp];
						info->hUBlkVal[counter*bs*bs+itmp*bs+jtmp]=Aa[At*bs*bs+itmp*bs+jtmp];//column-major
					}
				}
				At++;
			}
			ft++;
			counter++;
		}
	}
	// until now, we have filled hURowPtr, hUColVal, and hUBlkVal, but hURowPtr stores 
	// the number of non-zero elements for each row, we just need a transformation
	// ...................................
	for(irow=n-1;irow>=0;irow--)
	{
		info->hURowPtr[irow+1]=info->hURowPtr[irow];
	}
	info->hURowPtr[0]=0;
	for(irow=1;irow<=n;irow++)
	{
		info->hURowPtr[irow]=info->hURowPtr[irow]+info->hURowPtr[irow-1];
	}
	
	// now we have L(k) and U(k) in point-block CSR format. We also need the CSR format 
	// of Fact(k). So we merge L(k) and U(k).
	//PetscInt Factnnz=Lnnz+Unnz;
	PetscInt Factnnz=Lnnz+Unnz-n;  //both Lnnz and Unnz contains diagonal elements, so we eleminate repeated ones. 2020.1.1
	info->Factnnz=Factnnz;
	PetscInt lst, led;
	PetscInt ust, ued;

	/////////////////////////////////////////BEGIN SPECIFIC DATA STRUCTURE ////////////////////////////////////
	//PetscInt idx;
	if(!ds->hFactRowPtr)
	{
		ds->hFactRowPtr=(PetscInt *)malloc(sizeof(PetscInt)*(n+1));
		memset(ds->hFactRowPtr,0,sizeof(PetscInt)*(n+1));
	}
	if(!ds->hFactColVal)
	{
		ds->hFactColVal=(PetscInt *)malloc(sizeof(PetscInt)*Factnnz);
		memset(ds->hFactColVal,0,sizeof(PetscInt)*Factnnz);
	}
	if(!ds->hFactBlkVal)
	{
		ds->hFactBlkVal=(PetscReal *)malloc(sizeof(PetscReal)*Factnnz*bs*bs);
	}
	if(ds->hFactBlkVal){memset(ds->hFactBlkVal,0,sizeof(PetscReal)*Factnnz*bs*bs);}
	
	counter=0;
	if(!info->init_gpu_called) // more than once, we don't have to fill FactRowPtr and FactColPtr again
	{
		for(irow=0;irow<n;irow++)
		{
			lst=info->hLRowPtr[irow];
			led=info->hLRowPtr[irow+1];
			ust=info->hURowPtr[irow];
			ued=info->hURowPtr[irow+1];
			//info->hFactRowPtr[irow]=(led-lst)+(ued-ust);
			ds->hFactRowPtr[irow]=(led-lst-1)+(ued-ust);// 2020.1.1
			//for(idx=lst;idx<led;idx++)
			for(idx=lst;idx<(led-1);idx++)  // 
			{
				ds->hFactColVal[counter]=info->hLColVal[idx];
				counter++;
			}
			for(idx=ust;idx<ued;idx++)
			{
				ds->hFactColVal[counter]=info->hUColVal[idx];
				counter++;
			}
		}
		if(counter != (Lnnz+Unnz-n)){printf("error:counter != Annz, please have a check!\n");}
		// hFactRowPtr
		for(irow=n-1;irow>=0;irow--)
		{
			ds->hFactRowPtr[irow+1]=ds->hFactRowPtr[irow];
		}
		ds->hFactRowPtr[0]=0;
		for(irow=1;irow<=n;irow++)
		{
			ds->hFactRowPtr[irow]=ds->hFactRowPtr[irow]+ds->hFactRowPtr[irow-1];
		}
		
	}

	if(0)// debugging printing
	{
		printf("i for Fact structure:");
		for(irow=0;irow<(n+1);irow++){printf("%d ",ds->hFactRowPtr[irow]);}
		printf("\n");
		printf("j for Fact structure:");
		for(irow=0;irow<n;irow++)
		{
			printf("[%d]:",irow);
			for(itmp=ds->hFactRowPtr[irow];itmp<ds->hFactRowPtr[irow+1];itmp++)
			{printf("%d ",ds->hFactColVal[itmp]);}
		}
		printf("\n");
	}

	// each non-linear time step, we have to update FactBlkVal  
	counter=0;
	for(irow=0;irow<n;irow++)
	{
		// number of non-zero elements each row : L + U
		lst=info->hLRowPtr[irow];
		led=info->hLRowPtr[irow+1];
		ust=info->hURowPtr[irow];
		ued=info->hURowPtr[irow+1];
		// copy L:
		//for(idx=lst;idx<led;idx++)
		for(idx=lst;idx<(led-1);idx++)  //2020.1.1  excluding repeated diagonal elements
		{
			for(itmp=0;itmp<bs*bs;itmp++)
			{
				ds->hFactBlkVal[counter*bs*bs+itmp]=info->hLBlkVal[idx*bs*bs+itmp];
			}
			counter++;
		}
		//copy U:
		for(idx=ust;idx<ued;idx++)
		{
			for(itmp=0;itmp<bs*bs;itmp++)
			{
				ds->hFactBlkVal[counter*bs*bs+itmp]=info->hUBlkVal[idx*bs*bs+itmp];
			}
			counter++;
		}
	}
	
	// we have a check 
	if(counter != (Lnnz+Unnz-n)){printf("ERROR:counter != Lnnz+Unnz, please check\n");}
	
}



void pre_ilu_asynchronous(PetscInt *Ai, PetscInt *Aj, PetscInt *Adiag, PetscReal *Aa,
			PetscInt *facti, PetscInt *factj, PetscInt *factdiag, PetscReal *facta, PetscInt n)
{
	PetscInt irow,idx_st,idx_ed,idx, lt, At, ft;
	PetscInt itmp,jtmp;
	PetscInt bs=info->bs;
	PetscInt colcnt,counter;
	PetscInt Lnnz=info->Lnnz;
	PetscInt Unnz=info->Unnz;
	DS_PBILU_ASYN * ds=(DS_PBILU_ASYN *)info->DS_PBILU;
	
	PetscBool is_sweep_num;
	PetscBool is_asyn_use_exactilu_asfirst=PETSC_FALSE;
	ds->sweep_num=5;
	//If the user does not supply the option ivalue is NOT changed. 
	//Thus you should ALWAYS initialize the ivalue if you access it without first checking if the set flag is true.
	PetscOptionsGetInt(NULL,NULL,"-asyn_pbilu_sweep_num",&ds->sweep_num,&is_sweep_num);
	if(!is_sweep_num){printf("you can specify -asyn_pbilu_sweep_num to set the number of sweeps the asynchronous ILU factorization performs, default is 5\n");}	
	
	PetscOptionsGetBool(NULL,NULL,"-asyn_use_exactilu_asfirst", &ds->asyn_use_exactilu_asfirst, &is_asyn_use_exactilu_asfirst);	
	if(is_asyn_use_exactilu_asfirst && info->cusparse_precond)
	{
		printf("ERROR: the -asyn_use_exactilu_asfirst is only used in iterative_precond and bisai_precond,it can't be used in cusparse_precond\n");
	}



	// L: fact->a[idx] = A->a[idx]
	// NOTE: lower triangular matrix does not contain diagonal elements 
	// every non-linear step, we have to update the non-zero value in L, because A represents
	// Jacobi matrix which changes every non-linear step
	if(!info->init_gpu_called)
	{
		// set hLRowPtr:
		for(irow=0;irow<=n;irow++)
		{
			info->hLRowPtr[irow]=facti[irow]+irow;	
		}
		// set hColVal:
		colcnt=0;
		for(irow=0;irow<n;irow++)
		{
			idx_st=facti[irow];
			idx_ed=facti[irow+1];
			for(idx=idx_st;idx<idx_ed;idx++)
			{
				info->hLColVal[colcnt]=factj[idx];
				colcnt++;
			}
			// add one more element: diagonal element
			info->hLColVal[colcnt]=irow;
			colcnt++;
		}
	}
	for(irow=0;irow<n;irow++)
	{
		idx_st=info->hLRowPtr[irow];
		idx_ed=info->hLRowPtr[irow+1];
		lt=idx_st;
		At=Ai[irow];
		while(lt < idx_ed && At < (Adiag[irow]+1)) // 2020.1.1
		{
			if(info->hLColVal[lt] == Aj[At])   // 2020.1.1
			{
				if(Aj[At] == irow)
				{
					// 2020.1.1  initial with identity matrix
					for(itmp=0;itmp<bs;itmp++)
					{
						info->hLBlkVal[lt*bs*bs+itmp*bs+itmp]=1.0;	
					}		
				}
				else
				{
					for(itmp=0;itmp<bs;itmp++)
					{
						for(jtmp=0;jtmp<bs;jtmp++)
						{
							// row-major within blocks NOT COLUMN MAJOR
							//info->hLBlkVal[lt*bs*bs+itmp*bs+jtmp]=Aa[At*bs*bs+jtmp*bs+itmp];
							info->hLBlkVal[lt*bs*bs+itmp*bs+jtmp]=Aa[At*bs*bs+itmp*bs+jtmp]; // column-major within blocks
						}
					}
				}
				At++;
			}
			lt++; //2020.1.1
		}
	}
	//if(!info->rank){printf("\n");}
	// we copy fact->i  fact->j  fact->a  to hLRowPtr hLColVal hLBlkVal, we copy only once, because 
	// the symbolic non-zero structure generally remains unchanged during non-linear process
	//U: 
	//NOTE: upper triangular matrix contains diagonal elements, but the order PETSc stores
	// lower tri-matrix for each row in factmat is like non-diag0, non-diag1,...., diag_element
	counter=0;
	for(irow=0;irow<n;irow++)
	{	
		idx_st=factdiag[irow+1]+1;
		idx_ed=factdiag[irow];  // it is the diagonal element
		info->hURowPtr[irow]=idx_ed-idx_st+1; // temporarily stores the number of element each row
		ft=idx_st;
		At=Adiag[irow];
		
		// handle diagonal element
		info->hUColVal[counter]=Aj[At];
		for(itmp=0;itmp<bs;itmp++)
		{
			for(jtmp=0;jtmp<bs;jtmp++)
			{
				//info->hUBlkVal[counter*bs*bs+itmp*bs+jtmp]=Aa[At*bs*bs+jtmp*bs+itmp];//row major format NOT COLUMN MAJOR for asynchronous ILU factorization
				info->hUBlkVal[counter*bs*bs+itmp*bs+jtmp]=Aa[At*bs*bs+itmp*bs+jtmp];//column major format NOT ROW  MAJOR for asynchronous ILU factorization
			}
		}
		counter++;
		At++;   // indicating the first element after diag element 
		// so we don't need to consider diag element each row 
		while(ft<idx_ed) // excluding diag element in fact
		{
			// copy ft in fact to hUColVal
			info->hUColVal[counter]=factj[ft];
			// copy or not copy double values depends
			//if(fact->j[ft] == A->j[At] && At< A->i[irow+1])
			if(At < Ai[irow+1] && factj[ft] == Aj[At])
			{
				for(itmp=0;itmp<bs;itmp++)
				{
					for(jtmp=0;jtmp<bs;jtmp++)
					{
						//info->hUBlkVal[counter*bs*bs+itmp*bs+jtmp]=Aa[At*bs*bs+jtmp*bs+itmp];//row major format NOT COLUMN MAJOR for asynchronous ILU factorization
						info->hUBlkVal[counter*bs*bs+itmp*bs+jtmp]=Aa[At*bs*bs+itmp*bs+jtmp];//column major format NOT ROW MAJOR for asynchronous ILU factorization
					}
				}
				At++;
			}
			ft++;
			counter++;
		}
	}
	// until now, we have filled hURowPtr, hUColVal, and hUBlkVal, but hURowPtr stores 
	// the number of non-zero elements for each row, we just need a transformation
	// ...................................
	for(irow=n-1;irow>=0;irow--)
	{
		info->hURowPtr[irow+1]=info->hURowPtr[irow];
	}
	info->hURowPtr[0]=0;
	for(irow=1;irow<=n;irow++)
	{
		info->hURowPtr[irow]=info->hURowPtr[irow]+info->hURowPtr[irow-1];
	}
	
	// now we have L(k) and U(k) in point-block CSR format. We also need the CSR format 
	// of Fact(k). So we merge L(k) and U(k).
	PetscInt Factnnz=Lnnz+Unnz-n;  //both Lnnz and Unnz contains diagonal elements, so we eleminate repeated ones. 2020.1.1
	info->Factnnz=Factnnz;

	// SETUP DS_PBILU_ASYN * ds  (data structure)
	PetscInt lst, led;
	PetscInt ust, ued;
	if(!ds->hFactRowPtr)
	{
		ds->hFactRowPtr=(PetscInt *)malloc(sizeof(PetscInt)*(n+1));
		memset(ds->hFactRowPtr,0,sizeof(PetscInt)*(n+1));
	}
	if(!ds->hFactColVal)
	{
		ds->hFactColVal=(PetscInt *)malloc(sizeof(PetscInt)*Factnnz);
		memset(ds->hFactColVal,0,sizeof(PetscInt)*Factnnz);
	}
	if(!ds->hFactBlkVal)
	{
		ds->hFactBlkVal=(PetscReal *)malloc(sizeof(PetscReal)*Factnnz*bs*bs);
	}
	if(ds->hFactBlkVal){memset(ds->hFactBlkVal,0,sizeof(PetscReal)*Factnnz*bs*bs);}
	
	counter=0;
	if(!info->init_gpu_called) // more than once, we don't have to fill FactRowPtr and FactColPtr again
	{
		for(irow=0;irow<n;irow++)
		{
			lst=info->hLRowPtr[irow];
			led=info->hLRowPtr[irow+1];
			ust=info->hURowPtr[irow];
			ued=info->hURowPtr[irow+1];
			//info->hFactRowPtr[irow]=(led-lst)+(ued-ust);
			ds->hFactRowPtr[irow]=(led-lst-1)+(ued-ust);// 2020.1.1
			//for(idx=lst;idx<led;idx++)
			for(idx=lst;idx<(led-1);idx++)  // 
			{
				ds->hFactColVal[counter]=info->hLColVal[idx];
				counter++;
			}
			for(idx=ust;idx<ued;idx++)
			{
				ds->hFactColVal[counter]=info->hUColVal[idx];
				counter++;
			}
		}
		if(counter != (Lnnz+Unnz-n)){printf("error:counter != Annz, please have a check!\n");}
		// hFactRowPtr
		for(irow=n-1;irow>=0;irow--)
		{
			ds->hFactRowPtr[irow+1]=ds->hFactRowPtr[irow];
		}
		ds->hFactRowPtr[0]=0;
		for(irow=1;irow<=n;irow++)
		{
			ds->hFactRowPtr[irow]=ds->hFactRowPtr[irow]+ds->hFactRowPtr[irow-1];
		}
		
	}

	if(0)// debugging printing
	{
		printf("i for Fact structure:");
		for(irow=0;irow<(n+1);irow++){printf("%d ",ds->hFactRowPtr[irow]);}
		printf("\n");
		printf("j for Fact structure:");
		for(irow=0;irow<n;irow++)
		{
			printf("[%d]:",irow);
			for(itmp=ds->hFactRowPtr[irow];itmp<ds->hFactRowPtr[irow+1];itmp++)
			{printf("%d ",ds->hFactColVal[itmp]);}
		}
		printf("\n");
	}

	// each non-linear time step, we have to update FactBlkVal, this part will be executed many times
	counter=0;
	for(irow=0;irow<n;irow++)
	{
		// number of non-zero elements each row : L + U
		lst=info->hLRowPtr[irow];
		led=info->hLRowPtr[irow+1];
		ust=info->hURowPtr[irow];
		ued=info->hURowPtr[irow+1];
		// copy L:
		//for(idx=lst;idx<led;idx++)
		for(idx=lst;idx<(led-1);idx++)  //2020.1.1  excluding repeated diagonal elements
		{
			for(itmp=0;itmp<bs*bs;itmp++)
			{
				ds->hFactBlkVal[counter*bs*bs+itmp]=info->hLBlkVal[idx*bs*bs+itmp];
			}
			counter++;
		}
		//copy U:
		for(idx=ust;idx<ued;idx++)
		{
			for(itmp=0;itmp<bs*bs;itmp++)
			{
				ds->hFactBlkVal[counter*bs*bs+itmp]=info->hUBlkVal[idx*bs*bs+itmp];
			}
			counter++;
		}
	}
	
	// we have a check 
	if(counter != (Lnnz+Unnz-n)){printf("ERROR:counter != Lnnz+Unnz, please check\n");}

	// The following,  we do CSR_to_CSC and CSC_to_CSR transformation
	/*  we don't need CSC version for U, because the CSR to CSC version transformation is completed on GPU
	if(!ds->hCSC_UColPtr)
	{
		ds->hCSC_UColPtr=(PetscInt *)malloc(sizeof(PetscInt)*(n+1));
		memset(ds->hCSC_UColPtr,0,sizeof(PetscInt)*(n+1));
	}	 
	if(!ds->hCSC_URowVal)
	{
		ds->hCSC_URowVal=(PetscInt *)malloc(sizeof(PetscInt)*Unnz);
		memset(ds->hCSC_URowVal,0,sizeof(PetscInt)*Unnz);
	}
	if(!ds->hCSC_UBlkVal)
	{
		ds->hCSC_UBlkVal=(PetscReal *)malloc(sizeof(PetscReal)*Unnz*bs*bs);
	}
	if(ds->hCSC_UBlkVal){memset(ds->hCSC_UBlkVal,0,sizeof(PetscReal)*Unnz*bs*bs);}

	// we fill hCSC_UColPtr, hCSC_URowVal only once
	PetscInt col,icol,pos;
	if(!info->init_gpu_called)
	{
		for(irow=0;irow<n;irow++)
		{
			idx_st=info->hURowPtr[irow];
			idx_ed=info->hURowPtr[irow+1];
			for(idx=idx_st;idx<idx_ed;idx++)
			{
				col=info->hUColVal[idx];
				ds->hCSC_UColPtr[col]++;		
			}
		}	
		// 
		for(icol=n-1;icol>=0;icol--){ds->hCSC_UColPtr[icol+1]=ds->hCSC_UColPtr[icol];}
		ds->hCSC_UColPtr[0]=0;
		for(icol=1;icol<=n;icol++){ds->hCSC_UColPtr[icol]=ds->hCSC_UColPtr[icol]+ds->hCSC_UColPtr[icol-1];}
		
	}			
	
	PetscInt *tmpcounter=(PetscInt *)malloc(sizeof(PetscInt)*n);
	memset(tmpcounter,0,sizeof(PetscInt)*n);
	for(irow=0;irow<n;irow++)
	{
		idx_st=info->hURowPtr[irow];
		idx_ed=info->hURowPtr[irow+1];
		for(idx=idx_st;idx<idx_ed;idx++)
		{
			col=info->hUColVal[idx];
			pos=ds->hCSC_UColPtr[col]+tmpcounter[col];
			ds->hCSC_URowVal[pos]=irow;
			for(itmp=0;itmp<bs*bs;itmp++)
			{
				ds->hCSC_UBlkVal[pos*bs*bs+itmp]=info->hUBlkVal[idx*bs*bs+itmp];
			}
			tmpcounter[col]++;
		}
	}
	// now we need to establish a CSC_to_CSR integer map that help transform a CSC upper
	// triangular matrix into a CSR format easily on GPU. THis step should be performed only once
	if(!ds->hcsc_to_csr_map)
	{
		ds->hcsc_to_csr_map=(PetscInt *)malloc(sizeof(PetscInt)*Unnz);
		
		memset(tmpcounter,0,sizeof(PetscInt)*n);
		for(icol=0;icol<n;icol++)
		{
			idx_st=ds->hCSC_UColPtr[icol];
			idx_ed=ds->hCSC_UColPtr[icol+1];
			for(idx=idx_st;idx<idx_ed;idx++)
			{
				irow=ds->hCSC_URowVal[idx];
				pos=info->hURowPtr[irow]+tmpcounter[irow];
				ds->hcsc_to_csr_map[idx]=pos;
				tmpcounter[irow]++;
			}
		}
		// debug
	} 
	free(tmpcounter);
	*/	
	if(!ds->hUDiagRowPtr)
	{
		ds->hUDiagRowPtr=(PetscInt *)malloc(sizeof(PetscInt)*(n+1));
		// set values
		for(irow=0;irow<=n;irow++)
		{
			ds->hUDiagRowPtr[irow]=irow;
		}
	}
	if(!ds->hUDiagColVal)
	{
		ds->hUDiagColVal=(PetscInt *)malloc(sizeof(PetscInt)*n);
		// set values
		for(irow=0;irow<n;irow++)
		{
			ds->hUDiagColVal[irow]=irow;
		}
	}
	if(!ds->hUDiagVal){ds->hUDiagVal=(PetscReal *)malloc(sizeof(PetscReal)*n*bs*bs);}

	if(ds->hUDiagVal){memset(ds->hUDiagVal,0,sizeof(PetscReal)*n*bs*bs);}


	// could use point-block ilu factorization in the first newton step
	// at the same time, it also could provide very good inital values for L and U in the following newton steps
	
	if(is_asyn_use_exactilu_asfirst && !info->init_gpu_called)
	{
		// copy exact ilu factorization from petsc
		// Step1:copy L and U in BCSR from PETSc (fact), ONLY FOR THE FIRST  non-linear (Newton )step
		if(!ds->hExactLBlkVal){ds->hExactLBlkVal=(PetscReal *)malloc(sizeof(PetscReal)*Lnnz*bs*bs);}
		if(ds->hExactLBlkVal){memset(ds->hExactLBlkVal,0,sizeof(PetscReal)*Lnnz*bs*bs);}
		if(!ds->hExactUBlkVal){ds->hExactUBlkVal=(PetscReal *)malloc(sizeof(PetscReal)*Unnz*bs*bs);}
		if(ds->hExactUBlkVal){memset(ds->hExactUBlkVal,0,sizeof(PetscReal)*Unnz*bs*bs);}
		// copy petsc's L's values to hExactLBlkVal
		colcnt=0;
		for(irow=0;irow<n;irow++)
		{
			idx_st=facti[irow];
			idx_ed=facti[irow+1];
			for(idx=idx_st;idx<idx_ed;idx++)
			{
				// copy fact->a values: column-major within block
				for(itmp=0;itmp<bs*bs;itmp++){ds->hExactLBlkVal[colcnt*bs*bs+itmp]=facta[idx*bs*bs+itmp];}
				colcnt++;
			}
			// set values to identities
			for(itmp=0;itmp<bs;itmp++){ds->hExactLBlkVal[colcnt*bs*bs+itmp*bs+itmp]=1.0;}
			colcnt++;
		}
		//U: copy petsc's U's values to hExactUBlkVal 
		//NOTE: upper triangular matrix contains diagonal elements, but the order PETSc stores
		// upper tri-matrix for each row in factmat is like non-diag0, non-diag1,...., diag_element
		counter=0;
		for(irow=0;irow<n;irow++)
		{	
			idx_st=factdiag[irow+1]+1;
			idx_ed=factdiag[irow];  // it is the diagonal element
			for(itmp=0;itmp<bs*bs;itmp++)
			{
				ds->hExactUBlkVal[counter*bs*bs+itmp]=facta[idx_ed*bs*bs+itmp]; // the diagonal is inversed 
			}
			counter++;
			// so we don't need to consider diag element each row 
			for(idx=idx_st;idx<idx_ed;idx++)
			{
				for(itmp=0;itmp<bs*bs;itmp++)
				{
					ds->hExactUBlkVal[counter*bs*bs+itmp]=facta[idx*bs*bs+itmp];
				}
				counter++;
			}
		}

	}
	


	if(info->cusparse_precond)
	{
		if(!ds->hUInvDiagVal)
		{
			ds->hUInvDiagVal=(PetscReal *)malloc(sizeof(PetscReal)*n*bs*bs);
		}

		if(!ds->hUStarBlkVal)
		{
			ds->hUStarBlkVal=(PetscReal *)malloc(sizeof(PetscReal)*Unnz*bs*bs);
		}
			
	}

	if(info->bisai_precond)
	{
		DS_PRECOND_BISAI * ds_bisai =(DS_PRECOND_BISAI *)info->DS_PRECOND;
		ds_bisai->bisai_dense_level = 2;
		PetscBool is_bisai_dense_level= PETSC_FALSE;
		PetscOptionsGetInt(NULL,NULL,"-bisai_dense_level",&ds_bisai->bisai_dense_level,&is_bisai_dense_level);
		if(!is_bisai_dense_level){printf("the dense level for L and U's inverses can be set by -bisai_dense_level, the default is 2\n");}	
		// because we don't have the sizes of InvL and InvU, the host memory allocation are done on
		// InitialGMRES_GPU function.
		// if asyn_use_exactilu_asfirst is true, we do nothing, because the hExactUBlkVal's diagonal blocks are already inversed
	}
	if(info->iterative_precond)
	{
		DS_PRECOND_ITERATIVE * ds_iter = (DS_PRECOND_ITERATIVE *)info->DS_PRECOND;
		if(!ds_iter->hUdiagRowPtr){ds_iter->hUdiagRowPtr=(PetscInt *)malloc(sizeof(PetscInt)*(n+1));} // Invdiag
		if(!ds_iter->hUdiagColVal){ds_iter->hUdiagColVal=(PetscInt *)malloc(sizeof(PetscInt)*n);}      // Invdiag
		//if(!ds_iter->hUdiagBlkVal){ds_iter->hUdiagBlkVal=(PetscReal *)malloc(sizeof(PetscReal)*n*bs*bs);} // don't need to save inverse on CPU

		// since ASYN includes the hUDiagRowPtr and hUDiagColVal, we copy from them,
		if(ds_iter->hUdiagRowPtr){memcpy(ds_iter->hUdiagRowPtr,ds->hUDiagRowPtr,sizeof(PetscInt)*(n+1));}
		if(ds_iter->hUdiagColVal){memcpy(ds_iter->hUdiagColVal,ds->hUDiagColVal,sizeof(PetscInt)*n);}
	}


}








