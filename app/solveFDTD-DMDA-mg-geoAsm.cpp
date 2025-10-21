// #include <petscksp.h>
// #include <petscpc.h>     
//
#include<stdio.h>
#include <stdlib.h>
#include <math.h>
//
#include <petscdm.h>
#include <petscdmda.h>
#include <sys/time.h>
#include <petsc.h>
#include <petsc/private/pcimpl.h>
#include <petsc/private/kspimpl.h>
#include <../src/mat/impls/baij/mpi/mpibaij.h>
#include <../src/mat/impls/baij/seq/baij.h>
#include "KSPSolve_GMRES_CPU.h"
#include "KSPSolve_GMRES_GPU.h"
#include <petscversion.h>

#if PETSC_VERSION_LT(3, 14, 0)
  #define PetscCall(func) do { PetscErrorCode ierr = (func); CHKERRQ(ierr); } while (0)
  #define PetscCallMPI(func) do { PetscErrorCode ierr = (func); CHKERRMPI(ierr); } while (0)
  #define PetscCheck(condition, comm, errcode, msg) if (!(condition)) SETERRQ(comm, errcode, msg)
#endif


static char help[] = "Solves \n\n";

static PetscErrorCode PCMGSetupViaCoarsen(PC pc, DM da_fine);
PETSC_EXTERN PetscErrorCode PCCreate_GEOASM(PC pc);
PetscErrorCode PCASMprintsubA(PC pc);

int main(int argc, char **argv)
{
  Vec         x0,x, b, r;
  Mat         A, A_baij; 
  DM        da, da_asm;  
  PetscScalar norm;
  PetscInt rank,size;
  PetscInt nn=10,i,j,k, idtmp, nsize, geoasm_overlap=1;
  PetscScalar deltat =8.0; 
  char nameD[PETSC_MAX_PATH_LEN] = "", namex[PETSC_MAX_PATH_LEN] = "";  // subdomain D and x  
  char namex_g[PETSC_MAX_PATH_LEN] = "";    // global solution x
  PetscBool flg= PETSC_FALSE;  
  float vtmp;  
  //
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "initialize\n"));    
  //
  PetscInt lengthx, lengthd=1277952; //nonzero num
  PetscInt npx = size, npy =1, npz  = 1;
  PetscCall(PetscOptionsGetString(NULL, NULL, "-fD", nameD, sizeof(nameD), &flg));
  PetscCall(PetscOptionsGetString(NULL, NULL, "-fx", namex, sizeof(namex), &flg));  
  PetscCall(PetscOptionsGetString(NULL, NULL, "-fx_g", namex_g, sizeof(namex_g), &flg));  
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-nn", &nn, NULL));  // side length in matlab ( including ghost cells)
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-nsize", &nsize, NULL));  // side length (subdomain size) = nn-2
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-dt", &deltat, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-nnz", &lengthd, NULL));  // nonzeoro num in D
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-npx", &npx, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-npy", &npy, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-npz", &npz, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-geoasm_overlap", &geoasm_overlap, NULL));
 
  PetscCheck(nn-nsize==2, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "error: nn-nsize !=2 ! " );
  PetscCheck(npx*npy*npz==size, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "error: npx*npy*npz !=size ! " );
 
  lengthx =nn*nn*nn*3;  // problem size
  PetscScalar xin[lengthx];
  // PetscScalar Dval[lengthd];
  PetscInt   row, col;
  // PetscInt Dcol[lengthd];
  PetscScalar vv;

  double *Dval;
  Dval = (double *)malloc(lengthd * sizeof(double));
  PetscInt *Drow, *Dcol;
  Drow = (PetscInt *)malloc(lengthd * sizeof(PetscInt));
  Dcol = (PetscInt *)malloc(lengthd * sizeof(PetscInt));

  //
  //read subdomain D from file 
  if(rank== 0){
	  FILE * fp1 = fopen(nameD, "r");
	  for(i = 0 ;i< lengthd; i++) {
		  fscanf(fp1,"%d %d %g",&row, &col, &vtmp);
		  Dval[i] = (double)vtmp;
		  Drow[i] = row-1;
		  Dcol[i] = col-1;
	  }	  
	  fclose(fp1);
  } 
  MPI_Bcast(Dval,lengthd,MPI_DOUBLE,0,PETSC_COMM_WORLD);
  MPI_Bcast(Drow,lengthd,MPI_INT,0,PETSC_COMM_WORLD);
  MPI_Bcast(Dcol,lengthd,MPI_INT,0,PETSC_COMM_WORLD);
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "read local D from file %s !\n", nameD));
  //
  if(strcmp(namex, "") != 0){  
	  if(rank== 0){
		  FILE * fp2 = fopen(namex, "r"); 
		  for(i = 0 ;i< lengthx; i++) {
			  fscanf(fp2,"%g",&vtmp);
			  xin[i] = (double)vtmp;
		  } 
		  fclose(fp2);
	  } 
	  MPI_Bcast(xin,lengthx,MPI_DOUBLE,0,PETSC_COMM_WORLD);
	  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "read local x from file %s !\n", namex));  
  }
  if(strcmp(namex_g, "") != 0){ 
      PetscInt nvar = nsize*nsize*nsize*3 ;  
	  if(rank== 0){
		  FILE * fp2 = fopen(namex_g, "r"); 
		  rewind(fp2);
		  for(j=0;j<size;j++){
			  for(i = 0 ; i< nvar; i++) {
				  fscanf(fp2,"%g",&vtmp);
				  xin[i] = (double)vtmp;
			  }
			  if(j>0) {
				  MPI_Send(xin, nvar, MPI_DOUBLE, j, 0, PETSC_COMM_WORLD);
				  //PetscCall(PetscPrintf(PETSC_COMM_WORLD, "send global x to processor %d !\n", j));  
			  }
		  } 
		  rewind(fp2);
		  for(i = 0 ; i< nvar; i++) {
			  fscanf(fp2,"%g",&vtmp);
			  xin[i] = (double)vtmp;
		  }		  
		  fclose(fp2);
	  }else{
		  MPI_Recv(xin, nvar, MPI_DOUBLE, 0, 0, PETSC_COMM_WORLD, MPI_STATUS_IGNORE);
	  } 
	  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "read global x from file %s !\n", namex_g));  
  }  
  //	

  PetscCall(DMDACreate3d(PETSC_COMM_WORLD, DM_BOUNDARY_PERIODIC, DM_BOUNDARY_PERIODIC, DM_BOUNDARY_PERIODIC, DMDA_STENCIL_BOX, npx*nsize, npy*nsize, npz*nsize, npx,npy,npz, 3, 1, NULL,NULL,NULL, &da));  
  PetscCall(DMSetFromOptions(da));
  PetscCall(DMSetUp(da));  
  
  PetscCall(DMCreateGlobalVector(da, &x0));   
  PetscScalar  ***var_x0;
  PetscCall(DMDAVecGetArray(da, x0, &var_x0));
  PetscInt xs, ys, zs, xm, ym, zm;
  PetscCall(DMDAGetCorners(da, &xs, &ys, &zs, &xm, &ym, &zm));  
 
  idtmp =0;
  for (k = zs; k < zs + zm; k++) {
	  for (j = ys; j < ys + ym; j++) {
		  for (i = xs; i < xs + xm; i++) {
			  var_x0[k][j][3*i+0] = xin[ idtmp++];
			  var_x0[k][j][3*i+1] = xin[ idtmp++];
			  var_x0[k][j][3*i+2] = xin[ idtmp++];			  
		  }
      }
  }
  PetscCall(DMDAVecRestoreArray(da, x0, &var_x0)); 
  
  PetscInt nlocalsize;
  PetscCall(VecGetLocalSize(x0,&nlocalsize));
  PetscCall(MatCreateAIJ(PETSC_COMM_WORLD,nlocalsize,nlocalsize,PETSC_DECIDE,PETSC_DECIDE,13,NULL,13,NULL,&A));  
  ISLocalToGlobalMapping map;  // local to global mapping object    
  PetscCall(DMGetLocalToGlobalMapping(da,&map));
  PetscCall(MatSetLocalToGlobalMapping(A,map,map));  

  for (i =0; i< lengthd; i++){
	  row = Drow[i]+1;  //in matlab from 1 to nn+2
	  col = Dcol[i]-1;  //in local from -1 to nn
	  vv = Dval[i];
	  PetscInt ii,jj,kk;
	  kk = (row-1)/(3*(nsize+2)*(nsize+2))+1 ;
	  jj = (row-1)%(3*(nsize+2)*(nsize+2))/(3*(nsize+2))+1;
	  ii = (row-1)%(3*(nsize+2))/3+1;
	  ii = ii -2;  // from[1,nx+2] to [-1,nx]
	  jj = jj -2;
	  kk = kk -2;
	  if(ii>=0 && ii<nsize &&jj>=0 && jj<nsize &&kk>=0 && kk<nsize ){
		  row = row-1; 
		  col = col+1;
	      PetscCall(MatSetValuesLocal(A, 1, &row, 1, &col, &vv, INSERT_VALUES));
	  }
  }

  
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));  

  // f(inputE) = inputE+deltat**2/4*curl_bw(curl_fw(inputE))
  // df/dE = A = I+ deltat**2/2*D
  // D=1/2* d(curl_bw(curl_fw(E)))/d(E)
  PetscCall(MatScale(A, deltat*deltat*0.5));  
  PetscCall(MatShift(A, 1.0));


  PetscCall(MatSetBlockSize(A, 1));

    // 将矩阵 A 转换为 MATMPIBAIJ 格式，块大小为 1
  PetscCall(MatConvert(A, MATMPIBAIJ, MAT_INITIAL_MATRIX, &A_baij));
 

  // print the row and col of A_baij
  PetscInt row_A_baij, col_A_baij;
  PetscCall(MatGetLocalSize(A_baij, &row_A_baij, &col_A_baij));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "A_baij row: %d, col: %d\n", row_A_baij, col_A_baij));


 // 验证矩阵类型 and print the size of block
  const char *type;
  PetscCall(MatGetType(A_baij, &type));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "A_baij type: %s\n", type));
  PetscInt bs;
  PetscCall(MatGetBlockSize(A_baij, &bs));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "A_baij block size: %d\n", bs));  

  PetscCall(MatCreateVecs(A_baij, &b, NULL));  
  PetscCall(MatMult(A_baij, x0, b));
  PetscCall(VecDuplicate(b, &x));   


  
// petsc solver
  KSP         ksp;
  PC          pc;  
  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));  	
  PetscCall(KSPSetNormType(ksp, KSP_NORM_UNPRECONDITIONED));
  PetscCall(KSPSetTolerances(ksp,1.E-12,PETSC_DEFAULT,PETSC_DEFAULT,500));	 
  PetscCall(KSPSetOperators(ksp, A_baij, A_baij));
  PetscCall(KSPGetPC(ksp, &pc));  
  if(rank==0){
	  printf("rank=%d, using geoasm\n", rank);
  }
  PetscCall(PCRegister("geoasm", PCCreate_GEOASM));
  PetscCall(PCSetType(pc, "geoasm"));
  PetscCall(DMDACreate3d(PETSC_COMM_WORLD, DM_BOUNDARY_PERIODIC, DM_BOUNDARY_PERIODIC, DM_BOUNDARY_PERIODIC, DMDA_STENCIL_BOX, npx*nsize, npy*nsize, npz*nsize, npx,npy,npz, 3, geoasm_overlap, NULL,NULL,NULL, &da_asm));  
  PetscCall(DMSetFromOptions(da_asm));
  PetscCall(DMSetUp(da_asm));      
  PetscCall(PCSetDM(pc, NULL)); 
  PetscCall(PCSetDM(pc, da_asm)); 
  PetscCall(PCGASMSetOverlap(pc, geoasm_overlap));  

  PetscCall(KSPSetFromOptions(ksp));  
  //
  PetscBool ismg,isbddc,ismatis;
  PetscCall(PetscObjectTypeCompare((PetscObject)pc, PCBDDC, &isbddc));
  PetscCall(PetscObjectTypeCompare((PetscObject)pc, PCMG, &ismg));
  PetscCall(PetscObjectTypeCompare((PetscObject)A, MATIS, &ismatis));
  if (isbddc && !ismatis) {
	  Mat J;
      PetscCall(MatConvert(A, MATIS, MAT_INITIAL_MATRIX, &J));
      PetscCall(KSPSetOperators(ksp, A, J));
      PetscCall(MatDestroy(&J));
  }
  if(ismg){
	  PetscCall(KSPSetDM(ksp, da));
      PetscCall(KSPSetDMActive(ksp, PETSC_FALSE)); 
	  PCMGSetupViaCoarsen(pc, da);
  } 
 
  //
  PetscCall(KSPSetUp(ksp));  
  
 
  CreateGMRES_INFO();
 // we use the BiCGSTAB_GPU
   //ksp->ops->solve = KSPSolve_BiCGSTAB_GPU;
   //ksp->ops->solve = KSPSolve_CG_GPU;
   ksp->ops->solve = KSPSolve_GMRES_GPU;

  struct timeval start_time, end_time;
  PetscCall(gettimeofday(&start_time, NULL));
  PetscCall(KSPSolve(ksp, b, x));  
  PetscCall(gettimeofday(&end_time, NULL));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Time taken: %f milliseconds\n", (end_time.tv_sec - start_time.tv_sec) * 1000 + (end_time.tv_usec - start_time.tv_usec) / 1000.0));
  
  DestroyGMRES_INFO();
  //
  //PCASMprintsubA(pc);
  //
  //PetscCall(VecView(x,PETSC_VIEWER_STDOUT_WORLD)) ;

  free(Dval);
  free(Drow);
  free(Dcol);

  PetscCall(VecDuplicate(x, &r));
  PetscCall(VecCopy(x, r)); 
  PetscCall(VecAXPY(r, -1.0, x0));
  PetscCall(VecNorm(r, NORM_2, &norm));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "x-x0 norm %g\n", (double)norm));  
 
  PetscCall(DMDestroy(&da));   
  PetscCall(DMDestroy(&da_asm));   
  PetscCall(VecDestroy(&b));  
  PetscCall(VecDestroy(&r));  
  PetscCall(VecDestroy(&x)); 
  PetscCall(VecDestroy(&x0)); 
  PetscCall(MatDestroy(&A));  
  PetscCall(KSPDestroy(&ksp));   
 
  PetscCall(PetscFinalize());
  return 0;
}  


static PetscErrorCode PCMGSetupViaCoarsen(PC pc, DM da_fine)
{
  PetscInt              nlevels, k;
  PETSC_UNUSED PetscInt finest;
  DM                   *da_list, *daclist;
  Mat                   R;

  PetscFunctionBeginUser;
  nlevels = 1;
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-levels", &nlevels, 0));

  PetscCall(PetscMalloc1(nlevels, &da_list));
  for (k = 0; k < nlevels; k++) da_list[k] = NULL;
  PetscCall(PetscMalloc1(nlevels, &daclist));
  for (k = 0; k < nlevels; k++) daclist[k] = NULL;

  /* finest grid is nlevels - 1 */
  finest     = nlevels - 1;
  daclist[0] = da_fine;
  PetscCall(PetscObjectReference((PetscObject)da_fine));
  PetscCall(DMCoarsenHierarchy(da_fine, nlevels - 1, &daclist[1]));
  for (k = 0; k < nlevels; k++) {
    da_list[k] = daclist[nlevels - 1 - k];
	PetscCall(DMDASetInterpolationType(da_list[k], DMDA_Q0));  //DMDA_Q1  DMDA_Q0
    PetscCall(DMDASetUniformCoordinates(da_list[k], 0.0, 1.0, 0.0, 1.0, 0.0, 1.0));
  }

  PetscCall(PCMGSetLevels(pc, nlevels, NULL));
  PetscCall(PCMGSetType(pc, PC_MG_MULTIPLICATIVE));
  PetscCall(PCMGSetGalerkin(pc, PC_MG_GALERKIN_PMAT));

  for (k = 1; k < nlevels; k++) {
    PetscCall(DMCreateInterpolation(da_list[k - 1], da_list[k], &R, NULL));
    PetscCall(PCMGSetInterpolation(pc, k, R));
    PetscCall(MatDestroy(&R));
  }

  /* tidy up */
  for (k = 0; k < nlevels; k++) PetscCall(DMDestroy(&da_list[k]));
  PetscCall(PetscFree(da_list));
  PetscCall(PetscFree(daclist));
  PetscFunctionReturn(0);
}

PetscErrorCode PCASMprintsubA(PC pc)
{
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "ASM output !\n")); 
  KSP      *subksp;        
  PetscInt  i, nlocal, first;
  PC        subpc; 
  PetscCall(PCASMGetSubKSP(pc, &nlocal, &first, &subksp));
  for (i = 0; i < nlocal; i++) {
	  Mat AA,BB;
	  PetscInt asize;
	  PetscCall(KSPGetOperators(subksp[i], &AA,&BB));
	  PetscCall( MatGetSize(AA, &asize,NULL));
	  PetscErrorCode ierr;
	  MatInfo info;
      ierr = MatGetInfo(AA, MAT_GLOBAL_SUM, &info);CHKERRQ(ierr);
	  char straa[] = "amat", strbb[] ="pmat"; 
	  char resultaa[10], resultbb[10], numstr[10];
	  sprintf(numstr, "%d", first);
	  strcpy(resultaa, straa); 
	  strcat(resultaa, numstr);
	  strcpy(resultbb, strbb); 
	  strcat(resultbb, numstr); 
	  PetscCall(PetscSynchronizedPrintf(PETSC_COMM_WORLD, "domain id = %d ,size=%d, nnznum=%f  \n", first, asize, info.nz_used)); 
	  PetscCall(PetscSynchronizedFlush(PETSC_COMM_WORLD, PETSC_STDOUT));
      PetscViewer viewer;
	  PetscCall(PetscViewerASCIIOpen(PETSC_COMM_SELF , resultaa, &viewer));
	  PetscCall(MatView(AA, viewer));
	  PetscCall(PetscViewerDestroy(&viewer));
	  PetscCall(PetscViewerASCIIOpen(PETSC_COMM_SELF , resultbb, &viewer));
	  PetscCall(MatView(BB, viewer));
	  PetscCall(PetscViewerDestroy(&viewer));	   
  }
  PetscFunctionReturn(0);	
}


// mpirun -np 4 ./solveFDTD-dmda  -nnz 35460 -nsize 8 -nn 10 -dt 2.0 -npx 2 -npy 2 -npz 1 -fD D.txt -fx x.txt
// mpirun -np 8 ./solveFDTD-dmda  -nnz 215892 -nsize 16 -nn 18 -dt 16.0 -npx 2 -npy 2 -npz 2 -fD D-18-boundp.txt -fx x-18.txt
// mpirun  ./solveFDTD-dmda  -nnz 1532856 -nsize 32 -nn 34 -dt 16.0 -npx 4 -npy 4 -npz 4 -fD D-34-boundp.txt -fx_g x-128-subd64order.txt -ksp_view -ksp_monitor_true_residual -ksp_type bcgs -pc_type none
// mpirun -np 8 ./solveFDTD-dmda  -nnz 1277952 -nsize 32 -nn 34 -dt 16.0 -npx 2 -npy 2 -npz 2 -fD D-34-boundp.txt -fx_g x-64-subd8order.txt -ksp_view -ksp_monitor_true_residual -ksp_type bcgs -pc_type none -ksp_rtol 1.E-12 > out-size64-np8-bcgs-pcnone &
// mpirun -np 8 ./solveFDTD-mg  -nnz 1532856 -nsize 32 -nn 34 -dt 16.0 -npx 2 -npy 2 -npz 2 -fD D-34-boundp.txt -fx_g x-64-subd8order.txt -ksp_view -ksp_monitor_true_residual -ksp_type bcgs -pc_type mg -pc_mg_levels 2 -ksp_rtol 1.E-12 > out-size64-np8-bcgs-mg2 &

