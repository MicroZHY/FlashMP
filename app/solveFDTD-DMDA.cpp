static char help[] = "Block Jacobi preconditioner for solving a linear system in parallel with KSP.\n\
The code indicates the\n\
procedures for setting the particular block sizes and for using different\n\
linear solvers on the individual blocks.\n\n";


#include <petsc.h>
//#include <petscksp.h>
#include <sys/time.h>
#include <petsc/private/pcimpl.h>
#include <petsc/private/kspimpl.h>
#include <../src/mat/impls/baij/mpi/mpibaij.h>
#include <../src/mat/impls/baij/seq/baij.h>
#include "KSPSolve_GMRES_CPU.h"
#include "KSPSolve_GMRES_GPU.h"
// scale the matrix and right hand side

#include <petscversion.h>

#if PETSC_VERSION_LT(3, 14, 0)
  #define PetscCall(func) do { PetscErrorCode ierr = (func); CHKERRQ(ierr); } while (0)
  #define PetscCallMPI(func) do { PetscErrorCode ierr = (func); CHKERRMPI(ierr); } while (0)
  #define PetscCheck(condition, comm, errcode, msg) if (!(condition)) SETERRQ(comm, errcode, msg)
#endif

PETSC_EXTERN PetscErrorCode PCCreate_GEOASM(PC pc);
PetscErrorCode PCASMprintsubA(PC pc);
 

int main(int argc, char **argv)
{
  Vec         x0, x, b, r;
  Mat         A, A_baij; // A_baij 是转换后的 MATMPIBAIJ 格式矩阵
  
  DM          da, da_asm;  
  PetscScalar norm;
  PetscInt    rank, size;
  PetscInt    nn = 10, i, j, k, idtmp, nsize, geoasm_overlap=1;
  PetscScalar deltat = 8.0; 
  char        nameD[PETSC_MAX_PATH_LEN] = "", namex[PETSC_MAX_PATH_LEN] = "";  // 子域 D 和 x
  char        namex_g[PETSC_MAX_PATH_LEN] = "";    // 全局解 x
  PetscBool   flg = PETSC_FALSE;  
  float       vtmp;  

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  printf("rank: %d, size: %d\n", rank, size);
  
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "initialize\n"));    
  
  PetscInt lengthx, lengthd = 1277952; // 非零元素数量
  PetscInt npx = size, npy = 1, npz = 1;
  PetscCall(PetscOptionsGetString(NULL, NULL, "-fD", nameD, sizeof(nameD), &flg));
  PetscCall(PetscOptionsGetString(NULL, NULL, "-fx", namex, sizeof(namex), &flg));  
  PetscCall(PetscOptionsGetString(NULL, NULL, "-fx_g", namex_g, sizeof(namex_g), &flg));  
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-nn", &nn, NULL));  // Matlab 中的边长（包含幽灵单元）
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-nsize", &nsize, NULL));  // 子域边长 = nn-2
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-dt", &deltat, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-nnz", &lengthd, NULL));  // D 中的非零元素数量
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-npx", &npx, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-npy", &npy, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-npz", &npz, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-geoasm_overlap", &geoasm_overlap, NULL));
 
  PetscCheck(nn - nsize == 2, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "error: nn-nsize !=2 ! ");
  PetscCheck(npx * npy * npz == size, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "error: npx*npy*npz !=size ! ");
  
  // if (rank == 0) {
  //   printf("nn: %d, nsize: %d, npx: %d, npy: %d, npz: %d\n", nn, nsize, npx, npy, npz);
  //   printf("lengthx: %d, lengthd: %d\n", lengthx, lengthd);
  // }

  lengthx = nn * nn * nn * 3;  // 问题规模
  PetscScalar xin[lengthx];
  // PetscScalar Dval[lengthd];
  double *Dval;
  Dval = (double *)malloc(lengthd * sizeof(double));
  PetscInt *Drow, *Dcol;
  Drow = (PetscInt *)malloc(lengthd * sizeof(PetscInt));
  Dcol = (PetscInt *)malloc(lengthd * sizeof(PetscInt));
  PetscInt  row, col;
  
  PetscScalar vv;
  if (rank == 0) {
    printf("i am here\n");
  }

  // 从文件读取子域 D
  if (rank == 0) {
    FILE *fp1 = fopen(nameD, "r");
    for (i = 0; i < lengthd; i++) {
      fscanf(fp1, "%d %d %g", &row, &col, &vtmp);
      Dval[i] = (double)vtmp;
      Drow[i] = row - 1;
      Dcol[i] = col - 1;
    }	  
    fclose(fp1);
  } 
  MPI_Bcast(Dval, lengthd, MPI_DOUBLE, 0, PETSC_COMM_WORLD);
  MPI_Bcast(Drow, lengthd, MPI_INT, 0, PETSC_COMM_WORLD);
  MPI_Bcast(Dcol, lengthd, MPI_INT, 0, PETSC_COMM_WORLD);
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "read local D from file %s !\n", nameD));
  

  if (rank == 0) {
    printf("i am here 2\n");
  }
  // 读取本地 x
  if (strcmp(namex, "") != 0) {  
    if (rank == 0) {
      FILE *fp2 = fopen(namex, "r"); 
      for (i = 0; i < lengthx; i++) {
        fscanf(fp2, "%g", &vtmp);
        xin[i] = (double)vtmp;
      } 
      fclose(fp2);
    } 
    MPI_Bcast(xin, lengthx, MPI_DOUBLE, 0, PETSC_COMM_WORLD);
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "read local x from file %s !\n", namex));  
  }
  if (rank == 0) {
    printf("i am here 3\n");
  }

  // 读取全局 x
  if (strcmp(namex_g, "") != 0) { 
    PetscInt nvar = nsize * nsize * nsize * 3;  
    if (rank == 0) {
      FILE *fp2 = fopen(namex_g, "r"); 
      rewind(fp2);
      for (j = 0; j < size; j++) {
        for (i = 0; i < nvar; i++) {
          fscanf(fp2, "%g", &vtmp);
          xin[i] = (double)vtmp;
        }
        if (j > 0) {
          MPI_Send(xin, nvar, MPI_DOUBLE, j, 0, PETSC_COMM_WORLD);
        }
      } 
      rewind(fp2);
      for (i = 0; i < nvar; i++) {
        fscanf(fp2, "%g", &vtmp);
        xin[i] = (double)vtmp;
      }		  
      fclose(fp2);
    } else {
      MPI_Recv(xin, nvar, MPI_DOUBLE, 0, 0, PETSC_COMM_WORLD, MPI_STATUS_IGNORE);
    } 
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "read global x from file %s !\n", namex_g));  
  }  
  if (rank == 0) {
    printf("i am here 4\n");
  }

  // 创建 3D DMDA
  PetscCall(DMDACreate3d(PETSC_COMM_WORLD, DM_BOUNDARY_PERIODIC, DM_BOUNDARY_PERIODIC, DM_BOUNDARY_PERIODIC, DMDA_STENCIL_BOX, npx * nsize, npy * nsize, npz * nsize, npx, npy, npz, 3, 1, NULL, NULL, NULL, &da));  
  PetscCall(DMSetFromOptions(da));
  PetscCall(DMSetUp(da));  

  // 初始化 x0
  PetscCall(DMCreateGlobalVector(da, &x0));   
  PetscScalar ***var_x0;
  PetscCall(DMDAVecGetArray(da, x0, &var_x0));
  PetscInt xs, ys, zs, xm, ym, zm;
  PetscCall(DMDAGetCorners(da, &xs, &ys, &zs, &xm, &ym, &zm));  
 
  idtmp = 0;
  for (k = zs; k < zs + zm; k++) {
    for (j = ys; j < ys + ym; j++) {
      for (i = xs; i < xs + xm; i++) {
        var_x0[k][j][3 * i + 0] = xin[idtmp++];
        var_x0[k][j][3 * i + 1] = xin[idtmp++];
        var_x0[k][j][3 * i + 2] = xin[idtmp++];			  
      }
    }
  }
  PetscCall(DMDAVecRestoreArray(da, x0, &var_x0)); 
  if (rank == 0) {
    printf("i am here 5\n");
  }
  
  // 创建矩阵 A
  PetscInt nlocalsize;
  PetscCall(VecGetLocalSize(x0, &nlocalsize));
  PetscCall(MatCreateAIJ(PETSC_COMM_WORLD, nlocalsize, nlocalsize, PETSC_DECIDE, PETSC_DECIDE, 13, NULL, 13, NULL, &A));  
  ISLocalToGlobalMapping map;    
  PetscCall(DMGetLocalToGlobalMapping(da, &map));
  PetscCall(MatSetLocalToGlobalMapping(A, map, map));  

  // 填充矩阵 A
  for (i = 0; i < lengthd; i++) {
    row = Drow[i] + 1;  // Matlab 中从 1 到 nn+2
    col = Dcol[i] - 1;  // 本地从 -1 到 nn
    vv = Dval[i];
    PetscInt ii, jj, kk;
    kk = (row - 1) / (3 * (nsize + 2) * (nsize + 2)) + 1;
    jj = (row - 1) % (3 * (nsize + 2) * (nsize + 2)) / (3 * (nsize + 2)) + 1;
    ii = (row - 1) % (3 * (nsize + 2)) / 3 + 1;
    ii = ii - 2;  // 从 [1, nx+2] 到 [-1, nx]
    jj = jj - 2;
    kk = kk - 2;
    if (ii >= 0 && ii < nsize && jj >= 0 && jj < nsize && kk >= 0 && kk < nsize) {
      row = row - 1; 
      col = col + 1;
      PetscCall(MatSetValuesLocal(A, 1, &row, 1, &col, &vv, INSERT_VALUES));
    }
  }

  // 组装矩阵 A
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));  

  // 调整矩阵 A: A = I + deltat^2/2 * D
  PetscCall(MatScale(A, deltat * deltat * 0.5));  
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

  // 创建向量 b 和 x
  PetscCall(MatCreateVecs(A_baij, &b, NULL));  
  PetscCall(MatMult(A_baij, x0, b));
  PetscCall(VecDuplicate(b, &x));   
  PetscCall(VecSet(x, 0.0));

  // 配置 KSP 求解器
  KSP ksp;
  PC  pc;  
  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));  	
  // PetscCall(KSPSetNormType(ksp, KSP_NORM_UNPRECONDITIONED));
  // PetscCall(KSPSetTolerances(ksp, 1.E-5, PETSC_DEFAULT, PETSC_DEFAULT, 500));	  
  PetscCall(KSPSetOperators(ksp, A_baij, A_baij));
  // PetscCall(KSPGetPC(ksp, &pc));  


 


  PetscCall(KSPSetFromOptions(ksp));
  PetscCall(KSPSetUp(ksp));  	  

  CreateGMRES_INFO();
 // we use the BiCGSTAB_GPU
   ksp->ops->solve = KSPSolve_BiCGSTAB_GPU;
  //ksp->ops->solve = KSPSolve_CG_GPU;
   // ksp->ops->solve = KSPSolve_GMRES_GPU;
   //ksp->ops->solve = KSPSolve_BiCG_GPU;
  // 记录求解时间
  struct timeval start_time, end_time;
  PetscCall(gettimeofday(&start_time, NULL));
  PetscCall(KSPSolve(ksp, b, x));  
  PetscCall(gettimeofday(&end_time, NULL));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Time taken: %f milliseconds\n", (end_time.tv_sec - start_time.tv_sec) * 1000 + (end_time.tv_usec - start_time.tv_usec) / 1000.0));
  
  DestroyGMRES_INFO();
  // 计算残差
  PetscCall(VecDuplicate(x, &r));
  PetscCall(VecCopy(x, r)); 
  PetscCall(VecAXPY(r, -1.0, x0));
  PetscCall(VecNorm(r, NORM_2, &norm));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "x-x0 norm %g\n", (double)norm));  

  // 清理资源
  PetscCall(DMDestroy(&da));   
  PetscCall(VecDestroy(&b));  
  PetscCall(VecDestroy(&r));  
  PetscCall(VecDestroy(&x)); 
  PetscCall(VecDestroy(&x0)); 
  PetscCall(MatDestroy(&A));  
  PetscCall(MatDestroy(&A_baij));  
  PetscCall(KSPDestroy(&ksp));   

  free(Dval);
  free(Drow);
  free(Dcol);
 
  PetscCall(PetscFinalize());
  return 0;
}
