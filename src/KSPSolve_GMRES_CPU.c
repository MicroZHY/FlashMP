#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <../src/ksp/ksp/impls/gmres/gmresimpl.h>
#include "KSPSolve_GMRES_CPU.h"
PetscErrorCode KSPGMRESBuildSoln_MARK(PetscScalar *nrs,Vec vs,Vec vdest,KSP ksp,PetscInt it)
{
  PetscScalar    tt;
  PetscErrorCode ierr;
  PetscInt       ii,k,j;
  KSP_GMRES      *gmres = (KSP_GMRES*)(ksp->data);

  PetscFunctionBegin;
  /* Solve for solution vector that minimizes the residual */

  /* If it is < 0, no gmres steps have been performed */
  if (it < 0) {
    ierr = VecCopy(vs,vdest);CHKERRQ(ierr); /* VecCopy() is smart, exists immediately if vguess == vdest */
    PetscFunctionReturn(0);
  }
  if (*HH(it,it) != 0.0) {
    nrs[it] = *GRS(it) / *HH(it,it);
  } else {
    ksp->reason = KSP_DIVERGED_BREAKDOWN;

    ierr = PetscInfo2(ksp,"Likely your matrix or preconditioner is singular. HH(it,it) is identically zero; it = %D GRS(it) = %g\n",it,(double)PetscAbsScalar(*GRS(it)));CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  for (ii=1; ii<=it; ii++) {
    k  = it - ii;
    tt = *GRS(k);
    for (j=k+1; j<=it; j++) tt = tt - *HH(k,j) * nrs[j];
    if (*HH(k,k) == 0.0) {
      ksp->reason = KSP_DIVERGED_BREAKDOWN;

      ierr = PetscInfo1(ksp,"Likely your matrix or preconditioner is singular. HH(k,k) is identically zero; k = %D\n",k);CHKERRQ(ierr);
      PetscFunctionReturn(0);
    }
    nrs[k] = tt / *HH(k,k);
  }

  /* Accumulate the correction to the solution of the preconditioned problem in TEMP */
  ierr = VecSet(VEC_TEMP,0.0);CHKERRQ(ierr);
  ierr = VecMAXPY(VEC_TEMP,it+1,nrs,&VEC_VV(0));CHKERRQ(ierr);

  ierr = KSPUnwindPreconditioner(ksp,VEC_TEMP,VEC_TEMP_MATOP);CHKERRQ(ierr);
  /* add solution to previous solution */
  if (vdest != vs) {
    ierr = VecCopy(vs,vdest);CHKERRQ(ierr);
  }
  ierr = VecAXPY(vdest,1.0,VEC_TEMP);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/*
   Do the scalar work for the orthogonalization.  Return new residual norm.
 */
PetscErrorCode KSPGMRESUpdateHessenberg_MARK(KSP ksp,PetscInt it,PetscBool hapend,PetscReal *res)
{
  PetscScalar *hh,*cc,*ss,tt;
  PetscInt    j;
  KSP_GMRES   *gmres = (KSP_GMRES*)(ksp->data);

  PetscFunctionBegin;
  hh = HH(0,it);
  cc = CC(0);
  ss = SS(0);

  /* Apply all the previously computed plane rotations to the new column
     of the Hessenberg matrix */
  for (j=1; j<=it; j++) {
    tt  = *hh;
    *hh = PetscConj(*cc) * tt + *ss * *(hh+1);
    hh++;
    *hh = *cc++ * *hh - (*ss++ * tt);
  }

  /*
    compute the new plane rotation, and apply it to:
     1) the right-hand-side of the Hessenberg system
     2) the new column of the Hessenberg matrix
    thus obtaining the updated value of the residual
  */
  if (!hapend) {
    tt = PetscSqrtScalar(PetscConj(*hh) * *hh + PetscConj(*(hh+1)) * *(hh+1));
    if (tt == 0.0) {
      ksp->reason = KSP_DIVERGED_NULL;
      PetscFunctionReturn(0);
    }
    *cc        = *hh / tt;
    *ss        = *(hh+1) / tt;
    *GRS(it+1) = -(*ss * *GRS(it));
    *GRS(it)   = PetscConj(*cc) * *GRS(it);
    *hh        = PetscConj(*cc) * *hh + *ss * *(hh+1);
    *res       = PetscAbsScalar(*GRS(it+1));
  } else {
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

PetscErrorCode KSPSolve_GMRES_CPU(KSP ksp)
{
  PetscErrorCode ierr;
  PetscInt       its,itcount;
  KSP_GMRES      *gmres     = (KSP_GMRES*)ksp->data;
  PetscBool      guess_zero = ksp->guess_zero;

  PC  pc;
  Mat 			Amat,Pmat;
  PetscReal		res_norm,res,hapbnd,tt;
  PetscBool		hapend;
  PetscInt		it;  // it and its in inner cycle
  PetscInt		max_k=gmres->max_k;
  PetscInt		j;
  PetscScalar	*hh,*hes,*lhh;

  PetscReal *hveci=NULL;
  PetscReal *hvecj=NULL;
  PetscInt irow;

  PetscInt		rank;
  struct timeval tstart,tend;
  double telapse;
  PetscFunctionBegin;
  if (ksp->calc_sings && !gmres->Rsvd) SETERRQ(PetscObjectComm((PetscObject)ksp),PETSC_ERR_ORDER,"Must call KSPSetComputeSingularValues() before KSPSetUp() is called");

  ierr     = PetscObjectSAWsTakeAccess((PetscObject)ksp);CHKERRQ(ierr);
  ksp->its = 0;
  ierr     = PetscObjectSAWsGrantAccess((PetscObject)ksp);CHKERRQ(ierr);

  itcount     = 0;
  gmres->fullcycle = 0;
  ksp->reason = KSP_CONVERGED_ITERATING;
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);

  while (!ksp->reason) {
	//******************************S1: calculating: b-Ax:**************************//   
	//VEC_TEMP=A*vec_sol, VEC_TEMP_MATOP=vec_rhs, VEC_TMP_MATOP=VEC_TMP_MATOP-VEC_TEMP
	if(!ksp->pc){KSPGetPC(ksp,&ksp->pc);}
	PCGetOperators(ksp->pc,&Amat,&Pmat);
  
	MPI_Barrier(MPI_COMM_WORLD);
  	gettimeofday(&tstart,NULL);
	if(itcount)
	{
		MatMult(Amat,ksp->vec_sol,VEC_TEMP);
		VecCopy(ksp->vec_rhs,VEC_TEMP_MATOP);VecAXPY(VEC_TEMP_MATOP,-1.0,VEC_TEMP);
		VecCopy(VEC_TEMP_MATOP,VEC_VV(0));
	}
	else{VecCopy(ksp->vec_rhs,VEC_TEMP_MATOP);VecCopy(ksp->vec_rhs,VEC_VV(0));}

	//******************************       END OF S1       ************************//

	//**************S2:Check norm AND enter KSPGMRESCycle(&its,ksp); *******************//
	//******************************S2-1: CHECK NORM       *****************************//
	it=0;	its=0;	hapend=PETSC_FALSE;
	ierr=VecNormalize(VEC_VV(0),&res_norm);CHKERRQ(ierr);
	KSPCheckNorm(ksp,res_norm);
	res     = res_norm;
	*GRS(0) = res_norm;
   /* check for the convergence */
	ierr=PetscObjectSAWsTakeAccess((PetscObject)ksp);CHKERRQ(ierr);
	ksp->rnorm = res;
	ierr=PetscObjectSAWsGrantAccess((PetscObject)ksp);CHKERRQ(ierr);
	gmres->it  = (it - 1);
	ierr=KSPLogResidualHistory(ksp,res);CHKERRQ(ierr);
	ierr=KSPMonitor(ksp,ksp->its,res);CHKERRQ(ierr);
	if (!res) {
		ksp->reason = KSP_CONVERGED_ATOL;
		ierr=PetscInfo(ksp,"Converged due to zero residual norm on entry\n");CHKERRQ(ierr);
	    PetscFunctionReturn(0);
	   }
   	ierr=(*ksp->converged)(ksp,ksp->its,res,&ksp->reason,ksp->cnvP);CHKERRQ(ierr);
	if(!rank){printf("it=%d,res=%22.15e\n",it,res);}
	
	//******************************S2-2:  ENTER CYCLE ******************************//
	while (!ksp->reason && it < max_k && ksp->its < ksp->max_it) {
		if (it) {
			ierr=KSPLogResidualHistory(ksp,res);CHKERRQ(ierr);
			ierr=KSPMonitor(ksp,ksp->its,res);CHKERRQ(ierr);
			//if(!rank){printf("it=%d,res=%22.15e\n",it,res);}
	  	}
	  	gmres->it = (it - 1);
	  	// NOTE: we need to make sure gmres->preallocatevec is 1 
	  	// PCApply: M^-1 *VV(it)= tmp  VV(it+1)=A*tmp -> operations on VV(it+1).....
	  	// if we do not use PC, that means PCNONE and tmp=VV(it)
	  	// PCNONE CASE: VV(it+1)=A*VV(it); need to call PCApply when PCASM or PCBJacobi
	//  	ierr=MatMult(Amat,VEC_VV(it),VEC_VV(it+1));CHKERRQ(ierr);
	//
	//	try right preconditioner:
		//pc=ksp->pc;
		PCSetUp(ksp->pc);
		PCApply(ksp->pc,VEC_VV(it),VEC_TEMP_MATOP);
		//if(it==0)
		if(0)
		{
			VecGetArray(VEC_TEMP_MATOP,&hveci);
			VecGetArray(VEC_VV(it),&hvecj);
			for(irow=0;irow<54;irow++)
			{printf("rank=%d,hveci[%d]=%22.15lf,hvecj[%d]=%22.15lf\n",rank,irow,hveci[irow],irow,hvecj[irow]);}
			VecRestoreArray(VEC_TEMP_MATOP,&hveci);
			VecRestoreArray(VEC_VV(it),&hvecj);
		}
		MatMult(Amat,VEC_TEMP_MATOP,VEC_VV(it+1));
		
		// do gramschmidtOrthogonalization
		if(!gmres->orthogwork){PetscMalloc1(gmres->max_k+2,&gmres->orthogwork);}
		lhh=gmres->orthogwork;
		 /* update Hessenberg matrix and do unmodified Gram-Schmidt */
		hh  = HH(0,it);
		hes = HES(0,it);
		/* Clear hh and hes since we will accumulate values into them */
		for (j=0; j<=it; j++) {
		    hh[j]  = 0.0;
		    hes[j] = 0.0;
		}
		VecMDot(VEC_VV(it+1),it+1,&(VEC_VV(0)),lhh); // <v,vnew> 
		for(j=0;j<=it;j++){lhh[j] = -lhh[j];}
		VecMAXPY(VEC_VV(it+1),it+1,lhh,&VEC_VV(0));
	 	// note lhh[j] is -<v,vnew> , hence the subtraction
		for (j=0; j<=it; j++) { hh[j]  -= lhh[j];hes[j] -= lhh[j];}
		if(ksp->reason){break;}
		// vv(i+1) . vv(i+1)
		VecNormalize(VEC_VV(it+1),&tt);
		KSPCheckNorm(ksp,tt);
		// save the magnitude
		*HH(it+1,it)  = tt;
		*HES(it+1,it) = tt;
		// check for the happy breakdown 
		hapbnd = PetscAbsScalar(tt / *GRS(it));
		if (hapbnd > gmres->haptol){ hapbnd = gmres->haptol;}
		if (tt < hapbnd) {
		   PetscInfo2(ksp,"Detected happy breakdown, current hapbnd = %14.12e tt = %14.12e\n",(double)hapbnd,(double)tt);
		   hapend = PETSC_TRUE;}
		KSPGMRESUpdateHessenberg_MARK(ksp,it,hapend,&res);   

		it++;
		gmres->it=(it-1);
		ksp->its++;
		ksp->rnorm=res;
		if(ksp->reason){break;}
		(*ksp->converged)(ksp,ksp->its,res,&ksp->reason,ksp->cnvP);

		/* Catch error in happy breakdown and signal convergence and break from loop */
		if (hapend) {
		   if (ksp->normtype == KSP_NORM_NONE) {ksp->reason = KSP_CONVERGED_HAPPY_BREAKDOWN;}
		   else if (!ksp->reason) {
		     if (ksp->errorifnotconverged) {SETERRQ1(PetscObjectComm((PetscObject)ksp),PETSC_ERR_NOT_CONVERGED,"You reached the happy break down, but convergence was not indicated. Residual norm = %g",(double)res);}
		     else {ksp->reason = KSP_DIVERGED_BREAKDOWN;break;}
		   }
		 }

	}
	//************************END OF S2***********************************
//	gettimeofday(&tend,NULL);
//	telapse=(double) ((tend.tv_sec*1000000.0 + tend.tv_usec)-(tstart.tv_sec*1000000.0+tstart.tv_usec));
//        telapse=telapse/1000.0;
//        printf("rank=%d,telapse=%22.15lf ms \n",rank,telapse);	
	//*************************S3: We need to build solution *************
   	/* Monitor if we know that we will not return for a restart */
    if (it && (ksp->reason || ksp->its >= ksp->max_it)) 
   	{
    	 KSPLogResidualHistory(ksp,res);
     	KSPMonitor(ksp,ksp->its,res);
		//if(!rank){printf("it=%d,res=%22.15e\n",it,res);}
   	}
   	its = it;

   	//build solution
    KSPGMRESBuildSoln_MARK(GRS(0),ksp->vec_sol,ksp->vec_sol,ksp,it-1);

    if (its == gmres->max_k) {gmres->fullcycle++;}
    itcount += its;
    if (itcount >= ksp->max_it) {if (!ksp->reason) {ksp->reason = KSP_DIVERGED_ITS;} break;}

	// every future call to KSPInitialResidual() will have nonzero guess
    ksp->guess_zero = PETSC_FALSE; 
	gettimeofday(&tend,NULL);
	telapse=(double) ((tend.tv_sec*1000000.0 + tend.tv_usec)-(tstart.tv_sec*1000000.0+tstart.tv_usec));
        telapse=telapse/1000.0;
        printf("rank=%d,telapse=%22.15lf ms \n",rank,telapse);	
  }
  ksp->guess_zero = guess_zero; // restore if user provided nonzero initial guess
  PetscFunctionReturn(0);
}


