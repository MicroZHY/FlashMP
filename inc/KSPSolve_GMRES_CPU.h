#ifndef __KSPSOLVE_GMRES_CPU__
#define __KSPSOLVE_GMRES_CPU__

#include <petscksp.h>
PetscErrorCode KSPGMRESBuildSoln_MARK(PetscScalar *nrs,Vec vs,Vec vdest,KSP ksp,PetscInt it);
PetscErrorCode KSPGMRESUpdateHessenberg_MARK(KSP ksp,PetscInt it,PetscBool hapend,PetscReal *res);
// PetscErrorCode KSPSolve_GMRES_CPU(KSP ksp);

#ifdef __cplusplus
extern "C"
{
PetscErrorCode KSPSolve_GMRES_CPU(KSP ksp);
}
#endif

#endif
