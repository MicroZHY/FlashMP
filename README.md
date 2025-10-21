# FlashMP: Fast Discrete Transform-Based Solver for Preconditioning Maxwell's Equations on GPUs

[![arXiv](https://img.shields.io/badge/arXiv-2508.07193-b31b1b.svg)](https://arxiv.org/abs/2508.07193)
[![Conference](https://img.shields.io/badge/ICCD%202025-Accepted-green.svg)](https://www.iccd-conf.com/home.html)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## 📖 Overview

FlashMP is a high-performance preconditioning system designed for efficiently solving Maxwell's equations on GPUs. This project implements a novel discrete transform-based subdomain exact solver that achieves significant performance improvements for large-scale electromagnetic simulations.

### 🏆 Key Achievements


- **Performance**: Up to 16× reduction in iteration counts on AMD MI60 GPU clusters
- **Speedup**: 2.5× to 4.9× speedup compared to state-of-the-art libraries like Hypre
- **Scalability**: 84.1% parallel efficiency on 1000 GPUs

## 🚀 Core Features

### Algorithm Innovation
- **Discrete Transform Subdomain Solver**: Based on SVD decomposition of forward difference operators
- **Low-Rank Boundary Correction**: Using Woodbury formula for boundary condition handling
- **Tensor Product Optimization**: Efficient 3D tensor transformation operations

### GPU Acceleration
- **AMD GPU Support**: Optimized for ROCm/HIP platform
- **Memory Management**: Pre-allocated GPU memory to avoid allocation overhead
- **Sparse Matrix Optimization**: Efficient sparse operations using hipsparse

### Parallel Computing
- **PETSc Integration**: Full distributed computing support
- **Domain Decomposition**: Geometric domain decomposition preconditioner
- **MPI Optimization**: Efficient inter-process communication

## 📁 Project Structure

```
fdtd/
├── app/                                    # Main applications
│   ├── solveFDTD-DMDA-mg-geoAsm.cpp       # Main solver with FlashMP
│   ├── solveFDTD-DMDA.cpp                 # Simplified solver
│   ├── makefile                           # Build configuration
│   ├── subcast.sbatch                     # Job submission script
│   └── geoasm_subcast.sbatch              # Geometric ASM job script
├── inc/                                   # Header files
│   ├── fast_solve.h                       # FlashMP core algorithm
│   ├── KSPSolve_GMRES_GPU.h              # GPU GMRES solver
│   ├── KSPSolve_GMRES_CPU.h              # CPU GMRES solver
│   └── CudaTimer.h                       # GPU timer utilities
├── src/                                   # Source code
│   ├── fast_solve.cpp                    # FlashMP implementation
│   ├── KSPSolve_GMRES_GPU.cpp            # GPU GMRES implementation
│   ├── KSPSolve_GMRES_CPU.c              # CPU GMRES implementation
│   ├── geoasm.c                          # Geometric ASM preconditioner
│   ├── pbilu_fact_impl.cpp               # Point-block ILU factorization
│   ├── pre_ilu_impl.cpp                  # Pre-ILU implementation
│   ├── precond_impl.cpp                  # Preconditioner implementation
│   └── CudaTimer.cpp                     # GPU timer implementation
└── obj/                                   # Compiled object files
```

## 🛠️ Requirements

### Hardware Requirements
- **GPU**: AMD MI60 or compatible ROCm GPU
- **CPU**: Multi-core processor (32+ cores recommended)
- **Memory**: 128GB+ recommended
- **Network**: High-speed interconnect for multi-node parallel execution

### Software Dependencies
- **OS**: Linux (CentOS 7.6+ recommended)
- **Compilers**: GCC 7.0+, HIPCC (ROCm)
- **MPI**: OpenMPI 4.0+
- **Math Libraries**: 
  - PETSc 3.14+
  - ROCm 4.0+
  - hipblas, hipsparse
  - rocblas

## 📦 Installation

### 1. Environment Setup

```bash
# Install ROCm (example for ROCm 4.0)
wget https://repo.radeon.com/rocm/apt/4.0/pool/main/r/rocm-dkms/rocm-dkms_4.0.0.40100-1_all.deb
sudo dpkg -i rocm-dkms_4.0.0.40100-1_all.deb

# Install PETSc
wget https://ftp.mcs.anl.gov/pub/petsc/release-snapshots/petsc-lite-3.14.0.tar.gz
tar -xzf petsc-lite-3.14.0.tar.gz
cd petsc-3.14.0
./configure --with-hip=1 --with-hipc=hipcc --with-cc=mpicc --with-cxx=mpicxx
make all
```

### 2. Build FlashMP

```bash
# Clone the repository
git clone https://github.com/yourusername/flashmp.git
cd flashmp

# Update paths in makefile
vim app/makefile
# Update PETSC_DIR, HIP_BASE_PATH, and other paths

# Compile the project
cd app
make clean
make

# Run a test
mpirun -np 4 ./solveFDTD-DMDA-mg-geoAsm -nnz 35460 -nsize 8 -nn 10 -dt 2.0 -npx 2 -npy 2 -npz 1
```

### 3. Verify Installation

```bash
# Check GPU availability
rocm-smi

# Run simple test
mpirun -np 1 ./solveFDTD-DMDA-mg-geoAsm -nsize 16 -nn 18 -dt 16.0
```

## 🎯 Usage

### Basic Usage

```bash
# Single GPU execution
mpirun -np 1 ./solveFDTD-DMDA-mg-geoAsm \
    -nsize 32 -nn 34 -dt 16.0 \
    -fD D-34-boundp.txt -fx_g x-64-subd8order.txt \
    -ksp_type gmres -pc_type geoasm

# Multi-GPU parallel execution
mpirun -np 8 ./solveFDTD-DMDA-mg-geoAsm \
    -nsize 32 -nn 34 -dt 16.0 \
    -npx 2 -npy 2 -npz 2 \
    -fD D-34-boundp.txt -fx_g x-64-subd8order.txt \
    -ksp_type gmres -pc_type geoasm -geoasm_overlap 1
```

### Parameter Description

| Parameter | Description | Example Value |
|-----------|-------------|---------------|
| `-nsize` | Subdomain size | 32 |
| `-nn` | Total grid size | 34 |
| `-dt` | Time step size | 16.0 |
| `-npx, -npy, -npz` | Process distribution | 2, 2, 2 |
| `-geoasm_overlap` | ASM overlap layers | 1-3 |
| `-ksp_type` | Solver type | gmres, bcgs |
| `-pc_type` | Preconditioner type | geoasm |

### Advanced Configuration

```bash
# Using FlashMP preconditioner
mpirun -np 4 ./solveFDTD-DMDA-mg-geoAsm \
    -nsize 32 -nn 34 -dt 16.0 \
    -npx 2 -npy 2 -npz 1 \
    -ksp_type gmres -pc_type geoasm \
    -geoasm_overlap 2 \
    -ksp_rtol 1.E-12 \
    -ksp_monitor_true_residual
```

## 📊 Performance Benchmarks

### Test Environment
- **Hardware**: AMD MI60 GPU cluster
- **Software**: ROCm 4.0, PETSc 3.14
- **Scale**: 32³ to 1000 GPUs

### Performance Results

| Configuration | Iterations | Speedup | Parallel Efficiency |
|---------------|------------|---------|-------------------|
| NOPRE | 193 | 1.0× | 63.4% |
| FlashMP (overlap=1) | 20 | 3.05× | 77.8% |
| FlashMP (overlap=2) | 15 | 4.06× | 81.4% |
| FlashMP (overlap=3) | 12 | 4.56× | 84.1% |

### Running Benchmarks

```bash
# Weak scalability test
for np in 8 64 216 512 1000; do
    mpirun -np $np ./solveFDTD-DMDA-mg-geoAsm \
        -nsize 32 -nn 34 -dt 16.0 \
        -npx $((np/4)) -npy 2 -npz 2 \
        -ksp_type gmres -pc_type geoasm \
        -geoasm_overlap 2
done
```

## 🔬 Algorithm Principles

### FlashMP Core Algorithm

FlashMP achieves efficient solving through four main steps:

1. **Component Transformation**: Using SVD decomposition of forward difference operators
   ```
   D^f = U S V^T
   ```

2. **Point-wise Field Solving**: Decoupling 3n³×3n³ system into n³ 3×3 small systems
   ```
   B_ijk * [e_x, e_y, e_z]^T = [r_x, r_y, r_z]^T
   ```

3. **Component Inverse Transformation**: Restoring original variables

4. **Boundary Error Correction**: Using Woodbury formula for boundary conditions

### Complexity Analysis

- **Computational Complexity**: O(n⁴) vs O(n⁶) for traditional methods
- **Memory Complexity**: O(n⁴) vs O(n⁶) for traditional methods
- **Actual Speedup**: 128× computation reduction, 322× memory reduction

## 📚 Related Papers

- **Main Paper**: [FlashMP: Fast Discrete Transform-Based Solver for Preconditioning Maxwell's Equations on GPUs](https://arxiv.org/abs/2508.07193)
- **Conference**: The 43rd IEEE International Conference on Computer Design (ICCD 2025)
- **DOI**: https://doi.org/10.48550/arXiv.2508.07193

## 🤝 Contributing

We welcome contributions of all kinds!

### How to Contribute
1. Fork the project
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Reporting Issues
- Use GitHub Issues to report bugs
- Provide detailed reproduction steps
- Include system information and error logs

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Thanks to the Strategic Priority Research Program of Chinese Academy of Sciences (Grant NO.XDB0500101)
- Thanks to AMD for providing GPU hardware support
- Thanks to the PETSc development team for technical support
- Thanks to all contributors and users for feedback

## 📞 Contact

- **Homepage**: https://microzhy.github.io/
- **Paper Link**: https://arxiv.org/abs/2508.07193
- **Email**: zhanghaoyuan@cnic.cn

## 🔗 Related Links

- [PETSc Official Website](https://petsc.org/release/)
- [ROCm Development Platform](https://www.amd.com/en/products/software/rocm.html)
- [ICCD 2025 Conference](https://www.iccd-conf.com/home.html)

---

**⭐ If this project helps you, please give us a star!**