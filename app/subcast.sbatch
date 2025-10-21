#!/bin/bash

#作业模板
#时间：2021.4

#SBATCH -J cast                 #指定作业名称
#SBATCH -p  kshdnormal         #指定队列/分区名称
##SBATCH -p kshcnormal
#SBATCH -n 64 
#SBATCH -N 16
#SBATCH --gres=dcu:4
#SBATCH --ntasks-per-node=4        #指定每节点的任务数量
#SBATCH --cpus-per-task=7          #指定每个任务使用CPU数量
####SBATCH --mem=100G          #指定每节点申请的内存大小。可以根据程序需要，自由指定。最大值为100GB。
#SBATCH --exclusive                 #指定独占节点
#SBATCH -o log/test_jobid%j.log         #指定标准输出文件名称
#SBATCH -e log/test_jobid%j.log             #指定报错信息输出文件名称



echo "Start time: `date`"                           #显示开始时间
echo "SLURM_JOB_ID: $SLURM_JOB_ID"                  #显示作业号
echo "SLURM_NNODES: $SLURM_NNODES"                  #显示节点数 
echo "SLURM_NTASKS: $SLURM_NTASKS"                  #显示总任务数
echo "SLURM_TASKS_PER_NODE: $SLURM_TASKS_PER_NODE"      #显示每节点的任务数
echo "SLURM_CPUS_PER_TASK: $SLURM_CPUS_PER_TASK"        #显示每个任务使用的CPU数量
echo "SLURM_JOB_PARTITION: $SLURM_JOB_PARTITION"        #显示队列/分区名称
echo "SLURM_SUBMIT_DIR:$SLURM_SUBMIT_DIR"           #显示提交作业目录的路径
echo "SLURM_NODELIST:$SLURM_NODELIST"               #显示执行节点列表名称
# 完全关闭UCX警告（推荐）
export UCX_WARNINGS=disable
export UCX_LOG_LEVEL=error


#执行程序

#mpirun  ./solveFDTD-DMDA  -nnz 1277952 -nsize 30 -nn 32 -dt 16.0 -npx 4 -npy 4 -npz 4 -fD D-32-boundp.txt -fx x-32.txt -ksp_view -ksp_monitor_true_residual -ksp_type bcgs -pc_type none  
#mpirun  ./solveFDTD-DMDA  -nnz 1532856 -nsize 32 -nn 34 -dt 16.0 -npx 4 -npy 4 -npz 4 -fD D-34-boundp.txt -fx_g x-128-subd64order.txt -ksp_rtol 1.0E-12 -ksp_view -ksp_monitor_true_residual -ksp_type bcgs -pc_type none  
#mpirun  ./solveFDTD-DMDA  -nnz 1532856 -nsize 32 -nn 34 -dt 16.0 -npx 4 -npy 4 -npz 4 -fD D-34-boundp.txt -fx_g x-128-subd64order.txt -ksp_rtol 1.0E-12 -ksp_view -ksp_monitor_true_residual -ksp_type bcgs -pc_type bjacobi -sub_pc_type lu  
#mpirun  ./solveFDTD-DMDA  -nnz 1532856 -nsize 32 -nn 34 -dt 16.0 -npx 4 -npy 4 -npz 4 -fD D-34-boundp.txt -fx_g x-128-subd64order.txt -ksp_rtol 1.0E-12 -ksp_view -ksp_monitor_true_residual -ksp_type bcgs  -pc_type asm -pc_asm_overlap 3  -sub_pc_type lu  


mpirun   ./solveFDTD-DMDA  -nnz 1532856 -nsize 32 -nn 34 -dt 16.0 -npx 4 -npy 4 -npz 4 -fD D-34-boundp.txt -fx_g x-128-subd64order.txt -ksp_rtol 1.0E-12   -ksp_monitor_true_residual  -vecscatter_type mpi1  -ksp_type gmres -ksp_pc_side right  -pc_type bjacobi  -ksp_gmres_preallocate   -ksp_max_it 500 -sub_ksp_type preonly -sub_pc_type ilu   -ksp_converged_reason     

#mpirun   ./solveFDTD-DMDA  -nnz 1532856 -nsize 32 -nn 34 -dt 16.0 -npx 4 -npy 4 -npz 4 -fD D-34-boundp.txt -fx_g x-128-subd64order.txt -ksp_rtol 1.0E-12   -ksp_monitor_true_residual  -vecscatter_type mpi1  -ksp_type gmres -ksp_pc_side right  -pc_type asm -pc_asm_overlap 1   -ksp_gmres_preallocate   -ksp_max_it 500 -sub_ksp_type preonly -sub_pc_type ilu   -ksp_converged_reason     

#mpirun   ./solveFDTD-DMDA  -nnz 1532856 -nsize 32 -nn 34 -dt 16.0 -npx 4 -npy 4 -npz 4 -fD D-34-boundp.txt -fx_g x-128-subd64order.txt -ksp_rtol 1.0E-12   -ksp_monitor_true_residual  -vecscatter_type mpi1  -ksp_type gmres -ksp_pc_side right  -pc_type asm -pc_asm_overlap 2   -ksp_gmres_preallocate   -ksp_max_it 500 -sub_ksp_type preonly -sub_pc_type ilu   -ksp_converged_reason  

#mpirun   ./solveFDTD-DMDA  -nnz 1532856 -nsize 32 -nn 34 -dt 16.0 -npx 4 -npy 4 -npz 4 -fD D-34-boundp.txt -fx_g x-128-subd64order.txt -ksp_rtol 1.0E-12   -ksp_monitor_true_residual  -vecscatter_type mpi1  -ksp_type gmres -ksp_pc_side right  -pc_type asm -pc_asm_overlap 3   -ksp_gmres_preallocate   -ksp_max_it 500 -sub_ksp_type preonly -sub_pc_type ilu   -ksp_converged_reason    


#mpirun   ./solveFDTD-DMDA  -nnz 1532856 -nsize 32 -nn 34 -dt 16.0 -npx 4 -npy 4 -npz 4 -fD D-34-boundp.txt -fx_g x-128-subd64order.txt -ksp_rtol 1.0E-12 -ksp_view -ksp_monitor_true_residual -ksp_type gmres   -pc_type none   -ksp_max_it 500
#mpirun  ./solveFDTD-DMDA  -nnz 1532856 -nsize 32 -nn 34 -dt 16.0 -npx 4 -npy 4 -npz 4 -fD D-34-boundp.txt -fx_g x-128-subd64order.txt -ksp_rtol 1.0E-12 -ksp_view -ksp_monitor_true_residual -ksp_type bcgs  -ksp_converged_reason  -pc_type bjacobi -sub_pc_type lu  -ksp_max_it 500
#mpirun  ./solveFDTD-DMDA  -nnz 1532856 -nsize 32 -nn 34 -dt 16.0 -npx 4 -npy 4 -npz 4 -fD D-34-boundp.txt -fx_g x-128-subd64order.txt -ksp_rtol 1.0E-12 -ksp_view -ksp_monitor_true_residual -ksp_type gmres    -pc_type asm -pc_asm_overlap 0  -sub_pc_type lu   -ksp_max_it 500



echo "End time: `date`"  #显示结束时间


