#ifndef CUDA_TIMER_H
#define CUDA_TIMER_H

 
#include <hip/hip_runtime.h>
#include <stdio.h>

class CudaTimer {
public:
    enum OperationType {
        TRIANGULAR_SOLVE = 0,
        DENSE_SOLVE = 1,
        SPMV_SCHUR = 2,
        NUM_OPERATIONS
    };

    static const int MAX_LEVELS = 6;

    // 获取单例实例（线程安全版本需加锁）
    static CudaTimer& getInstance();

    // 设置当前操作的上下文
    void setContext(int level, OperationType op);

    // 计时控制接口
    void start();
    void stop();

    // 统计管理接口
    static void resetStats();
    static void printStats(const char* output_filename);

    // 删除拷贝构造函数和赋值运算符
    CudaTimer(const CudaTimer&) = delete;
    void operator=(const CudaTimer&) = delete;

private:
    // 私有构造函数和析构函数
    CudaTimer();
    ~CudaTimer();

    hipEvent_t start_event_;
    hipEvent_t stop_event_;
    int current_level_;        // 0-based
    OperationType current_op_;
    bool is_running_;

    // 性能统计存储
    static double time_stats_[NUM_OPERATIONS][MAX_LEVELS];
    static int call_counts_[NUM_OPERATIONS][MAX_LEVELS];
    static const char* op_names_[NUM_OPERATIONS];
};

#endif // CUDA_TIMER_H
