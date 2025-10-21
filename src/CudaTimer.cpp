#include "CudaTimer.h"
#include <cstdio>

// 初始化静态成员
double CudaTimer::time_stats_[CudaTimer::NUM_OPERATIONS][CudaTimer::MAX_LEVELS] = {0};
int CudaTimer::call_counts_[CudaTimer::NUM_OPERATIONS][CudaTimer::MAX_LEVELS] = {0};
const char *CudaTimer::op_names_[] = {
    "Triangular Solve", "Dense Solve", "SpMV (Schur)"};

CudaTimer::CudaTimer() : current_level_(0), current_op_(TRIANGULAR_SOLVE), is_running_(false)
{
    hipError_t err = hipEventCreate(&start_event_);
    if (err != hipSuccess)
    {
        printf("CUDA Error in hipEventCreate(start): %s\n", hipGetErrorString(err));
    }

    err = hipEventCreate(&stop_event_);
    if (err != hipSuccess)
    {
        printf("CUDA Error in hipEventCreate(stop): %s\n", hipGetErrorString(err));
    }
}

CudaTimer::~CudaTimer()
{
    // 无论计时器是否运行，都应该销毁事件对象
    hipError_t err = hipEventDestroy(start_event_);
    if (err != hipSuccess)
    {
        printf("CUDA Error in hipEventDestroy(start): %s\n", hipGetErrorString(err));
    }

    err = hipEventDestroy(stop_event_);
    if (err != hipSuccess)
    {
        printf("CUDA Error in hipEventDestroy(stop): %s\n", hipGetErrorString(err));
    }
}

CudaTimer &CudaTimer::getInstance()
{
    static CudaTimer instance; // C++11保证线程安全的本地静态变量
    return instance;
}

void CudaTimer::setContext(int user_level, OperationType op)
{
    // 输入层级是1-based，转换为0-based存储
    current_level_ = user_level - 1;

    // 层级有效性检查
    if (current_level_ < 0)
    {
        printf("WARNING: Invalid level %d, using minimum level 1\n", user_level);
        current_level_ = 0;
    }
    else if (current_level_ >= MAX_LEVELS)
    {
        printf("WARNING: Exceed max level %d, using max level %d\n",
               user_level, MAX_LEVELS);
        current_level_ = MAX_LEVELS - 1;
    }

    current_op_ = op;
}

void CudaTimer::start()
{
    if (!is_running_)
    {
        // 确保之前的操作已完成
        hipDeviceSynchronize();

        hipError_t err = hipEventRecord(start_event_);
        if (err != hipSuccess)
        {
            printf("CUDA Error in hipEventRecord(start): %s\n", hipGetErrorString(err));
            return;
        }

        is_running_ = true;
    }
    else
    {
        printf("WARNING: Timer is already running!\n");
    }
}

void CudaTimer::stop()
{
    if (is_running_)
    {
        hipError_t err = hipEventRecord(stop_event_);
        if (err != hipSuccess)
        {
            printf("CUDA Error in hipEventRecord(stop): %s\n", hipGetErrorString(err));
            return;
        }

        err = hipEventSynchronize(stop_event_);
        if (err != hipSuccess)
        {
            printf("CUDA Error in hipEventSynchronize: %s\n", hipGetErrorString(err));
            return;
        }

        float elapsed_ms = 0.0f;
        err = hipEventElapsedTime(&elapsed_ms, start_event_, stop_event_);
        if (err != hipSuccess)
        {
            printf("CUDA Error in hipEventElapsedTime: %s\n", hipGetErrorString(err));
            return;
        }

        // 累加统计信息
        time_stats_[current_op_][current_level_] += elapsed_ms;
        call_counts_[current_op_][current_level_]++;

        // 调试输出（可以通过编译选项控制是否启用）
#ifdef CUDA_TIMER_DEBUG
        printf("[PERF] %s @ Level %d: %.2f ms (Total: %.2f ms)\n",
               op_names_[current_op_], current_level_ + 1,
               elapsed_ms, time_stats_[current_op_][current_level_]);
#endif

        is_running_ = false;
    }
    else
    {
        printf("WARNING: Timer is not running!\n");
    }
}

void CudaTimer::resetStats()
{
    for (int op = 0; op < NUM_OPERATIONS; ++op)
    {
        for (int lvl = 0; lvl < MAX_LEVELS; ++lvl)
        {
            time_stats_[op][lvl] = 0.0;
            call_counts_[op][lvl] = 0;
        }
    }
}

// void CudaTimer::printStats() {
//     printf("\n=== Performance Statistics ===\n");
//     printf("%-6s | %-20s | %-20s | %-20s\n",
//            "Level", "Triangular Solve", "Dense Solve", "SpMV (Schur)");
//     printf("%-6s | %-8s %-8s | %-8s %-8s | %-8s %-8s\n",
//            "", "Calls", "Time", "Calls", "Time", "Calls", "Time");
//     printf("---------------------------------------------------------------\n");

//     for (int lvl = 0; lvl < MAX_LEVELS; ++lvl) {
//         printf("%-6d | ", lvl + 1);  // 显示1-based层级

//         for (int op = 0; op < NUM_OPERATIONS; ++op) {
//             if (call_counts_[op][lvl] > 0) {
//                 printf("%-8d %-8.2f | ",
//                        call_counts_[op][lvl], time_stats_[op][lvl]);
//             } else {
//                 printf("%-8s %-8s | ", "-", "-");
//             }
//         }
//         printf("\n");
//     }

//     // 总计计算
//     printf("\n[Total]\n");
//     for (int op = 0; op < NUM_OPERATIONS; ++op) {
//         double total_time = 0.0;
//         int total_calls = 0;
//         for (int lvl = 0; lvl < MAX_LEVELS; ++lvl) {
//             total_time += time_stats_[op][lvl];
//             total_calls += call_counts_[op][lvl];
//         }
//         printf("%-12s: Calls=%-6d TotalTime=%-10.2fms Avg=%.2fms\n",
//                op_names_[op], total_calls, total_time,
//                total_calls > 0 ? total_time / total_calls : 0.0);
//     }
//     printf("===============================================================\n");
// }

void CudaTimer::printStats(const char *output_filename)
{
    // 打开文件以追加模式写入
    FILE *file = fopen(output_filename, "a");
    if (file == NULL)
    {
        printf("Error: Unable to open the output file.\n");
        return;
    }

    // 打印和写入标题
    printf("\n=== Performance Statistics ===\n");
    fprintf(file, "\n=== Performance Statistics ===\n");

    printf("%-6s | %-20s | %-20s | %-20s\n",
           "Level", "Triangular Solve", "Dense Solve", "SpMV (Schur)");
    fprintf(file, "%-6s | %-20s | %-20s | %-20s\n",
            "Level", "Triangular Solve", "Dense Solve", "SpMV (Schur)");

    printf("%-6s | %-8s %-8s | %-8s %-8s | %-8s %-8s\n",
           "", "Calls", "Time", "Calls", "Time", "Calls", "Time");
    fprintf(file, "%-6s | %-8s %-8s | %-8s %-8s | %-8s %-8s\n",
            "", "Calls", "Time", "Calls", "Time", "Calls", "Time");

    printf("---------------------------------------------------------------\n");
    fprintf(file, "---------------------------------------------------------------\n");

    // 遍历每一层
    for (int lvl = 0; lvl < MAX_LEVELS; ++lvl)
    {
        printf("%-6d | ", lvl); // 显示1-based层级
        fprintf(file, "%-6d | ", lvl);

        for (int op = 0; op < NUM_OPERATIONS; ++op)
        {
            if (call_counts_[op][lvl] > 0)
            {
                printf("%-8d %-8.2f | ",
                       call_counts_[op][lvl], time_stats_[op][lvl]);
                fprintf(file, "%-8d %-8.2f | ",
                        call_counts_[op][lvl], time_stats_[op][lvl]);
            }
            else
            {
                printf("%-8s %-8s | ", "-", "-");
                fprintf(file, "%-8s %-8s | ", "-", "-");
            }
        }
        printf("\n");
        fprintf(file, "\n");
    }

    // 总计计算
    printf("\n[Total]\n");
    fprintf(file, "\n[Total]\n");

    for (int op = 0; op < NUM_OPERATIONS; ++op)
    {
        double total_time = 0.0;
        int total_calls = 0;
        for (int lvl = 0; lvl < MAX_LEVELS; ++lvl)
        {
            total_time += time_stats_[op][lvl];
            total_calls += call_counts_[op][lvl];
        }
        printf("%-12s: Calls=%-6d TotalTime=%-10.2fms Avg=%.2fms\n",
               op_names_[op], total_calls, total_time,
               total_calls > 0 ? total_time / total_calls : 0.0);
        fprintf(file, "%-12s: Calls=%-6d TotalTime=%-10.2fms Avg=%.2fms\n",
                op_names_[op], total_calls, total_time,
                total_calls > 0 ? total_time / total_calls : 0.0);
    }

    printf("===============================================================\n");
    fprintf(file, "===============================================================\n");

    // 关闭文件
    fclose(file);
}
