#ifndef UTILS_CUH
#define UTILS_CUH

#define cudaCheckErrors(msg)                                                                                                                                   \
    do                                                                                                                                                         \
    {                                                                                                                                                          \
        cudaError_t __err = cudaGetLastError();                                                                                                                \
        if (__err != cudaSuccess)                                                                                                                              \
        {                                                                                                                                                      \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", msg, cudaGetErrorString(__err), __FILE__, __LINE__);                                            \
            exit(1);                                                                                                                                           \
        }                                                                                                                                                      \
    }                                                                                                                                                          \
    while (0)

#endif