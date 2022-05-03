template<typename T>
__inline__ __device__ T cu_reduce_warp(T val) {
    val += __shfl_down_sync(0xffffffff, val, 16);
    val += __shfl_down_sync(0xffffffff, val, 8);
    val += __shfl_down_sync(0xffffffff, val, 4);
    val += __shfl_down_sync(0xffffffff, val, 2);
    val += __shfl_down_sync(0xffffffff, val, 1);
    return val;
}

template<typename T>
 void cu_reduce_warps(T const * inputs, unsigned int input_size, T * outputs) {
    T sum = 0;
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; 
            i < input_size; 
            i += blockDim.x * gridDim.x)
        sum += inputs[i];

    __shared__ T shared[32];
    unsigned int lane = threadIdx.x % warpSize;
    unsigned int wid = threadIdx.x / warpSize;

    sum = cu_reduce_warp(sum);
    if (lane == 0)
        shared[wid] = sum;

    // Wait for all partial reductions
    __syncthreads();

    sum = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;
    if (wid == 0)
        sum = cu_reduce_warp(sum);

    if (threadIdx.x == 0)
        outputs[blockIdx.x] = sum;
}
