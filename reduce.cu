

template<
    const int THREAD_PER_BLOCK,
    const int Block_num,
    const int Num_block
>
__global__ void reduce_bassline (float* input ,float* output){ // input is one dimensional
    __shared__ input_s[THREAD_PER_BLOCK ];
    unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;
    unsigned int tid = threadIdx.x;
    unsigned int warp_id = tid / 32;
    
    //load global memory to shared memory
    input_s[tid] = input[idx];
    __syncthreads();
    //reduce
    float ans;
    for(int stride = THREAD_PER_BLOCK / 2; stride > 32;stirde >>= 1){
        if(tid < stride){
            input_s[tid] += input_s[tid + stride];
        }
        __syncthreads();
    }
    
    //unroll 
    if(tid < 32){
        input_s[tid] += input_s[tid + 32];
        input_s[tid] += input_s[tid + 16];
        input_s[tid] += input_s[tid + 8];
        input_s[tid] += input_s[tid + 4];
        input_s[tid] += input_s[tid + 2];
        input_s[tid] += input_s[tid + 1];
    }
    //write back
    if(tid == 0){
        ans = input_s[0];
        output[blockIdx.x] = ans;
    }
    

}

template<const int blockSize>
__device__  __forceinline__ float shflin(float* sum){
    if(blockSize >= 32) sum += __shfl_down_sync(0xffffffff,sum,16);
    if(blockSize >= 16) sum += __shfl_down_sync(0xffffffff,sum,8);
    if(blockSize >= 8) sum += __shfl_down_sync(0xffffffff,sum,4);
    if(blockSize >= 4) sum += __shfl_down_sync(0xffffffff,sum,2);
    if(blockSize >= 2) sum += __shfl_down_sync(0xffffffff,sum,1);
    return sum;
}


template<const int blockSize>
__global__  void reduce(float* input,float* output,unsigned int n){
    unsigned int idx = threadIdx.y * (blockDim.x * 2)  + threadIdx.x;
    unsigned int tid = threadIdx.x;
    unsigned int gridSize = blockSize * 2 * gridDim.x;
    float sum = 0;
    while(i < n){
        input[i] += input[i + blockSize];
        i += gridSize;
    }
    static __shared__ ans[32];
    const int warpID = tid / 32;
    const int laneID = tid % 32;
    sum = shflin<blockSize>(sum);
    if(laneID == 0) ans[warpID] = sum;
    __syncthreads();
    
    sum = (threadIdx.x < (blockDim.x / 32) ) ? ans[laneID] : 0;

    //final reduce in first block
    if(warpID == 0) sum += shflin<blockSize / 32>(sum);

    if(tid == 0) output[blockIdx.x] = sum;
}