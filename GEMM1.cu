#include<cuda_runtime.h>
template<
    const int Block_A_X,
    const int Block_A_Y,
    const int Block_B_X,
    const int Thread_A,
    const int Thread_B
    >
#define OFFSET(row,col,width) (row * width + col)
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*> (&(pointer))[0])
__global__ void GEMM(float* A,float* B,float* C,const int M,const int N,const int K){
    //make data idx
    int tid = threadIdx,x;
    int thread_per_row_inblockA = Block_A_X / Thread_A;
    int thread_per_row_inblockB = Block_B_X / Thread_B;
    int thread_per_col_inblockA = Block_A_Y / Thread_A;
    int thread_per_col_inblockB = Block_A_X / Thread_B;
    const int threads_num_inblockA = thread_per_row_inblockA * thread_per_col_inblockA;
    const int threads_num_inblockB = thread_per_row_inblockB * thread_Per_col_inblockB; 
    const int A_data_X = tid % thread_per_row_inblockA * 4;
    const int A_data_Y = tid / thread_per_row_inblockA * 4;
    const int B_data_X = tid % thread_per_row_inblockB * 4;
    const int B_data_Y = tid / thread_per_row_inblockB * 4;
    
    // allocate memory
    __shared__ float  As[2][Block_A_X][block_A_Y];
    __shared__ float  Bs[2][Block_A_X][Block_B_X];
    
    float regA_cal[2][Thread_A];
    float regB_cal[2][Thread_B];
    float temp_ans[Thread_A][Thread_B]

    const int idg_A = (Block_A_X * Block_A_Y) / threads_num_inblockA; 
    const int idg_B = (Block_A_X * Block_B_X) / threads_num_inblockB;
    float regA_mov[idg_A];
    float regB_mov[idg_B];

    const int stride_A = threads_num_inblockA / thread_per_row_inblockA;
    const int stride_B = threads_num_inblockB / thread_per_row_inblockB;

    A = &(A[(Block_A_Y * blockIdx.y) * K ]);
    B = &(B[Block_B_X * blockIdx.x]);
    //load global memory A to share memory0 
    #pragma unroll
    for(int i = 0 ; i < Block_A_Y ; i += stride_A){
        int idx_i = i / stride_A * 4;
        FETCH_FLOAT4(regA_mov[j]) = FETCH_FLOAT4(A[OFFSET(
            A_data_Y + i,//row
            A_data_X,//col
            K//width
        )] );
    
        As[0][A_data_X][A_data_Y + i] = regA_mov[idx_i];
        As[0][A_data_X + 1][A_data_Y + i] = regA_mov[idx_i + 1];
        As[0][A_data_X + 2][A_data_Y + i] = regA_mov[idx_i + 2];
        As[0][A_data_X + 3][A_data_Y + i] = regA_mov[idx_i + 3];
    }
    //load global memory B to shared memory0
    #pragma unroll
    for(int i = 0;i < Block_B_X , i += stride_B){
        int idx_i = i / stride_B * 4;
        FETCH_FLOAT4(regB_mov[idx_i]) = FETCH_FLOAT4(B[OFFSET(
            B_data_Y,//row
            B_data_X,//col
            N//width
        )]);
        Bs[0][B_data_Y][B_data_X + i] = regB_mov[idx_i];
        Bs[0][B_data_Y][B_data_X + i + 1] = regB_mov[idx_i + 1];
        Bs[0][B_data_Y][B_data_X + i + 2] = regB_mov[idx_i + 2];
        Bs[0][B_data_Y][B_data_X + i + 3] = regB_mov[idx_i + 3];
    }
    //load share memory to register
    for(int i = 0;i < Thread_A; i += 4){
        FETCH_FLOAT4(regA_cal[0][i]) = FETCH_FLOAT4(As[0][0][Thread_A * threadIdx.y + i]);
    }
    for(int i = 0;i< Thread_B;i += 4){
        FETCH_FLOAT4(regB_cal[0][i]) = FETCH_FLOAT4(Bs[0][0][Thread_B * threadIdx.x + i]);
    }
    int tile = 0;
    int write_idx = 1;
    int load_idx = 0;
    do{
        tile += Block_A_X;
        //load GA to share
        if(tile < K){
            for(int i = 0;i < Block_A_Y; i += stride_A){
            int idx_i = i / stride_A * 4;
            FETCH_FLOAT4(regA_mov[idx_i]) = FETCH_FLOAT49(A[OFFSET(
                A_data_Y + i,//row
                A_data_X + tile,//col
                K//width
                )]);
            }

            for(int i = 0 ;i < Block_A_X;i += stride_B){
                int idx_i = i / stride_B * 4;
                FETCH_FLOAT4(regB_mov[idx_i]) = FETCH_FLOAT4(B[OFFSET(
                    tile + B_data_X + i,//row
                    B_data_Y,//col
                    N//width
                    )]);
            }
        }
        

        //load to register
        for(int j = 0;j < Block_A_X; j++){
            for(int thread_y = 0; thread_y < Thread_A; thread_y += 4){
                FETCH_FLOAT4(regA_cal[(j+!)%2][thread_y]) = FETCH_FLOAT4(As[load_idx][j+1][Thread_A * threadIdx.y + thread_y]);
            }
            for(int thread_x = 0;thread_x < Thread_B;thread_x += 4){
                FETCH_FLOAT4(regB_cal[(j+!)%2][thread_x]) = FETCH_FLOAT4(Bs[load_idx][j+1][Thread_B * threadIdx.x + thread_x]);
            }
            //calculation
            for(int thread_y = 0; thread_y < Thread_A; thread_y++){
                for(int thread_x = 0;thread_x < Thread_B;thread_x++){
                    temp_ans[thread_y][thread_x] += regA_cal[j%2][thread_y] * regB_cal[j%2][thread_x];
                }
            }
        }
        //
        if(tile < K){
            for(int i = 0;i < Block_A_Y; i += stride_A){
                int idx_i = i / stride_A * 4;
                As[write_idx][A_data_X][A_data_Y + i] = regA_mov[idx_i];
                As[write_idx][A_data_X + 1][A_data_Y + i] = regA_mov[idx_i + 1];
                As[write_idx][A_data_X + 2][A_data_Y + i] = regA_mov[idx_i + 2];
                As[write_idx][A_data_X + 3][A_data_Y + i] = regA_mov[idx_i + 3];
            }

            for(int i = 0 ;i < Block_A_X;i += stride_B){
                int idx_i = i / stride_B * 4;
                Bs[write_idx][A_data_X][B_data_X + i] = regB_mov[idx_i];
                Bs[write_idx][A_data_X][B_data_X + 1 + i] = regB_mov[idx_i + 1];
                Bs[write_idx][A_data_X][B_data_X + 2 + i] = regB_mov[idx_i + 2];
                Bs[write_idx][A_data_X][B_data_X + 3 + i] = regB_mov[idx_i + 3];
            }
        }   

        load_idx = write_idx ^ 1;
        write_idx ^= 1;
    }while(tile < K)
    //write back
        for(int thread_y = 0;thread_y < Thread_A; thread_y++){
            for(int thread_x = 0;thread_x < Thread_B;thread_x += 4){
                FETCH_FLOAT4( C[OFFSET(
                    Block_A_Y * blockIdx.y + Thread_A * threadIdx.y + thread_y,//row
                    Block_B_X * blockIdx.x + Thread_B * threadIdx.x + thread_x,//col
                    N//width
                    )]) = FETCH_FLOAT4(temp_ans[thread_y][thread_X]);
            }
        }




}