#include <cuda_runtime.h>
#include <cstdio>
#include <algorithm>
#include <climits>

extern "C" {

__global__ void dijkstra_step(int* grid, int* dist, int* updated, int rows, int cols){
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx >= rows*cols) return;
    if(updated[idx]==0) return;

    int r = idx/cols, c = idx%cols;
    updated[idx] = 0;
    int dr[] = {-1,1,0,0};
    int dc[] = {0,0,-1,1};
    for(int k=0;k<4;k++){
        int nr=r+dr[k], nc=c+dc[k];
        if(nr>=0 && nr<rows && nc>=0 && nc<cols){
            int nidx = nr*cols+nc;
            int new_dist = dist[idx] + grid[nidx];
            int old = atomicMin(&dist[nidx], new_dist);
            if(new_dist < old) updated[nidx] = 1;
        }
    }
}

bool gpu_available(){
    int n=0;
    cudaGetDeviceCount(&n);
    return n>0;
}

// GPU wrapper
double gpu_dijkstra_time_ms(int* grid_h, int rows, int cols, int start, int goal, int* dist_h){
    int N = rows*cols;
    int *grid_d=nullptr,*dist_d=nullptr,*upd_d=nullptr;
    cudaMalloc(&grid_d,sizeof(int)*N);
    cudaMalloc(&dist_d,sizeof(int)*N);
    cudaMalloc(&upd_d,sizeof(int)*N);

    cudaMemcpy(grid_d,grid_h,sizeof(int)*N,cudaMemcpyHostToDevice);
    std::vector<int> dist(N,INT_MAX), updated(N,0);
    dist[start]=0; updated[start]=1;
    cudaMemcpy(dist_d, dist.data(), sizeof(int)*N,cudaMemcpyHostToDevice);
    cudaMemcpy(upd_d, updated.data(), sizeof(int)*N,cudaMemcpyHostToDevice);

    int block=256, gridSize=(N+block-1)/block;
    cudaEvent_t start_ev, stop_ev;
    cudaEventCreate(&start_ev); cudaEventCreate(&stop_ev);
    cudaEventRecord(start_ev);

    for(int iter=0; iter<N; iter++){
        dijkstra_step<<<gridSize,block>>>(grid_d, dist_d, upd_d, rows, cols);
        cudaDeviceSynchronize();
    }

    cudaEventRecord(stop_ev);
    cudaEventSynchronize(stop_ev);
    float ms=0; cudaEventElapsedTime(&ms,start_ev,stop_ev);

    cudaMemcpy(dist_h, dist_d, sizeof(int)*N, cudaMemcpyDeviceToHost);
    cudaFree(grid_d); cudaFree(dist_d); cudaFree(upd_d);
    cudaEventDestroy(start_ev); cudaEventDestroy(stop_ev);
    return double(ms);
}

} // extern "C"
