#include <iostream>
#include <vector>
#include <queue>
#include <chrono>
#include <random>
#include <fstream>
#include <omp.h>

extern "C" {
bool gpu_available();
double gpu_dijkstra_time_ms(int* grid, int rows, int cols, int start, int goal, int* dist_out);
}

// Utilities
using Clock = std::chrono::high_resolution_clock;

inline int idx(int r, int c, int cols) { return r * cols + c; }

struct Node {
    int pos;
    int dist;
    bool operator>(const Node& o) const { return dist > o.dist; }
};

// Serial Dijkstra
double dijkstra_cpu(const std::vector<int>& grid, int rows, int cols, int start, int goal, std::vector<int>& dist) {
    int N = rows * cols;
    dist.assign(N, 1e9);
    dist[start] = 0;
    std::priority_queue<Node, std::vector<Node>, std::greater<Node>> pq;
    pq.push({start, 0});

    auto t0 = Clock::now();
    while(!pq.empty()) {
        Node cur = pq.top(); pq.pop();
        int r = cur.pos / cols, c = cur.pos % cols;
        if(cur.dist > dist[cur.pos]) continue;

        std::vector<int> dr = { -1, 1, 0, 0 };
        std::vector<int> dc = { 0, 0, -1, 1 };
        for(int k=0;k<4;k++){
            int nr = r + dr[k], nc = c + dc[k];
            if(nr>=0 && nr<rows && nc>=0 && nc<cols){
                int nidx = idx(nr,nc,cols);
                int w = grid[nidx];
                if(dist[nidx] > dist[cur.pos] + w){
                    dist[nidx] = dist[cur.pos] + w;
                    pq.push({nidx, dist[nidx]});
                }
            }
        }
    }
    auto t1 = Clock::now();
    return std::chrono::duration<double>(t1 - t0).count() * 1000.0;
}

// Parallelized Dijkstra using OpenMP (naive approach)
double dijkstra_cpu_parallel(const std::vector<int>& grid, int rows, int cols, int start, int goal, std::vector<int>& dist){
    int N = rows * cols;
    dist.assign(N, 1e9);
    dist[start] = 0;

    std::vector<bool> visited(N,false);

    auto t0 = Clock::now();
    for(int iter=0; iter<N; ++iter){
        int cur = -1;
        int min_dist = 1e9;
        #pragma omp parallel
        {
            int local_min = 1e9, local_idx = -1;
            #pragma omp for nowait
            for(int i=0;i<N;i++){
                if(!visited[i] && dist[i]<local_min){
                    local_min = dist[i];
                    local_idx = i;
                }
            }
            #pragma omp critical
            {
                if(local_min < min_dist){
                    min_dist = local_min;
                    cur = local_idx;
                }
            }
        }
        if(cur==-1 || cur==goal) break;
        visited[cur] = true;
        int r = cur / cols, c = cur % cols;
        std::vector<int> dr = { -1, 1, 0, 0 };
        std::vector<int> dc = { 0, 0, -1, 1 };
        for(int k=0;k<4;k++){
            int nr = r + dr[k], nc = c + dc[k];
            if(nr>=0 && nr<rows && nc>=0 && nc<cols){
                int nidx = idx(nr,nc,cols);
                int w = grid[nidx];
                int new_dist = dist[cur]+w;
                #pragma omp critical
                if(new_dist<dist[nidx]) dist[nidx]=new_dist;
            }
        }
    }
    auto t1 = Clock::now();
    return std::chrono::duration<double>(t1 - t0).count() * 1000.0;
}

int main(){
    int rows=512, cols=512;
    std::vector<int> grid(rows*cols);
    std::mt19937 rng(123);
    std::uniform_int_distribution<int> d(1,5);
    for(auto &x:grid) x = d(rng);

    int start = 0, goal = rows*cols-1;
    std::vector<int> dist_serial, dist_par;

    std::ofstream csv("results.csv");
    csv << "method,ms\n";

    double ms = dijkstra_cpu(grid, rows, cols, start, goal, dist_serial);
    std::cout << "CPU serial: " << ms << " ms\n";
    csv << "cpu_serial," << ms << "\n";

    double ms_par = dijkstra_cpu_parallel(grid, rows, cols, start, goal, dist_par);
    std::cout << "CPU parallel: " << ms_par << " ms\n";
    csv << "cpu_parallel," << ms_par << "\n";

    // GPU
    if(gpu_available()){
        std::vector<int> dist_gpu(rows*cols);
        double ms_gpu = gpu_dijkstra_time_ms(grid.data(), rows, cols, start, goal, dist_gpu.data());
        std::cout << "GPU: " << ms_gpu << " ms\n";
        csv << "gpu," << ms_gpu << "\n";
    }

    csv.close();
    return 0;
}
