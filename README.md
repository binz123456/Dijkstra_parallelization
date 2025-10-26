# Parallel Pathfinding (Dijkstra / GPU & CPU)

## Overview

This project implements **Dijkstra’s algorithm** for 2D grid-based pathfinding, with multiple versions:

- **CPU serial**: baseline Dijkstra
- **CPU parallel**: OpenMP-optimized Dijkstra
- **GPU**: CUDA-based Dijkstra with parallel frontier expansion  

The goal is to **optimize execution** and demonstrate **speedup** across CPU and GPU implementations.

**Applications:** Autonomous Driving (ADAS), robotics, games, trajectory planning.

---

## Features

- **CPU Serial Dijkstra** – simple, easy to understand.
- **CPU Parallel Dijkstra** – optimized using OpenMP.
- **GPU Dijkstra** – CUDA kernel with atomic updates for parallel distance calculation.
- **Randomly generated grids** – weights 1–5 for demonstration.
- **Timing harness** – measures execution time in milliseconds.
- **Speedup visualization** – Python script generates a bar chart comparing methods.

## Build Instructions
mkdir -p build
cd build
cmake ..

make -j

python3 plot_results.py results.csv

## Run all with scripts
chmod +x run.sh
./run.sh



