#!/usr/bin/env bash
set -e
mkdir -p build
cd build
cmake ..
make -j
cd ..

echo "Running pathfinding benchmark..."
./build/path_gpu
python3 plot_results.py results.csv
