# OpenMP Matrix Parallelization

This repository contains a report and source code for parallelizing matrix multiplication using OpenMP. It explores different parallelization strategies and evaluates performance on various matrix sizes and thread counts.

## Features
- Sequential and parallel matrix multiplication.
- OpenMP scheduling strategies: Static, Dynamic, and Guided.
- Performance analysis: execution time and speedup.

## How to Use
1. Compile the code with an OpenMP-enabled compiler:
   ```bash
   gcc -fopenmp openMP.c -o openMP
2. Run the program:
   ```bash
./openMP
