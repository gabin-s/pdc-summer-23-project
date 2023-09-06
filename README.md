# PDC Summer School - project 2023
This repository contains code for the handout of the PDC Summer School 2023 project.

## Description of the project
Simplified simulation of high-energy particle storms

EduHPC 2018: Peachy assignment

(c) 2018 Arturo Gonzalez-Escribano, Eduardo Rodriguez-Gutiez
Group Trasgo, Universidad de Valladolid (Spain)

We implemented three parallelized version of the sequential code: OpenMP, MPI, CUDA.
The general parallelization strategy is to split the workload across the whole domain (`layer`) across several processors, use communication when required, and collectively perform reduction operation.
Details are provided in the [report](report.pdf).

## Compiling
The code has been evaluated using the following versions of libraries and compilers:
- OpenMP: Apple clang/14.0.3, OpenMP/16.0.6
- MPI: Cray clang/14.0.1, Cray MPICH/8.1.17 (through module PrgEnv-cray/8.3.3)
- CUDA: compiler 11.5, runtime 11.7, driver 515.65.01

Please adjust the Makefile to you own setup in order to compile the code.