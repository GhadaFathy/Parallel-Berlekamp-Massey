# Parallel implementation of Berlekamp-Massey algorithm 
This implementation includes the two published papers:
 https://ieeexplore.ieee.org/abstract/document/7394405
 https://link.springer.com/article/10.1007/s10586-019-02961-x
 
CUDA C and NVIDIA Tesla M2090 are used. 
The repository includes:
* Source code of Berlekamp-Massey algorithm (BMA).
* The makefile
* Paper 1: Evaluation of CUDA Memory Fence Performance; Berlekamp-Massey Case Study
* Paper 2: Exploring the parallel capabilities of GPU: Berlekamp-Massey algorithm case study


The code is documented and designed to be easy to extend. If you use it in your research, please consider citing this repository (BibTeX below).
## Stetting 
The proposed experiments execute BMA using 512 threads per block.
The number of active blocks and active threads are calculated as discussed in the parallel BMA implementation
section. 
The syndrome lengths were varied from 1 Kbit to 64 Mbits.
## Requirements
GCC compiler, CUDA 7 or higher.
The operating system is Linux (CentOS).
## Compile and Run
- To compile the code you need to edit the makefile with CUDA installation Path.
- $ cd the_code_dir
- $ make
- $ ./parallel_bma

## Citation
Use this BibTeX to cite this repository:
```
1- Ali, Hanan, Zeinab Fayez, Ghada M. Fathy, and Walaa Sheta.
   "Evaluation of CUDA memory fence performance; Berlekamp-Massey case study." 
   In 2015 IEEE International Symposium on Signal Processing and Information Technology (ISSPIT), pp. 586-590. IEEE, 2015.
   
2- Ali, Hanan, Ghada M. Fathy, Zeinab Fayez, and Walaa Sheta. 
   "Exploring the parallel capabilities of GPU: Berlekamp-Massey algorithm case study."
   Cluster Computing 23, no. 2 (2020): 1007-1024.

