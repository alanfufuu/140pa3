Last name of Student 1:
First name of Student 1:
Email of Student 1:
GradeScope account name of Student 1: 
Last name of Student 2:
First name of Student 2:
Email of Student 2:
GradeScope account name of Student 2: 


----------------------------------------------------------------------------
Report for Question 1 

List your code change for this question 

idx = blockIdx.x * blockDim.x + threadIdx.x;

result = cudaMalloc((void **) &A_d, A_size);
if (result) {
    printf("Error in cudaMalloc. Error code is %d.\n", result);
    return -1;
}
result = cudaMemcpy(A_d, A, A_size, cudaMemcpyHostToDevice);
if (result) {
    printf("Error in cudaMemcpy. Error code is %d.\n", result);
    return -1;
}

result = cudaMemcpy(y, y_d, row_size, cudaMemcpyDeviceToHost);
if (result) {
    printf("Error in cudaMemcpy. Error code is %d.\n", result);
    return -1;
}

mult_vec<<<num_blocks, threads_per_block>>>(N, rows_per_thread, y_d, d_d, A_d, x_d, diff_d);



Parallel time for n=4K, t=1K,  4x128  threads

Parallel time for n=4K, t=1K,  8x128  threads

Parallel time for n=4K, t=1K,  16x128 threads

Parallel time for n=4K, t=1K,  32x128 threads


Do you see a trend of  speedup improvement  with more threads? We expect a good speedup and explain the reason.


----------------------------------------------------------------------------


Report for Question 2 
List your code change for this question

idx = blockIdx.x * blockDim.x + threadIdx.x;

for (int i = 0; i < rows_per_thread; i++) {
    int row_index = idx * rows_per_thread + i;
    float sum = d[row_index];
    for (int j = 0; j < n; j++) {
        sum += A[row_index * n + j] * y[j];
    }
    y[row_index] = sum;
}



Let the default number of asynchronous iterations be 5 in a batch as specified in it_mult_vec.h.
List reported parallel time and the number of actual iterations executed  for n=4K, t=1K, 8x128  threads with asynchronous Gauss Seidel


List reported parallel time and the number of actual iterations executed  for n=4K, t=1K,  32x128 threads with asynchronous Gauss Seidel


Is the number of iterations  executed by  above parallel asynchronous Gauss Seidel-Seidel method  bigger or smaller  than that
of the sequential Gauss Seidel-Seidel code under the same converging error threshold (1e-3)?  
Explain the reason based on the running trace of above two thread configurations that more threads may not yield more time reduction in this case. 



Make sure you attach the  output trace  of your code below in running the tests of the unmodified it_mult_vec_test.cu on Expanse GPU for Q1 and Q2

----------------------------------------------------------------------------

Report for Question 3

List your solution to call  cublasDgemm() in Method 1.

cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, size_N, size_N, size_N, &alpha, d_A, size_N, d_B, size_N, &beta, d_C, size_N);

List the latency and GFLOPs of the above 3 version of implementation and the number of Cuda threads used in executing Method 3 
when matrix dimension N varies as 50, 200, 800,  and 1600.  


List the highest gigaflops you have observed with V100 from this question and the highest gigaflops  you have observed from PA2 MKL GEMM code  when N=1600.  
Compute the ratio between these two numbers as the speedup of V100 over a CPU host. 





