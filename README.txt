Last name of Student 1: Fu
First name of Student 1: Alan
Email of Student 1: a_fu@ucsb.edu
GradeScope account name of Student 1: a_fu@ucsb.edu
Last name of Student 2: N/A
First name of Student 2:N/A
Email of Student 2: N/A
GradeScope account name of Student 2: N/A


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
4.108857 seconds

Parallel time for n=4K, t=1K,  8x128  threads
2.055702 seconds

Parallel time for n=4K, t=1K,  16x128 threads
1.038297 seconds

Parallel time for n=4K, t=1K,  32x128 threads
0.562406 seconds

Do you see a trend of  speedup improvement  with more threads? We expect a good speedup and explain the reason.
Yes, there is a clear trend of speedup improvement with more threads. Each time the number of threads doubles, the execution time is roughly cut in half.
This speedup occurs because the Jacobi method is very parallel.
Each thread computes y[i] = d[i] + A[i]*x independently for its assigned rows. There are
no data dependencies between rows within a single iteration, so doubling the number of
threads halves the number of rows each thread must process, which halves the
computation time.

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
Time: 0.038458 seconds, Iterations: 15

List reported parallel time and the number of actual iterations executed  for n=4K, t=1K,  32x128 threads with asynchronous Gauss Seidel
time: 0.440662 seconds, Iterations: 1025

Is the number of iterations  executed by  above parallel asynchronous Gauss Seidel-Seidel method  bigger or smaller  than that
of the sequential Gauss Seidel-Seidel code under the same converging error threshold (1e-3)?  
Explain the reason based on the running trace of above two thread configurations that more threads may not yield more time reduction in this case. 

It is based on the thread configuration. With 8x128 threads , the async
Gauss-Seidel converged in 15 iterations, comparable to or fewer than
the sequential Gauss-Seidel method. With 32x128 threads, the async
method needed 1025 iterations and did not converge, which is a lot more than
sequential Gauss-Seidel.This is because each thread only rows one row in 32x128, so every row depends on values from other threads
that may be not yet updated. this will cause the method to converge a lot slower, as opposed to 8x128, where each thread owns 4 rows and updates them with
fresh values, resulting in faster convergence. 



Make sure you attach the  output trace  of your code below in running the tests of the unmodified it_mult_vec_test.cu on Expanse GPU for Q1 and Q2
>>>>>>>>>>>>>>>>>>>>>>>>>
Start running itmv tests.
>>>>>>>>>>>>>>>>>>>>>>>>>
Test 1:n=4, t=1, 1x2 threads:
With totally 1*2 threads, matrix size being 4, t being 1
Time cost in seconds: 0.103557
Final error (|y-x|): 1.750000.
# of iterations executed: 1.
Final y[0]=1.750000. y[n-1]=1.750000
Test 2:n=4, t=2, 1x2 threads:
With totally 1*2 threads, matrix size being 4, t being 2
Time cost in seconds: 0.000251
Final error (|y-x|): 1.312500.
# of iterations executed: 2.
Final y[0]=0.437500. y[n-1]=0.437500
Test 3:n=8, t=1, 1x2 threads:
With totally 1*2 threads, matrix size being 8, t being 1
Time cost in seconds: 0.000225
Final error (|y-x|): 1.875000.
# of iterations executed: 1.
Final y[0]=1.875000. y[n-1]=1.875000
Test 4:n=8, t=2, 1x2 threads:
With totally 1*2 threads, matrix size being 8, t being 2
Time cost in seconds: 0.000250
Final error (|y-x|): 1.640625.
# of iterations executed: 2.
Final y[0]=0.234375. y[n-1]=0.234375
Test 8a:n=4, t=1, 1x1 threads/Gauss-Seidel:
With totally 1*1 threads, matrix size being 4, t being 1
Time cost in seconds: 0.000230
Final error (|y-x|): 1.000193.
# of iterations executed: 5.
Final y[0]=1.000089. y[n-1]=1.000193
Test 8b:n=4, t=2, 1x1 threads/Gauss-Seidel:
With totally 1*1 threads, matrix size being 4, t being 2
Time cost in seconds: 0.000226
Final error (|y-x|): 1.000193.
# of iterations executed: 5.
Final y[0]=1.000089. y[n-1]=1.000193
Test 8c:n=8, t=1, 1x1 threads/Gauss-Seidel:
With totally 1*1 threads, matrix size being 8, t being 1
Time cost in seconds: 0.000230
Final error (|y-x|): 1.001155.
# of iterations executed: 5.
Final y[0]=1.001155. y[n-1]=0.999790
Test 8d:n=8, t=2, 1x1 threads/Gauss-Seidel:
With totally 1*1 threads, matrix size being 8, t being 2
Time cost in seconds: 0.000233
Final error (|y-x|): 1.001155.
# of iterations executed: 5.
Final y[0]=1.001155. y[n-1]=0.999790
Test 9: n=4K t=1K 32x128 threads:
With totally 32*128 threads, matrix size being 4096, t being 1024
Time cost in seconds: 0.562406
Final error (|y-x|): 1.557740.
# of iterations executed: 1024.
Final y[0]=0.221225. y[n-1]=0.221225
Test 9a: n=4K t=1K 16x128 threads:
With totally 16*128 threads, matrix size being 4096, t being 1024
Time cost in seconds: 1.038297
Final error (|y-x|): 1.557740.
# of iterations executed: 1024.
Final y[0]=0.221225. y[n-1]=0.221225
Test 9b: n=4K t=1K 8x128 threads:
With totally 8*128 threads, matrix size being 4096, t being 1024
Time cost in seconds: 2.055702
Final error (|y-x|): 1.557740.
# of iterations executed: 1024.
Final y[0]=0.221225. y[n-1]=0.221225
Test 9c: n=4K t=1K 4x128 threads:
With totally 4*128 threads, matrix size being 4096, t being 1024
Time cost in seconds: 4.108857
Final error (|y-x|): 1.557740.
# of iterations executed: 1024.
Final y[0]=0.221225. y[n-1]=0.221225
Test 11: n=4K t=1K 32x128 threads/Async:
With totally 32*128 threads, matrix size being 4096, t being 1024
Time cost in seconds: 0.440662
Final error (|y-x|): 0.001880.
# of iterations executed: 1025.
Final y[0]=1.000965. y[n-1]=1.000969
Test 11a: n=4K t=1K 8x128 threads/Async:
With totally 8*128 threads, matrix size being 4096, t being 1024
Time cost in seconds: 0.038458
Final error (|y-x|): 0.000000.
# of iterations executed: 15.
Early exit due to convergence, even asked for 1024 iterations.
Asynchronous code actually runs 15 iterations.
Final y[0]=1.000000. y[n-1]=1.000000
Summary: Failed 0 out of 14 tests
----------------------------------------------------------------------------

Report for Question 3

List your solution to call  cublasDgemm() in Method 1.

Warm-up call:
  cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, size_N, size_N, size_N,
              &alpha, d_A, size_N, d_B, size_N, &beta, d_C, size_N);

Main timed call:
  handle_cublas_error(
      cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, size_N, size_N, size_N,
                  &alpha, d_A, size_N, d_B, size_N, &beta, d_C, size_N),
      "cublasDgemm (Main)", __FILE__, __LINE__);

List the latency and GFLOPs of the above 3 version of implementation and the number of Cuda threads used in executing Method 3 
when matrix dimension N varies as 50, 200, 800,  and 1600.  

N=50:
  Method 1 (cuBLAS dgemm):   Latency: 0.020 ms,  GFLOPS: 12.207
  Method 2 (dgemv loop):     Latency: 0.155 ms,  GFLOPS: 1.617
  Method 3 (Naive GEMM):     Latency: 0.009 ms,  GFLOPS: 27.127,  CUDA threads: 3*3*400 = 3600

N=200:
  Method 1 (cuBLAS dgemm):   Latency: 0.028 ms,  GFLOPS: 578.704
  Method 2 (dgemv loop):     Latency: 3.315 ms,  GFLOPS: 4.827
  Method 3 (Naive GEMM):     Latency: 0.029 ms,  GFLOPS: 558.036,  CUDA threads: 10*10*400 = 40000

N=800:
  Method 1 (cuBLAS dgemm):   Latency: 0.226 ms,  GFLOPS: 4524.887
  Method 2 (dgemv loop):     Latency: 13.255 ms, GFLOPS: 77.256
  Method 3 (Naive GEMM):     Latency: 1.040 ms,  GFLOPS: 984.252,  CUDA threads: 40*40*400 = 640000

N=1600:
  Method 1 (cuBLAS dgemm):   Latency: 1.295 ms,  GFLOPS: 6324.111
  Method 2 (dgemv loop):     Latency: 46.684 ms, GFLOPS: 175.477
  Method 3 (Naive GEMM):     Latency: 8.433 ms,  GFLOPS: 971.463,  CUDA threads: 80*80*400 = 2560000


List the highest gigaflops you have observed with V100 from this question and the highest gigaflops  you have observed from PA2 MKL GEMM code  when N=1600.  
Compute the ratio between these two numbers as the speedup of V100 over a CPU host. 

Highest V100 GFLOPS: 6324.111 GFLOPS (cuBLAS dgemm at N=1600)
Highest PA2 MKL GEMM GFLOPS at N=1600: 43.03 GFLOPS (MKL DGEMM, 1 thread)
Speedup ratio (V100 / CPU): 6324.111 / 43.03 = 147.0x



Execution trace for Q3:

CUDA matrix-matrix multiply (50x50 matrices) with column major
1. cuBLAS dgemm (optimized library)    Latency: 0.020 ms   GFLOPS:  12.207
2. cuBLAS dgemv in a loop for GEMM     Latency: 0.155 ms   GFLOPS:  1.617
3. Naive GEMM (parallelized 3 loops)   Latency: 0.009 ms   GFLOPS:  27.127

Mid-point verification looks OK: DGEMM=13.0289, DGEMV=13.0289, Naive=13.0289

CUDA matrix-matrix multiply (200x200 matrices) with column major
1. cuBLAS dgemm (optimized library)    Latency: 0.028 ms   GFLOPS:  578.704
2. cuBLAS dgemv in a loop for GEMM     Latency: 3.315 ms   GFLOPS:  4.827
3. Naive GEMM (parallelized 3 loops)   Latency: 0.029 ms   GFLOPS:  558.036

Mid-point verification looks OK: DGEMM=46.8633, DGEMV=46.8633, Naive=46.8633

CUDA matrix-matrix multiply (800x800 matrices) with column major
1. cuBLAS dgemm (optimized library)    Latency: 0.226 ms   GFLOPS:  4524.887
2. cuBLAS dgemv in a loop for GEMM     Latency: 13.255 ms   GFLOPS:  77.256
3. Naive GEMM (parallelized 3 loops)   Latency: 1.040 ms   GFLOPS:  984.252

Mid-point verification looks OK: DGEMM=195.4754, DGEMV=195.4754, Naive=195.4754

CUDA matrix-matrix multiply (1600x1600 matrices) with column major
1. cuBLAS dgemm (optimized library)    Latency: 1.295 ms   GFLOPS:  6324.111
2. cuBLAS dgemv in a loop for GEMM     Latency: 46.684 ms   GFLOPS:  175.477
3. Naive GEMM (parallelized 3 loops)   Latency: 8.433 ms   GFLOPS:  971.463

Mid-point verification looks OK: DGEMM=389.1097, DGEMV=389.1097, Naive=389.1097

CUDA matrix-matrix multiply (3200x3200 matrices) with column major
1. cuBLAS dgemm (optimized library)    Latency: 9.298 ms   GFLOPS:  7048.458
2. cuBLAS dgemv in a loop for GEMM     Latency: 314.196 ms   GFLOPS:  208.583
3. Naive GEMM (parallelized 3 loops)   Latency: 58.957 ms   GFLOPS:  1111.594

Mid-point verification looks OK: DGEMM=807.8201, DGEMV=807.8201, Naive=807.8201
