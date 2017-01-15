## Evaluation

* convLayerCPU() will do the computation with C++ and store the output in the outCPU
* checker() will check whether the values stored in outCPU and outGPU are the same
* clock_gettime() is used to measure your preformance
* Lunch your CUDA kernels within two clock_gettime() functions (You are allowed to lunch multiple kernels in this project)
* Put cudaDeviceSynchronize() before the last clock_gettime()
* You must pass the checking to ensure your result is correct!
* We will compare the execution time to get the speedup
* Speedup = convLayerCPU_execTime / convLayerGPU_execTime


**- Version : Global Barrier** 

| kernel         | blocks        | threads      | thread per block  |
| ------         | -----------   |---------     |-------------------|
| Conv kernel    | 128 x 2 x 2   | 512 x 32 x 32|        1024       | 

        
**- Version : DualKernel_200(Final Version)**

| kernel         | gridDim       | thread      | thread per block  |kernel time|
| ------         | -----------   |---------     |-------------------|---|
| Conv kernel    | 128 x 2 x 2   | 512 x 32 x 32|        1024       | 199.8 ms|
| Pooling kernel | 128 x 1 x 1 | 512x16x16    |        1024       | 0.159ms|

** nvprof
```
==25417== Profiling result:
Time(%)      Time     Calls       Avg       Min       Max  Name
 99.24%  198.67ms         1  198.67ms  198.67ms  198.67ms  convLayerGPU(short*, short*, int*, int*, int*)
  0.49%  973.45us         2  486.72us  175.58us  797.87us  [CUDA memcpy HtoD]
  0.20%  398.87us         2  199.43us  80.254us  318.62us  [CUDA memcpy DtoH]
  0.08%  154.59us         1  154.59us  154.59us  154.59us  MaxPoolingGPU(int*, int*)

==25417== API calls:
Time(%)      Time     Calls       Avg       Min       Max  Name
 71.42%  198.95ms         1  198.95ms  198.95ms  198.95ms  cudaDeviceSynchronize
 27.78%  77.383ms         5  15.477ms  75.499us  77.078ms  cudaMalloc
  0.66%  1.8506ms         4  462.65us  202.54us  800.87us  cudaMemcpy
  0.09%  241.72us        83  2.9120us     698ns  81.016us  cuDeviceGetAttribute
  0.01%  41.346us         2  20.673us  8.8700us  32.476us  cudaLaunch
  0.01%  31.359us         1  31.359us  31.359us  31.359us  cuDeviceTotalMem
  0.01%  27.518us         1  27.518us  27.518us  27.518us  cuDeviceGetName
  0.00%  10.059us         4  2.5140us  1.1180us  6.5660us  cudaFree
  0.00%  9.9170us         7  1.4160us     698ns  4.9590us  cudaSetupArgument
  0.00%  6.0760us         2  3.0380us     977ns  5.0990us  cudaConfigureCall
  0.00%  2.6550us         2  1.3270us     769ns  1.8860us  cuDeviceGetCount
  0.00%  1.7460us         2     873ns     769ns     977ns  cuDeviceGet
```

````
CPU time for executing a typical convolutional layer = 16701.2ms
GPU time for executing a typical convolutional layer = 190.447.ms
Congratulations! You pass the check.
Speedup: 87.694 
````    
    
- **Kernel Latency**
![](https://i.imgur.com/CEDDBj0.jpg)



- **Kernel Memory**
![](https://i.imgur.com/otBV3Hs.jpg)
