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
