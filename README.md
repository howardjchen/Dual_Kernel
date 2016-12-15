<h1> NCTU IEE 2016 Fall </br> Computer Architecture Final Project </h1>

Part-I: Use CUDA to accelerate the operations of a typical convolutional layer in often-used. </br>
(You can find the description slides [here](https://docs.google.com/presentation/d/1uYAh4sU3ZA39zQfRGr596CdbRKgjEh4FnfDEz4eQwuU/edit?usp=sharing))
## Three sub-directory
### ./data
This directory contains the input data for the base program
* /data/filt.txt - Store the values of filters
* /data/inNeu.txt - Store the values of input neurons

### ./innerProduct
This is the example to show you how to use CUDA to accelerate Inner Product
#### Usage
    
    cd ./innerProduct
    make
    make run
    
### ./device
The program under this directory can show the device information
#### Usage
    
    cd ./device
    make
    make run
    
## Usage of the base program
### Compile the code

    make
    
### Run the code

    make run
## Task

* Implement convLayerGPU() with CUDA
* Store your result in the outGPU
* Use NVIDIA Visual Profiler to analyze and improve your code

## Evaluation

* convLayerCPU() will do the computation with C++ and store the output in the outCPU
* checker() will check whether the values stored in outCPU and outGPU are the same
* clock_gettime() is used to measure your preformance
* Lunch your CUDA kernels within two clock_gettime() functions (You are allowed to lunch multiple kernels in this project)
* Put cudaDeviceSynchronize() before the last clock_gettime()
* You must pass the checking to ensure your result is correct!
* We will compare the execution time to get the speedup
    
        Speedup = convLayerCPU_execTime / convLayerGPU_execTime
        
## Grading Policy

* Completeness (30%)
    * Your result must be correct (Pass the check) (10%)
    * You get speedup compared to convLayerCPU() (10%)
    * You use NVIDIA Visual Profiler (NVVP) to help you (10%)
* Report (40%)
    * Describe your implementation algorithm and explain your results (10%)
    * Show how you use NVVP to help you find and solve perf. issues (10%)
    * Discussions on the optimizations you do (10%)
    * Feedback of this project (10%)
* Performance Rank (30%)
    * We will rank your CUDA kernels’ performance on GTX 680
    * The fastest one will get 30 points and the last one will get 1 points for this part
* Delay is not acceptable!

## Other Rules
* It’s team work, 1 ~ 3 people in one team
    * Register [here](https://docs.google.com/spreadsheets/d/1o-Tpq2UEE8jDqwkoMaVHfYQvgkfbu5n_KWtzuctjJ7c/edit?usp=sharing) before deadline
* Compress your code and report into one zip file and upload to E3
    * Name your package as: LeaderID_FP1.zip
    * One team only need to upload one package to E3
    * Please name your report as: LeaderID_Report_FP1.pdf
    * Make sure TA can compile and run your code with "make" and "make run" on the provided server
* Any CUDA library is forbidden to use in this project

## Useful Reference
* LeNet: [Gradient Based Learning Applied to Document Recognition](http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf)
* AlexNet: [ImageNet Classification with Deep Convolutional Neural Networks](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)
* CNN: [Standford CS231n Convolutional Neural Networks for Visual Recognition](http://cs231n.github.io/convolutional-networks/)
* CUDA Tutorial: [CUDA C/C++ Basics](http://www.nvidia.com/docs/io/116711/sc11-cuda-c-basics.pdf)
* CNN with CUDA: [Optimizing Convolution Operations in CUDA with Adaptive Tiling convolution on gpu](http://www.few.vu.nl/~bwn200/papers/werkhoven-a4mmc2011.pdf)
* GPU Profiling: [GPU Performance Analysis and Optimisation](http://people.maths.ox.ac.uk/gilesm/cuda/lecs/NV_Profiling_lowres.pdf)
* GPU Profiling: [CUDA Profiling Documentation](http://docs.nvidia.com/cuda/profiler-users-guide/index.html#axzz4PPDcxdt6)

TA: Chien-Yu Lin </br>
Email: myislin@gmail.com
