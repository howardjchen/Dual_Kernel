#include <stdio.h> 

int main() {
  int nDevices;

  cudaGetDeviceCount(&nDevices);
  for (int i = 0; i < nDevices; i++) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    printf("Device Number: %d\n", i);

    printf("Device name: %s\n", prop.name);
		printf("TotalGlobalMem: %d mB\n", prop.totalGlobalMem/1024/1024);
		printf("SharedMemPerBlock: %d kB\n", prop.sharedMemPerBlock/1024);
		printf("MaxThreadsPerBlock: %d\n", prop.maxThreadsPerBlock);
		printf("MaxThreadsDim [x]: %d [y]: %d [z]: %d\n", 
					prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
		printf("MaxGridSize [x]: %d [y]: %d [z]: %d\n", 
					prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
		printf("MultiProcessorCount: %d\n", prop.multiProcessorCount);
		printf("MaxThreadsPerMultiProcessor: %d\n", prop.maxThreadsPerMultiProcessor);
  }
}
