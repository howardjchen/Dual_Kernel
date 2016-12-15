// This program executes a typical convolutional layer in regular CNNs
#include <iostream>
#include "cnnConvLayer.h"
#include <stdio.h>
#include <unistd.h>
using namespace std;

#define xDim 512
#define yDim 32
#define zDim 32

#define xThreadDim 16
#define yThreadDim 16
#define zThreadDim 4


int outputsize = 512*16*16;
int Outputsize = xDim*yDim*zDim;

int *devoutNeu;
int *devPooling;
short *devFilt;
short *devinNeu;
int *outResult = new int[outputsize]();
int *outResult_neu = new int[Outputsize]();

// This is the CPU version, please don't modify it
void convLayerCPU()
{
	// declarations for bunch of indexing parameters
	int fn, sli, fmy, fmx, y, x;
	int sum, ifmy, ifmx, ofmy, ofmx;
	int filtIdx, inNeuIdx, outNeuIdx, outIdx;
	int filtVol = FMDEPTH * FILTSIZE * FILTSIZE;
	int filtArea = FILTSIZE * FILTSIZE;
	int fmArea = FMSIZE *FMSIZE;
	int outArea = FMSIZE/2 * FMSIZE/2;


	cout << "convolutioning..." << endl;

	// Convolution
	for(fn = 0; fn < FILTNUM; fn++) //512
	{
		for(fmy = 0; fmy < FMSIZE; fmy += STRIDE) //32
		{
			for(fmx = 0; fmx < FMSIZE; fmx += STRIDE)  //32
			{
				sum = 0;
				for(sli = 0; sli < FMDEPTH; sli++)  //512
				{
					for(y = 0; y < FILTSIZE; y++)  //3
					{
						for(x = 0; x < FILTSIZE; x++)  //3
						{
							ifmy = fmy - FILTSIZE / 2 + y;		//no dependancy
							ifmx = fmx - FILTSIZE / 2 + x;		//no dependancy
							filtIdx = (fn * filtVol) + (sli * filtArea) + (y * FILTSIZE) + x;	//no dependancy
							inNeuIdx = sli*fmArea + ifmy*FMSIZE + ifmx;							//no dependancy
							if(ifmy >= 0 && ifmy < FMSIZE && ifmx >= 0 && ifmx < FMSIZE)		
								sum += filt[filtIdx] * inNeu[inNeuIdx];
						}
					}
				}
				// Activation - ReLU
				outNeuIdx = fn*fmArea + fmy*FMSIZE + fmx;
				if(sum <= 0)
					outNeu[outNeuIdx] = 0;
				else
					outNeu[outNeuIdx] = sum;
			}
		}
	}


 	cout << "Pooling....." << endl;
	// Max Pooling with Window Size 2x2
	int max, tmpVal;
	for(sli = 0; sli < FILTNUM; sli++)
	{
		for(fmy = 0; fmy < FMSIZE/2 ; fmy += 1)
		{
			for(fmx = 0; fmx < FMSIZE/2 ; fmx += 1)
			{
				outNeuIdx = sli*fmArea + fmy*2*FMSIZE + fmx*2;
				max = outNeu[outNeuIdx];
				for(y = 0; y < 2; y++)
				{
					for(x = 0; x < 2; x++)
					{
						ofmy = fmy*2 + y;
						ofmx = fmx*2 + x;
						outNeuIdx = sli*fmArea + ofmy*FMSIZE + ofmx;
						tmpVal = outNeu[outNeuIdx];	
						if(tmpVal > max)
							max = tmpVal;
					}
				}
				outIdx = sli*outArea + fmy*FMSIZE/2 + fmx;
				outCPU[outIdx] = max;
			}
		}
	}
}


void initGPU()
{
	int outNeuVol = FILTNUM * FMSIZE * FMSIZE;  //512x32x32
	int outPolVol = FILTNUM * FMSIZE/2 * FMSIZE/2;  //512x16x16
	int filtTensorVol = sizeof(short)*FILTNUM*FMDEPTH*FILTSIZE*FILTSIZE; //512x512x3x3
	int inNeuVol = sizeof(short)*FMDEPTH*FMSIZE*FMSIZE;	//512x32x32

	cudaMalloc(&devoutNeu, sizeof(int)*outNeuVol);
	cudaMalloc(&devPooling, sizeof(int)*outPolVol);
	cudaMalloc(&devFilt, filtTensorVol);
	cudaMalloc(&devinNeu, inNeuVol);

	cudaMemcpy(devFilt, filt, filtTensorVol, cudaMemcpyHostToDevice);
	cudaMemcpy(devinNeu, inNeu, inNeuVol, cudaMemcpyHostToDevice);
}


/***	Implement your CUDA Kernel here	***/
__global__
void convLayerGPU(short *FILT, short *InNeu, int *outNeural, int *outPooling)
{
	int threadX = threadIdx.x + blockIdx.x * blockDim.x;
	int threadY = threadIdx.y + blockIdx.y * blockDim.y;
	int threadZ = threadIdx.z + blockIdx.z * blockDim.z;
	//int xall = blockDim.x * gridDim.x;
	//int yall = blockDim.y * gridDim.y;
	//int GlobalThreadId = threadX + threadY * xall + threadZ * xall * yall;
	//int GlobalBlockId = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.y * gridDim.x;

	int sli,y, x;
	int ifmy, ifmx;
	int filtIdx, inNeuIdx, outNeuIdx;
	int filtVol = 4608;  	//512x3x3
	int filtArea = 9;		//3x3
	int fmArea = 1024;	//32x32
	int outArea = 256;	//32/2*32/2
	int sum = 0;

	for(sli = 0; sli < 512; sli++)  //512
	{
		for(y = 0; y < 3; y++)  //3
		{
			for(x = 0; x < 3; x++)  //3
			{
				ifmy = threadY - 3 / 2 + y;		//no dependancy
				ifmx = threadZ - 3 / 2 + x;		//no dependancy
				filtIdx = (threadX * filtVol) + (sli * filtArea) + (y * 3) + x;//no dependancy
				inNeuIdx = sli * fmArea + ifmy * 32 + ifmx;					//no dependancy
				if(ifmy >= 0 && ifmy < 32 && ifmx >= 0 && ifmx < 32)		
					sum += FILT[filtIdx] * InNeu[inNeuIdx];
			}
		}
	}

	// Activation - ReLU
	outNeuIdx = threadX * fmArea + threadY*32 + threadZ;

	if(sum <= 0)
		outNeural[outNeuIdx] = 0;
	else
		outNeural[outNeuIdx] = sum;



	__syncthreads();

 /*========== Max Pooling with Window Size 2x2 =================*/
	
	if(threadX == 0 && threadY == 0 && threadZ == 0 )  //asking 1 thread to do pooling
	{		
		int max, tmpVal, py, px;
		int  ofmy, ofmx, outIdx; // pooling varable
		int xx,yy,slii;

		for(slii = 0; slii < 512; slii++)	//FILTNUM
		{
			for(py = 0; py < 16 ; py += 1) //FMSIZE/2
			{
				for(px = 0; px < 16 ; px += 1)  //FMSIZE/2
				{
					outNeuIdx = slii*fmArea + py*2*32 + px*2;
					max = outNeural[outNeuIdx];
					for(yy = 0; yy < 2; yy++)
					{
						for(xx = 0; xx < 2; xx++)
						{
							ofmy = py*2 + yy;
							ofmx = px*2 + xx;
							outNeuIdx = slii*fmArea + ofmy*32 + ofmx;
							tmpVal = outNeural[outNeuIdx];	
							if(tmpVal > max)
								max = tmpVal;
						}
					}
					outIdx = slii*outArea + py*32/2 + px;
					outPooling[outIdx] = max;
				}
			}
		}
	}
}




/*
__global__ 
void MaxPoolingGPU(int *out)  // Max Pooling with Window Size 2x2
{
	int threadX = threadIdx.x + blockIdx.x * blockDim.x;
	int threadY = threadIdx.y + blockIdx.y * blockDim.y;
	int threadZ = threadIdx.z + blockIdx.z * blockDim.z;
	int xall = blockDim.x * gridDim.x;
	int yall = blockDim.y * gridDim.y;
	int GlobalThreadId = threadX + threadY * xall + threadZ * xall * yall;
	int GlobalBlockId = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.y * gridDim.x;
	
	int max, tmpVal, outNeuIdx, x, y;
	int fmArea = 32 *32;
	int outArea = 32/2 * 32/2;
	int  ofmy, ofmx, outIdx; // pooling varable

	outNeuIdx = threadX*fmArea + threadY*2*32 + threadZ*2;
	max = outNeu[outNeuIdx];
	for(y = 0; y < 2; y++)
	{
		for(x = 0; x < 2; x++)
		{
			ofmy = threadY*2 + y;
			ofmx = threadZ*2 + x;
			outNeuIdx = threadX*fmArea + ofmy*32 + ofmx;
			tmpVal = outNeu[outNeuIdx];	
			if(tmpVal > max)
				max = tmpVal;
		}
	}
	outIdx = threadX*outArea + threadY*32/2 + threadZ;
	out[outIdx] = max;
}
*/

int main()
{
	float convLayerCPUExecTime, convLayerGPUExecTime;
	init();
		


	timespec time_begin, time_end;                                                 
  	clock_gettime(CLOCK_REALTIME, &time_begin);
	convLayerCPU();
  	clock_gettime(CLOCK_REALTIME, &time_end);
	convLayerCPUExecTime = timespec_diff_us(time_begin, time_end);
	cout << " ================ Result ===================" << endl;
	cout << "CPU time for executing a typical convolutional layer = " <<  convLayerCPUExecTime / 1000 << "ms" << endl;



 	initGPU();
 	dim3 threadPerBlock(xThreadDim, yThreadDim, zThreadDim);
 	dim3 numBlocks(xDim/xThreadDim, yDim/yThreadDim, zDim/zThreadDim);
 	clock_gettime(CLOCK_REALTIME, &time_begin);


	convLayerGPU<<<numBlocks,threadPerBlock>>>(devFilt, devinNeu, devoutNeu, devPooling); 


	cudaDeviceSynchronize(); 
  	clock_gettime(CLOCK_REALTIME, &time_end);
	convLayerGPUExecTime = timespec_diff_us(time_begin, time_end);
	cout << "GPU time for executing a typical convolutional layer = " << convLayerGPUExecTime / 1000 << "ms" << endl;


	int outSize = sizeof(int)*outputsize;
	cudaMemcpy(outGPU, devPooling, outSize, cudaMemcpyDeviceToHost);
	
	//int OutSize = sizeof(int)*Outputsize; 
	//cudaMemcpy(outResult_neu, devoutNeu, OutSize, cudaMemcpyDeviceToHost);


	// check the Output of Neu 
	/*for (int i = 0; i < 512*32*32; ++i)
	{
		if (outNeu[i] == outResult_neu[i])
		{
			printf("wrong at =  %d \n", i);
			break;
		}
	}
	printf("PASS!!!\n");*/
	// check the Output of GPU 
	/*for (int i = 0; i < 512*16*16; ++i)
	{
		if (outCPU[i] != outGPU[i])
		{
			printf("wrong at =  %d \n", i);
			break;
		}
	}
	printf("PASS!!!\n");*/



	if(checker())
	{
		cout << "Congratulations! You pass the check." << endl;
		cout << "Speedup: " << (float)convLayerCPUExecTime / convLayerGPUExecTime << endl;
	}
	else
		cout << "Sorry! Your result is wrong." << endl;

	cudaFree(&devoutNeu);
	cudaFree(&devPooling);
	cudaFree(&devFilt);
	cudaFree(&devinNeu);

	delete [] outResult;
	delete [] outResult_neu;
	ending();
	
	return 0;
}
