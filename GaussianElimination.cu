//==========================================================================================================
// A small snippet of code to solve equation of types Ax=B using Gaussian Elimniation
// Author - Anmol Gupta, Naved Ansari
// Course - EC513 - Introduction to Computer Architecture
// Boston University
//==========================================================================================================

//==========================================================================================================
// Command to compile the code
//nvcc -o GaussianElimination GaussianElimination.cu
//==========================================================================================================

#include <cstdio>
#include <cstdlib>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include "cuPrintf.cu"

// Assertion to check for errors
#define CUDA_SAFE_CALL(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=true)
{
	if (code != cudaSuccess) 
	{
		fprintf(stderr,"CUDA_SAFE_CALL: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

#define MINVAL   10.0 //specify the range to initialize matrix A and vector B. 
#define MAXVAL  100.0    
#define MAXSIZE 100  //set the number of variables here
#define OPTIONS 2  // just for the timing thing, left it parametrized for future use. 
#define ITERS 1 //just for the timing thing, left it parametrized for future use. 
#define PRINT_TIME 1

//#define GIG 1000000000
//#define CPG 2.5 //set processor frequency here
#define GIG 1000000
#define NPM 0.001	// Nano second per Microsecond
 
#define mat_elem(a, y, x, n) (a + ((y) * (n) + (x)))

//swap function to implement pivoting on CPU
void swap_row(float *a, float *b, int r1, int r2, int n)
{
	float tmp, *p1, *p2;
	int i;
 
	if (r1 == r2) return;
	for (i = 0; i < n; i++) {
		p1 = mat_elem(a, r1, i, n);
		p2 = mat_elem(a, r2, i, n);
		tmp = *p1, *p1 = *p2, *p2 = tmp;
	}
	tmp = b[r1], b[r1] = b[r2], b[r2] = tmp;
}

//the main worker function on CPU
void gauss_eliminate(float *a, float *b, float *x, int n) 
{
#define A(y, x) (*mat_elem(a, y, x, n))
	int j, col, row, max_row,dia;
	float max, tmp;
	
	//for every column element
	for (dia = 0; dia < n; dia++) {
		max_row = dia, max = A(dia, dia);
		
		//sort the rows from max to min using swap
		for (row = dia + 1; row < n; row++)
			if ((tmp = fabs(A(row, dia))) > max)
				max_row = row, max = tmp;
		swap_row(a, b, dia, max_row, n);
		
		#pragma omp parallel for
		for (row = dia + 1; row < n; row++) {
			tmp = A(row, dia) / A(dia, dia);
			for (col = dia+1; col < n; col++)
				A(row, col) -= tmp * A(dia, col);
			A(row, dia) = 0;
			b[row] -= tmp * b[dia];
		}
	}
	for (row = n - 1; row >= 0; row--) {
		tmp = b[row];
		for (j = n - 1; j > row; j--)
			tmp -= x[j] * A(row, j);
		x[row] = tmp / A(row, row);
	}
#undef A
}
//GPU function
/*
//GPU code version 1, single block
__global__ void gauss_elimination_cuda(float *a_d, float *b_d ,int size) {
	int idx = threadIdx.x;
	int idy = threadIdx.y;

	__shared__ float temp[40][40];
	temp[idy][idx] = a_d[(idy * (size+1)) + idx];
	__syncthreads();
	//cuPrintf("T idy=%d, idx=%d, temp=%f\n", idy, idx, a_d[(idy * (size+1)) + idx]);

	for(int column = 0; column < size-1; column++){
		if(idy > column && idx >= column){
			float t = temp[column][idx] - (temp[column][column] / temp[idy][column]) * temp[idy][idx];
			__syncthreads();
			temp[idy][idx] = t;
		}
		__syncthreads();
	}

	b_d[idy*(size+1) + idx] = temp[idy][idx];
}

//GPU code version 2, single block, each thread works on a row
__global__ void gauss_elimination_cuda_new(float *a_d, float *b_d ,int size) {
	int i, j;
	int idy = threadIdx.x;

	__shared__ float temp[MAXSIZE+10][MAXSIZE+10];
	//copy to share
	for(i=0; i<size+1; i++){
		temp[idy][i] = a_d[(idy * (size+1)) + i];
		//cuPrintf("T idy=%d, num = %d, temp=%f\n", idy, i, temp[idy][i]);
	}
	__syncthreads();
	

	//loop through every row, calculate every column in parallel
	for(i=1; i<size; i++){
		//cuPrintf("\nthread %d(idy) going to loop %d(i)\n", idy, i);
		if(idy >= i){
			float t[MAXSIZE+10];
			//perform calculation
			for(j=0; j<size+1; j++){
				if(j >= i-1){
					t[j] = temp[i-1][j] - (temp[i-1][i-1] / temp[idy][i-1]) * temp[idy][j];
					//cuPrintf("calculate No %d, answer %f\n", j, t);
					
				}
			}
			__syncthreads();
			//store data
			for(j=0; j<size+1; j++){
				if(j >= i-1){
					temp[idy][j] = t[j];
				}
			}
		}
		__syncthreads();
	}

	//copy to host
	for(i=0; i<size+1; i++){
		b_d[idy * (size+1) + i] = temp[idy][i];
	}
}
*/
//GPU code final version, multiple blocks
__global__ void gauss_elimination_cuda_new_blocks(float *a_d, float *b_d ,int size) {
	int i;
	int idx = threadIdx.x;
	int idy = blockIdx.x;
	cuPrintf("idy=%d, idx=%d\n", idy, idx);
	cuPrintf("---%f %f\n", b_d[2], b_d[5]);
	__syncthreads();
	for(i=0; i<=idy; i++){
		if(i == 0){
			__syncthreads();
		}else{
			if(idx >= i-1){
			
			cuPrintf("idy%d idx%d eq : %f - %f/%f * %f\n", idy, idx,a_d[(i-1)*(size+1) + idx], a_d[(i-1)*(size+1) + i-1], a_d[idy*(size+1) + i-1], a_d[idy*(size+1) + idx]);

			b_d[idy*(size+1) + idx] = a_d[(i-1)*(size+1) + idx] - (a_d[(i-1)*(size+1) + i-1]/a_d[idy*(size+1) + i-1]) * a_d[idy*(size+1) + idx];

			cuPrintf("answer%f\n", b_d[idy*(size+1) + idx]);
			
			__syncthreads();
			
			a_d[idy*(size+1) + idx]  = b_d[idy*(size+1) + idx];
			__syncthreads();
			cuPrintf("answer%f\n", b_d[idy*(size+1) + idx]);
			}
		}
	}
	cuPrintf("---%f %f\n", b_d[2], b_d[5]);
	
}


int main(void)
{
	struct timespec diff(struct timespec start, struct timespec end);
	struct timespec time1, time2;
	struct timespec time_stamp[OPTIONS][ITERS+1];

	cudaEvent_t start, stop;
	float elapsed_gpu;
	
	float fRand(float fMin, float fMax); //for random values
	
	int i, j, len;
	len = MAXSIZE; 

	
	float a[len*len], b[len], x[len];
	for (i = 0; i < len*len; i++)
		a[i] = fRand((float)(MINVAL),(float)(MAXVAL)); //initialize with random values
	for (i = 0; i < len; i++)
    		b[i] = fRand((float)(MINVAL),(float)(MAXVAL)); //same thing as a
  	for (i = 0; i < len; i++)
    		x[i] = 0.0; //make it zero

	/*
	float a[] = {6,3,2,4};
	float b[] = {12,10};
	float x[len];
	printf("\nPrinting A\n"); //just for verification purposes m
	for (i = 0; i < len; i++) {
    		printf("\n");
    		for (j = 0; j < len; j++)
      			printf("%.4lf ", a[i*len + j]);
	}
	printf("\n \nPrinting B\n"); 
	for(i=0;i<len;i++)
		printf("%.4lf  ",b[i]);
	
	printf("\n \nPrinting X\n"); //not needed, because x is all 0
	for(i=0;i<len;i++)
		printf("%.4lf",x[i]);
	printf("\n\n");
*/

	//GPU prepare and run -----------------------------------------
	float *in_h;
	float *out_h;
	float *in_d;
	float *out_d;

	size_t allocSize = len*(len+1) * sizeof(float);
	in_h = (float *)malloc(allocSize);
	out_h = (float *)malloc(allocSize);

	CUDA_SAFE_CALL(cudaMalloc((void **)&in_d, allocSize));
	CUDA_SAFE_CALL(cudaMalloc((void **)&out_d, allocSize));

	//copy the a,b into single in_h array
	for(i = 0; i < len; i++){
		for(j = 0; j < len; j++){
			in_h[i*(len+1) + j] = a[i*len + j];
		}
	}
	for(i = 0; i < len; i++){
		in_h[(i+1)*(len+1) - 1] = b[i];
	}

#if PRINT_TIME
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
#endif
	cudaPrintfInit(); 
	CUDA_SAFE_CALL(cudaMemcpy(in_d, in_h, allocSize, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(out_d, in_h, allocSize, cudaMemcpyHostToDevice));
	
	//argument for version1
	//dim3 dimBlock(len+1, len, 1);
	//dim3 dimGrid(1);

	//argument for version2
	//dim3 dimBlock(len);
	//dim3 dimGrid(1);

	//argument for final version
	dim3 dimBlock(len+1);
	dim3 dimGrid(len);

	gauss_elimination_cuda_new_blocks<<<dimGrid, dimBlock>>>(in_d, out_d, len);	
	
	CUDA_SAFE_CALL(cudaPeekAtLastError());
	CUDA_SAFE_CALL(cudaMemcpy(out_h, out_d, allocSize, cudaMemcpyDeviceToHost));
	cudaPrintfDisplay();
	cudaPrintfEnd();
		
#if PRINT_TIME
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed_gpu, start, stop);
	printf("\n\nGPU time: %f (msec)\n", elapsed_gpu);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
#endif
	/*
	printf("\nPrinting out_h\n"); //just for verification purposes m
	for (i = 0; i < len; i++) {
    		printf("\n");
    		for (j = 0; j < len+1; j++)
      			printf("%.4lf ", out_h[i*len + j]);
	}
	*/
//Using Back substitution method
	float *result, sum, rvalue;
	result = (float*)malloc(sizeof(float)*(len));
	for(int i = 0; i < len; i++) {
		result[i] = 1.0;
	}
	for(int i = len - 1; i >= 0; i--) {
		sum = 0.0; 
		for(j = len-1; j > i; j--) {
			sum = sum + result[j] * out_h[i*(len+1) + j];
		}
		rvalue = out_h[i*(len+1) + len] - sum;
		result[i] = rvalue / out_h[i*(len+1) + j];
	}
//calling the CPU function here -------------------------------------------------
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time1);
	gauss_eliminate(a, b, x, len);
    	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time2);
    	time_stamp[0][0] = diff(time1,time2);


	//print result ------------------------------------------------------
	printf("\n \nPrinting CPU Answers\n");
	for (i = 0; i < len; i++)
		printf("%g\n", x[i]);
	printf("\n \nPrinting GPU Answers\n");
	for(int i = 0; i < len+1; i++) {
		printf("%g\n", result[i]);
	} 
	
printf("CPU time: %ld (msec)\n", (long int)(GIG * time_stamp[0][0].tv_sec + NPM * time_stamp[0][0].tv_nsec));
printf("\nGPU time: %f (msec)\n", elapsed_gpu);

	free(in_h);
	free(out_h);
	cudaFree(in_d);
	cudaFree(out_d);

	return 0;
}


float fRand(float fMin, float fMax) //the random function generates same random variable, perhaps there's some way to make it more random. 
{
    float f = (float)random() / RAND_MAX;
    return fMin + f * (fMax - fMin);
}

struct timespec diff(struct timespec start, struct timespec end) //good ol timing function 
{
  struct timespec temp;
  if ((end.tv_nsec-start.tv_nsec)<0) {
    temp.tv_sec = end.tv_sec-start.tv_sec-1;
    temp.tv_nsec = 1000000000+end.tv_nsec-start.tv_nsec;
  } else {
    temp.tv_sec = end.tv_sec-start.tv_sec;
    temp.tv_nsec = end.tv_nsec-start.tv_nsec;
  }
  return temp;
}
