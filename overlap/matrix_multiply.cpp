/*
 * multimat.cpp
 *
 *  Created on: 23/02/2015
 *      Author: Diego Nieto Muñoz
 */

#include <CL/cl_platform.h>
#include <math.h>
#include <string.h>
#include <string>
#include <sys/time.h>
#include <stddef.h>
#include <cstdlib>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include "matrixMul.h"
#include <assert.h>

#define TOL 0.000001

//#define CHECK

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

cl_context context = 0;
cl_command_queue inQueue = 0;
cl_command_queue outQueue = 0;
cl_command_queue kernelQueue = 0;
cl_program program = 0;
cl_mem memObjects[6] = { 0 };

/**
 * @brief Clean OpenCL resources
 */
void cleanUp() {
	clReleaseMemObject(memObjects[0]);
	//clReleaseMemObject(memObjects[1]);
	//clReleaseMemObject(memObjects[2]);
	//clReleaseMemObject(memObjects[3]);
	//clReleaseMemObject(memObjects[4]);
	//clReleaseMemObject(memObjects[5]);
	clReleaseCommandQueue(kernelQueue);
	clReleaseCommandQueue(inQueue);
	clReleaseCommandQueue(outQueue);
	clReleaseProgram(program);
	clReleaseContext(context);
}

/**
 * @brief Print the matrix values
 */
template<typename T>
void printMat(T *mat, int m, int n, const char *msg = NULL) {
	int i, j;
	std::setprecision(6);
	std::cout << "--------------------------------" << std::endl;
	if (msg != NULL) {
		std::cout << msg << std::endl;
		std::cout << "--------------------------------" << std::endl;
	}
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			std::cout << mat[i * n + j] << ' ';
		}
		std::cout << std::endl;
	}
	std::cout << "--------------------------------" << std::endl;
}

/**
 * @brief This function check if the OpenCL function return an error
 */
inline void clCheckError(cl_int clError, const std::string errorString) {
	if (clError != CL_SUCCESS) {
		cleanUp();
		std::cerr << errorString << ": " << clError << std::endl;
		exit( EXIT_FAILURE);
	}
}

/**
 * @brief Show device parameters
 */
void clShowDeviceInfo(cl_device_id device, size_t *maxWorkGroupSize) {
	cl_int clError;
	cl_uint maxComputeUnits;
	cl_ulong globalMemSize;
	cl_ulong localMemSize;

	char *deviceName = new char[1024];
	char *deviceVendor = new char[1024];

	// Get vendor name
	clGetDeviceInfo(device, CL_DEVICE_VENDOR, sizeof(char) * 1024, deviceVendor,
	NULL);

	// Get device name
	clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(char) * 1024, deviceName,
			NULL);

	// Get max work group size
	clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t),
			maxWorkGroupSize, NULL);

	// Get max compute units (STREAM MULTIPROCESSORS)
	clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint),
			&maxComputeUnits, NULL);

	// Get global memory size
	clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong),
			&globalMemSize, NULL);

	// Get local memory size
	clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong),
			&localMemSize, NULL);

	std::cout << "## GPU DEVICE INFO ##############################"
			<< std::endl << std::endl;
	std::cout << "-- " << deviceVendor << " : " << deviceName << " --"
			<< std::endl << std::endl;
	std::cout << "- Number of max work group items: " << *maxWorkGroupSize
			<< std::endl;
	std::cout << "- Number of max compute units:    " << maxComputeUnits
			<< std::endl;
	std::cout << "- Global memory size in bytes:    " << globalMemSize
			<< std::endl;
	std::cout << "- Local memory size in bytes:     " << globalMemSize
			<< std::endl;
	std::cout << "#################################################"
			<< std::endl;
}

/**
 * @brief gemm CPU version
 */
void mxm(int m, int n, int p, const float *left, const float *right,
		float *dest) {
	int i, j, k;
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			dest[i * n + j] += 0; // Added the equal
			for (k = 0; k < p; k++) {
				dest[i * n + j] += left[i * p + k] * right[k * n + j];
			}
		}
	}
}

int check(float *m1, float *m2, const unsigned int dim) {
	for (int i = 0; i < dim; i++)
		if (abs(m1[i] - m2[i]) > TOL) {
			std::cout << "m1[" << i << "]=" << m1[i] << ", m2[" << i << "]="
					<< m2[i] << std::endl;
			return i;
		}
	return -1;
}

void init_matrix(float *mat_A, float *mat_B, unsigned int dim) {
	for (int i = 0; i < dim; i++)
		for (int j = 0; j < dim; j++) {
			mat_A[i * dim + j] = 1.f / (j + 1);
			mat_B[i * dim + j] = 1.f / (j + 1 % 512);
		}
}

int main(int argc, char *argv[]) {
	const char* filename = "matrixMul.cl";
	struct timeval t0, t1, t_gpu, t_cpu, t0_full, t1_full, t_full, t0_minit,
			t1_minit, t_minit, t0_total, t1_total, t_total;

	// Context properties
	cl_platform_id platform;
	cl_uint numPlatforms;
	cl_device_id device;

	// Device info
	size_t maxWorkGroupSize;

	// CL program declarations
	cl_kernel kernel = 0;
	cl_int clError;
	cl_event event;
	const unsigned int iterations = 10;

	gettimeofday(&t0_total, NULL);

	if (argc != 4) {
		std::cout << "Use: " << argv[0]
				<< " <dim> <pinned=0|1> <blockFlag=0|1>, i.e. " << argv[0]
				<< " 512 1 0" << std::endl;
		exit( EXIT_FAILURE);
	}
	cl_int dim = argc > 1 ? atoi(argv[1]) : 512;
	bool pinned = argc > 2 ? atoi(argv[2]) : false;
	cl_bool blockFlag = (argc > 3 ? atoi(argv[3]) : 0) > 0 ? CL_TRUE : CL_FALSE;

	cl_int uiWA = dim;
	cl_int uiWB = dim;
	size_t totalBytes = sizeof(float) * dim * dim;

	/************** Matrix definition **************/
	float *mat_A;
	float *mat_B;
	float *d_mat_C; // Computed on device
	float *h_mat_C = (float*) malloc(totalBytes); // To check
	if (!pinned) {
		mat_A = (float*) malloc(totalBytes);
		mat_B = (float*) malloc(totalBytes);
		d_mat_C = (float*) malloc(totalBytes);

		init_matrix(mat_A, mat_B, dim);
	}

	/************** Create a Context **************/
	clCheckError(clGetPlatformIDs(1, &platform, &numPlatforms),
			"Failed to find any OpenCL platforms.");

	clCheckError(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL),
			"Failed to get any GPU device");
	clShowDeviceInfo(device, &maxWorkGroupSize);

	context = clCreateContext(0, 1, &device, NULL, NULL, &clError);
	clCheckError(clError, "Failed to create the context");

	/************** Create a Command Queue **************/
	kernelQueue = clCreateCommandQueue(context, device, 0, &clError);
	clCheckError(clError, "Failed to create a command kernel queue");
	inQueue = clCreateCommandQueue(context, device, 0, &clError);
	clCheckError(clError, "Failed to create a command in queue");
	outQueue = clCreateCommandQueue(context, device, 0, &clError);
	clCheckError(clError, "Failed to create a command out queue");

	/************** Create a Program **************/
	std::ifstream kernelFile(filename, std::ios::in);
	if (!kernelFile.is_open()) {
		std::cerr << "Failed to open file for reading: " << filename
				<< std::endl;
		return ( EXIT_FAILURE);
	}

	std::ostringstream oss;
	oss << kernelFile.rdbuf();

	std::string srcStdStr = oss.str();
	const char *srcStr = srcStdStr.c_str();
	program = clCreateProgramWithSource(context, 1, (const char**) &srcStr,
	NULL, NULL);
	if (program == NULL) {
		std::cerr << "Failed to create CL program from source." << std::endl;
		return ( EXIT_FAILURE);
	}
	clError = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	if (clError != CL_SUCCESS) {
		// Determine the reason for the error
		char buildLog[16384];
		clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
				sizeof(buildLog), buildLog, NULL);
		std::cerr << "Error in kernel: " << std::endl;
		std::cerr << buildLog;
		clReleaseProgram(program);
		return ( EXIT_FAILURE);
	}

	/************** Create Kernel **************/
	kernel = clCreateKernel(program, "matrixMul_kernel2", &clError);
	clCheckError(clError, "Failed to create kernel");

	gettimeofday(&t0_full, NULL);

	/************** Create Memory objects **************/
	// ALLOC AND COPY WAY
	//		memObjects[0] = clCreateBuffer(context,
	//				CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
	//				totalBytes, mat_A, &clError);
	//		clCheckError(clError, "Failed to create input A buffer");
	//
	//		memObjects[1] = clCreateBuffer(context,
	//				CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
	//				totalBytes, mat_B, &clError);
	//		clCheckError(clError, "Failed to create input B buffer");
	//
	//		memObjects[2] = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
	//				totalBytes, NULL, &clError);
	//		clCheckError(clError, "Failed to create output C buffer");
	memObjects[0] = clCreateBuffer(context,
	CL_MEM_READ_ONLY, totalBytes, mat_A, &clError);
	clCheckError(clError, "Failed to create input A buffer");

	memObjects[1] = clCreateBuffer(context,
	CL_MEM_READ_ONLY, totalBytes, mat_B, &clError);
	clCheckError(clError, "Failed to create input B buffer");

	memObjects[2] = clCreateBuffer(context, CL_MEM_READ_WRITE, totalBytes, NULL,
			&clError);
	clCheckError(clError, "Failed to create output C buffer");

	if (pinned) {
		// Mem objects from 3 to 6 are the link for NVIDIA
		memObjects[3] = clCreateBuffer(context,
		CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, totalBytes, NULL, &clError);
		clCheckError(clError, "Failed to create input link A buffer");

		memObjects[4] = clCreateBuffer(context,
		CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, totalBytes, NULL, &clError);
		clCheckError(clError, "Failed to create input link B buffer");

		memObjects[5] = clCreateBuffer(context,
				CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, totalBytes, NULL,
				&clError);
		clCheckError(clError, "Failed to create output link C buffer");

		// Map input A
		mat_A = (float *) clEnqueueMapBuffer(inQueue, memObjects[3], CL_TRUE,
				CL_MAP_WRITE, 0, totalBytes, 0, NULL, NULL, &clError);
		clCheckError(clError, "Failed to map A buffer");

		// Map input B
		mat_B = (float *) clEnqueueMapBuffer(inQueue, memObjects[4], CL_TRUE,
				CL_MAP_WRITE, 0, totalBytes, 0, NULL, NULL, &clError);

		// Map input C
		d_mat_C = (float *) clEnqueueMapBuffer(outQueue, memObjects[5], CL_TRUE,
				CL_MAP_READ, 0, totalBytes, 0, NULL, NULL, &clError);
		clCheckError(clError, "Failed to map C buffer");
	}

	// Initialize matrix data
	gettimeofday(&t0_minit, NULL);
	init_matrix(mat_A, mat_B, dim);
	gettimeofday(&t1_minit, NULL);
	timersub(&t1_minit, &t0_minit, &t_minit);

	/************** Set Kernel Arguments **************/
	clError = clSetKernelArg(kernel, 0, sizeof(cl_mem), &memObjects[2]);
	clError |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &memObjects[0]);
	clError |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &memObjects[1]);
	clError |= clSetKernelArg(kernel, 3,
			sizeof(float) * BLOCK_SIZE * BLOCK_SIZE, 0);
	clError |= clSetKernelArg(kernel, 4,
			sizeof(float) * BLOCK_SIZE * BLOCK_SIZE, 0);
	clError |= clSetKernelArg(kernel, 5, sizeof(cl_int), (void *) &uiWA);
	clError |= clSetKernelArg(kernel, 6, sizeof(cl_int), (void *) &uiWB);
	clCheckError(clError, "Failed to set kernel args");

	gettimeofday(&t0, NULL);

	cl_event evTransIn[3], evKernel;

	clCheckError(
			clEnqueueWriteBuffer(outQueue, memObjects[2], CL_TRUE, 0,
					totalBytes, d_mat_C, 0, NULL, &evTransIn[2]),
			"Error when writing C buffer");

	for (unsigned int k = 0; k < iterations; k++) {
		/************** Write input buffer **************/
		//mat_A[5] = k;
		//mat_B[5] = k+1;
		clCheckError(
				clEnqueueWriteBuffer(inQueue, memObjects[0], blockFlag, 0,
						totalBytes, mat_A, 0, NULL, evTransIn),
				"Error when writing A buffer");
		clCheckError(
				clEnqueueWriteBuffer(inQueue, memObjects[1], blockFlag, 0,
						totalBytes, mat_B, 0, NULL, &evTransIn[1]),
				"Error when writing B buffer");

		/************** Launch Kernel **************/
		size_t localWorkSize[2] = { BLOCK_SIZE, BLOCK_SIZE };
		size_t globalWorkSize[2] = { dim, dim };

		clError = clEnqueueNDRangeKernel(kernelQueue, kernel, 2, NULL,
				globalWorkSize, localWorkSize, 2, evTransIn, &evKernel);
		clCheckError(clError, "Failed to launch kernel");

		/************** Read output buffer **************/
		clCheckError(
				clEnqueueReadBuffer(outQueue, memObjects[2], blockFlag, 0,
						totalBytes, d_mat_C, 1, &evKernel, NULL),
				"Failed to read result");
		//printMat(d_mat_C, dim, dim);
		//std::cout << "####" << std::endl;
	}

	/************** Measure end time **************/
	gettimeofday(&t1_full, NULL);
	timersub(&t1_full, &t0_full, &t_full);
	timersub(&t_full, &t_minit, &t_full);
	gettimeofday(&t1, NULL);
	timersub(&t1, &t0, &t_gpu);

	/************** Check results **************/
#ifdef CHECK
	if ( dim < 1025 ) {
		for ( unsigned int i=0; i<iterations; i++ )
		mxm(dim, dim, dim, mat_A, mat_B, h_mat_C);

		int line = check(h_mat_C, d_mat_C, dim * dim);
		if (line == -1)
		std::cout << std::endl << "PASS" << std::endl << std::endl;
		else
		std::cout << std::endl <<"FAIL: " << line << std::endl << std::endl;
	}
#endif

	//printMat(mat_A, dim, dim);
	//printMat(mat_B, dim, dim);
	//printMat(h_mat_C, dim, dim);
	//printMat(d_mat_C, dim, dim);

	/************** Release the resources **************/
	//clFinish(inQueue);
	//clFinish(kernelQueue);
	//clFinish(outQueue);
	if (pinned) {
		clCheckError(
				clEnqueueUnmapMemObject(inQueue, memObjects[3], mat_A, 0, NULL,
						&event), "Failed to unmap A buffer");

		clCheckError(
				clEnqueueUnmapMemObject(inQueue, memObjects[4], mat_B, 0, NULL,
						&event), "Failed to unmap B buffer");

		clCheckError(
				clEnqueueUnmapMemObject(outQueue, memObjects[5], d_mat_C, 0,
						NULL, &event), "Failed to unmap C buffer");
	}
	cleanUp();
	gettimeofday(&t1_total, NULL);

	timersub(&t1_total, &t0_total, &t_total);

	std::cout << std::endl
			<< "## PARAMETERS ###################################" << std::endl;
	std::cout << "WorkGroup size = " << BLOCK_SIZE * BLOCK_SIZE << std::endl;
	std::cout << "Threads        = " << dim * dim << std::endl;
	std::cout << "Iterations     = " << iterations << std::endl;
	std::cout << std::setprecision(5) << "GPU =       " << t_gpu.tv_sec << "s, "
			<< t_gpu.tv_usec << "us" << std::endl;
	std::cout << std::setprecision(5) << "GPU + I/O = " << t_full.tv_sec
			<< "s, " << t_full.tv_usec << "us" << std::endl;
	std::cout << std::setprecision(5) << "TOTAL = " << t_total.tv_sec << "s, "
			<< t_total.tv_usec << "us" << std::endl;

	return EXIT_SUCCESS;
}
