/*
 * pi_calc.cpp
 *
 *  Created on: 27/02/2014
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

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#define PI25DT 3.141592653589793238462643
#define NBUFFERS 1

cl_context context = 0;
cl_command_queue commandQueue = 0;
cl_program program = 0;
cl_mem memObjects[NBUFFERS] = { 0 };

/*
 * This function determines pi value in CPU
 */
float pi_cpu(int niter) {
	float pi, x;
	const float h = 1.0 / (float) niter;

	pi = 0.0;
	for(int i=0; i<niter; i++) {
		x = h * ((float) i - 0.5);
	    pi += 4.0 / (1.0 + x*x);
	}

	return h * pi;
}

/*
 * Clean OpenCL resources
 */
void cleanUp() {
	clReleaseMemObject(memObjects[0]);
	clReleaseCommandQueue(commandQueue);
	clReleaseProgram(program);
	clReleaseContext(context);
}

/*
 * This function check if the OpenCL function return an error
 */
inline void clCheckError(cl_int clError, const std::string errorString) {
	if (clError != CL_SUCCESS) {
		cleanUp();
		std::cerr << errorString << std::endl;
		exit( EXIT_FAILURE );
	}
}

/*
 * Show device parameters
 */
void clShowDeviceInfo(cl_device_id device, size_t *maxWorkGroupSize) {
	cl_int clError;
	cl_uint maxComputeUnits;
	cl_ulong globalMemSize;
	cl_ulong localMemSize;

	char *deviceName = new char[1024];
	char *deviceVendor = new char[1024];

	// Get vendor name
	clGetDeviceInfo(device, CL_DEVICE_VENDOR, sizeof(char)*1024, deviceVendor,
			NULL);

	// Get device name
	clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(char)*1024, deviceName, NULL);

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

int main(int argc, char *argv[]) {
	const char* filename1 = "pi_kernel.cl";
	const char* filename2 = "pi_kernel2.cl";
	char filename[1024];
	struct timeval t0, t1, t_gpu, t_cpu;
	int niter, chunks, workGroups;

	// Context properties
	cl_platform_id platform;
	cl_uint numPlatforms;
	cl_device_id device;

	// Device info
	size_t maxWorkGroupSize;

	// CL program declarations
	cl_kernel kernel = 0;
	cl_int clError;

	if (argc < 2) {
		std::cerr << "Type: " << argv[0] << " <n iterations> [chunk-size]"
				<< std::endl;
		return 0;
	} else {
		niter = atoi(argv[1]);
	}

	chunks = argc == 3 ? atoi(argv[2]) : 64;
	if(argc != 4)
		strcpy(filename, filename1);
	else
		strcpy(filename, filename2);

	/************** Create a Context **************/
	clCheckError(clGetPlatformIDs(1, &platform, &numPlatforms),
			"Failed to find any OpenCL platforms.");

	clCheckError(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL),
			"Failed to get any GPU device");
	clShowDeviceInfo(device, &maxWorkGroupSize);
	workGroups = ceil(niter / maxWorkGroupSize / chunks);

	context = clCreateContext(0, 1, &device, NULL, NULL, &clError);
	clCheckError(clError, "Failed to create the context");

	/************** Create a Command Queue **************/
	commandQueue = clCreateCommandQueue(context, device, 0, &clError);
	clCheckError(clError, "Failed to create a command queue");

	/************** Create a Program **************/
	std::ifstream kernelFile(filename, std::ios::in);
	if (!kernelFile.is_open()) {
		std::cerr << "Failed to open file for reading: " << filename
				<< std::endl;
		return( EXIT_FAILURE );
	}

	std::ostringstream oss;
	oss << kernelFile.rdbuf();

	std::string srcStdStr = oss.str();
	const char *srcStr = srcStdStr.c_str();
	program = clCreateProgramWithSource(context, 1, (const char**) &srcStr,
	NULL, NULL);
	if (program == NULL) {
		std::cerr << "Failed to create CL program from source." << std::endl;
		return( EXIT_FAILURE );
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
		return( EXIT_FAILURE );
	}

	/************** Create Kernel **************/
	kernel = clCreateKernel(program, "Pi", &clError);
	clCheckError(clError, "Failed to create kernel");

	/************** Create Memory objects **************/
	memObjects[0] = clCreateBuffer(context, CL_MEM_READ_WRITE,
				sizeof(float)*workGroups, NULL, &clError);
	clCheckError(clError, "Failed to create output buffer");

	/************** Set Kernel Arguments **************/
	clError  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &memObjects[0]);
	clError |= clSetKernelArg(kernel, 1, sizeof(float)*(maxWorkGroupSize), NULL);
	clError |= clSetKernelArg(kernel, 2, sizeof(uint), &niter);
	clError |= clSetKernelArg(kernel, 3, sizeof(uint), &chunks);
	clCheckError(clError, "Failed to set kernel args");

	gettimeofday(&t0, NULL);

	/************** Launch Kernel **************/
	size_t globalWorkSize[1] = { niter / chunks	 };
	size_t localWorkSize[1] = { maxWorkGroupSize };
	clError = clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL,
			globalWorkSize, localWorkSize, 0, NULL, NULL);
	clCheckError(clError, "Failed to launch kernel");

	/************** Read output buffer **************/
	float pi = 0;
	float *pi_partial = new float[workGroups];
	clError = clEnqueueReadBuffer(commandQueue, memObjects[0], CL_TRUE, 0,
			sizeof(float)*workGroups, pi_partial, 0, NULL, NULL);
	clCheckError(clError, "Failed to read result");
	for(int i=0; i<workGroups; i++) {
		pi += pi_partial[i];
	}
	pi *= (1.0/(float)niter);

	gettimeofday(&t1, NULL);
	timersub(&t1, &t0, &t_gpu);

	/************** Release the resources **************/
	cleanUp();

	std::cout << std::endl <<
			"## GPU Results ##################################" << std::endl;
	std::cout << std::setprecision(15) << "Result: " << pi << std::endl;
	std::cout << std::setprecision(15) << "Error:  " << fabs(pi - PI25DT) << std::endl;

	std::cout << std::endl <<
			"## CPU Results ##################################" << std::endl;

	gettimeofday(&t0, NULL);
	float pi_cpu_value = pi_cpu(niter);
	gettimeofday(&t1, NULL);
	timersub(&t1, &t0, &t_cpu);

	std::cout << std::setprecision(15) << "Result: " << pi_cpu_value << std::endl;
	std::cout << std::setprecision(15) << "Error:  " << fabs(pi_cpu_value - PI25DT) << std::endl;

	std::cout << std::endl <<
			"## GPU timing vs CPU timing #####################" << std::endl;
	std::cout << std::setprecision(5) << "GPU = " << t_gpu.tv_usec << "us"
			<< std::endl;
	std::cout << std::setprecision(5) << "CPU = " << t_cpu.tv_usec << "us"
			<< std::endl;

	std::cout << std::endl <<
			"## PARAMETERS ###################################" << std::endl;
	std::cout << "WorkGroups     = " << workGroups << std::endl;
	std::cout << "WorkGroup size = " << maxWorkGroupSize << std::endl;
	std::cout << "Threads        = " << niter/chunks << std::endl;
	std::cout << "Chunk size     = " << chunks << std::endl;
	std::cout << "Iterations     = " << niter << std::endl;

	return EXIT_SUCCESS;
}
