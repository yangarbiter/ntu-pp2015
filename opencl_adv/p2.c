/* header */
#define CL_USE_DEPRECATED_OPENCL_2_0_APIS 
#include <stdio.h>
#include <assert.h>
#include <CL/cl.h>

#define N 768
#define Blk 768
#define BSIDE (N / Blk)
#define MAXGPU 11
#define MAXK 1024
#define MAXLOG 4096
#define DEVICENUM 3
#define ITEMPERDEVICE (N * N / DEVICENUM)
#define NANO2SECOND 1000000000.0

cl_uint A[N][N], B[N][N], C[N][N];
/* main */
int main(int argc, char *argv[])
{
  printf("Hello, OpenCL\n");
  cl_int status;
  cl_platform_id platform_id;
  cl_uint platform_id_got;
  status = clGetPlatformIDs(1, &platform_id, 
			    &platform_id_got);
  assert(status == CL_SUCCESS && platform_id_got == 1);
  printf("%d platform found\n", platform_id_got);
  /* getdevice */
  cl_device_id GPU[MAXGPU];
  cl_uint GPU_id_got;
  cl_device_id CPU[MAXGPU];
  cl_uint CPU_id_got;
  status = 
    clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 
		   MAXGPU, GPU, &GPU_id_got);
  status = 
    clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_CPU, 
		   MAXGPU, CPU, &CPU_id_got);
  assert(status == CL_SUCCESS && 
	 GPU_id_got >= DEVICENUM);
  GPU[1] = CPU[0];
  printf("There are %d GPU devices\n", GPU_id_got); 
  /* getcontext */
  cl_context context = 
    clCreateContext(NULL, 2, GPU, NULL, NULL, 
		    &status);
  assert(status == CL_SUCCESS);
  /* commandqueue */
  cl_command_queue commandQueue[DEVICENUM];
  for (int device = 0; device < 2; device++) {
    commandQueue[device] = 
      clCreateCommandQueue(context, GPU[device],
			   CL_QUEUE_PROFILING_ENABLE, 
			   &status);
    assert(status == CL_SUCCESS);
  }
  /* kernelsource */
  FILE *kernelfp = fopen(argv[1], "r");
  assert(kernelfp != NULL);
  char kernelBuffer[MAXK];
  const char *constKernelSource = kernelBuffer;
  size_t kernelLength = 
    fread(kernelBuffer, 1, MAXK, kernelfp);
  printf("The size of kernel source is %zu\n", kernelLength);
  cl_program program =
    clCreateProgramWithSource(context, 1, &constKernelSource, 
			      &kernelLength, &status);
  assert(status == CL_SUCCESS);
  FILE *kernelfp2 = fopen("m2.cl", "r");
  assert(kernelfp2 != NULL);
  char kernelBuffer2[MAXK];
  const char *constKernelSource2 = kernelBuffer2;
  size_t kernelLength2 = 
    fread(kernelBuffer2, 1, MAXK, kernelfp2);
  printf("The size of kernel2 source is %zu\n", kernelLength2);
  cl_program program2 =
    clCreateProgramWithSource(context, 1, &constKernelSource2, 
			      &kernelLength2, &status);
  assert(status == CL_SUCCESS);
  /* buildprogram */
  status = 
    clBuildProgram(program, 1, GPU, NULL, 
		   NULL, NULL);
  status = 
    clBuildProgram(program2, 1, GPU+1, NULL, 
		   NULL, NULL);
  if (status != CL_SUCCESS) {
    char log[MAXLOG];
    size_t logLength;
    //for (int device = 0; device < DEVICENUM; device++) {
    //  clGetProgramBuildInfo(program, GPU[device], 
	//		    CL_PROGRAM_BUILD_LOG,
	//		    MAXLOG, log, &logLength);
    //  puts(log);
    //}
      clGetProgramBuildInfo(program, GPU[0], 
			    CL_PROGRAM_BUILD_LOG,
			    MAXLOG, log, &logLength);
      puts(log);
      clGetProgramBuildInfo(program2, GPU[1], 
			    CL_PROGRAM_BUILD_LOG,
			    MAXLOG, log, &logLength);
      puts(log);
    exit(-1);
  }
  printf("Build program completes\n");
  /* createkernel */
  cl_kernel kernel = clCreateKernel(program, "mul", &status);
  assert(status == CL_SUCCESS);

  cl_kernel kernel2 = clCreateKernel(program2, "mul2", &status);
  assert(status == CL_SUCCESS);
  printf("Build kernel completes\n");
  /* vector */
  for (int i = 0; i < N; i++) 
    for (int j = 0; j < N; j++) {
      A[i][j] = i + j;
      B[i][j] = i - j;
    }
  /* createbuffer */
  cl_mem bufferA[DEVICENUM];
    bufferA[0] = 
      clCreateBuffer(context, 
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        2*ITEMPERDEVICE * sizeof(cl_uint), 
        ((cl_uint *)A), 
        &status);
    assert(status == CL_SUCCESS);
    bufferA[1] = 
      clCreateBuffer(context, 
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        ITEMPERDEVICE * sizeof(cl_uint), 
        ((cl_uint *)A) + ITEMPERDEVICE * 2, 
        &status);
    assert(status == CL_SUCCESS);
  //for (int device = 0; device < DEVICENUM; device++) {
  //  bufferA[device] = 
  //    clCreateBuffer(context, 
  //      CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
  //      ITEMPERDEVICE * sizeof(cl_uint), 
  //      ((cl_uint *)A) + device * ITEMPERDEVICE * 2, 
  //      &status);
  //  assert(status == CL_SUCCESS);
  //}
  /* bufferB */
  cl_mem bufferB = 
    clCreateBuffer(context, 
      CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
      N * N * sizeof(cl_uint), B, &status);
  assert(status == CL_SUCCESS);
  /* bufferC */
  cl_mem bufferC[DEVICENUM];
    bufferC[0] = 
      clCreateBuffer(context, 
        CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR,
        2*ITEMPERDEVICE * sizeof(cl_uint), 
        ((cl_uint *) C), 
        &status);
    assert(status == CL_SUCCESS);
    bufferC[1] = 
      clCreateBuffer(context, 
        CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR,
        ITEMPERDEVICE * sizeof(cl_uint), 
        ((cl_uint *) C) + ITEMPERDEVICE * 2, 
        &status);
    assert(status == CL_SUCCESS);
  //for (int device = 0; device < DEVICENUM; device++) {
  //  if(device == 1){
  //      bufferC[device] = 
  //        clCreateBuffer(context, 
  //          CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR,
  //          ITEMPERDEVICE * sizeof(cl_uint), 
  //          ((cl_uint *) C) + device * ITEMPERDEVICE * 2, 
  //          &status);
  //      assert(status == CL_SUCCESS);
  //  }
  //}
  printf("Build buffers completes\n");
  /* NDRange */
  size_t globalThreads[] = 
    {(size_t)(N*2 / DEVICENUM), (size_t)N};
  size_t localThreads[] = {BSIDE, BSIDE};
  cl_event events[DEVICENUM];
    status = clSetKernelArg(kernel, 0, sizeof(cl_mem), 
    		    (void*)(&bufferA[0]));
    status |= clSetKernelArg(kernel, 1, sizeof(cl_mem), 
    		    (void*)&bufferB);
    status |= clSetKernelArg(kernel, 2, sizeof(cl_mem), 
    		    (void*)(&bufferC[0]));
    assert(status == CL_SUCCESS);
    printf("Set kernel arguments completes\n");
    printf("%d\n", status);
    assert(status == CL_SUCCESS);

    status = clSetKernelArg(kernel2, 0, sizeof(cl_mem), 
    		    (void*)(&bufferA[1]));
    status |= clSetKernelArg(kernel2, 1, sizeof(cl_mem), 
    		    (void*)&bufferB);
    status |= clSetKernelArg(kernel2, 2, sizeof(cl_mem), 
    		    (void*)(&bufferC[1]));
    assert(status == CL_SUCCESS);
    printf("Set kernel2 arguments completes\n");
/* startkernel */
    status = 
      clEnqueueNDRangeKernel(commandQueue[0], 
    		     kernel, 2, NULL, 
    		     globalThreads, localThreads, 
    		     0, NULL, &(events[0]));
  size_t globalThreads2[] = 
    {(size_t)(N / DEVICENUM), (size_t)N};
    status = 
      clEnqueueNDRangeKernel(commandQueue[1], 
    		     kernel2, 2, NULL, 
    		     globalThreads2, localThreads, 
    		     0, NULL, &(events[1]));
    assert(status == CL_SUCCESS);
  /* setarg */
  //for (int device = 0; device < DEVICENUM; device++) {
  //  status = clSetKernelArg(kernel, 0, sizeof(cl_mem), 
  //  		    (void*)(&bufferA[device]));
  //  assert(status == CL_SUCCESS);
  //  status = clSetKernelArg(kernel, 1, sizeof(cl_mem), 
  //  		    (void*)&bufferB);
  //  assert(status == CL_SUCCESS);
  //  status = clSetKernelArg(kernel, 2, sizeof(cl_mem), 
  //  		    (void*)(&bufferC[device]));
  //  assert(status == CL_SUCCESS);
  //  printf("Set kernel arguments completes\n");
  //  /* startkernel */
  //  status = 
  //    clEnqueueNDRangeKernel(commandQueue[device], 
  //  		     kernel, 2, NULL, 
  //  		     globalThreads, localThreads, 
  //  		     0, NULL, &(events[device]));
  //  assert(status == CL_SUCCESS);
  //}
  /* waitforevent */
  clWaitForEvents(2, events); 
  printf("Kernel execution completes.\n");
  /* getbase */
  cl_ulong base[2];
  status = 
    clGetEventProfilingInfo(events[0], 
      CL_PROFILING_COMMAND_QUEUED, 
      sizeof(cl_ulong), &(base[0]), NULL);
  assert(status == CL_SUCCESS);
  status = 
    clGetEventProfilingInfo(events[1], 
      CL_PROFILING_COMMAND_QUEUED, 
      sizeof(cl_ulong), &(base[1]), NULL);
  assert(status == CL_SUCCESS);
  /* gettime */
  for (int device = 0; device < 2; device++) {
    cl_ulong timeEnterQueue, timeSubmit, timeStart, 
      timeEnd;
    status = 
      clGetEventProfilingInfo(events[device], 
        CL_PROFILING_COMMAND_QUEUED, 
        sizeof(cl_ulong), &timeEnterQueue, NULL);
    assert(status == CL_SUCCESS);
    status = 
      clGetEventProfilingInfo(events[device], 
        CL_PROFILING_COMMAND_SUBMIT, 
        sizeof(cl_ulong), &timeSubmit, NULL);
    assert(status == CL_SUCCESS);
    /* getrest */
    status = 
      clGetEventProfilingInfo(events[device], 
        CL_PROFILING_COMMAND_START, 
        sizeof(cl_ulong), &timeStart, NULL);
    assert(status == CL_SUCCESS);
    status = 
      clGetEventProfilingInfo(events[device], 
        CL_PROFILING_COMMAND_END, 
         sizeof(cl_ulong), &timeEnd, NULL); 
    assert(status == CL_SUCCESS);
    /* printtime */
    printf("\nkernel entered queue at %f\n", 
	   (timeEnterQueue - base[device]) / NANO2SECOND);
    printf("kernel submitted to device at %f\n", 
	   (timeSubmit - base[device]) / NANO2SECOND);
    printf("kernel started at %f\n", 
	   (timeStart - base[device]) / NANO2SECOND);
    printf("kernel ended  at %f\n", 
	   (timeEnd - base[device]) / NANO2SECOND);
    printf("kernel queued time %f seconds\n", 
	   (timeSubmit - timeEnterQueue) / NANO2SECOND);
    printf("kernel submission time %f seconds\n", 
	   (timeStart - timeSubmit) / NANO2SECOND);
    printf("kernel execution time %f seconds\n", 
	 (timeEnd - timeStart) / NANO2SECOND);
  }  
  /* checkandfree */
  for (int i = 0; i < N; i++){
    for (int j = 0; j < N; j++) {
      int sum = 0;
      for (int k = 0; k < N; k++)
        sum += A[i][k] * B[k][j];
      assert(C[i][j] == sum);
    }
  }

  clReleaseContext(context);	/* context etcmake */
  for (int device = 0; device < 2; device++) {
    clReleaseCommandQueue(commandQueue[device]);
    clReleaseMemObject(bufferA[device]);	/* buffers */
    clReleaseMemObject(bufferC[device]);
  }
  clReleaseProgram(program);
  clReleaseKernel(kernel);
  clReleaseMemObject(bufferB);
  return 0;
}
/* end */
