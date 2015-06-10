/* header */
#define CL_USE_DEPRECATED_OPENCL_2_0_APIS 
#include <stdio.h>
#include <assert.h>
#include <CL/cl.h>

#define N 512
#define Blk 512
#define BSIDE (N / Blk)
#define MAXGPU 10
#define MAXK 1024
#define MAXLOG 4096
#define DEVICENUM 2
#define NANO2SECOND 1000000000.0

cl_uint A[N][N], B[N][N], C[N][N], D[N][N];
cl_uint E[N][N], F[N][N], G[N][N], H[N][N];
cl_uint I[N][N], J[N][N], X[N][N], Y[N][N];
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
  status = 
    clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 
		   MAXGPU, GPU, &GPU_id_got);
  assert(status == CL_SUCCESS && 
	 GPU_id_got >= DEVICENUM);
  printf("There are %d GPU devices\n", GPU_id_got); 
  /* getcontext */
  cl_context context = 
    clCreateContext(NULL, DEVICENUM, GPU, NULL, NULL, 
		    &status);
  assert(status == CL_SUCCESS);
  /* commandqueue */
  cl_command_queue commandQueue[DEVICENUM];
  for (int device = 0; device < DEVICENUM; device++) {
    commandQueue[device] = 
      clCreateCommandQueue(context, GPU[device],
			   CL_QUEUE_PROFILING_ENABLE, 
			   &status);
    assert(status == CL_SUCCESS);
  }
  /* kernelsource */
  FILE *mkernelfp = fopen("mul.cl", "r");
  assert(mkernelfp != NULL);
  char mkernelBuffer[MAXK];
  const char *mconstKernelSource = mkernelBuffer;
  size_t mkernelLength = fread(mkernelBuffer, 1, MAXK, mkernelfp);
  printf("The size of kernel source is %zu\n", mkernelLength);
  cl_program program_mul =
    clCreateProgramWithSource(context, 1, &mconstKernelSource, &mkernelLength, &status);
  assert(status == CL_SUCCESS);

  FILE *akernelfp = fopen("add.cl", "r");
  assert(akernelfp != NULL);
  char akernelBuffer[MAXK];
  const char *aconstKernelSource = akernelBuffer;
  size_t akernelLength = fread(akernelBuffer, 1, MAXK, akernelfp);
  printf("The size of kernel source is %zu\n", akernelLength);
  cl_program program_add =
    clCreateProgramWithSource(context, 1, &aconstKernelSource, &akernelLength, &status);
  assert(status == CL_SUCCESS);

  /* buildprogram */
  status = clBuildProgram(program_mul, DEVICENUM, GPU, NULL, NULL, NULL);
  status = clBuildProgram(program_add, DEVICENUM, GPU, NULL, NULL, NULL);
  if (status != CL_SUCCESS) {
    char log[MAXLOG];
    size_t logLength;
    for (int device = 0; device < DEVICENUM; device++) {
      clGetProgramBuildInfo(program_mul, GPU[device], 
			    CL_PROGRAM_BUILD_LOG, MAXLOG, log, &logLength);
      puts(log);
      clGetProgramBuildInfo(program_add, GPU[device], 
			    CL_PROGRAM_BUILD_LOG, MAXLOG, log, &logLength);
      puts(log);
    }
    exit(-1);
  }
  printf("Build program completes\n");
  /* createkernel */
  cl_kernel kernel_mul = clCreateKernel(program_mul, "mul", &status);
  cl_kernel kernel_add = clCreateKernel(program_add, "add", &status);
  assert(status == CL_SUCCESS);
  printf("Build kernel completes\n");
  /* vector */
  for (int i = 0; i < N; i++) 
    for (int j = 0; j < N; j++) {
      A[i][j] = i + j;
      B[i][j] = i - j;
      C[i][j] = i;
      D[i][j] = j;
      E[i][j] = i + j;
      F[i][j] = i - j;
    }
  /* createbuffer */
  cl_mem bufferA =
    clCreateBuffer(context, 
      CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
      N*N * sizeof(cl_uint), A, &status);
  assert(status == CL_SUCCESS);
  cl_mem bufferB = 
    clCreateBuffer(context, 
      CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
      N * N * sizeof(cl_uint), B, &status);
  assert(status == CL_SUCCESS);
  cl_mem bufferC = 
    clCreateBuffer(context, 
      CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
      N * N * sizeof(cl_uint), C, &status);
  assert(status == CL_SUCCESS);
  cl_mem bufferD = 
    clCreateBuffer(context, 
      CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
      N * N * sizeof(cl_uint), D, &status);
  assert(status == CL_SUCCESS);
  cl_mem bufferE = 
    clCreateBuffer(context, 
      CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
      N * N * sizeof(cl_uint), E, &status);
  assert(status == CL_SUCCESS);
  cl_mem bufferF = 
    clCreateBuffer(context, 
      CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
      N * N * sizeof(cl_uint), F, &status);
  assert(status == CL_SUCCESS);

  /* bufferC */
  cl_mem bufferG =
    clCreateBuffer(context, 
      CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
      N * N * sizeof(cl_uint), G, &status);
  assert(status == CL_SUCCESS);
  cl_mem bufferH =
    clCreateBuffer(context, 
      CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
      N * N * sizeof(cl_uint), H, &status);
  assert(status == CL_SUCCESS);
  cl_mem bufferI =
    clCreateBuffer(context, 
      CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
      N * N * sizeof(cl_uint), I, &status);
  assert(status == CL_SUCCESS);
  cl_mem bufferJ =
    clCreateBuffer(context, 
      CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
      N * N * sizeof(cl_uint), J, &status);
  assert(status == CL_SUCCESS);

  /* Ans */
  cl_mem bufferX =
    clCreateBuffer(context, 
      CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR,
      N * N * sizeof(cl_uint), X, &status);
  assert(status == CL_SUCCESS);
  cl_mem bufferY =
    clCreateBuffer(context, 
      CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR,
      N * N * sizeof(cl_uint), Y, &status);
  assert(status == CL_SUCCESS);
  printf("Build buffers completes\n");

  /* NDRange */
  size_t globalThreads[] = 
    {(size_t)N, (size_t)N};
  size_t localThreads[] = {BSIDE, BSIDE};
  cl_event events[6];

  /* G=A*B H=C*D */
  /* setarg */
  status |= clSetKernelArg(kernel_mul, 0, sizeof(cl_mem), (void*)(&bufferA));
  status |= clSetKernelArg(kernel_mul, 1, sizeof(cl_mem), (void*)&bufferB);
  status = clSetKernelArg(kernel_mul, 2, sizeof(cl_mem), (void*)(&bufferG));
  assert(status == CL_SUCCESS);
  printf("Set kernel arguments completes\n");
  /* startkernel */
  status = clEnqueueNDRangeKernel(commandQueue[0], kernel_mul, 2, NULL, 
       globalThreads, localThreads, 0, NULL, &(events[0]));
  assert(status == CL_SUCCESS);

  /* setarg */
  status |= clSetKernelArg(kernel_mul, 0, sizeof(cl_mem), (void*)(&bufferC));
  status |= clSetKernelArg(kernel_mul, 1, sizeof(cl_mem), (void*)&bufferD);
  status = clSetKernelArg(kernel_mul, 2, sizeof(cl_mem), (void*)(&bufferH));
  assert(status == CL_SUCCESS);
  printf("Set kernel arguments completes\n");
  /* startkernel */
  status = clEnqueueNDRangeKernel(commandQueue[1], kernel_mul, 2, NULL, 
       globalThreads, localThreads, 0, NULL, &(events[1]));
  assert(status == CL_SUCCESS);

  //clFinish(commandQueue[0]);
  //clFinish(commandQueue[1]);

  /* I=G*E J=H*F */
  /* setarg */
  status |= clSetKernelArg(kernel_mul, 0, sizeof(cl_mem), (void*)(&bufferG));
  status |= clSetKernelArg(kernel_mul, 1, sizeof(cl_mem), (void*)&bufferE);
  status = clSetKernelArg(kernel_mul, 2, sizeof(cl_mem), (void*)(&bufferI));
  assert(status == CL_SUCCESS);
  printf("Set kernel arguments completes\n");
  /* startkernel */
  status = clEnqueueNDRangeKernel(commandQueue[0], kernel_mul, 2, NULL, 
       globalThreads, localThreads, 2, events, &(events[2]));
  assert(status == CL_SUCCESS);

  /* setarg */
  status |= clSetKernelArg(kernel_mul, 0, sizeof(cl_mem), (void*)(&bufferH));
  status |= clSetKernelArg(kernel_mul, 1, sizeof(cl_mem), (void*)&bufferF);
  status = clSetKernelArg(kernel_mul, 2, sizeof(cl_mem), (void*)(&bufferJ));
  assert(status == CL_SUCCESS);
  printf("Set kernel arguments completes\n");
  /* startkernel */
  status = clEnqueueNDRangeKernel(commandQueue[1], kernel_mul, 2, NULL, 
       globalThreads, localThreads, 2, events, &(events[3]));
  assert(status == CL_SUCCESS);

  //clFinish(commandQueue[0]);
  //clFinish(commandQueue[1]);

  /* X=G+H Y=I+J */
  /* setarg */
  status |= clSetKernelArg(kernel_add, 0, sizeof(cl_mem), (void*)(&bufferG));
  status |= clSetKernelArg(kernel_add, 1, sizeof(cl_mem), (void*)&bufferH);
  status = clSetKernelArg(kernel_add, 2, sizeof(cl_mem), (void*)(&bufferX));
  assert(status == CL_SUCCESS);
  printf("Set kernel arguments completes\n");
  /* startkernel */
  status = clEnqueueNDRangeKernel(commandQueue[0], kernel_add, 2, NULL, 
       globalThreads, localThreads, 2, events+2, &(events[4]));
  assert(status == CL_SUCCESS);

  /* setarg */
  status |= clSetKernelArg(kernel_add, 0, sizeof(cl_mem), (void*)(&bufferI));
  status |= clSetKernelArg(kernel_add, 1, sizeof(cl_mem), (void*)&bufferJ);
  status = clSetKernelArg(kernel_add, 2, sizeof(cl_mem), (void*)(&bufferY));
  assert(status == CL_SUCCESS);
  printf("Set kernel arguments completes\n");
  /* startkernel */
  status = clEnqueueNDRangeKernel(commandQueue[1], kernel_add, 2, NULL, 
       globalThreads, localThreads, 2, events+2, &(events[5]));
  assert(status == CL_SUCCESS);

  //clFinish(commandQueue[0]);
  //clFinish(commandQueue[1]);

  /* waitforevent */
  clWaitForEvents(2, events+4); 
  printf("Kernel execution completes.\n");
  /* getbase */
  cl_ulong base;
  status = 
    clGetEventProfilingInfo(events[0], 
      CL_PROFILING_COMMAND_QUEUED, sizeof(cl_ulong), &base, NULL);
  assert(status == CL_SUCCESS);
  /* gettime */
  for (int device = 0; device < 6; device++) {
    cl_ulong timeEnterQueue, timeSubmit, timeStart, timeEnd;
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
	   (timeEnterQueue - base) / NANO2SECOND);
    printf("kernel submitted to device at %f\n",
	   (timeSubmit - base) / NANO2SECOND);
    printf("kernel started at %f\n",
	   (timeStart - base) / NANO2SECOND);
    printf("kernel ended  at %f\n",
	   (timeEnd - base) / NANO2SECOND);
    printf("kernel queued time %f seconds\n",
	   (timeSubmit - timeEnterQueue) / NANO2SECOND);
    printf("kernel submission time %f seconds\n",
	   (timeStart - timeSubmit) / NANO2SECOND);
    printf("kernel execution time %f seconds\n",
	 (timeEnd - timeStart) / NANO2SECOND);
  }  
  /* checkandfree */
  int P[N][N], Q[N][N];
  for (int i = 0; i < N; i++)
    for (int j = 0; j < N; j++) {
      int sum1=0, sum2=0;
      for (int k = 0; k < N; k++){
        sum1 += A[i][k] * B[k][j];
        sum2 += C[i][k] * D[k][j];
      }
      P[i][j] = sum1;
      Q[i][j] = sum2;
      assert(G[i][j] == P[i][j]);
      assert(H[i][j] == Q[i][j]);
      assert(X[i][j] == sum1+sum2);
    }
  for (int i = 0; i < N; i++)
    for (int j = 0; j < N; j++) {
      int sum1=0, sum2=0;
      for (int k = 0; k < N; k++){
        sum1 += P[i][k] * E[k][j];
        sum2 += Q[i][k] * F[k][j];
      }
      assert(Y[i][j] == sum1+sum2);
    }

  clReleaseContext(context);	/* context etcmake */
  for (int device = 0; device < DEVICENUM; device++) {
    clReleaseCommandQueue(commandQueue[device]);
  }
  clReleaseProgram(program_add);
  clReleaseProgram(program_mul);
  clReleaseKernel(kernel_add);
  clReleaseKernel(kernel_mul);
  clReleaseMemObject(bufferA);	/* buffers */
  clReleaseMemObject(bufferB);	/* buffers */
  clReleaseMemObject(bufferC);
  clReleaseMemObject(bufferD);	/* buffers */
  clReleaseMemObject(bufferE);	/* buffers */
  clReleaseMemObject(bufferF);
  return 0;
}
/* end */
