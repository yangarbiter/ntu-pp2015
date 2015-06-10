/* constant */
#define N 512
__kernel void add(__global int A[N][N], __global int B[N][N], __global int C[N][N])
{
  int globalRow = get_global_id(0);
  int globalCol = get_global_id(1);

  C[globalRow][globalCol] = A[globalRow][globalCol] + B[globalRow][globalCol];
}
/* end */
