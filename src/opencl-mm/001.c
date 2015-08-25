//#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// every working-item calculates one element of matrix

__kernel void matrix_mul(const __global float* A,
			 const __global float* B,
			 __global float* C,
			 uint m, uint n, uint p)
{
  // get indexes
  uint row = get_global_id(0);
  uint col = get_global_id(1);

  // set C value to 0 
  C[row * p + col] = 0;

  // multiply elements of matrices: A (row) and B (column)
  for (uint k = 0; k < n; ++k)
    C[row * p + col] += A[row * n + k] * B[k * p + col];
}

