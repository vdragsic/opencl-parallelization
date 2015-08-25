//#pragma OPENCL EXTENSION cl_khr_fp64 : enable

// every working-group calculates block of elements

__kernel void matrix_mul(const __global float* A,
			 const __global float* B,
			 __global float* C,
			 uint m, uint n, uint p)
{
  // get block size, block is square sized
  uint blockSize = get_local_size(0);

  // get group row and column id
  uint row = get_group_id(0);
  uint col = get_group_id(1);

  // get local ids of working-items
  uint x = get_local_id(0);
  uint y = get_local_id(1);

  // calculate position in matrix --> upper-left position of block + local ids
  uint pos_x = (row * blockSize + x) * p;
  uint pos_y = col * blockSize + y;

  // set value in matrix C to 0
  C[pos_x  + pos_y] = 0;

  // multiply elements of matrices: A (row) with B (column)
  for( uint k = 0; k < n; ++k)
    C[pos_x + pos_y] += A[pos_x + k] * B[k * p + pos_y];
}
