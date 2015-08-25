//#pragma OPENCL EXTENSION cl_khr_fp64 : enable

// every working-group calculates block of elements
// blocks are copied to local memory

__kernel void matrix_mul(const __global float* A,
			 const __global float* B,
			 __global float* C,
			 uint m, uint n, uint p,
			 __local float* subA,
			 __local float* subB)
{
  // get number of groups and block size
  uint numGroups = get_num_groups(0);
  uint blockSize = get_local_size(0);

  // get group row and column id
  uint row = get_group_id(0);
  uint col = get_group_id(1);

  // get local ids of working-items
  uint x = get_local_id(0);
  uint y = get_local_id(1);

  // set value in matrix C to 0
  float subC = 0;

  // iterate through all blocks
  for(int blockA = row * blockSize * n , blockB = col * blockSize;
      blockA <= row * blockSize * n + numGroups * blockSize - 1;
      blockA += blockSize, blockB += blockSize * p)
    {
      // copy elements from matrices A and B to submatrices in local memory
      // every working-item copies one element
      subA[x * blockSize + y] = A[blockA + x * n + y];
      subB[x * blockSize + y] = B[blockB + x * p + y];

      // wait for all elements to be copied to local memory
      barrier(CLK_LOCAL_MEM_FENCE);
      
      // multiply elements of submatrices
      for(int k = 0; k < blockSize; ++k)
	subC += subA[x * blockSize + k] * subB[k * blockSize + y];

      // wait for all working-items to finish
      barrier(CLK_LOCAL_MEM_FENCE);
    }

  // set element in matrix C
  C[get_global_id(0) * get_global_size(0) + get_global_id(1)] = subC;
}
