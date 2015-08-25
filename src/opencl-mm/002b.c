/*
DO NOT USE!!! --> global_work_size should be set to matrix dimensions divided by local_work_size
 */

__kernel void matrix_mul(const __global int* A,
			 const __global int* B,
			 __global int* C,
			 uint m, uint n, uint p)
{
  // block is square size so getting 1 dimension is just fine :)
  uint block = get_local_size(0); 

  uint row = get_global_id(0);
  uint col = get_global_id(1);

  uint pos = row * p * block + col * block;  

  for (uint i = 0; i < block; ++i)
    for (uint j = 0; j < block; ++j)
      {
	C[pos + p * i + j] = 0;
	for (uint k = 0; k < n; ++k)
	  C[pos + p * i + j] += A[(row * block + i) * n + k] * B[k * p + col * block + j];
      }
}

