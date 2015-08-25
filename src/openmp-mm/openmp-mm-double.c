#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <assert.h>
#include "omp.h"

// prints given matrix --> used for checking
void print_matrix(double **matrix, int rank)
{
  unsigned int i, j;

  printf("\n----------");

  for (i = 0; i < rank; ++i)
    {
      printf("\n");
      for (j = 0; j < rank; ++j)
	printf("%5.2f ", matrix[i][j]);
    }

  printf("\n");
}

// main function
int main(int argc, char* argv[])
{
  unsigned int i, j, k, rank;
  double starttime, endtime;
  double sum;

  // get rank
  printf("Matrices rank [8,2048]: ");
  scanf("%d", &rank);
  assert( rank >= 8 && rank <= 2048);

  // allocate memory
  // malloc is used because it allocates memory on heap
  // A[rank] would allocate memory on stack and segfault for higher ranks like 1024
  double **A = malloc( sizeof(double*) * rank);
  double **B = malloc( sizeof(double*) * rank);
  double **C = malloc( sizeof(double*) * rank);
  
  for(i=0; i<rank; ++i) {
    A[i] = malloc( sizeof(double) * rank );
    B[i] = malloc( sizeof(double) * rank );
    C[i] = malloc( sizeof(double) * rank );
  }

  // seed random
  srand(time(0));

  // init values: fill A & B with random values
  for (i = 0; i < rank; ++i)
    for (j = 0; j < rank; ++j)
      {
	A[i][j] = (double)(rand()) / RAND_MAX;
	B[i][j] = (double)(rand()) / RAND_MAX;
      }
  // --- multithreaded OpenMP ------------------------------------

  // get start time
  starttime = omp_get_wtime();

  // matrix multiplication: C = A * B
  #pragma omp parallel for private(i, j, k, sum) //schedule(dynamic, 32)
  for (i = 0; i < rank; ++i)
    for (j = 0; j < rank; ++j)
      {
	sum = 0;

	for (k = 0; k < rank; ++ k)
	  sum += A[i][k] * B[k][j];

	C[i][j] = sum;
      }

  // get end time
  endtime = omp_get_wtime();
  
  // print elapsed time
  printf("OpenMP elapsed time: \t\t%f sec\n", endtime - starttime);

  // --- singlethread C ------------------------------------------

  // get start time
  starttime = omp_get_wtime();

  // matrix multiplication: C = A * B
  for (i = 0; i < rank; ++i)
    for (j = 0; j < rank; ++j)
      {
	sum = 0;

	for (k = 0; k < rank; ++ k)
	  sum += A[i][k] * B[k][j];

	C[i][j] = sum;
      }

  // get end time
  endtime = omp_get_wtime();
  
  // print elapsed time
  printf("Singlethread C elapsed time: \t%f sec\n", endtime - starttime);

  // -------------------------------------------------------------

  // print matrices for result checking
  // print_matrix((double **)A, rank);
  // print_matrix((double **)B, rank);
  // print_matrix((double **)C, rank);

  // free memory on heap
  for (i = 0; i < rank; ++i)
    {
      free(A[i]);
      free(B[i]);
      free(C[i]);
    }
  free(A);
  free(B);
  free(C);


  return EXIT_SUCCESS;
}
