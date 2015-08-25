#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define RANK 512

// print matrix, used only for checking
void print_matrix(double *matrix)
{
  unsigned int i;

  printf("\n----------");

  for (i = 0; i < RANK*RANK; i++)
    {
      if (i % RANK == 0) printf("\n");
      printf("%5.2f ", matrix[i]);
    }

  printf("\n");
}

// main function
int main(int argc, char *argv[])
{
  int myid, numprocs;
  unsigned int i, j, njobs, start, end, row, col;
  double starttime, endtime;

  // seed random
  srand (time(0));

  // allocate memory for matrices on heap
  double *A = malloc(sizeof(double) * RANK * RANK);
  double *B = malloc(sizeof(double) * RANK * RANK);
  double *C = malloc(sizeof(double) * RANK * RANK);
  double *D = malloc(sizeof(double) * RANK * RANK);

  // init MPI
  MPI_Init (&argc, &argv);
  MPI_Comm_size (MPI_COMM_WORLD, &numprocs);
  MPI_Comm_rank (MPI_COMM_WORLD, &myid);
    
  // master populates A and B with random values [0, 1]
  if (myid == 0)
    {
      for (i = 0; i < RANK*RANK; i++)
	{
	  A[i] = (double)(rand()) / RAND_MAX;
	  B[i] = (double)(rand()) / RAND_MAX; 
	}
    }

  // get number of jobs per proc
  njobs = (int)(RANK*RANK / numprocs);

  // calcucalte start&end indexes
  start = myid * njobs;
  end = (myid == numprocs-1) ? RANK * RANK - 1 : (myid + 1) * njobs - 1;

  // set C to 0
  for (i = 0; i < RANK*RANK; i++)
    C[i] = 0;

  // broadcast whole matrices A and B
  // isn't optimal way od communication, but it doesn't matter for this test
  MPI_Bcast (A ,RANK * RANK, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast (B ,RANK * RANK, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  // sync & start timer
  MPI_Barrier(MPI_COMM_WORLD);
  if (myid == 0) starttime = MPI_Wtime();

  // matrices multiplication: C = A * B
  // printf ("Proc %d calculates range [%d, %d]\n", myid, start, end);
  for (i = start; i <= end; ++i)
    {
      row = i / RANK;
      col = i % RANK;

      for (j = 0; j < RANK; ++j)
	  C[i] +=  A[row * RANK + j] * B[j * RANK + col];
    }

  // collect results, sum values in C into D on server
  MPI_Reduce(C, D, RANK*RANK, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  // stop timer & display result
  if (myid == 0)
    {
      endtime = MPI_Wtime();
      printf("Calculation took %f seconds!\n", endtime - starttime);
    }

  /*
  if (myid == 0)
    {
      print_matrix(A);
      print_matrix(B);
      print_matrix(D);
    }
  */

  MPI_Finalize();

  // free memory
  free(A);
  free(B);
  free(C);
  free(D);

  return EXIT_SUCCESS;
}


/*
-- compile MPI program
mpicc -Wall mpi-mm-double.c -o mpi-mm-double

-- run MPI program
mpirun -np 2 mpi-mm-double

-- run MPI program in gdb
mpirun -np 2 --debug --debugger gdb mpi-mm-double
*/

