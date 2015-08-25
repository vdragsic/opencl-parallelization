#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <CL/cl.h>

#define SIZE 1024

// Izvorni programski kod OpenCL programske jezgre
const char* oclSource[] = {
  "__kernel void sum_arrays(__global int *in_a, "
  "                         __global int *in_b, __global int *out_c)",
  "{",
  "    uint n = get_global_id(0);",
  "    out_c[n] = in_a[n] + in_b[n];",
  "}",

};

int main (int argc, char **argv)
{
  // Alokacija memorije za dva ulazna niza
  int array_1[SIZE], array_2[SIZE];

  // Inicijalizacija nasumičnih vrijednosti ulaznih nizova
  srandom(time(0));
  for (int i=0; i<SIZE; ++i)
    {
      array_1[i] = ((float)random() / RAND_MAX) * 100;
      array_2[i] = ((float)random() / RAND_MAX) * 100;
    }

  // Stvaranje OpenCL konteksta za GPU OpenCL uređaje
  cl_context oclContext = clCreateContextFromType(0, CL_DEVICE_TYPE_GPU,  NULL, NULL,NULL);

  // Dobivanje veličine memorije za listu uređaja u kontekstu
  size_t paramSize;
  clGetContextInfo(oclContext, CL_CONTEXT_DEVICES, 0, NULL, &paramSize);

  // Dobivanje liste uređaja u kontekstu
  cl_device_id* oclDevices = (cl_device_id*)malloc(paramSize);
  clGetContextInfo(oclContext, CL_CONTEXT_DEVICES, paramSize, oclDevices, NULL);

  // Stvaranje programskog redoslijeda u kontekstu za prvi uređaj (0)
  cl_command_queue oclCommandQueue = clCreateCommandQueue(oclContext, oclDevices[0], 0, NULL);

  // Stvaranje memorijskih objekata međuspremnika, uređaj ih može samo čitati
  // Alokacija memorije na uređaja i kopiranje vrijednosti iz ulaznih nizova na domaćinu
  cl_mem oclInBuffer_1 = clCreateBuffer(oclContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int) * SIZE, array_1, NULL);
  cl_mem oclInBuffer_2 = clCreateBuffer(oclContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int) * SIZE, array_2, NULL);

  // Stvaranje memorijskog objekta međuspremnika, uređaj može samo pisati u njega
  cl_mem oclOutBuffer = clCreateBuffer(oclContext, CL_MEM_WRITE_ONLY, sizeof(int) * SIZE, NULL, NULL);
  
  // Stvaranje objekta programa iz izvornog programskog koda
  cl_program oclProgram = clCreateProgramWithSource(oclContext, 6, oclSource, NULL, NULL);

  // Prevođenje programa za sve uređaje u konektstu
  // (engl JIT, Just In Time Compilation)
  clBuildProgram(oclProgram, 0, NULL, NULL, NULL, NULL);

  // Stvaranje poveznice sa funkcijom (sum_arrays) unutar programske jezgre
  cl_kernel oclSumArrays = clCreateKernel(oclProgram, "sum_arrays", NULL);

  // Postavljanje argumenata za programsku jezgru u GPU memoriju
  clSetKernelArg(oclSumArrays, 0, sizeof(cl_mem), (void*)&oclInBuffer_1);
  clSetKernelArg(oclSumArrays, 1, sizeof(cl_mem), (void*)&oclInBuffer_2);
  clSetKernelArg(oclSumArrays, 2, sizeof(cl_mem), (void*)&oclOutBuffer);

  // Pokretanje izvođenja programske jezgre
  // Argumentima se specificira programski redoslijed, objekt programske jezgre i veličina indeksnog prostora
  size_t WorkSize[1] = {SIZE}; // index space
  clEnqueueNDRangeKernel(oclCommandQueue, oclSumArrays, 1, NULL, WorkSize, NULL, 0, NULL, NULL);

  // Alokacija memorije na domaćinu za rezultate
  // Kopiranje vrijednosti iz objekta međuspremnika u memoriju domaćina
  int array_res[SIZE];
  clEnqueueReadBuffer(oclCommandQueue, oclOutBuffer, CL_TRUE, 0, sizeof(int) * SIZE, array_res,0, NULL, NULL);

  // Ispis rezultata 
  for (int i=0; i<SIZE; i++)
      printf("%d\t%d\t%d\n", array_1[i], array_2[i], array_res[i]);

  // Oslobađanje memorije objekata
  free(oclDevices);
  clReleaseKernel(oclSumArrays);
  clReleaseProgram(oclProgram);
  clReleaseCommandQueue(oclCommandQueue);
  clReleaseContext(oclContext);
  clReleaseMemObject(oclInBuffer_1);
  clReleaseMemObject(oclInBuffer_2);
  clReleaseMemObject(oclOutBuffer);

  return 0;
}
