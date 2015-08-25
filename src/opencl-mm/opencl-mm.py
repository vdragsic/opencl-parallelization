#!/usr/bin/python

import pyopencl as cl
import numpy as np
import sys, getopt

class MatrixMul:
    """
    Base class for openCL tests!.
    """
    
    def LoadKernelSrc(self, filename):
        """
        Tries to load kernel source code from file and returns it.
        """
        
        try:
            f = open(filename, 'r')
            kernel = f.read()
        except IOError, err:
            print str(err)
            sys.exit(2)

        return kernel

    def __init__(self, ctx, queue, dim, block=1, dataType=np.float32, A=False, B=False):
        """
        Init function, executed every time an object is created.
        Sets values for kernel run.
        """
        
        # get dimensions
        (self.m, self.n, self.p) = dim

        self.ctx = ctx
        self.queue = queue
        self.block = block
        
        # generate random-filled test matrices
        self.A = A if A.any() else GenMat(self.m, self.n, dataType)
        self.B = B if B.any() else GenMat(self.n, self.p, dataType)
        self.C = np.zeros([self.m,self.p], dtype=dataType)

        # allocate memory for openCL buffers: matrices A, B, C
        mf = cl.mem_flags
        self.A_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.A)
        self.B_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.B)
        self.C_buf = cl.Buffer(ctx, mf.WRITE_ONLY, self.m * self.p * self.A.itemsize)    

    def execute(self):
        """
        Runs test openCL kernel and returns elapsed time.
        """

        kernel = self.LoadKernelSrc(self.src)

        # build opencl kernel
        prg = cl.Program(self.ctx, kernel).build()

        exec_evt = prg.matrix_mul(self.queue, (self.m, self.p,),
                                       self.A_buf, self.B_buf, self.C_buf,
                                       np.uint32(self.m), np.uint32(self.n), np.uint32(self.p),
                                       local_size=(self.block, self.block,),
                                       ).wait()

        # read result from opencl buffer
        cl.enqueue_read_buffer(self.queue, self.C_buf, self.C).wait()

        # return elapsed time in seconds
        return 1e-9 * (exec_evt.profile.end - exec_evt.profile.start)


class MM_001(MatrixMul):

    src = "001.c"

    def __str__(self):
        return "Test 001"

    
class MM_002(MatrixMul):

    src = "002.c"

    def __str__(self):
        return "Test 002"


class MM_003(MatrixMul):

    src = "003.c"

    # this test uses openCL local memory so arguments to kernel are a bit different.
    def execute(self):

        kernel = self.LoadKernelSrc(self.src)

        # build opencl kernel
        prg = cl.Program(self.ctx, kernel).build()#(options="-cl-mad-enable -cl-fast-relaxed-math")

        # define openCL local memory size
        localMem = cl.LocalMemory(self.block * self.block * self.A.itemsize)

        exec_evt = prg.matrix_mul(self.queue, (self.m, self.p,),
                                       self.A_buf, self.B_buf, self.C_buf,
                                       np.uint32(self.m), np.uint32(self.n), np.uint32(self.p),
                                       localMem, localMem, # alloc local memory
                                       local_size=(self.block, self.block,),
                                       ).wait()

        # read result from opencl buffer
        cl.enqueue_read_buffer(self.queue, self.C_buf, self.C).wait()

        # return elapsed time in seconds
        return 1e-9 * (exec_evt.profile.end - exec_evt.profile.start)

    def __str__(self):
        return "Test 003"


def opencl_matrix_mul(dim, block=1, verbose=False, dataType=np.float32):
    """
    Runs tests and print elapsed time for each test.
    
    dim   - (m, n, p,), shape of matrices : [M x N] * [N x P]
    block - int, openCL block size, [size x size]
    """

    # generate matrices A and B --> every test operates on same data
    (m, n, p) = dim
    mat_A = GenMat(m, n, dataType)
    mat_B = GenMat(n, p, dataType)

    # print matrices
    if verbose:
        print "Matrix A: %s" % mat_A
        print "\nMatrix B: %s" % mat_B

    # detect devices
    for platform in cl.get_platforms():
        for device in platform.get_devices():
            print "------------------------------------"
            print "Device name:", device.name
            print "Device type:", cl.device_type.to_string(device.type)
            print "Device memory: ", device.global_mem_size//1024//1024, 'MB'
            print "Device max clock speed:", device.max_clock_frequency, 'MHz'
            print "Device compute units:", device.max_compute_units

            # create opencl context and put it in program queue
            ctx = cl.Context([device])
            queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)

            # run every test
            for MM_test in [MM_001, MM_002, MM_003]:

                # preform test
                test = MM_test(ctx, queue, dim, block, A=mat_A, B=mat_B, dataType=dataType)
                
                # print result
                print "\n%s \nTime elapsed: %g s\n" % (test, test.execute())
                
                if verbose:
                    print test.C


### OpenCL relevant code ends here ---------------------------------------------


def GenMat(m, n, dataType=np.float32):
    """
    Generates matrix of dimensions MxN filled with random values of given datatype.
    """

    if issubclass(dataType, np.int32):
        return np.random.random_integers(0,10,(m,n)).astype(dataType)
    else:
        return np.random.rand(m,n).astype(dataType) * 10


# used only for testing ... 
def matrix_mul(A, B):
    """
    Multiplies matrices A * B and returns result.
    Doesn't checks dimensions or anything else, only used for verifying openCL output.
    """
    
    (m, n) = A.shape
    (n1, p) = B.shape

    C = np.zeros([m,p], dtype=np.int32)

    for i in range(m):
        for j in range(p):
            for k in range(n):
                C[i,j] += A[i,k] * B[k,j]

    return C
                

def usage():
    
    print """
    Parametars:
      [-d | --dimensions] MxNxP  : Matrices dimensions, [m n] * [n p]
      [-b | --block] N           : Block size, N x N
      [-t | --datatype] type     : DataType [int|float|double] (default float)
      [-v | --verbose]           : Prints matrices for checking purpose
      [-h | --help]              : Shows this help    
    """    

def main(argv=None):
    """
    Main function. Parses parametars and calls matrix multiplication tests.
    Take a look on usage function above.    
    """

    # default values
    dim_m, dim_n, dim_p = 16, 16, 16
    dim_b = 1
    datatype = np.float32;
    verbose = False

    # print help argument
    print "For help use --help argument!\n"

    # get arguments
    try:
        args, opts = getopt.getopt(argv, 'd:b:t:vh', ['dimensions=', 'block=', 'datatype=', 'verbose', 'help'])
        
    except getopt.GetoptError, err:
        print str(err)
        usage()
        sys.exit(2)

    # parse arguments
    for o,a in args:

        #help
        if o in ("-h", "--help"):
            usage()
            sys.exit()
            
        #dimenzions
        elif o in ("-d", "--dimensions"):
            dim_m, dim_n, dim_p = map(lambda x: int(x), a.split('x')) # str -> int
            
        # block size
        elif o in ("-b", "--block"):
            dim_b = int(a)
            
        # datatype
        elif o in ("-t", "--datatype"):
            if a == "int":
                datatype = np.int32
            elif a == "double":
                datatype = np.float64
            else:
                datatype = np.float32

        # verbose
        elif o in ("-v", "--verbose"):
            verbose = True
        else:
            assert False, "unhandled option"

    # function call
    opencl_matrix_mul((dim_m, dim_n, dim_p,), dim_b, verbose, datatype)

# call main function
if __name__ == '__main__':
    main(sys.argv[1:])  

    

