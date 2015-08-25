#!/usr/bin/python

import pyopencl as cl
import numpy as np
import sys, getopt


def opencl_nbody(size, block_size, iters):

    # first try to load kernel source from file
    try:
        f = open("opencl-nbody.cl", 'r')
        kernel = f.read()
    except IOError, err:
        print str(err)
        sys.exit(2)

    # simulation parametars
    deltatime = 0.001
    eps = 0.001

    # init particle's position and velocity
    particles = np.random.rand(size,4).astype(np.float32) #* (size / 1024)
    velocity = np.zeros((size,4), dtype=np.float32)

    # create opencl context and put it in program queue
    ctx = cl.Context( cl.get_platforms()[0].get_devices() ) # quick fix
    queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)

    # make buffers
    mf = cl.mem_flags
    particles_buf = cl.Buffer(ctx, mf.COPY_HOST_PTR, hostbuf=particles)
    velocity_buf = cl.Buffer(ctx, mf.COPY_HOST_PTR, hostbuf=velocity)

    # --- just for checking
    #print particles
    #print " --- " 

    # define OpenCL local memory size (blocksize * vectorsize * itemsize)
    local_buf = cl.LocalMemory(block_size * 4 * particles.itemsize ) 

    # build program
    prg = cl.Program(ctx, kernel).build()

    # execute kernel
    exec_evt = prg.nbody_simulation(queue, (size,),
                                    np.int32(iters), np.float32(deltatime), np.float32(eps),
                                    particles_buf,
                                    velocity_buf,
                                    local_buf,
                                    local_size=(block_size,),
                                    ).wait()
    
    # read results
    cl.enqueue_read_buffer(queue, particles_buf, particles).wait()

    # --- just for checking
    #print particles

    # print elased time in seconds
    print "Time elapsed: %g s\n" % (1e-9 * (exec_evt.profile.end - exec_evt.profile.start))
    

def usage():
    
    print """Parameters:
      [-n | --size] N   : Number of particles (default: 1024) (power of 2)
      [-b | --block] N  : Block size (default: 4)
      [-i | --iters] N  : Number of iterations (default: 128)
      [-h | --help]     : Shows this help
    """    

def main(argv=None):
    """
    Main function. Parses parametars and calls matrix multiplication tests.
    Take a look on usage function above.    
    """

    # default values
    size = 1024
    block_size = 4
    iters = 1024
    step = 16

    # print help argument
    print "For help use --help switch!\n"

    # get arguments
    try:
        args, opts = getopt.getopt(argv, 'n:b:i:s:h', ['size=', 'block=', 'iters=', 'step=', 'help'])
        
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
            
        # dimenzions
        elif o in ("-n", "--size"):
            size = int(a)
            
        # block size
        elif o in ("-b", "--block"):
            block_size = int(a)
            
        # iters
        elif o in ("-i", "--iters"):
            iters  = int(a)
            
        # step
        elif o in ("-s", "--step"):
            step = int(a)
            
        else:
            assert False, "unhandled option"

    # function call
    opencl_nbody(size, block_size, iters)


# call main function
if __name__ == '__main__':
    main(sys.argv[1:])  

    

