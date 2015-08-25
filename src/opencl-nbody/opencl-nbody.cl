
__kernel void nbody_simulation(int iters, float _dt, float limit,
			       __global float4* _position,
			       __global float4* _velocity,
			       __local float4* block) 
{       
  // get sizes
  int l_size = get_local_size(0);
  int n_groups = get_num_groups(0);
  
  // get global and local thread id
  int g_tid = get_global_id(0);
  int l_tid = get_local_id(0);

  // define interval --> deltatime
  const float4 dt = (float4)(_dt, _dt, _dt, 0.0f);

  // iterate
  for (int i=0; i < iters; i++)
    {
      // init values (position, velocity, acceleration)
      float4 position = _position[g_tid];
      float4 velocity = _velocity[g_tid];  
      float4 a = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
  
      // for every group
      for (int gid=0; gid < n_groups; ++gid) 
	{
	  // copy from global to local mem and sync
	  block[l_tid] = _position[gid * l_size + l_tid];
	  barrier(CLK_LOCAL_MEM_FENCE);
    
	  // for every particle in group
	  for (int k=0; k<l_size; ++k) 
	    {	    
	      // calculate distance
	      float4 d = block[k] - position;

	      // calculate acceleration (F = m * a) ...
	      float invr = rsqrt(pown(d.x,2) + pown(d.y,2) + pown(d.z,2) + limit);
	      a += block[k].w * invr * invr * invr * d;
	    }
	
	  // sync
	  barrier(CLK_LOCAL_MEM_FENCE);    
      } 
     
      // calculate new position and velocity
      position += dt * velocity + 0.5f*dt*dt*a;
      velocity += dt * a;

      // update global values
      _position[g_tid] = position;
      _velocity[g_tid] = velocity;  
      
      // sync
      barrier(CLK_LOCAL_MEM_FENCE);
    }
}
