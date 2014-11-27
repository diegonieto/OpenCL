/*
 * pi_kernel2.cl
 *
 *  Created on: 27/02/2014
 *      Author: Diego Nieto Muñoz
 */

__kernel void Pi(__global float *workGroupBuffer,         // 0..NumWorkGroups-1
				 __local float *scratch,  // 0..workGroupSize-1
				 const uint niter,        // Total iterations
				 const uint chunk)        // Chunk size
{
	const uint lid = get_local_id(0);
	const uint gid = get_global_id(0);

	const float h = (1.0/(float)niter);
	float partial_sum = 0.0;

	// Each thread compute chunk iterations
	for(uint i=gid*chunk; i<(gid*chunk)+chunk; i++) {
		float x = h * ((float) i - 0.5);
		partial_sum += 4.0 / (1.0 + x * x);
	}

	// Each thread store its partial sum in the workgroup array
	scratch[lid] = partial_sum;

	// Synchronize all threads within the workgroup
	barrier(CLK_LOCAL_MEM_FENCE);

	float local_pi = 0;

	// Only thread 0 of each workgroup perform the reduction
	// of that workgroup
	if(lid == 0) {
		const uint length = lid + get_local_size(0);
		for (uint i = lid; i<length; i++) {
			local_pi += scratch[i];
		}
		// It store the workgroup sum
		// Final reduction, between block, is done out by CPU
		workGroupBuffer[get_group_id(0)] = local_pi;
	}
}
