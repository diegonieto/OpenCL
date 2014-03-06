/*
 * pi_kernel2.cl
 *
 *  Created on: 27/02/2014
 *      Author: Diego Nieto Muñoz
 */

__kernel void Pi(__global float *workGroupBuffer,
				 __local float *scratch,
				 const uint niter,
				 const uint chunk)
{
	const uint lid = get_local_id(0);
	const uint gid = get_global_id(0);

	const float h = (1.0/(float)niter);
	float partial_sum = 0.0;

	for(uint i=gid*chunk; i<(gid*chunk)+chunk; i++) {
		float x = h * ((float) i - 0.5);
		partial_sum += 4.0 / (1.0 + x * x);
	}

	scratch[lid] = partial_sum;

	barrier(CLK_LOCAL_MEM_FENCE);

	float local_pi = 0;

	const uint local_size = get_local_size(0);

	for(uint stride = 2;  stride - 1 < local_size; stride *= 2) {
		if(lid % stride == 0) {
			scratch[lid] += scratch[lid+stride/2];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	if(lid == 0) {
		workGroupBuffer[get_group_id(0)] = scratch[0];
	}
}
