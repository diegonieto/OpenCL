/*
 * pi_kernel2.cl
 *
 *  Created on: 27/02/2014
 *      Author: Diego Nieto Muñoz
 *
 * -------------------
 * Reduction approach:
 * -------------------
 *  Example with work group equal eight
 *  MEM POSITIONS [ | | | | | | | ]
 *                 | / | / | / | /     iteration 1
 *                 |/  |/  |/  |/
 *  MEM POSITIONS [ | | | | | | | ]
 *                 |   /   |   /
 *                 |  /    |  /        iteration 2
 *                 | /     | /
 *                 |/      |/
 *  MEM POSITIONS [ | | | | | | | ]
 *                 |       /
 *                 |      /
 *                 |     /
 *                 |    /              iteration 3
 *                 |   /
 *                 |  /
 *                 | /
 *                 |/
 *  MEM POSITIONS [ | | | | | | | ]
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

	const uint local_size = get_local_size(0);

	// Perform the reduction applying the above technique	
	for(uint stride = 2;  stride - 1 < local_size; stride *= 2) {
		if(lid % stride == 0) {
			scratch[lid] += scratch[lid+stride/2];
		}
		// Synchronize all threads within the block 
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	if(lid == 0) {
		// The final value is stored in scratch[0] so
		// some thread write it
		workGroupBuffer[get_group_id(0)] = scratch[0];
	}
}
