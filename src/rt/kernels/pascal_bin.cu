/*
*  Copyright (c) 2009-2011, NVIDIA Corporation
*  All rights reserved.
*
*  Redistribution and use in source and binary forms, with or without
*  modification, are permitted provided that the following conditions are met:
*      * Redistributions of source code must retain the above copyright
*        notice, this list of conditions and the following disclaimer.
*      * Redistributions in binary form must reproduce the above copyright
*        notice, this list of conditions and the following disclaimer in the
*        documentation and/or other materials provided with the distribution.
*      * Neither the name of NVIDIA Corporation nor the
*        names of its contributors may be used to endorse or promote products
*        derived from this software without specific prior written permission.
*
*  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
*  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
*  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
*  DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
*  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
*  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
*  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
*  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
*  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
*  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include "CudaTracerKernels.hpp"

//------------------------------------------------------------------------

#define STACK_SIZE              24          // Size of the traversal stack in local memory.
#define BLOCK_HEIGHT             2
#define NUM_SUBWARPS            16
#define SUBWARP_WIDTH            2
#define CUDA_INF_F __int_as_float(0x7f800000)

extern "C" __device__ int g_warpCounter;    // Work counter for persistent threads.

//------------------------------------------------------------------------

extern "C" __global__ void queryConfig(void)
{
	g_config.bvhLayout = BVHLayout_Bin;
	g_config.blockWidth = 32;
	g_config.blockHeight = BLOCK_HEIGHT;
	g_config.usePersistentThreads = 1;
	g_config.desiredWarps = 960;
}

//------------------------------------------------------------------------

TRACE_FUNC
{

// Traversal stack in CUDA shared memory.
__shared__ volatile int traversalStack[NUM_SUBWARPS][BLOCK_HEIGHT][STACK_SIZE];

// Live state during traversal, stored in registers.
float   origx, origy, origz;            // Ray origin.
float   tmin;
float   hitT;
int     rayidx;
int     stackPtr = -1;
float   oodx, oody, oodz;
float   dirx, diry, dirz;
float   idirx, idiry, idirz;
int     hitIndex = -1;

const int offset = (threadIdx.x & 0x00000001);
const int subwarp = (threadIdx.x >> 1);
const int subwarp_mask = (0x00000003 << (threadIdx.x & 0xfffffffe));

// Initialize persistent threads.
// Persistent threads: fetch and process rays in a loop.
do
{
	// Fetch new rays from the global pool using lane 0.
	if (stackPtr < 0)
	{
		if (threadIdx.x == 0)
			rayidx = atomicAdd(&g_warpCounter, NUM_SUBWARPS);
		rayidx = __shfl(rayidx, 0) + subwarp;

		if (rayidx >= numRays)
			break;

		// Fetch ray.

		const float4 o = FETCH_GLOBAL(rays, rayidx * 2 + 0, float4);
		const float4 d = FETCH_GLOBAL(rays, rayidx * 2 + 1, float4);
		origx = o.x;
		origy = o.y;
		origz = o.z;
		tmin = o.w;
		dirx = d.x;
		diry = d.y;
		dirz = d.z;
		hitT = d.w;
		const float ooeps = exp2f(-80.0f); // Avoid div by zero.
		idirx = 1.0f / (fabsf(d.x) > ooeps ? d.x : copysignf(ooeps, d.x));
		idiry = 1.0f / (fabsf(d.y) > ooeps ? d.y : copysignf(ooeps, d.y));
		idirz = 1.0f / (fabsf(d.z) > ooeps ? d.z : copysignf(ooeps, d.z));
		oodx = origx * idirx;
		oody = origy * idiry;
		oodz = origz * idirz;
		// Setup traversal.
		stackPtr = 0;
		hitIndex = -1;
		if (!offset)
			traversalStack[subwarp][threadIdx.y][0] = 0;

	}

	// Traversal loop.

	while (stackPtr >= 0)
	{
	
		const int curr = traversalStack[subwarp][threadIdx.y][stackPtr--];
		if (curr >= 0)
		{
			// Fetch AABBs of the two child nodes.
			const float4 xy = tex1Dfetch(t_nodesA, curr + offset);
			const float4 zi = tex1Dfetch(t_nodesA, curr + SUBWARP_WIDTH + offset);

			// Intersect the ray against the child nodes.
			const float c0lox = xy.x * idirx - oodx;
			const float c0hix = xy.y * idirx - oodx;
			const float c0loy = xy.z * idiry - oody;
			const float c0hiy = xy.w * idiry - oody;		
			const float c0loz = zi.x * idirz - oodz;
			const float c0hiz = zi.y * idirz - oodz;
			int link = float_as_int(zi.z);

			const float c0min = spanBeginKepler(c0lox, c0hix, c0loy, c0hiy, c0loz, c0hiz, tmin);
			const bool hit = c0min <= spanEndKepler(c0lox, c0hix, c0loy, c0hiy, c0loz, c0hiz, hitT);
			

			float dist = hit ? c0min : CUDA_INF_F;
			const int  hits = __popc(__ballot(hit)&subwarp_mask);
			if (!hits) continue;
			stackPtr += hits;


			//sort hits
			swap(dist, link, 0x01, bfe(threadIdx.x, 0));

			if (dist < CUDA_INF_F)
				traversalStack[subwarp][threadIdx.y][stackPtr - offset] = link;
		}
		else {

			// Load triangle

			int triAddr = ~curr + offset * 3;
			const float4 v00 = tex1Dfetch(t_trisA, triAddr + 0);
			const float4 v11 = tex1Dfetch(t_trisA, triAddr + 1);
			const float4 v22 = tex1Dfetch(t_trisA, triAddr + 2);

			const float Oz = v00.w - origx*v00.x - origy*v00.y - origz*v00.z;
			const float invDz = 1.0f / (dirx*v00.x + diry*v00.y + dirz*v00.z);
			float t = Oz * invDz;
			bool hit = false;

			if (t > tmin && t < hitT)
			{
				// Compute and check barycentric u.

				const float Ox = v11.w + origx*v11.x + origy*v11.y + origz*v11.z;
				const float Dx = dirx*v11.x + diry*v11.y + dirz*v11.z;
				const float u = Ox + t*Dx;

				if (u >= 0.0f)
				{
					// Compute and check barycentric v.

					const float Oy = v22.w + origx*v22.x + origy*v22.y + origz*v22.z;
					const float Dy = dirx*v22.x + diry*v22.y + dirz*v22.z;
					const float v = Oy + t*Dy;

					if (v >= 0.0f && u + v <= 1.0f)
					{
						// Record intersection.
						// Closest intersection not required => terminate.

						hit = true;
					}
				}
			}

			// Sort triangles
	
			const int hits = __ballot(hit)&subwarp_mask;
			if (!hits) continue;

			t = hit ? t : CUDA_INF_F;

			{
				const float tmp_t = __shfl_xor(t, 1);
				const int tmp_addr = __shfl_xor(triAddr, 1);

				triAddr = tmp_t < t ? tmp_addr : triAddr;
				t = fminf(t, tmp_t);
			}

			hitIndex = triAddr;
			hitT = t;

			if (anyHit && hits)
			{
				stackPtr = -1;
				break;
			}

		}
	}

	if (offset == 0) {
		if (hitIndex == -1) { STORE_RESULT(rayidx, -1, hitT); }
		else { STORE_RESULT(rayidx, FETCH_TEXTURE(triIndices, hitIndex, int), hitT); }
	}

} while (true);
}

//------------------------------------------------------------------------
