/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

 /**
  * Kernel for testing that creating and reading from a 3D texture happens correctly.
  * The 3D texture is just written to a linear buffer and the linear buffer is then
  * expected to contain the data of the texture, basically just a copy of it.
  */

#include "Device/includes/FixIntellisense.h"

#include <Orochi/Orochi.h>

#ifdef __KERNELCC__
GLOBAL_KERNEL_SIGNATURE(void) Test3DTexture(oroTextureObject_t texture_3D, int tex_size, float* out_buffer)
#else
GLOBAL_KERNEL_SIGNATURE(void) Test3DTexture(oroTextureObject_t texture_3D, int tex_size, float* out_buffer, int x, int y, int z)
#endif
{
#ifdef __KERNELCC__
    const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const uint32_t z = blockIdx.z * blockDim.z + threadIdx.z;
#endif
    if (x >= tex_size || y >= tex_size || z >= tex_size)
        return;

    const uint32_t thread_index = (x + y * tex_size + z * tex_size * tex_size);

    out_buffer[thread_index * 4] = tex3D<float4>(texture_3D, x + 0.35f, y + 0.35f, z + 0.35f).y;
}
