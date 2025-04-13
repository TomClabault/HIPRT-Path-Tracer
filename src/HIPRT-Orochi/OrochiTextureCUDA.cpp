/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifdef OROCHI_ENABLE_CUEW

#include "cuda_runtime_api.h"
#include "contrib/cuew/include/cuew.h"

#include "Utils/Utils.h"

void create_texture_from_array_cuda(void* m_texture_array, void* m_texture, void* filtering_mode, void* address_mode, bool read_mode_float_normalized)
{
	// Resource descriptor
	cudaResourceDesc resource_descriptor = {};
	resource_descriptor.resType = cudaResourceTypeArray;
	resource_descriptor.res.array.array = reinterpret_cast<cudaArray_t>(m_texture_array);

	cudaTextureDesc texture_descriptor = {};
	texture_descriptor.addressMode[0] = *reinterpret_cast<cudaTextureAddressMode*>(address_mode);
	texture_descriptor.addressMode[1] = *reinterpret_cast<cudaTextureAddressMode*>(address_mode);
	texture_descriptor.addressMode[2] = *reinterpret_cast<cudaTextureAddressMode*>(address_mode);
	texture_descriptor.filterMode = *reinterpret_cast<cudaTextureFilterMode*>(filtering_mode);
	texture_descriptor.normalizedCoords = true;
	texture_descriptor.readMode = read_mode_float_normalized ? cudaTextureReadMode::cudaReadModeNormalizedFloat : cudaTextureReadMode::cudaReadModeElementType;
	texture_descriptor.sRGB = false;

	cudaError_t error = cudaCreateTextureObject_oro(reinterpret_cast<cudaTextureObject_t*>(m_texture), &resource_descriptor, &texture_descriptor, nullptr);
	if (error != cudaError::cudaSuccess)
		Utils::debugbreak();
}

#endif
