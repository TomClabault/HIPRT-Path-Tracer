/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef RENDERER_PERCOMPUTED_EMISSIVE_TRIANGLE_DATA_SOA_HOST_H
#define RENDERER_PERCOMPUTED_EMISSIVE_TRIANGLE_DATA_SOA_HOST_H

#include "HostDeviceCommon/PrecomputedEmissiveTrianglesDataSoADevice.h"

#include "Renderer/CPUGPUCommonDataStructures/GenericSoA.h"

template <template <typename> typename DataContainer>
using PrecomputedEmissiveTrianglesDataSoAHost = GenericSoA<DataContainer, float3, float3, float3>;

namespace PrecomputedEmissiveTrianglesDataSoAHostHelpers
{
	enum
	{
		VERTEX_A_BUFFER,
		AB_BUFFER,
		AC_BUFFER,
	};

	template <template <typename> typename DataContainer>
	PrecomputedEmissiveTrianglesDataSoADevice to_device(PrecomputedEmissiveTrianglesDataSoAHost<DataContainer>& petd_host)
	{
		PrecomputedEmissiveTrianglesDataSoADevice petd_device;

		petd_device.triangles_A = petd_host.template get_buffer<0>().data();
		petd_device.triangles_AB = petd_host.template get_buffer<1>().data();
		petd_device.triangles_AC = petd_host.template get_buffer<2>().data();

		return petd_device;
	}
};

#endif
