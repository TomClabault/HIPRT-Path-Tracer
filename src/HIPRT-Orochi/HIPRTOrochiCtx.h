/* 
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt 
 */

#ifndef HIPRT_OROCHI_CTX_H
#define HIPRT_OROCHI_CTX_H

#include <memory>

#include <hiprt/hiprt.h>
#include <hiprt/impl/Context.h>
#include <Orochi/Orochi.h>

#include "HIPRT-Orochi/HIPRTOrochiUtils.h"

struct HIPRTOrochiCtx
{
	HIPRTOrochiCtx() {}

	HIPRTOrochiCtx(int device_index)
	{
		init(device_index);
	}

	void init(int device_index)
	{
		if (static_cast<oroError>(oroInitialize((oroApi)(ORO_API_HIP | ORO_API_CUDA), 0)) != oroSuccess)
		{
			std::cerr << "Unable to initialize Orochi... Is CUDA/HIP installed?" << std::endl;

			int trash = std::getchar();
			std::exit(1);
		}

		OROCHI_CHECK_ERROR(oroInit(0));
		OROCHI_CHECK_ERROR(oroDeviceGet(&orochi_device, device_index));
		OROCHI_CHECK_ERROR(oroCtxCreate(&orochi_ctx, 0, orochi_device));

		OROCHI_CHECK_ERROR(oroGetDeviceProperties(&device_properties, orochi_device));

		std::cout << "hiprt ver." << HIPRT_VERSION_STR << std::endl;
		std::cout << "Executing on '" << device_properties.name << "'" << std::endl;
		if (std::string(device_properties.name).find("NVIDIA") != std::string::npos)
			hiprt_ctx_input.deviceType = hiprtDeviceNVIDIA;
		else
			hiprt_ctx_input.deviceType = hiprtDeviceAMD;

		hiprt_ctx_input.ctxt = oroGetRawCtx(orochi_ctx);
		hiprt_ctx_input.device = oroGetRawDevice(orochi_device);
		hiprtSetLogLevel(hiprtLogLevelError);

		HIPRT_CHECK_ERROR(hiprtCreateContext(HIPRT_API_VERSION, hiprt_ctx_input, hiprt_ctx));
	}

	hiprtContextCreationInput hiprt_ctx_input = { nullptr, -1, hiprtDeviceAMD };

	oroCtx orochi_ctx = nullptr;
	oroDevice orochi_device = -1;
	oroDeviceProp device_properties = {};

	hiprtContext hiprt_ctx = nullptr;
};

#endif
