/* 
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt 
 */

#ifndef HIPRT_OROCHI_CTX_H
#define HIPRT_OROCHI_CTX_H

#include "hiprt/hiprt.h"
#include "Orochi/Orochi.h"
#include "HIPRT-Orochi/HIPRTOrochiUtils.h"

struct HIPRTOrochiCtx
{
	void init(int device_index)
	{
		OROCHI_CHECK_ERROR(static_cast<oroError>(oroInitialize((oroApi)(ORO_API_HIP | ORO_API_CUDA), 0)));

		OROCHI_CHECK_ERROR(oroInit(0));
		OROCHI_CHECK_ERROR(oroDeviceGet(&orochi_device, device_index));
		OROCHI_CHECK_ERROR(oroCtxCreate(&orochi_ctx, 0, orochi_device));

		oroDeviceProp props;
		OROCHI_CHECK_ERROR(oroGetDeviceProperties(&props, orochi_device));

		std::cout << "hiprt ver." << HIPRT_VERSION_STR << std::endl;
		std::cout << "Executing on '" << props.name << "'" << std::endl;
		if (std::string(props.name).find("NVIDIA") != std::string::npos)
			hiprt_ctx_input.deviceType = hiprtDeviceNVIDIA;
		else
			hiprt_ctx_input.deviceType = hiprtDeviceAMD;

		hiprt_ctx_input.ctxt = oroGetRawCtx(orochi_ctx);
		hiprt_ctx_input.device = oroGetRawDevice(orochi_device);
		hiprtSetLogLevel(hiprtLogLevelError);

		HIPRT_CHECK_ERROR(hiprtCreateContext(HIPRT_API_VERSION, hiprt_ctx_input, hiprt_ctx));
	}

	hiprtContextCreationInput hiprt_ctx_input;
	oroCtx orochi_ctx;
	oroDevice orochi_device;

	hiprtContext hiprt_ctx;
};

#endif
