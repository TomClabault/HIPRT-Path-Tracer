/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef HIPRT_OROCHI_CTX_H
#define HIPRT_OROCHI_CTX_H

#include <memory>

#include <hiprt/hiprt.h>
// #include <hiprt/impl/Context.h>
#include <Orochi/Orochi.h>

#include "HIPRT-Orochi/HIPRTOrochiUtils.h"
#include "UI/ImGui/ImGuiLogger.h"
#include "Utils/Utils.h"

extern ImGuiLogger g_imgui_logger;

struct HIPRTOrochiCtx
{
	HIPRTOrochiCtx() {}

	HIPRTOrochiCtx(int device_index)
	{
		init(device_index);
	}

#ifdef _WIN32
	Utils::AddEnvVarError add_CUDA_PATH_to_PATH()
	{
		// On Windows + NVIDIA, adding the CUDA_PATH to the PATH environment variable just to be sure
		// that CUDA's DLLs are found in case the user indeed has installer the CUDA toolkit but their PATH
		// environment variable is not set correctly.
		return Utils::windows_add_ENV_var_to_PATH(L"CUDA_PATH", L"\\bin;");
	}
#endif // #ifdef _WIN32

	void init(int device_index)
	{
		this->device_index = device_index;
		g_imgui_logger.add_line(ImGuiLoggerSeverity::IMGUI_LOGGER_INFO, "Initializing Orochi...");

#ifdef OROCHI_ENABLE_CUEW
#ifdef _WIN32
		Utils::AddEnvVarError error = add_CUDA_PATH_to_PATH();
#endif // #ifdef _WIN32
#endif // #ifdef OROCHI_ENABLE_CUEW

#ifdef OROCHI_ENABLE_CUEW
		int error_initialize = oroInitialize((oroApi)(ORO_API_CUDA), 0);
#else // #ifdef OROCHI_ENABLE_CUEW
		int error_initialize = oroInitialize((oroApi)(ORO_API_HIP), 0);
#endif // #ifdef OROCHI_ENABLE_CUEW
		if (error_initialize != oroSuccess)
		{
			switch (error_initialize)
			{
				// Unable to load HIP/CUDA
			case ORO_API_HIPDRIVER:
				g_imgui_logger.add_line(ImGuiLoggerSeverity::IMGUI_LOGGER_ERROR, "Unable to load HIP... Are your drivers up-to-date?");
				break;

			case ORO_API_CUDADRIVER:
				g_imgui_logger.add_line(ImGuiLoggerSeverity::IMGUI_LOGGER_ERROR, "Unable to load CUDA... Are your drivers up-to-date?");
				break;

				// Unable to load HIP/CUDA
			case ORO_API_HIPRTC:
				g_imgui_logger.add_line(ImGuiLoggerSeverity::IMGUI_LOGGER_ERROR,
										"Unable to load HIPRTC... Is the HIP SDK (Windows) or ROCm + HIP (Linux) installed?");
				break;

			case ORO_API_CUDARTC:
				g_imgui_logger.add_line(ImGuiLoggerSeverity::IMGUI_LOGGER_ERROR, "Unable to load CUDARTC... Is the CUDA Toolkit installed + is the CUDA_PATH "
																				 "environment variable set? (or have {CUDA_TOOLKIT_FOLDER/bin} in your "
																				 "PATH environment variable)");
				break;
			}

			int trash = std::getchar();
			std::exit(1);
		}

		OROCHI_CHECK_ERROR(oroInit(0));
		OROCHI_CHECK_ERROR(oroDeviceGet(&orochi_device, device_index));
		OROCHI_CHECK_ERROR(oroCtxCreate(&orochi_ctx, 0, orochi_device));

		OROCHI_CHECK_ERROR(oroGetDeviceProperties(&device_properties, orochi_device));

		g_imgui_logger.add_line(ImGuiLoggerSeverity::IMGUI_LOGGER_INFO, "HIPRT ver.%s", HIPRT_VERSION_STR);
		g_imgui_logger.add_line(ImGuiLoggerSeverity::IMGUI_LOGGER_INFO, "Executing on '%s'\n", device_properties.name);
		if (std::string(device_properties.name).find("NVIDIA") != std::string::npos)
			hiprt_ctx_input.deviceType = hiprtDeviceNVIDIA;
		else
			hiprt_ctx_input.deviceType = hiprtDeviceAMD;

		hiprt_ctx_input.ctxt   = oroGetRawCtx(orochi_ctx);
		hiprt_ctx_input.device = oroGetRawDevice(orochi_device);

		HIPRT_CHECK_ERROR(hiprtCreateContext(HIPRT_API_VERSION, hiprt_ctx_input, hiprt_ctx));

		HIPRT_CHECK_ERROR(hiprtSetLogLevel(hiprt_ctx, hiprtLogLevelError));
	}

	bool has_hardware_ray_tracing_support() const
	{
		std::string deviceName = device_properties.name;
		std::string archName   = device_properties.gcnArchName;

		uint32_t archNumber = 0;
		if (archName.substr(0, 3) == "gfx")
		{
			std::string numberPart = archName.substr(3);
			archNumber			   = std::stoi(numberPart);
		}
		return (archNumber >= 1030 && deviceName.find("NVIDIA") == std::string::npos);
	}

	hiprtContextCreationInput hiprt_ctx_input = { nullptr, -1, hiprtDeviceAMD };
	int device_index						  = 0;

	oroCtx orochi_ctx				= nullptr;
	oroDevice orochi_device			= -1;
	oroDeviceProp device_properties = {};

	hiprtContext hiprt_ctx = nullptr;
};

#endif // #ifndef HIPRT_OROCHI_CTX_H
