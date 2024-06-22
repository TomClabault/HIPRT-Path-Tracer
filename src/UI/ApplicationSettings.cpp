/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "UI/ApplicationSettings.h"

const std::string ApplicationSettings::PATH_TRACING_KERNEL = "FullPathTracer";
const std::string ApplicationSettings::CAMERA_RAYS_FUNC_NAME = "CameraRays";
const std::string ApplicationSettings::RESTIR_DI_INITIAL_CANDIDATES_FUNC_NAME = "ReSTIR_DI_InitialCandidates";
const std::string ApplicationSettings::RESTIR_DI_SPATIAL_REUSE_FUNC_NAME = "ReSTIR_DI_SpatialReuse";

const std::string ApplicationSettings::KERNEL_FILES[] = {DEVICE_KERNELS_DIRECTORY "/FullPathTracer.h", DEVICE_KERNELS_DIRECTORY "/CameraRays.h", DEVICE_KERNELS_DIRECTORY "/ReSTIR/ReSTIR_DI_InitialCandidates.h" , DEVICE_KERNELS_DIRECTORY "/ReSTIR/ReSTIR_DI_SpatialReuse.h" };
const std::string ApplicationSettings::KERNEL_FUNCTIONS[] = { PATH_TRACING_KERNEL, CAMERA_RAYS_FUNC_NAME, RESTIR_DI_INITIAL_CANDIDATES_FUNC_NAME, RESTIR_DI_SPATIAL_REUSE_FUNC_NAME };
