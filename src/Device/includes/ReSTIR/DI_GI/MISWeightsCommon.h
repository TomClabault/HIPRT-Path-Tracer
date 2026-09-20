/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_RESTIR_GI_PT_MIS_WEIGHTS_COMMON_H
#define DEVICE_RESTIR_GI_PT_MIS_WEIGHTS_COMMON_H

#include "Device/includes/ReSTIR/DI_GI/ReservoirsForwardDeclaration.h"
#include "HostDeviceCommon/KernelOptions/ReSTIRCommonOptions.h"

/**
 * The ReSTIRTypeStruct is used to automatically determine what SampleType to use
 * based on the ReSTIR variant template parameter
 *
 * This allows us to use the ReSTIRGIReservoirSample or ReSTIRPTReservoirSample type automatically
 * based
 * on which variant is being instantiated.
 * This sample type is then used in some of the specialization to pass to the target functions
 */
template <int ReSTIRVariant>
struct ReSTIRTypeStruct
{
};

template <>
struct ReSTIRTypeStruct<ReSTIR_VARIANT_GI>
{
	using SampleType	= ReSTIRGIReservoirSample;
	using ReservoirType = ReSTIRGIReservoir;
};

template <>
struct ReSTIRTypeStruct<ReSTIR_VARIANT_PT>
{
	using SampleType	= ReSTIRPTReservoirSample;
	using ReservoirType = ReSTIRPTReservoir;
};

template <int ReSTIRVariant>
using ReSTIRSampleType = typename ReSTIRTypeStruct<ReSTIRVariant>::SampleType;

template <int ReSTIRVariant>
using ReSTIRReservoirType = typename ReSTIRTypeStruct<ReSTIRVariant>::ReservoirType;

#endif // #ifndef DEVICE_RESTIR_GI_PT_MIS_WEIGHTS_COMMON_H
