/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_RESTIR_DI_MIS_WEIGHTS_COMMON_H
#define DEVICE_RESTIR_DI_MIS_WEIGHTS_COMMON_H

/**
 * Forward declarations
 */
struct ReSTIRDISample;
struct ReSTIRGISample;
struct ReSTIRDIReservoir;
struct ReSTIRGIReservoir;

 /**
 * The ReSTIRTypeStruct is used to automatically determine what SampleType to use
 * based on the 'IsReSTIRGI' template parameter
 *
 * This allows us to use the ReSTIRDISample type of ReSTIRGISample type automatically
 * based on whether or not we're instantiating the structures for ReSTIR DI or ReSTIR GI
 *
 * This sample type is then used in some of the specialization to pass to the target functions
 */
template <bool IsReSTIRGI>
struct ReSTIRTypeStruct {};

template <>
struct ReSTIRTypeStruct<false>
{
	using SampleType = ReSTIRDISample;
	using ReservoirType = ReSTIRDIReservoir;
};

template <>
struct ReSTIRTypeStruct<true>
{
	using SampleType = ReSTIRGISample;
	using ReservoirType = ReSTIRGIReservoir;
};

template <bool IsReSTIRGI>
using ReSTIRSampleType = typename ReSTIRTypeStruct<IsReSTIRGI>::SampleType;

template <bool IsReSTIRGI>
using ReSTIRReservoirType = typename ReSTIRTypeStruct<IsReSTIRGI>::ReservoirType;

HIPRT_HOST_DEVICE float symmetric_ratio_MIS_weights_difference_function(float target_function_at_center, float target_function_from_i)
{
	if (target_function_at_center == 0.0f && target_function_from_i == 0.0f)
		return 0.0f;
	else if (target_function_at_center == 0.0f || target_function_from_i == 0.0f)
		// Maximum difference
		return 0.0f;

	float ratio = hippt::min(target_function_at_center / target_function_from_i, target_function_from_i / target_function_at_center);

	// Pow beta=3
	return ratio * ratio * ratio;
}

#endif
