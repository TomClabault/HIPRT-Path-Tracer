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

 /**
 * The SampleTypeStruct is used to automatically determine what SampleType to use
 * based on the 'IsReSTIRGI' template parameter
 *
 * This allows us to use the ReSTIRDISample type of ReSTIRGISample type automatically
 * based on whether or not we're instantiating the structures for ReSTIR DI or ReSTIR GI
 *
 * This sample type is then used in some of the specialization to pass to the target functions
 */
template <bool IsReSTIRGI>
struct SampleTypeStruct {};

template <>
struct SampleTypeStruct<false>
{
	using Type = ReSTIRDISample;
};

template <>
struct SampleTypeStruct<true>
{
	using Type = ReSTIRGISample;
};

template <bool IsReSTIRGI>
using ReSTIRSampleType = SampleTypeStruct<IsReSTIRGI>::Type;

#endif
