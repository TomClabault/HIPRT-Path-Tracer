/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef HOST_DEVICE_COMMON_LIGHT_SAMPLE_INFORMATION_H
#define HOST_DEVICE_COMMON_LIGHT_SAMPLE_INFORMATION_H

#include "Device/includes/PDFConversion.h"

#include "HostDeviceCommon/Color.h"

struct LightSampleInformation
{
    int emissive_triangle_index = -1;
    float3 light_source_normal = { 0.0f, 1.0f, 0.0f };
    float light_area = 1.0f;
    ColorRGB32F emission;

    float3 point_on_light = make_float3(0.0f, 0.0f, 0.0f);

    float area_measure_pdf = 0.0f;

    /**
     * The inverse PDF template parameter may be set to true if the given 'area_measure_pdf' parameter is in fact the 
     * inverse of the PDF. 
     * 
     * This is a typical case when sampling lights with ReGIR as we only have the UCW in that case
     * and the UCW is the inverse of the PDF.
     * 
     * Because light sampling procedures divide by 'LightSampleInformation::area_measure_pdf', ReGIR stores 1.0f / UCW
     * in 'LightSampleInformation::area_measure_pdf' such that dividing by '1.0f / UCW' in fact multiplies by the UCW, which
     * is how we want to use an UCW
     * 
     * The issue with that 1.0f / UCW trick is that now, we're effectively going to have the geometry term for converting
     * to solid angle inverted. To end up with the proper PDF, we need to invert the solid angle conversion term, effectively
     * computing 'solid_angle_to_area' conversion instead of 'area_to_solid_angle'
     */
    template <bool inversePDF = false>
    HIPRT_HOST_DEVICE static float get_solid_angle_measure_pdf(float area_measure_pdf, float distance_to_light, float cosine_angle_at_light)
    {
        if constexpr (inversePDF)
            return solid_angle_to_area_pdf(area_measure_pdf, distance_to_light, cosine_angle_at_light);
        else
            return area_to_solid_angle_pdf(area_measure_pdf, distance_to_light, cosine_angle_at_light);
    }
};

#endif
