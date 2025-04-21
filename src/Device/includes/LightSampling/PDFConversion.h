/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_PDF_CONVERSION_H
#define DEVICE_INCLUDES_PDF_CONVERSION_H

#include "HostDeviceCommon/Math.h"

HIPRT_INLINE HIPRT_HOST_DEVICE float area_to_solid_angle_pdf(float area_pdf, float distance, float cos_theta)
{
    if (cos_theta < 1.0e-8f)
        return 0.0f;

    return area_pdf * hippt::square(distance) / cos_theta;
}

HIPRT_INLINE HIPRT_HOST_DEVICE float solid_angle_to_area_pdf(float area_pdf, float distance, float cos_theta)
{
    if (cos_theta < 1.0e-8f)
        return 0.0f;

    return area_pdf / hippt::square(distance) * cos_theta;
}

#endif
