#ifndef HIPRT_CAMERA_H
#define HIPRT_CAMERA_H

#include "Kernels/includes/HIPRT_maths.h"

struct HIPRTCamera
{
    float4x4 inverse_view;
    float4x4 inverse_projection;
    hiprtFloat3 position;
};

#endif