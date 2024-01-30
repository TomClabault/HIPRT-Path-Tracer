#ifndef HIPRT_CAMERA_H
#define HIPRT_CAMERA_H

#include "Kernels/includes/HIPRT_maths.h"

struct HIPRTCamera
{
    float4x4 view_matrix;

    // Distance to iamge plane
    float focal_length;
};

#endif