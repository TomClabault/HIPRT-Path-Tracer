#ifndef HIPRT_ONB_H
#define HIPRT_ONB_H

#include "Kernels/includes/HIPRT_common.h"
#include "Kernels/includes/HIPRT_maths.h"

__device__ void branchlessONB(const hiprtFloat3& n, hiprtFloat3& b1, hiprtFloat3& b2)
{
    float sign = n.z < 0 ? -1.0f : 1.0f;
    const float a = -1.0f / (sign + n.z);
    const float b = n.x * n.y * a;
    b1 = hiprtFloat3{ 1.0f + sign * n.x * n.x * a, sign * b, -sign * n.x };
    b2 = hiprtFloat3{ b, sign + n.y * n.y * a, -n.y };
}

__device__ hiprtFloat3 local_to_world_frame(const hiprtFloat3& normal, const hiprtFloat3& random_dir_local_space)
{
    hiprtFloat3 tangent, bitangent;
    branchlessONB(normal, tangent, bitangent);

    //Transforming from the random_direction in its local space to the space around the normal
    //given in parameter (the space with the given normal as the Z up vector)
    return random_dir_local_space.x * tangent + random_dir_local_space.y * bitangent + random_dir_local_space.z * normal;
}

#endif