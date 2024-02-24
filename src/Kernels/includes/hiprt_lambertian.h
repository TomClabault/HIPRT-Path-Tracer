#include "Kernels/includes/hiprt_color.h"
#include "Kernels/includes/HIPRT_common.h"
#include "Kernels/includes/HIPRT_maths.h"
#include "Kernels/includes/hiprt_onb.h"

inline __device__ hiprtFloat3 hiprt_cosine_weighted_direction_around_normal(const hiprtFloat3& normal, float& pdf, HIPRT_xorshift32_generator& random_number_generator)
{
    float rand_1 = random_number_generator();
    float rand_2 = random_number_generator();

    float sqrt_rand_2 = sqrt(rand_2);
    float phi = 2.0f * (float)M_PI * rand_1;
    float cos_theta = sqrt_rand_2;
    float sin_theta = sqrt(RT_MAX(0.0f, 1.0f - cos_theta * cos_theta));

    pdf = sqrt_rand_2 / (float)M_PI;

    //Generating a random direction in a local space with Z as the Up vector
    hiprtFloat3 random_dir_local_space = hiprtFloat3{ cos(phi) * sin_theta, sin(phi) * sin_theta, sqrt_rand_2 };
    return rotate_vector_around_normal(normal, random_dir_local_space);
}

inline __device__ HIPRTColor hiprt_lambertian_brdf(const HIPRTRendererMaterial& material, const hiprtFloat3& to_light_direction, const hiprtFloat3& view_direction, const hiprtFloat3& surface_normal)
{
    return material.diffuse * M_1_PI;
}