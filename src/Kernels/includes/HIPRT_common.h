#ifndef HIPRTRT_COMMON
#define HIPRTRT_COMMON

#include "Kernels/includes/HIPRT_camera.h"
#include "Kernels/includes/HIPRT_maths.h"
#include "Kernels/includes/HIPRT_scene_data.h"

#include <hiprt/hiprt_device.h>
#include <hiprt/hiprt_vec.h>

// The HIPRT_KERNEL_SIGNATURE is only useful to help Visual Studio's Intellisense
// Without this macro, all kernel functions would be declared as:
// extern "C" void __global__ my_function(......)
// but Visual Studio doesn't like the 'extern "C" void __global__' part and it
// breaks code coloration and autocompletion. It is however required for the shader
// compiler
// To circumvent this problem, we're only declaring the functions 'void' when in the text editor
// (when __KERNELCC__ is not defined) and we're correctly declaring the function with the full
// attributes when it's the shader compiler processing the function (when __KERNELCC__ is defined)
// We're also defining blockDim, blockIdx and threadIdx because they are udefined otherwise...
#ifdef __KERNELCC__
#define GLOBAL_KERNEL_SIGNATURE(returnType) extern "C" returnType __global__
#define DEVICE_KERNEL_SIGNATURE(returnType) extern "C" returnType __device__
#else
struct dummyVec3
{
    int x, y, z;
};

dummyVec3 blockDim, blockIdx, threadIdx;

#define GLOBAL_KERNEL_SIGNATURE(returnType) returnType
#define DEVICE_KERNEL_SIGNATURE(returnType) returnType
#endif

struct xorshift32_state {
    unsigned int a = 42;
};

struct xorshift32_generator
{
    xorshift32_generator(unsigned int seed) : m_state({ seed }) {}

    float operator()()
    {
        //Float in [0, 1[
        return RT_MIN(xorshift32() / (float)UINT_MAX, 1.0f - 1.0e-6f);
    }

    unsigned int xorshift32()
    {
        /* Algorithm "xor" from p. 4 of Marsaglia, "Xorshift RNGs" */
        unsigned int x = m_state.a;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        return m_state.a = x;
    }

    xorshift32_state m_state;
};

DEVICE_KERNEL_SIGNATURE(hiprtRay) get_camera_ray(HIPRTCamera camera, float x, float y, int res_x, int res_y)
{
    float x_ndc_space = x / res_x * 2 - 1;
    x_ndc_space *= (float)res_x / res_y; //Aspect ratio
    float y_ndc_space = y / res_y * 2 - 1;

    hiprtFloat3 ray_origin_view_space = { 0.0f, 0.0f, 0.0f };
    hiprtFloat3 ray_origin = matrix_X_point(camera.view_matrix, ray_origin_view_space);

    hiprtFloat3 ray_point_direction_ndc_space = { x_ndc_space, y_ndc_space, -camera.focal_length };
    hiprtFloat3 ray_point_direction_world_space = matrix_X_point(camera.view_matrix, ray_point_direction_ndc_space);

    hiprtFloat3 ray_direction = normalize(ray_point_direction_world_space - ray_origin);
    
    hiprtRay ray;
    ray.origin = ray_origin;
    ray.direction = ray_direction;

    return ray;
}

//void RenderKernel::ray_trace_pixel(int x, int y) const
//{
//    xorshift32_generator random_number_generator(31 + x * y * m_render_samples);
//    //Generating some numbers to make sure the generators of each thread spread apart
//    //If not doing this, the generator shows clear artifacts until it has generated
//    //a few numbers
//    for (int i = 0; i < 10; i++)
//        random_number_generator();
//
//    Color final_color = Color(0.0f, 0.0f, 0.0f);
//    for (int sample = 0; sample < m_render_samples; sample++)
//    {
//        //Jittered around the center
//        float x_jittered = (x + 0.5f) + random_number_generator() - 1.0f;
//        float y_jittered = (y + 0.5f) + random_number_generator() - 1.0f;
//
//        //TODO area sampling triangles
//        Ray ray = get_camera_ray(x_jittered, y_jittered);
//
//        Color throughput = Color(1.0f, 1.0f, 1.0f);
//        Color sample_color = Color(0.0f, 0.0f, 0.0f);
//        RayState next_ray_state = RayState::BOUNCE;
//        BRDF last_brdf_hit_type = BRDF::Uninitialized;
//
//        for (int bounce = 0; bounce < m_max_bounces; bounce++)
//        {
//            if (next_ray_state == BOUNCE)
//            {
//                HitInfo closest_hit_info;
//                bool intersection_found = INTERSECT_SCENE(ray, closest_hit_info);
//
//                if (intersection_found)
//                {
//                    int material_index = m_materials_indices_buffer[closest_hit_info.primitive_index];
//                    RendererMaterial material = m_materials_buffer[material_index];
//                    last_brdf_hit_type = material.brdf_type;
//
//                    // --------------------------------------------------- //
//                    // ----------------- Direct lighting ----------------- //
//                    // --------------------------------------------------- //
//                    Color light_sample_radiance = sample_light_sources(ray, closest_hit_info, material, random_number_generator);
//                    Color env_map_radiance = sample_environment_map(ray, closest_hit_info, material, random_number_generator);
//
//                    // --------------------------------------- //
//                    // ---------- Indirect lighting ---------- //
//                    // --------------------------------------- //
//
//                    float brdf_pdf;
//
//                    Vector bounce_direction;
//                    Color brdf = brdf_dispatcher_sample(material, bounce_direction, ray.direction, closest_hit_info.normal_at_intersection, brdf_pdf, random_number_generator); //TODO relative IOR in the RayData rather than two incident and output ior values
//
//                    if (bounce == 0)
//                        sample_color += material.emission;
//                    sample_color += (light_sample_radiance + env_map_radiance) * throughput;
//
//                    if ((brdf.r == 0.0f && brdf.g == 0.0f && brdf.b == 0.0f) || brdf_pdf < 1.0e-8f || std::isinf(brdf_pdf))
//                    {
//                        next_ray_state = RayState::TERMINATED;
//
//                        break;
//                    }
//
//                    throughput *= brdf * std::max(0.0f, dot(bounce_direction, closest_hit_info.normal_at_intersection)) / brdf_pdf;
//
//                    //TODO RayData rather than having the normal, ray direction, is inside surface, ... as free variables in the code
//                    Point new_ray_origin = closest_hit_info.inter_point + closest_hit_info.normal_at_intersection * 1.0e-4f;
//                    ray = Ray(new_ray_origin, bounce_direction);
//                    next_ray_state = RayState::BOUNCE;
//                }
//                else
//                    next_ray_state = RayState::MISSED;
//            }
//            else if (next_ray_state == MISSED)
//            {
//                if (bounce == 1 || last_brdf_hit_type == BRDF::SpecularFresnel)
//                {
//                    //We're only getting the skysphere radiance for the first rays because the
//                    //syksphere is importance sampled
//                    // We're also getting the skysphere radiance for perfectly specular BRDF since those
//                    // are not importance sampled
//
//                    Color skysphere_color = sample_environment_map_from_direction(ray.direction);
//
//                    sample_color += skysphere_color * throughput;
//                }
//
//                break;
//            }
//            else if (next_ray_state == TERMINATED)
//                break;
//        }
//
//        final_color += sample_color;
//    }
//
//    final_color /= m_render_samples;
//    final_color.a = 0.0f;
//    m_frame_buffer[y * m_width + x] += final_color;
//
//    const float gamma = 2.2f;
//    const float exposure = 2.0f;
//    Color hdrColor = m_frame_buffer[y * m_width + x];
//
//    //Exposure tone mapping
//    Color tone_mapped = Color(1.0f, 1.0f, 1.0f) - exp(-hdrColor * exposure);
//    // Gamma correction
//    Color gamma_corrected = pow(tone_mapped, 1.0f / gamma);
//
//    m_frame_buffer[y * m_width + x] = gamma_corrected;
//}

#endif // !HIPRTRT_COMMON
