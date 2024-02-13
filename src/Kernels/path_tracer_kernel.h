#include "Kernels/includes/HIPRT_camera.h"
#include "Kernels/includes/HIPRT_common.h"
#include "Kernels/includes/hiprt_fix_vs.h"
//#include "Kernels/includes/hiprt_lambertian.h"
#include "Kernels/includes/HIPRT_maths.h"
#include "Kernels/includes/hiprt_render_data.h"

#include <hiprt/hiprt_device.h>
#include <hiprt/hiprt_vec.h>

__device__ void branchlessONB(const hiprtFloat3& n, hiprtFloat3& b1, hiprtFloat3& b2)
{
    float sign = n.z < 0 ? -1.0f : 1.0f;
    const float a = -1.0f / (sign + n.z);
    const float b = n.x * n.y * a;
    b1 = hiprtFloat3{ 1.0f + sign * n.x * n.x * a, sign * b, -sign * n.x };
    b2 = hiprtFloat3{ b, sign + n.y * n.y * a, -n.y };
}

__device__ hiprtFloat3 rotate_vector_around_normal(const hiprtFloat3& normal, const hiprtFloat3& random_dir_local_space)
{
    hiprtFloat3 tangent, bitangent;
    branchlessONB(normal, tangent, bitangent);

    //Transforming from the random_direction in its local space to the space around the normal
    //given in parameter (the space with the given normal as the Z up vector)
    return random_dir_local_space.x * tangent + random_dir_local_space.y * bitangent + random_dir_local_space.z * normal;
}

__device__ hiprtFloat3 hiprt_cosine_weighted_direction_around_normal(const hiprtFloat3& normal, float& pdf, HIPRT_xorshift32_generator& random_number_generator)
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

__device__ HIPRTColor hiprt_lambertian_brdf(const HIPRTRendererMaterial& material, const hiprtFloat3& to_light_direction, const hiprtFloat3& view_direction, const hiprtFloat3& surface_normal)
{
    return material.diffuse * M_1_PI;
}

__device__ bool trace_ray(hiprtGeometry geom, hiprtRay ray, HIPRTRenderData& render_data, HIPRTHitInfo& hit_info)
{
    //TODO use global stack for good traversal performance
    hiprtGeomTraversalClosest tr(geom, ray);
    hiprtHit				  hit = tr.getNextHit();

    if (hit.hasHit())
    {
        hit_info.inter_point = ray.origin + hit.t * ray.direction;
        // TODO hit.normal is in object space but we need world space normals. The line below assumes that all objects have
        // already been pretransformed in world space. This may not be true anymore with multiple level
        // acceleration structures
        hit_info.normal_at_intersection = normalize(hit.normal);
        hit_info.t = hit.t;
        hit_info.primitive_index = hit.primID;

        return true;
    }
    else
        return false;
}

GLOBAL_KERNEL_SIGNATURE(void) PathTracerKernel(hiprtGeometry geom, HIPRTRenderData render_data, HIPRTColor* pixels, int2 res, HIPRTCamera camera)
{
    const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const uint32_t index = (x + y * res.x);

    if (index >= res.x * res.y)
        return;


    // TODO try to use constructor
    HIPRT_xorshift32_generator random_number_generator;
    random_number_generator.m_state.a = (31 + x * y * (render_data.frame_number + 1));// = HIPRT_xorshift32_generator{ HIPRT_xorshift32_state{31 + x * y * render_data.frame_number} };
    //Generating some numbers to make sure the generators of each thread spread apart
    //If not doing this, the generator shows clear artifacts until it has generated
    //a few numbers
    for (int i = 0; i < 25; i++)
        random_number_generator();

    HIPRTColor final_color = HIPRTColor{ 0.0f, 0.0f, 0.0f };
    //TODO variable samples count per frame instead of one ?
    /*for (int sample = 0; sample < 1; sample++)
    {*/




    //Jittered around the center
    float x_jittered = (x + 0.5f) + random_number_generator() - 1.0f;
    float y_jittered = (y + 0.5f) + random_number_generator() - 1.0f;

    hiprtRay ray = camera.get_camera_ray(x_jittered, y_jittered, res);

    HIPRTColor throughput = HIPRTColor{ 1.0f, 1.0f, 1.0f };
    HIPRTColor sample_color = HIPRTColor{ 0.0f, 0.0f, 0.0f };
    HIPRTRayState next_ray_state = HIPRTRayState::HIPRT_BOUNCE;
    HIPRTBRDF last_brdf_hit_type = HIPRTBRDF::HIPRT_Uninitialized;

    for (int bounce = 0; bounce < 20; bounce++)//render_data.nb_bounces; bounce++)
    {
        if (next_ray_state == HIPRTRayState::HIPRT_BOUNCE)
        {
            HIPRTHitInfo closest_hit_info;
            bool intersection_found = trace_ray(geom, ray, render_data, closest_hit_info);

            if (intersection_found)
            {
                int material_index = render_data.material_indices[closest_hit_info.primitive_index];
                HIPRTRendererMaterial material = render_data.materials_buffer[material_index];

                last_brdf_hit_type = material.brdf_type;

                // --------------------------------------------------- //
                // ----------------- Direct lighting ----------------- //
                // --------------------------------------------------- //
                //TODO area sampling triangles
                /*Color light_sample_radiance = sample_light_sources(ray, closest_hit_info, material, random_number_generator);
                Color env_map_radiance = sample_environment_map(ray, closest_hit_info, material, random_number_generator);*/

                HIPRTColor light_sample_radiance;
                /*if (bounce > 0)
                    light_sample_radiance = HIPRTColor{ 1.0f, 1.0f, 1.0f, 1.0f };
                else*/
                    light_sample_radiance = HIPRTColor{ 0.0f, 0.0f, 0.0f, 0.0f };
                HIPRTColor env_map_radiance = HIPRTColor{ 0.0f, 0.0f, 0.0f, 0.0f };

                // --------------------------------------- //
                // ---------- Indirect lighting ---------- //
                // --------------------------------------- //

                float brdf_pdf;

                hiprtFloat3 bounce_direction;
                //Color brdf = brdf_dispatcher_sample(material, bounce_direction, ray.direction, closest_hit_info.normal_at_intersection, brdf_pdf, random_number_generator); //TODO relative IOR in the RayData rather than two incident and output ior values
                bounce_direction = hiprt_cosine_weighted_direction_around_normal(closest_hit_info.normal_at_intersection, brdf_pdf, random_number_generator);
                HIPRTColor brdf = hiprt_lambertian_brdf(material, bounce_direction, -ray.direction, closest_hit_info.normal_at_intersection);

                //if (bounce == 0)
                    sample_color = sample_color + material.emission * throughput;
                sample_color = sample_color + (light_sample_radiance + env_map_radiance) * throughput;

                if ((brdf.r == 0.0f && brdf.g == 0.0f && brdf.b == 0.0f) || brdf_pdf < 1.0e-8f || isinf(brdf_pdf))
                {
                    next_ray_state = HIPRTRayState::HIPRT_TERMINATED;

                    break;
                }

                throughput = throughput * brdf * RT_MAX(0.0f, dot(bounce_direction, closest_hit_info.normal_at_intersection)) / brdf_pdf;
                //pixels[y * res.x + x] = throughput; // HIPRTColor { throughput.x, throughput.y, throughput.z, 1.0f };
                //return;

                hiprtFloat3 new_ray_origin = closest_hit_info.inter_point + closest_hit_info.normal_at_intersection * 1.0e-4f;
                ray.origin = new_ray_origin; // Updating the next ray origin
                ray.direction = bounce_direction; // Updating the next ray direction

                next_ray_state = HIPRTRayState::HIPRT_BOUNCE;
            }
            else
                next_ray_state = HIPRTRayState::HIPRT_MISSED;
        }
        else if (next_ray_state == HIPRTRayState::HIPRT_MISSED)
        {
            //if (bounce == 1 || last_brdf_hit_type == HIPRTBRDF::HIPRT_SpecularFresnel)
            {
                //We're only getting the skysphere radiance for the first rays because the
                //syksphere is importance sampled
                // We're also getting the skysphere radiance for perfectly specular BRDF since those
                // are not importance sampled

                //Color skysphere_color = sample_environment_map_from_direction(ray.direction);
                HIPRTColor skysphere_color = HIPRTColor{ 2.0f, 2.0f, 2.0f };

                // TODO try overload +=, *=, ... operators
                sample_color = sample_color + skysphere_color * throughput;
            }

            break;
        }
        else if (next_ray_state == HIPRTRayState::HIPRT_TERMINATED)
            break;
    }

    final_color = final_color + sample_color;


    //}

    final_color = final_color / 1; //TODO this is 1 sample per frame
    final_color.a = 0.0f;

    if (render_data.frame_number == 0)
        pixels[y * res.x + x] = final_color;
    else
        pixels[y * res.x + x] = pixels[y * res.x + x] + final_color;
}