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

// TODO include in lambertian.h instead of here
__device__ HIPRTColor hiprt_lambertian_brdf(const HIPRTRendererMaterial& material, const hiprtFloat3& to_light_direction, const hiprtFloat3& view_direction, const hiprtFloat3& surface_normal)
{
    return material.diffuse * M_1_PI;
}

/**
 * Reflects a ray about a normal. This function requires that dot(ray_direction, surface_normal) > 0 i.e.
 * ray_direction and surface_normal are in the same hemisphere
 */
__device__ hiprtFloat3 reflect_ray(const hiprtFloat3& ray_direction, const hiprtFloat3& surface_normal)
{
    return -ray_direction + 2.0f * dot(ray_direction, surface_normal) * surface_normal;
}

__device__ float fresnel_dielectric(float cos_theta_i, float eta_i, float eta_t)
{
    // Computing cos_theta_t
    float sinThetaI = sqrt(1.0f - cos_theta_i * cos_theta_i);
    float sin_theta_t = eta_i / eta_t * sinThetaI;

    if (sin_theta_t >= 1.0f)
        // Total internal reflection, 0% refraction, all reflection
        return 1.0f;

    float cos_theta_t = sqrt(1.0f - sin_theta_t * sin_theta_t);
    float r_parallel = ((eta_t * cos_theta_i) - (eta_i * cos_theta_t)) / ((eta_t * cos_theta_i) + (eta_i * cos_theta_t));
    float r_perpendicular = ((eta_i * cos_theta_i) - (eta_t * cos_theta_t)) / ((eta_i * cos_theta_i) + (eta_t * cos_theta_t));
    return (r_parallel * r_parallel + r_perpendicular * r_perpendicular) / 2;
}

/**
 * Reflects a ray about a normal. This function requires that dot(ray_direction, surface_normal) > 0 i.e.
 * ray_direction and surface_normal are in the same hemisphere
 */
__device__ bool refract_ray(const hiprtFloat3& ray_direction, const hiprtFloat3& surface_normal, hiprtFloat3& refract_direction, float relative_eta)
{
    float NoI = dot(ray_direction, surface_normal);

    float sin_theta_i_2 = 1.0f - NoI * NoI;
    float root_term = 1.0f - sin_theta_i_2 / (relative_eta * relative_eta);
    if (root_term < 0.0f)
        return false;

    float cos_theta_t = sqrt(root_term);
    refract_direction = -ray_direction / relative_eta + (NoI / relative_eta - cos_theta_t) * surface_normal;

    return true;
}

__device__ HIPRTColor smooth_glass_bsdf(const HIPRTRendererMaterial& material, hiprtFloat3& out_bounce_direction, const hiprtFloat3& ray_direction, hiprtFloat3& surface_normal, float eta_i, float eta_t, float& pdf, HIPRT_xorshift32_generator& random_generator)
{
    float cos_theta_i = dot(surface_normal, -ray_direction);

    if (cos_theta_i < 0.0f)
    {
        // We're inside the surface, we're going to flip the eta and the normal for
        // the calculations that follow
        // Note that this also flips the normal for the caller of this function
        // since the normal is passed by reference. This is useful since the normal
        // will be used for offsetting the new ray origin for example
        cos_theta_i = -cos_theta_i;
        surface_normal = -surface_normal;

        float temp = eta_i;
        eta_i = eta_t;
        eta_t = temp;
    }

    // Computing the proportion of reflected light using fresnel equations
    // We're going to use the result to decide whether to refract or reflect the
    // ray
    float fresnel_reflect = fresnel_dielectric(cos_theta_i, eta_i, eta_t);
    if (random_generator() <= fresnel_reflect)
    {
        // Reflect the ray

        out_bounce_direction = reflect_ray(-ray_direction, surface_normal);
        pdf = fresnel_reflect;

        return HIPRTColor(fresnel_reflect) / dot(surface_normal, out_bounce_direction);
    }
    else
    {
        // Refract the ray

        hiprtFloat3 refract_direction;
        bool can_refract = refract_ray(-ray_direction, surface_normal, refract_direction, eta_t / eta_i);
        if (!can_refract)
        {
            // Shouldn't happen (?)
            return HIPRTColor(1000000.0f, 0.0f, 1000000.0f); //Omega pink
        }

        out_bounce_direction = refract_direction;
        surface_normal = -surface_normal;
        pdf = 1.0f - fresnel_reflect;

        // TODO use constructor
        return HIPRTColor(1.0f - fresnel_reflect) * material.diffuse / dot(out_bounce_direction, surface_normal);
    }
}

__device__ HIPRTColor brdf_dispatcher_sample(const HIPRTRendererMaterial& material, hiprtFloat3& bounce_direction, const hiprtFloat3& ray_direction, hiprtFloat3& surface_normal, float& brdf_pdf, HIPRT_xorshift32_generator& random_number_generator)
{
    if (material.brdf_type == HIPRTBRDF::HIPRT_SpecularFresnel)
        return smooth_glass_bsdf(material, bounce_direction, ray_direction, surface_normal, 1.0f, material.ior, brdf_pdf, random_number_generator); //TODO relative IOR in the RayData rather than two incident and output ior values
    else if (material.brdf_type == HIPRTBRDF::HIPRT_CookTorrance)
    {
        bounce_direction = hiprt_cosine_weighted_direction_around_normal(surface_normal, brdf_pdf, random_number_generator);
        return hiprt_lambertian_brdf(material, bounce_direction, -ray_direction, surface_normal);
    //return cook_torrance_brdf_importance_sample(material, -ray_direction, surface_normal, bounce_direction, brdf_pdf, random_number_generator);
    }

    return HIPRTColor(0.0f);
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

__device__ unsigned int wang_hash(unsigned int seed)
{
    seed = (seed ^ 61) ^ (seed >> 16);
    seed *= 9;
    seed = seed ^ (seed >> 4);
    seed *= 0x27d4eb2d;
    seed = seed ^ (seed >> 15);
    return seed;
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
    // Getting a random for the xorshift seed from the pixel index using wang_hash
    // + 1 used to avoid zeros
    random_number_generator.m_state.a = (wang_hash((index + 1) * (render_data.render_settings.frame_number + 1)));

    HIPRTColor final_color = HIPRTColor{ 0.0f, 0.0f, 0.0f };
    for (int sample = 0; sample < render_data.render_settings.samples_per_frame; sample++)
    {
        //Jittered around the center
        float x_jittered = (x + 0.5f) + random_number_generator() - 1.0f;
        float y_jittered = (y + 0.5f) + random_number_generator() - 1.0f;

        hiprtRay ray = camera.get_camera_ray(x_jittered, y_jittered, res);

        HIPRTColor throughput = HIPRTColor{ 1.0f, 1.0f, 1.0f };
        HIPRTColor sample_color = HIPRTColor{ 0.0f, 0.0f, 0.0f };
        HIPRTRayState next_ray_state = HIPRTRayState::HIPRT_BOUNCE;
        HIPRTBRDF last_brdf_hit_type = HIPRTBRDF::HIPRT_Uninitialized;

        for (int bounce = 0; bounce < render_data.render_settings.nb_bounces; bounce++)
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
                    //HIPRTColor light_sample_radiance = sample_light_sources(ray, closest_hit_info, material, random_number_generator);
                    //Color env_map_radiance = sample_environment_map(ray, closest_hit_info, material, random_number_generator);

                    HIPRTColor light_sample_radiance = HIPRTColor(0.0f);
                    HIPRTColor env_map_radiance = HIPRTColor(0.0f);

                    // --------------------------------------- //
                    // ---------- Indirect lighting ---------- //
                    // --------------------------------------- //

                    float brdf_pdf;
                    hiprtFloat3 bounce_direction;
                    HIPRTColor brdf = brdf_dispatcher_sample(material, bounce_direction, ray.direction, closest_hit_info.normal_at_intersection, brdf_pdf, random_number_generator); //TODO relative IOR in the RayData rather than two incident and output ior values

                    //float brdf_pdf;

                    //hiprtFloat3 bounce_direction;
                    ////Color brdf = brdf_dispatcher_sample(material, bounce_direction, ray.direction, closest_hit_info.normal_at_intersection, brdf_pdf, random_number_generator); //TODO relative IOR in the RayData rather than two incident and output ior values
                    //bounce_direction = hiprt_cosine_weighted_direction_around_normal(closest_hit_info.normal_at_intersection, brdf_pdf, random_number_generator);
                    //HIPRTColor brdf = hiprt_lambertian_brdf(material, bounce_direction, -ray.direction, closest_hit_info.normal_at_intersection);

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
                {
                    //if (bounce == 1 || last_brdf_hit_type == HIPRTBRDF::HIPRT_SpecularFresnel)
                    {
                        //We're only getting the skysphere radiance for the first rays because the
                        //syksphere is importance sampled
                        // We're also getting the skysphere radiance for perfectly specular BRDF since those
                        // are not importance sampled

                        //Color skysphere_color = sample_environment_map_from_direction(ray.direction);
                        HIPRTColor skysphere_color = HIPRTColor{ 1.0f, 1.0f, 1.0f };

                        // TODO try overload +=, *=, ... operators
                        sample_color = sample_color + skysphere_color * throughput;
                    }

                    next_ray_state = HIPRTRayState::HIPRT_MISSED;
                }
            }
            else if (next_ray_state == HIPRTRayState::HIPRT_MISSED)
                break;
            else if (next_ray_state == HIPRTRayState::HIPRT_TERMINATED)
                break;
        }

        final_color = final_color + sample_color;
    }

    final_color.a = 0.0f;
    if (render_data.render_settings.frame_number == 0)
        pixels[y * res.x + x] = final_color;
    else
        pixels[y * res.x + x] = pixels[y * res.x + x] + final_color;
}