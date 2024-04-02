#include "Kernels/includes/HIPRT_camera.h"
#include "Kernels/includes/HIPRT_common.h"
#include "Kernels/includes/hiprt_disney.h"
#include "Kernels/includes/hiprt_fix_vs.h"
#include "Kernels/includes/HIPRT_maths.h"
#include "Kernels/includes/hiprt_render_data.h"
#include "Kernels/includes/hiprt_sampling.h"

#include <hiprt/hiprt_device.h>
#include <hiprt/hiprt_vec.h>

__device__ float cook_torrance_brdf_pdf(const RendererMaterial& material, const hiprtFloat3& view_direction, const hiprtFloat3& to_light_direction, const hiprtFloat3& surface_normal)
{
    hiprtFloat3 microfacet_normal = normalize(view_direction + to_light_direction);

    float alpha = material.roughness * material.roughness;

    float VoH = RT_MAX(0.0f, dot(view_direction, microfacet_normal));
    float NoH = RT_MAX(0.0f, dot(surface_normal, microfacet_normal));
    float D = GGX_normal_distribution(alpha, NoH);

    return D * NoH / (4.0f * VoH);
}

__device__ Color cook_torrance_brdf(const RendererMaterial& material, const hiprtFloat3& to_light_direction, const hiprtFloat3& view_direction, const hiprtFloat3& surface_normal)
{
    Color brdf_color = Color(0.0f, 0.0f, 0.0f);
    Color base_color = material.diffuse;

    hiprtFloat3 halfway_vector = normalize(view_direction + to_light_direction);

    float NoV = RT_MAX(0.0f, dot(surface_normal, view_direction));
    float NoL = RT_MAX(0.0f, dot(surface_normal, to_light_direction));
    float NoH = RT_MAX(0.0f, dot(surface_normal, halfway_vector));
    float VoH = RT_MAX(0.0f, dot(halfway_vector, view_direction));

    if (NoV > 0.0f && NoL > 0.0f && NoH > 0.0f)
    {
        float metalness = material.metalness;
        float roughness = material.roughness;

        float alpha = roughness * roughness;

        ////////// Cook Torrance BRDF //////////
        Color F;
        float D, G;

        //F0 = 0.04 for dielectrics, 1.0 for metals (approximation)
        Color F0 = Color(0.04f * (1.0f - metalness)) + metalness * base_color;

        //GGX Distribution function
        F = fresnel_schlick(F0, VoH);
        D = GGX_normal_distribution(alpha, NoH);
        G = GGX_smith_masking_shadowing(alpha, NoV, NoL);

        Color kD = Color(1.0f - metalness); //Metals do not have a diffuse part
        kD = kD * (Color(1.0f) - F);//Only the transmitted light is diffused

        Color diffuse_part = kD * base_color / (float)M_PI;
        Color specular_part = (F * D * G) / (4.0f * NoV * NoL);

        brdf_color = diffuse_part + specular_part;
    }

    return brdf_color;
}

__device__ Color cook_torrance_brdf_importance_sample(const RendererMaterial& material, const hiprtFloat3& view_direction, const hiprtFloat3& surface_normal, hiprtFloat3& output_direction, float& pdf, Xorshift32Generator& random_number_generator)
{
    pdf = 0.0f;

    float metalness = material.metalness;
    float roughness = material.roughness;
    float alpha = roughness * roughness;

    float rand1 = random_number_generator();
    float rand2 = random_number_generator();

    float phi = 2.0f * (float)M_PI * rand1;
    float theta = acos((1.0f - rand2) / (rand2 * (alpha * alpha - 1.0f) + 1.0f));
    float sin_theta = sin(theta);

    // The microfacet normal is sampled in its local space, we'll have to bring it to the space
    // around the surface normal
    hiprtFloat3 microfacet_normal_local_space = hiprtFloat3(cos(phi) * sin_theta, sin(phi) * sin_theta, cos(theta));
    hiprtFloat3 microfacet_normal = local_to_world_frame(surface_normal, microfacet_normal_local_space);
    if (dot(microfacet_normal, surface_normal) < 0.0f)
        //The microfacet normal that we sampled was under the surface, this can happen
        return Color(0.0f);
    hiprtFloat3 to_light_direction = normalize(2.0f * dot(microfacet_normal, view_direction) * microfacet_normal - view_direction);
    hiprtFloat3 halfway_vector = microfacet_normal;
    output_direction = to_light_direction;

    Color brdf_color = Color(0.0f, 0.0f, 0.0f);
    Color base_color = material.diffuse;

    float NoV = RT_MAX(0.0f, dot(surface_normal, view_direction));
    float NoL = RT_MAX(0.0f, dot(surface_normal, to_light_direction));
    float NoH = RT_MAX(0.0f, dot(surface_normal, halfway_vector));
    float VoH = RT_MAX(0.0f, dot(halfway_vector, view_direction));

    if (NoV > 0.0f && NoL > 0.0f && NoH > 0.0f)
    {
        /////////// Cook Torrance BRDF //////////
        Color F;
        float D, G;

        //GGX Distribution function
        D = GGX_normal_distribution(alpha, NoH);

        //F0 = 0.04 for dielectrics, 1.0 for metals (approximation)
        Color F0 = Color(0.04f * (1.0f - metalness)) + metalness * base_color;
        F = fresnel_schlick(F0, VoH);
        G = GGX_smith_masking_shadowing(alpha, NoV, NoL);

        Color kD = Color(1.0f - metalness); //Metals do not have a diffuse part
        kD = kD * (Color(1.0f) - F);//Only the transmitted light is diffused

        Color diffuse_part = kD * base_color / (float)M_PI;
        Color specular_part = (F * D * G) / (4.0f * NoV * NoL);

        pdf = D * NoH / (4.0f * VoH);

        brdf_color = diffuse_part + specular_part;
    }

    return brdf_color;
}

__device__ Color smooth_glass_bsdf(const RendererMaterial& material, hiprtFloat3& out_bounce_direction, const hiprtFloat3& ray_direction, hiprtFloat3& surface_normal, float eta_i, float eta_t, float& pdf, Xorshift32Generator& random_generator)
{
    // Clamping here because the dot product can eventually returns values less
    // than -1 or greater than 1 because of precision errors in the vectors
    // (in previous calculations)
    float cos_theta_i = RT_MIN(RT_MAX(-1.0f, dot(surface_normal, -ray_direction)), 1.0f);

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

        return Color(fresnel_reflect) / dot(surface_normal, out_bounce_direction);
    }
    else
    {
        // Refract the ray

        hiprtFloat3 refract_direction;
        bool can_refract = refract_ray(-ray_direction, surface_normal, refract_direction, eta_t / eta_i);
        if (!can_refract)
        {
            // Shouldn't happen (?)
            return Color(1000000.0f, 0.0f, 1000000.0f); // Omega pink
        }

        out_bounce_direction = refract_direction;
        surface_normal = -surface_normal;
        pdf = 1.0f - fresnel_reflect;

        return Color(1.0f - fresnel_reflect) * material.diffuse / dot(out_bounce_direction, surface_normal);
    }
}

__device__ Color brdf_dispatcher_eval(const RendererMaterial& material, const hiprtFloat3& view_direction, const hiprtFloat3& surface_normal, const hiprtFloat3& to_light_direction, float& pdf)
{
    return disney_eval(material, view_direction, surface_normal, to_light_direction, pdf);
}

__device__ Color brdf_dispatcher_sample(const RendererMaterial& material, const hiprtFloat3& view_direction, hiprtFloat3& surface_normal, const hiprtFloat3& geometric_normal, hiprtFloat3& bounce_direction, float& brdf_pdf, Xorshift32Generator& random_number_generator)
{
    return disney_sample(material, view_direction, surface_normal, geometric_normal, bounce_direction, brdf_pdf, random_number_generator);
}

__device__ bool trace_ray(const HIPRTRenderData& render_data, hiprtRay ray, HitInfo& hit_info)
{
    hiprtGeomTraversalClosest tr(render_data.geom, ray);
    hiprtHit				  hit = tr.getNextHit();

    if (hit.hasHit())
    {
        hit_info.inter_point = ray.origin + hit.t * ray.direction;
        hit_info.primitive_index = hit.primID;

        // hit.normal is in object space, this simple approach will not work if using
        // multiple-levels BVH (TLAS/BLAS)
        hiprtFloat3 geometric_normal = normalize(hit.normal);
        
        int vertex_A_index = render_data.triangles_indices[hit_info.primitive_index * 3 + 0];
        if (render_data.normals_present[vertex_A_index])
        {
            // Smooth normal available for the triangle

            int vertex_B_index = render_data.triangles_indices[hit_info.primitive_index * 3 + 1];
            int vertex_C_index = render_data.triangles_indices[hit_info.primitive_index * 3 + 2];

            hiprtFloat3 smooth_normal = render_data.vertex_normals[vertex_B_index] * hit.uv.x
                + render_data.vertex_normals[vertex_C_index] * hit.uv.y
                + render_data.vertex_normals[vertex_A_index] * (1.0f - hit.uv.x - hit.uv.y);

            hit_info.shading_normal = normalize(smooth_normal);
        }
        else
            hit_info.shading_normal = geometric_normal;
        hit_info.geometric_normal = geometric_normal;

        hit_info.t = hit.t;
        hit_info.uv = hit.uv;

        return true;
    }
    else
        return false;
}

__device__ float power_heuristic(float pdf_a, float pdf_b)
{
    float pdf_a_squared = pdf_a * pdf_a;

    return pdf_a_squared / (pdf_a_squared + pdf_b * pdf_b);
}

__device__ hiprtFloat3 sample_random_point_on_lights(const HIPRTRenderData& render_data, Xorshift32Generator& random_number_generator, float& pdf, LightSourceInformation& light_info)
{
    int random_index = random_number_generator.random_index(render_data.emissive_triangles_count);
    int triangle_index = light_info.emissive_triangle_index = render_data.emissive_triangles_indices[random_index];
    

    hiprtFloat3 vertex_A = render_data.triangles_vertices[render_data.triangles_indices[triangle_index * 3 + 0]];
    hiprtFloat3 vertex_B = render_data.triangles_vertices[render_data.triangles_indices[triangle_index * 3 + 1]];
    hiprtFloat3 vertex_C = render_data.triangles_vertices[render_data.triangles_indices[triangle_index * 3 + 2]];

    float rand_1 = random_number_generator();
    float rand_2 = random_number_generator();

    float sqrt_r1 = sqrt(rand_1);
    float u = 1.0f - sqrt_r1;
    float v = (1.0f - rand_2) * sqrt_r1;

    hiprtFloat3 AB = vertex_B - vertex_A;
    hiprtFloat3 AC = vertex_C - vertex_A;

    hiprtFloat3 random_point_on_triangle = vertex_A + AB * u + AC * v;

    hiprtFloat3 normal = cross(AB, AC);
    float length_normal = length(normal);
    light_info.light_source_normal = normal / length_normal; // Normalization
    float triangle_area = length_normal * 0.5f;
    float nb_emissive_triangles = render_data.emissive_triangles_count;

    pdf = 1.0f / (nb_emissive_triangles * triangle_area);

    return random_point_on_triangle;
}

__device__ float triangle_area(const HIPRTRenderData& render_data, int triangle_index)
{
    hiprtFloat3 vertex_A = render_data.triangles_vertices[render_data.triangles_indices[triangle_index * 3 + 0]];
    hiprtFloat3 vertex_B = render_data.triangles_vertices[render_data.triangles_indices[triangle_index * 3 + 1]];
    hiprtFloat3 vertex_C = render_data.triangles_vertices[render_data.triangles_indices[triangle_index * 3 + 2]];

    hiprtFloat3 AB = vertex_B - vertex_A;
    hiprtFloat3 AC = vertex_C - vertex_A;

    return length(cross(AB, AC)) / 2.0f;
}

/**
 * Returns true if in shadow, false otherwise
 */
__device__ bool evaluate_shadow_ray(const HIPRTRenderData& render_data, hiprtRay ray, float t_max)
{
    ray.maxT = t_max - 1.0e-4f;

    hiprtGeomTraversalAnyHit traversal(render_data.geom, ray);
    hiprtHit aoHit = traversal.getNextHit();

    return aoHit.hasHit();
}

__device__ Color sample_light_sources(HIPRTRenderData& render_data, const hiprtFloat3& view_direction, HitInfo& closest_hit_info, const RendererMaterial& material, Xorshift32Generator& random_number_generator)
{
    if (material.brdf_type == BRDF::SpecularFresnel)
        // No sampling for perfectly specular materials
        return Color(0.0f);

    if (render_data.emissive_triangles_count == 0)
        // No emmisive geometry in the scene to sample
        return Color(0.0f);

    Color light_source_radiance_mis;
    float light_sample_pdf;
    LightSourceInformation light_source_info;
    hiprtFloat3 random_light_point = sample_random_point_on_lights(render_data, random_number_generator, light_sample_pdf, light_source_info);

    // Actually pushing the point towards the light here to avoid the shadow terminator problem
    hiprtFloat3 shadow_ray_origin = closest_hit_info.inter_point + closest_hit_info.shading_normal * 1.0e-4f;
    hiprtFloat3 shadow_ray_direction = random_light_point - shadow_ray_origin;
    float distance_to_light = length(shadow_ray_direction);
    hiprtFloat3 shadow_ray_direction_normalized = shadow_ray_direction / distance_to_light;


    hiprtRay shadow_ray;
    shadow_ray.origin = shadow_ray_origin;
    shadow_ray.direction = shadow_ray_direction_normalized;

    // abs() here to allow backfacing light sources
    float dot_light_source = abs(dot(light_source_info.light_source_normal, -shadow_ray.direction));
    if (dot_light_source > 0.0f)
    {
        bool in_shadow = evaluate_shadow_ray(render_data, shadow_ray, distance_to_light);

        if (!in_shadow)
        {
            const RendererMaterial& emissive_triangle_material = render_data.materials_buffer[render_data.material_indices[light_source_info.emissive_triangle_index]];

            light_sample_pdf *= distance_to_light * distance_to_light;
            light_sample_pdf /= dot_light_source;

            float pdf;
            Color brdf = brdf_dispatcher_eval(material, view_direction, closest_hit_info.shading_normal, shadow_ray.direction, pdf);
            if (pdf != 0.0f)
            {
                float mis_weight = power_heuristic(light_sample_pdf, pdf);

                Color Li = emissive_triangle_material.emission;
                float cosine_term = RT_MAX(dot(closest_hit_info.shading_normal, shadow_ray.direction), 0.0f);

                light_source_radiance_mis = Li * cosine_term * brdf * mis_weight / light_sample_pdf;
            }
        }
    }

    Color brdf_radiance_mis;

    hiprtFloat3 sampled_brdf_direction;
    float direction_pdf;
    Color brdf = brdf_dispatcher_sample(material, view_direction, closest_hit_info.shading_normal, closest_hit_info.geometric_normal, sampled_brdf_direction, direction_pdf, random_number_generator);
    if (brdf.r != 0.0f || brdf.g != 0.0f || brdf.b != 0.0f)
    {
        hiprtRay new_ray; 
        new_ray.origin = closest_hit_info.inter_point + closest_hit_info.shading_normal * 1.0e-5f;
        new_ray.direction = sampled_brdf_direction;

        HitInfo new_ray_hit_info;
        bool inter_found = trace_ray(render_data, new_ray, new_ray_hit_info);

        if (inter_found)
        {
            float cos_angle = RT_MAX(dot(new_ray_hit_info.shading_normal, -sampled_brdf_direction), 0.0f);
            if (cos_angle > 0.0f)
            {
                int material_index = render_data.material_indices[new_ray_hit_info.primitive_index];
                RendererMaterial material = render_data.materials_buffer[material_index];

                Color emission = material.emission;
                if (emission.r > 0 || emission.g > 0 || emission.b > 0)
                {
                    float distance_squared = new_ray_hit_info.t * new_ray_hit_info.t;
                    float light_area = triangle_area(render_data, new_ray_hit_info.primitive_index);

                    float light_pdf = distance_squared / (light_area * cos_angle);

                    float mis_weight = power_heuristic(direction_pdf, light_pdf);
                    float cosine_term = dot(closest_hit_info.shading_normal, sampled_brdf_direction);
                    brdf_radiance_mis = brdf * cosine_term * emission * mis_weight / direction_pdf;
                }
            }
        }
    }

    return light_source_radiance_mis + brdf_radiance_mis;
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

__device__ void debug_set_final_color(const HIPRTRenderData& render_data, int x, int y, int res_x, Color final_color)
{
    if (render_data.m_render_settings.sample_number == 0)
        render_data.pixels[y * res_x + x] = final_color;
    else
        render_data.pixels[y * res_x + x] = render_data.pixels[y * res_x + x] + final_color;
}

#define LOW_RESOLUTION_RENDER_DOWNSCALE 1
GLOBAL_KERNEL_SIGNATURE(void) PathTracerKernel(hiprtGeometry geom, HIPRTRenderData render_data, int2 res, HIPRTCamera camera)
{
    const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const uint32_t index = (x + y * res.x);

    if (index >= res.x * res.y)
        return;

    // 'Render low resolution' means that the user is moving the camera for example
    // so we're going to reduce the quality of the render for increased framerates
    // while moving
    if (render_data.m_render_settings.render_low_resolution)
    {
        // Reducing the number of bounces to 3
        render_data.m_render_settings.nb_bounces = 3;
        render_data.m_render_settings.samples_per_frame = 1;

        // If rendering at low resolution, only one pixel out of 
        // LOW_RESOLUTION_RENDER_DOWNSCALE x LOW_RESOLUTION_RENDER_DOWNSCALE will be rendered
        if (x & (LOW_RESOLUTION_RENDER_DOWNSCALE - 1) || y & (LOW_RESOLUTION_RENDER_DOWNSCALE - 1))
            return;
    }

    if (render_data.m_render_settings.sample_number == 0)
    {
        render_data.pixels[index] = Color(0.0f);
        render_data.denoiser_normals[index] = hiprtFloat3(1.0f, 1.0f, 1.0f);
        render_data.denoiser_albedo[index] = Color(0.0f, 0.0f, 0.0f);
    }

    Xorshift32Generator random_number_generator(wang_hash((index + 1) * (render_data.m_render_settings.sample_number + 1)));

    Color final_color = Color(0.0f, 0.0f, 0.0f);

    // This denoiser_blend variable is used when the rays hit delta function
    // BRDFs. In those cases, the denoiser albedo/normal is going to be
    // the Fresnel blend of the reflected and transmitted albedo/normal
    // so we're going to have to keep track of the blend coefficient across
    // the bounces
    Color denoiser_albedo = Color(0.0f, 0.0f, 0.0f);
    hiprtFloat3 denoiser_normal = hiprtFloat3{ 0.0f, 0.0f, 0.0f };
    for (int sample = 0; sample < render_data.m_render_settings.samples_per_frame; sample++)
    {
        //Jittered around the center
        float x_jittered = (x + 0.5f) + random_number_generator() - 1.0f;
        float y_jittered = (y + 0.5f) + random_number_generator() - 1.0f;

        hiprtRay ray = camera.get_camera_ray(x_jittered, y_jittered, res);

        Color throughput = Color{ 1.0f, 1.0f, 1.0f };
        Color sample_color = Color{ 0.0f, 0.0f, 0.0f };
        RayState next_ray_state = RayState::BOUNCE;
        BRDF last_brdf_hit_type = BRDF::Uninitialized;

        // Whether or not we've already written to the denoiser's buffers
        bool denoiser_AOVs_set = false;
        float denoiser_blend = 1.0f;

        for (int bounce = 0; bounce < render_data.m_render_settings.nb_bounces; bounce++)
        {
            if (next_ray_state == RayState::BOUNCE)
            {
                HitInfo closest_hit_info;
                bool intersection_found = trace_ray(render_data, ray, closest_hit_info);

                // Because I've had self-intersection issues in the past (offsetting the ray origin
                // from the surface along the normal by 1.0e-4f wasn't enough), I'm adding this "fail-safe" 
                // to make this kind of errors more visible and easily catchable in the future
                if (closest_hit_info.t < 0.01f && intersection_found)
                {
                    /*debug_set_final_color(render_data, x, y, res.x, Color(0.0f, 10000.0f, 0.0f));
                    return;*/
                }

                if (intersection_found)
                {
                    int material_index = render_data.material_indices[closest_hit_info.primitive_index];
                    RendererMaterial material = render_data.materials_buffer[material_index];
                    last_brdf_hit_type = material.brdf_type;

                    // --------------------------------------------------- //
                    // ----------------- Direct lighting ----------------- //
                    // --------------------------------------------------- //
                    Color light_sample_radiance = sample_light_sources(render_data, -ray.direction, closest_hit_info, material, random_number_generator);
                    //Color env_map_radiance = sample_environment_map(ray, closest_hit_info, material, random_number_generator);
                    Color env_map_radiance = Color(0.0f);

                    // --------------------------------------- //
                    // ---------- Indirect lighting ---------- //
                    // --------------------------------------- //

                    float brdf_pdf;
                    hiprtFloat3 bounce_direction;
                    Color brdf = brdf_dispatcher_sample(material, -ray.direction, closest_hit_info.shading_normal, closest_hit_info.geometric_normal, bounce_direction, brdf_pdf, random_number_generator);

                    if (last_brdf_hit_type == BRDF::SpecularFresnel)
                        // The fresnel blend coefficient is in the PDF
                        denoiser_blend *= brdf_pdf;

                    if (!denoiser_AOVs_set && last_brdf_hit_type != BRDF::SpecularFresnel)
                    {
                        denoiser_AOVs_set = true;

                        denoiser_albedo += material.diffuse * denoiser_blend;
                        denoiser_normal += closest_hit_info.shading_normal * denoiser_blend;
                    }

                    // Terminate ray if something went wrong according to the unforgivable laws of physic
                    // (sampling a direction below the surface for example)
                    if ((brdf.r == 0.0f && brdf.g == 0.0f && brdf.b == 0.0f) || brdf_pdf <= 0.0f)
                        break;

                    if (bounce == 0)
                        sample_color = sample_color + material.emission * throughput;
                    sample_color = sample_color + (light_sample_radiance + env_map_radiance) * throughput;

                    throughput *= brdf * abs(dot(bounce_direction, closest_hit_info.shading_normal)) / brdf_pdf;

                    hiprtFloat3 new_ray_origin = closest_hit_info.inter_point + closest_hit_info.shading_normal * 3.0e-3f;
                    ray.origin = new_ray_origin; // Updating the next ray origin
                    ray.direction = bounce_direction; // Updating the next ray direction

                    next_ray_state = RayState::BOUNCE;
                }
                else
                {
                    //if (bounce == 1 || last_brdf_hit_type == BRDF::SpecularFresnel)
                    {
                        //We're only getting the skysphere radiance for the first rays because the
                        //syksphere is importance sampled
                        // We're also getting the skysphere radiance for perfectly specular BRDF since those
                        // are not importance sampled

                        //Color skysphere_color = sample_environment_map_from_direction(ray.direction);
                        Color skysphere_color = Color(0.45f);

                        sample_color += skysphere_color * throughput;
                    }

                    next_ray_state = RayState::MISSED;
                }
            }
            else if (next_ray_state == RayState::MISSED)
                break;
        }

        // These 2 if() are basically anomally detectors
        // They will set pixels to very bright colors if somehow
        // weird samples are produced
        // This helps spot unrobustness in the renderer 
        //
        // - Pink : sample with negative color
        // - Yellow : NaN sample
        if (sample_color.r < 0 || sample_color.g < 0 || sample_color.b < 0)
        {
            debug_set_final_color(render_data, x, y, res.x, Color(1000000.0f, 0.0f, 1000000.0f));
            return;
        }
        else if (isnan(sample_color.r) || isnan(sample_color.g) || isnan(sample_color.b))
        {
            debug_set_final_color(render_data, x, y, res.x, Color(1000000.0f, 1000000.0f, 0.0f));
            return;
        }

        final_color += sample_color;
    }

    render_data.pixels[index] += final_color;
    
    // Handling denoiser's albedo and normals AOVs    
    // We don't need those when rendering at low resolution
    // hence why this is the else branch
    denoiser_albedo /= (float)render_data.m_render_settings.samples_per_frame;
    denoiser_normal /= (float)render_data.m_render_settings.samples_per_frame;
    render_data.denoiser_albedo[index] = (render_data.denoiser_albedo[index] * render_data.m_render_settings.frame_number + denoiser_albedo) / (render_data.m_render_settings.frame_number + 1.0f);

    hiprtFloat3 accumulated_normal = (render_data.denoiser_normals[index] * render_data.m_render_settings.frame_number + denoiser_normal) / (render_data.m_render_settings.frame_number + 1.0f);
    float normal_length = length(accumulated_normal);
    if (normal_length != 0.0f)
        render_data.denoiser_normals[index] = normalize(accumulated_normal);

    // Handling low resolution render
    // The framebuffer actually still is at full resolution, it's just that we cast
    // 1 ray every 4, 8 or 16 pixels (depending on the low resolution factor)
    // This means that we have "holes" in the rendered where rays will never be cast
    // this loop fills the wholes by copying the pixel that we rendered to its unrendered
    // neighbors
    if (render_data.m_render_settings.render_low_resolution)
    {
        // Copying the pixel we just rendered to the neighbors
        for (int _y = 0; _y < LOW_RESOLUTION_RENDER_DOWNSCALE; _y++)
        {
            for (int _x = 0; _x < LOW_RESOLUTION_RENDER_DOWNSCALE; _x++)
            {
                int _index = _y * res.x + _x + index;
                if (_y == 0 && _x == 0)
                    // This is ourselves
                    continue;
                else if (_index >= res.x * res.y)
                    // Outside of the framebuffer
                    return;
                else
                {
                    // Actually a valid pixel
                    render_data.pixels[_index] = render_data.pixels[index];

                    // Also handling the denoiser AOVs. Useful only when the user is moving the camera
                    // (and thus rendering at low resolution) while the denoiser's normals / albedo has
                    // been selected as the active viewport view
                    render_data.denoiser_albedo[_index] = render_data.denoiser_albedo[index];
                    render_data.denoiser_normals[_index] = render_data.denoiser_normals[index];
                }
            }
        }
    }
}