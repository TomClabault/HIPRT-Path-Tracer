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

__device__ HIPRTColor fresnel_schlick(HIPRTColor F0, float NoV)
{
    return F0 + (HIPRTColor(1.0f) - F0) * pow((1.0f - NoV), 5.0f);
}

__device__ float GGX_normal_distribution(float alpha, float NoH)
{
    //To avoid numerical instability when NoH basically == 1, i.e when the
    //material is a perfect mirror and the normal distribution function is a Dirac

    NoH = RT_MIN(NoH, 0.999999f);
    float alpha2 = alpha * alpha;
    float NoH2 = NoH * NoH;
    float b = (NoH2 * (alpha2 - 1.0f) + 1.0f);
    return alpha2 * M_1_PI / (b * b);
}

__device__ float G1_schlick_ggx(float k, float dot_prod)
{
    return dot_prod / (dot_prod * (1.0f - k) + k);
}

__device__ float GGX_smith_masking_shadowing(float roughness_squared, float NoV, float NoL)
{
    float k = roughness_squared / 2.0f;

    return G1_schlick_ggx(k, NoL) * G1_schlick_ggx(k, NoV);
}

__device__ float cook_torrance_brdf_pdf(const HIPRTRendererMaterial& material, const hiprtFloat3& view_direction, const hiprtFloat3& to_light_direction, const hiprtFloat3& surface_normal)
{
    hiprtFloat3 microfacet_normal = normalize(view_direction + to_light_direction);

    float alpha = material.roughness * material.roughness;

    float VoH = RT_MAX(0.0f, dot(view_direction, microfacet_normal));
    float NoH = RT_MAX(0.0f, dot(surface_normal, microfacet_normal));
    float D = GGX_normal_distribution(alpha, NoH);

    return D * NoH / (4.0f * VoH);
}

__device__ HIPRTColor cook_torrance_brdf(const HIPRTRendererMaterial& material, const hiprtFloat3& to_light_direction, const hiprtFloat3& view_direction, const hiprtFloat3& surface_normal)
{
    HIPRTColor brdf_color = HIPRTColor(0.0f, 0.0f, 0.0f);
    HIPRTColor base_color = material.diffuse;

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
        HIPRTColor F;
        float D, G;

        //F0 = 0.04 for dielectrics, 1.0 for metals (approximation)
        HIPRTColor F0 = HIPRTColor(0.04f * (1.0f - metalness)) + metalness * base_color;

        //GGX Distribution function
        F = fresnel_schlick(F0, VoH);
        D = GGX_normal_distribution(alpha, NoH);
        G = GGX_smith_masking_shadowing(alpha, NoV, NoL);

        HIPRTColor kD = HIPRTColor(1.0f - metalness); //Metals do not have a diffuse part
        kD = kD * (HIPRTColor(1.0f) - F);//Only the transmitted light is diffused

        HIPRTColor diffuse_part = kD * base_color / (float)M_PI;
        HIPRTColor specular_part = (F * D * G) / (4.0f * NoV * NoL);

        brdf_color = diffuse_part + specular_part;
    }

    return brdf_color;
}

__device__ HIPRTColor cook_torrance_brdf_importance_sample(const HIPRTRendererMaterial& material, const hiprtFloat3& view_direction, const hiprtFloat3& surface_normal, hiprtFloat3& output_direction, float& pdf, HIPRT_xorshift32_generator& random_number_generator)
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

    hiprtFloat3 microfacet_normal_local_space = hiprtFloat3(cos(phi) * sin_theta, sin(phi) * sin_theta, cos(theta));
    hiprtFloat3 microfacet_normal = rotate_vector_around_normal(surface_normal, microfacet_normal_local_space);
    if (dot(microfacet_normal, surface_normal) < 0.0f)
        //The microfacet normal that we sampled was under the surface, this can happen
        return HIPRTColor(0.0f);
    hiprtFloat3 to_light_direction = normalize(2.0f * dot(microfacet_normal, view_direction) * microfacet_normal - view_direction);
    hiprtFloat3 halfway_vector = microfacet_normal;
    output_direction = to_light_direction;

    HIPRTColor brdf_color = HIPRTColor(0.0f, 0.0f, 0.0f);
    HIPRTColor base_color = material.diffuse;

    float NoV = RT_MAX(0.0f, dot(surface_normal, view_direction));
    float NoL = RT_MAX(0.0f, dot(surface_normal, to_light_direction));
    float NoH = RT_MAX(0.0f, dot(surface_normal, halfway_vector));
    float VoH = RT_MAX(0.0f, dot(halfway_vector, view_direction));

    if (NoV > 0.0f && NoL > 0.0f && NoH > 0.0f)
    {
        /////////// Cook Torrance BRDF //////////
        HIPRTColor F;
        float D, G;


        //GGX Distribution function
        D = GGX_normal_distribution(alpha, NoH);

        //F0 = 0.04 for dielectrics, 1.0 for metals (approximation)
        HIPRTColor F0 = HIPRTColor(0.04f * (1.0f - metalness)) + metalness * base_color;
        F = fresnel_schlick(F0, VoH);
        G = GGX_smith_masking_shadowing(alpha, NoV, NoL);

        HIPRTColor kD = HIPRTColor(1.0f - metalness); //Metals do not have a diffuse part
        kD = kD * (HIPRTColor(1.0f) - F);//Only the transmitted light is diffused

        HIPRTColor diffuse_part = kD * base_color / (float)M_PI;
        HIPRTColor specular_part = (F * D * G) / (4.0f * NoV * NoL);

        pdf = D * NoH / (4.0f * VoH);

        brdf_color = diffuse_part + specular_part;
    }

    return brdf_color;
}

__device__ HIPRTColor smooth_glass_bsdf(const HIPRTRendererMaterial& material, hiprtFloat3& out_bounce_direction, const hiprtFloat3& ray_direction, hiprtFloat3& surface_normal, float eta_i, float eta_t, float& pdf, HIPRT_xorshift32_generator& random_generator)
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
            return HIPRTColor(1000000.0f, 0.0f, 1000000.0f); // Omega pink
        }

        out_bounce_direction = refract_direction;
        surface_normal = -surface_normal;
        pdf = 1.0f - fresnel_reflect;

        return HIPRTColor(1.0f - fresnel_reflect) * material.diffuse / dot(out_bounce_direction, surface_normal);
    }
}

__device__ HIPRTColor brdf_dispatcher_sample(const HIPRTRendererMaterial& material, hiprtFloat3& bounce_direction, const hiprtFloat3& ray_direction, hiprtFloat3& surface_normal, float& brdf_pdf, HIPRT_xorshift32_generator& random_number_generator)
{
    if (material.brdf_type == HIPRTBRDF::HIPRT_SpecularFresnel)
        return smooth_glass_bsdf(material, bounce_direction, ray_direction, surface_normal, 1.0f, material.ior, brdf_pdf, random_number_generator); //TODO relative IOR in the RayData rather than two incident and output ior values
    else if (material.brdf_type == HIPRTBRDF::HIPRT_CookTorrance)
    {
        return cook_torrance_brdf_importance_sample(material, -ray_direction, surface_normal, bounce_direction, brdf_pdf, random_number_generator);
        bounce_direction = hiprt_cosine_weighted_direction_around_normal(surface_normal, brdf_pdf, random_number_generator);
        return hiprt_lambertian_brdf(material, bounce_direction, -ray_direction, surface_normal);
    }

    return HIPRTColor(0.0f);
}

__device__ bool trace_ray(const HIPRTRenderData& render_data, hiprtRay ray, HIPRTHitInfo& hit_info)
{
    //TODO use global stack for good traversal performance
    hiprtGeomTraversalClosest tr(render_data.geom, ray);
    hiprtHit				  hit = tr.getNextHit();

    if (hit.hasHit())
    {
        hit_info.inter_point = ray.origin + hit.t * ray.direction;
        hit_info.primitive_index = hit.primID;
        // TODO hit.normal is in object space but we need world space normals. The line below assumes that all objects have
        // already been pretransformed in world space. This may not be true anymore with multiple level
        // acceleration structures
        /*hiprtFloat3 vertex_A = render_data.triangles_vertices[render_data.triangles_indices[hit_info.primitive_index * 3 + 0]];
        hiprtFloat3 vertex_B = render_data.triangles_vertices[render_data.triangles_indices[hit_info.primitive_index * 3 + 1]];
        hiprtFloat3 vertex_C = render_data.triangles_vertices[render_data.triangles_indices[hit_info.primitive_index * 3 + 2]];

        hiprtFloat3 AB = vertex_B - vertex_A;
        hiprtFloat3 AC = vertex_C - vertex_A;
        hit_info.normal_at_intersection = normalize(cross(AB, AC));*/

        int vertex_A_index = render_data.triangles_indices[hit_info.primitive_index * 3 + 0];
        if (render_data.normals_present[vertex_A_index])
        {
            // Smooth normal available for the triangle

            int vertex_B_index = render_data.triangles_indices[hit_info.primitive_index * 3 + 1];
            int vertex_C_index = render_data.triangles_indices[hit_info.primitive_index * 3 + 2];

            hiprtFloat3 smooth_normal = render_data.vertex_normals[vertex_B_index] * hit.uv.x 
                + render_data.vertex_normals[vertex_C_index] * hit.uv.y 
                + render_data.vertex_normals[vertex_A_index] * (1.0f - hit.uv.x - hit.uv.y);

            hit_info.normal_at_intersection = normalize(smooth_normal);
        }
        else
            hit_info.normal_at_intersection = normalize(hit.normal);

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

__device__ hiprtFloat3 sample_random_point_on_lights(const HIPRTRenderData& render_data, HIPRT_xorshift32_generator& random_number_generator, float& pdf, HIPRTLightSourceInformation& light_info)
{
    int random_index = random_number_generator() * render_data.emissive_triangles_count;
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

//// TODO rename xorshift32 generator without underscores for consistency
__device__ HIPRTColor sample_light_sources(HIPRTRenderData& render_data, const hiprtRay& ray, const HIPRTHitInfo& closest_hit_info, const HIPRTRendererMaterial& material, HIPRT_xorshift32_generator& random_number_generator)
{
    if (material.brdf_type == HIPRTBRDF::HIPRT_SpecularFresnel)
        // No sampling for perfectly specular materials
        return HIPRTColor(0.0f);

    HIPRTColor light_source_radiance_mis;
    if (render_data.emissive_triangles_count > 0)
    {
        float light_sample_pdf;
        HIPRTLightSourceInformation light_source_info;
        hiprtFloat3 random_light_point = sample_random_point_on_lights(render_data, random_number_generator, light_sample_pdf, light_source_info);

        hiprtFloat3 shadow_ray_origin = closest_hit_info.inter_point + closest_hit_info.normal_at_intersection * 1.0e-4f;
        hiprtFloat3 shadow_ray_direction = random_light_point - shadow_ray_origin;
        float distance_to_light = length(shadow_ray_direction);
        hiprtFloat3 shadow_ray_direction_normalized = normalize(shadow_ray_direction);

        hiprtRay shadow_ray;
        shadow_ray.origin = shadow_ray_origin;
        shadow_ray.direction = shadow_ray_direction_normalized;

        float dot_light_source = RT_MAX(dot(light_source_info.light_source_normal, -shadow_ray_direction_normalized), 0.0f);
        if (dot_light_source > 0.0f)
        {
            bool in_shadow = evaluate_shadow_ray(render_data, shadow_ray, distance_to_light);

            if (!in_shadow)
            {
                const HIPRTRendererMaterial& emissive_triangle_material = render_data.materials_buffer[render_data.material_indices[light_source_info.emissive_triangle_index]];

                light_sample_pdf *= distance_to_light * distance_to_light;
                light_sample_pdf /= dot_light_source;

                HIPRTColor brdf = cook_torrance_brdf(material, shadow_ray.direction, -ray.direction, closest_hit_info.normal_at_intersection);

                float cook_torrance_pdf = cook_torrance_brdf_pdf(material, -ray.direction, shadow_ray_direction_normalized, closest_hit_info.normal_at_intersection);
                if (cook_torrance_pdf != 0.0f)
                {
                    float mis_weight = power_heuristic(light_sample_pdf, cook_torrance_pdf);

                    HIPRTColor Li = emissive_triangle_material.emission;
                    float cosine_term = dot(closest_hit_info.normal_at_intersection, shadow_ray_direction_normalized);

                    light_source_radiance_mis = Li * cosine_term * brdf * mis_weight / light_sample_pdf;
                }
            }
        }
    }

    HIPRTColor brdf_radiance_mis;

    hiprtFloat3 sampled_brdf_direction;
    float direction_pdf;
    HIPRTColor brdf = cook_torrance_brdf_importance_sample(material, -ray.direction, closest_hit_info.normal_at_intersection, sampled_brdf_direction, direction_pdf, random_number_generator);
    if (brdf.r != 0.0f || brdf.g != 0.0f || brdf.b != 0.0f)
    {
        hiprtRay new_ray; 
        new_ray.origin = closest_hit_info.inter_point + closest_hit_info.normal_at_intersection * 1.0e-5f;
        new_ray.direction = sampled_brdf_direction;

        HIPRTHitInfo new_ray_hit_info;
        bool inter_found = trace_ray(render_data, new_ray, new_ray_hit_info);

        if (inter_found)
        {
            float cos_angle = RT_MAX(dot(new_ray_hit_info.normal_at_intersection, -sampled_brdf_direction), 0.0f);
            if (cos_angle > 0.0f)
            {
                int material_index = render_data.material_indices[new_ray_hit_info.primitive_index];
                HIPRTRendererMaterial material = render_data.materials_buffer[material_index];

                HIPRTColor emission = material.emission;
                if (emission.r > 0 || emission.g > 0 || emission.b > 0)
                {
                    float distance_squared = new_ray_hit_info.t * new_ray_hit_info.t;
                    float light_area = triangle_area(render_data, new_ray_hit_info.primitive_index);

                    float light_pdf = distance_squared / (light_area * cos_angle);

                    float mis_weight = power_heuristic(direction_pdf, light_pdf);
                    float cosine_term = dot(closest_hit_info.normal_at_intersection, sampled_brdf_direction);
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

__device__ void debug_set_final_color(const HIPRTRenderData& render_data, int x, int y, int res_x, HIPRTColor* pixels, HIPRTColor final_color)
{
    final_color.a = 0.0f;
    if (render_data.render_settings.frame_number == 0)
        pixels[y * res_x + x] = final_color;
    else
        pixels[y * res_x + x] = pixels[y * res_x + x] + final_color;
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
    // THe + 1 are used to avoid zeros
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
                bool intersection_found = trace_ray(render_data, ray, closest_hit_info);

                if (intersection_found)
                {
                    /*HIPRTColor debug_color(closest_hit_info.normal_at_intersection);
                    debug_set_final_color(render_data, x, y, res.x, pixels, HIPRTColor(abs(closest_hit_info.normal_at_intersection)));
                    return;*/



                    int material_index = render_data.material_indices[closest_hit_info.primitive_index];
                    HIPRTRendererMaterial material = render_data.materials_buffer[material_index];

                    last_brdf_hit_type = material.brdf_type;

                    // --------------------------------------------------- //
                    // ----------------- Direct lighting ----------------- //
                    // --------------------------------------------------- //
                    //TODO area sampling triangles
                    HIPRTColor light_sample_radiance = sample_light_sources(render_data, ray, closest_hit_info, material, random_number_generator);

                    //HIPRTColor env_map_radiance = sample_environment_map(ray, closest_hit_info, material, random_number_generator);

                    HIPRTColor env_map_radiance = HIPRTColor(0.0f);

                    // --------------------------------------- //
                    // ---------- Indirect lighting ---------- //
                    // --------------------------------------- //

                    float brdf_pdf;
                    hiprtFloat3 bounce_direction;
                    HIPRTColor brdf = brdf_dispatcher_sample(material, bounce_direction, ray.direction, closest_hit_info.normal_at_intersection, brdf_pdf, random_number_generator); //TODO relative IOR in the RayData rather than two incident and output ior values

                    if (bounce == 0)
                        sample_color = sample_color + material.emission * throughput;
                    sample_color = sample_color + (light_sample_radiance + env_map_radiance) * throughput;

                    if ((brdf.r == 0.0f && brdf.g == 0.0f && brdf.b == 0.0f) || brdf_pdf < 1.0e-8f || isinf(brdf_pdf))
                    {
                        next_ray_state = HIPRTRayState::HIPRT_TERMINATED;

                        break;
                    }

                    throughput = throughput * brdf * RT_MAX(0.0f, dot(bounce_direction, closest_hit_info.normal_at_intersection)) / brdf_pdf;

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

                        //HIPRTColor skysphere_color = sample_environment_map_from_direction(ray.direction);
                        HIPRTColor skysphere_color = HIPRTColor(1.0f);

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

    final_color = final_color / render_data.render_settings.samples_per_frame;
    final_color.a = 0.0f;
    if (render_data.render_settings.frame_number == 0)
        pixels[y * res.x + x] = final_color;
    else
        pixels[y * res.x + x] = (pixels[y * res.x + x] * render_data.render_settings.frame_number + final_color) / (float)(render_data.render_settings.frame_number + 1);
}