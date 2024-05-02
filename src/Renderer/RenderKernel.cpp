///*
// * Copyright 2024 Tom Clabault. GNU GPL3 license.
// * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
// */
//
//#include "HostDeviceCommon/HitInfo.h"
//#include "Device/includes/Disney.h"
//#include "Device/includes/ONB.h"
//#include "Device/includes/OrenNayar.h"
//#include "Device/includes/Sampling.h"
//#include "Renderer/RenderKernel.h"
//#include "Renderer/Triangle.h"
//
//#define DEBUG_PIXEL 0
//#define DEBUG_EXACT_COORDINATE 0
//#define DEBUG_PIXEL_X 105
//#define DEBUG_PIXEL_Y 508
//
//float3 point_mat4x4(const glm::mat4x4& mat, const float3& p)
//{
//    glm::vec4 pt = mat * (glm::vec4(p.x, p.y, p.z, 1.0f));
//    return float3(pt.x / pt.w, pt.y / pt.w, pt.z / pt.w);
//}
//
//float3 vec4_mat4x4(const glm::mat4x4& mat, const float3& v)
//{
//    glm::vec4 vt = mat * (glm::vec4(v.x, v.y, v.z, 0.0f));
//    return float3(vt.x / vt.w, vt.y / vt.w, vt.z / vt.w);
//}
//
////Ray get_camera_ray(float x, float y)
////{
////    float x_ndc_space = x / m_framebuffer_width * 2 - 1;
////    float y_ndc_space = y / m_framebuffer_height * 2 - 1;
////
////    float3 ray_origin_view_space(0.0f, 0.0f, 0.0f);
////    float3 ray_origin = point_mat4x4(glm::inverse(m_camera.get_view_matrix()), ray_origin_view_space);
////
////    float3 ray_point_direction_ndc_space = float3(x_ndc_space, y_ndc_space, 1.0f);
////    float3 ray_point_direction_view_space = point_mat4x4(glm::inverse(m_camera.projection_matrix), ray_point_direction_ndc_space);
////    float3 ray_point_direction_world_space = point_mat4x4(glm::inverse(m_camera.get_view_matrix()), ray_point_direction_view_space);
////
////    float3 ray_direction = hippt::normalize(ray_point_direction_world_space - ray_origin);
////
////    hippt::Ray ray;
////    ray.origin = origin;
////    ray.direction = ray_direction;
////    return ray;
////}
//
//void debug_set_final_color(int x, int y, ColorRGB final_color)
//{
//    m_frame_buffer[x + y * m_framebuffer_width] = final_color;
//}
//
//unsigned int wang_hash(unsigned int seed)
//{
//    seed = (seed ^ 61) ^ (seed >> 16);
//    seed *= 9;
//    seed = seed ^ (seed >> 4);
//    seed *= 0x27d4eb2d;
//    seed = seed ^ (seed >> 15);
//    return seed;
//}
//
//void ray_trace_pixel(int x, int y)
//{
//    int index = x + y * m_framebuffer_width;
//
//    int pixel_sample_count = m_pixels_sample_count[index];
//    if (pixel_sample_count > 128)
//    {
//        // Waiting for at least 16 samples to enable adaptative sampling
//        float luminance = m_frame_buffer[index].luminance();
//        float average_luminance = luminance / (pixel_sample_count + 1);
//        float squared_luminance = m_pixels_squared_luminance[index];
//
//        float pixel_variance = (squared_luminance - luminance * average_luminance) / (pixel_sample_count);
//
//        bool pixel_needs_sampling = 1.96f * sqrt(pixel_variance) / sqrt(pixel_sample_count + 1) > 0.02f * average_luminance;
//        if (!pixel_needs_sampling)
//            return;
//    }
//
//    Xorshift32Generator random_number_generator(wang_hash(((x + y * m_framebuffer_width) + 1) * (m_render_samples + 1)));
//
//    ColorRGB final_color = ColorRGB(0.0f, 0.0f, 0.0f);
//    for (int sample = 0; sample < m_render_samples; sample++)
//    {
//        //Jittered around the center
//        float x_jittered = (x + 0.5f) + random_number_generator() - 1.0f;
//        float y_jittered = (y + 0.5f) + random_number_generator() - 1.0f;
//
//        //TODO area sampling triangles
//        Ray ray = get_camera_ray(x_jittered, y_jittered);
//
//        ColorRGB throughput = ColorRGB(1.0f, 1.0f, 1.0f);
//        ColorRGB sample_color = ColorRGB(0.0f, 0.0f, 0.0f);
//        RayState next_ray_state = RayState::BOUNCE;
//        BRDF last_brdf_hit_type = BRDF::Uninitialized;
//
//        for (int bounce = 0; bounce < m_max_bounces; bounce++)
//        {
//            if (next_ray_state == RayState::BOUNCE)
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
//                    // For the BRDF calculations, bounces, ... to be correct, we need the normal to be in the same hemisphere as
//                    // the view direction. One thing that can go wrong is when we have an emissive triangle (typical area light)
//                    // and a ray hits the back of the triangle. The normal will not be facing the view direction in this
//                    // case and this will cause issues later in the BRDF.
//                    // Because we want to allow backfacing emissive geometry (making the emissive geometry double sided
//                    // and emitting light in both directions of the surface), we're negating the normal to make
//                    // it face the view direction (but only for emissive geometry)
//                    if (material.is_emissive() && hippt::dot(-ray.direction, closest_hit_info.geometric_normal) < 0)
//                    {
//                        closest_hit_info.geometric_normal = -closest_hit_info.geometric_normal;
//                        closest_hit_info.shading_normal = -closest_hit_info.shading_normal;
//                    }
//
//                    // --------------------------------------------------- //
//                    // ----------------- Direct lighting ----------------- //
//                    // --------------------------------------------------- //
//                    ColorRGB light_sample_radiance = sample_light_sources(-ray.direction, closest_hit_info, material, random_number_generator);
//                    ColorRGB env_map_radiance = ColorRGB(0.0f);// sample_environment_map(ray, closest_hit_info, material, random_number_generator);
//
//                    // --------------------------------------- //
//                    // ---------- Indirect lighting ---------- //
//                    // --------------------------------------- //
//
//                    float brdf_pdf;
//                    float3 bounce_direction;
//                    ColorRGB brdf = brdf_dispatcher_sample(material, -ray.direction, closest_hit_info.shading_normal, closest_hit_info.geometric_normal, bounce_direction, brdf_pdf, random_number_generator);
//                    
//                    if (bounce == 0)
//                        sample_color += material.emission;
//                    sample_color += (light_sample_radiance + env_map_radiance) * throughput;
//
//                    if ((brdf.r == 0.0f && brdf.g == 0.0f && brdf.b == 0.0f) || brdf_pdf <= 0.0f)
//                        break;
//
//                    throughput *= brdf * std::abs(hippt::dot(bounce_direction, closest_hit_info.shading_normal)) / brdf_pdf;
//
//                    int outside_surface = hippt::dot(bounce_direction, closest_hit_info.shading_normal) < 0 ? -1.0f : 1.0;
//                    float3 new_ray_origin = closest_hit_info.inter_point + closest_hit_info.shading_normal * 3.0e-3f * outside_surface;
//                    ray = Ray(new_ray_origin, bounce_direction);
//                    next_ray_state = RayState::BOUNCE;
//                }
//                else
//                    next_ray_state = RayState::MISSED;
//            }
//            else if (next_ray_state == RayState::MISSED)
//            {
//                //if (bounce == 1 || last_brdf_hit_type == BRDF::SpecularFresnel)
//                {
//                    //We're only getting the skysphere radiance for the first rays because the
//                    //syksphere is importance sampled
//                    // We're also getting the skysphere radiance for perfectly specular BRDF since those
//                    // are not importance sampled
//
//                    ColorRGB skysphere_color = ColorRGB(1.0f);
//                    //ColorRGB skysphere_color = sample_environment_map_from_direction(ray.direction);
//
//                    sample_color += skysphere_color * throughput;
//                }
//
//                break;
//            }
//        }
//
//        if (sample_color.r < 0 || sample_color.g < 0 || sample_color.b < 0)
//        {
//            std::cerr << "Sample color < 0" << std::endl;
//            std::cerr << "Exact_X, Exact_Y, Sample: " << x << ", " << y << ", " << sample << std::endl;
//            sample_color = ColorRGB(1000000.0f, 0.0f, 1000000.0f);
//        }
//        else if (std::isnan(sample_color.r) || std::isnan(sample_color.g) || std::isnan(sample_color.b))
//        {
//            std::cerr << "Sample color NaN" << std::endl;
//            std::cerr << "Exact_X, Exact_Y, Sample: " << x << ", " << y << ", " << sample << std::endl;
//            sample_color = ColorRGB(1000000.0f, 1000000.0f, 0.0f);
//        }
//
//        final_color += sample_color;
//    }
//
//    final_color /= m_render_samples;
//    m_frame_buffer[index] += final_color;
//
//    const float gamma = 2.2f;
//    const float exposure = 1.0f;
//    ColorRGB hdrColor = m_frame_buffer[index];
//
//    //Exposure tone mapping
//    ColorRGB tone_mapped = ColorRGB(1.0f, 1.0f, 1.0f) - exp(-hdrColor * exposure);
//    // Gamma correction
//    ColorRGB gamma_corrected = pow(tone_mapped, 1.0f / gamma);
//
//    m_frame_buffer[index] = gamma_corrected;
//}
//
//#include <atomic>
//#include <omp.h>
//
//void render()
//{
//    std::atomic<int> lines_completed = 0;
//#if DEBUG_PIXEL
//#if DEBUG_EXACT_COORDINATE
//    for (int y = DEBUG_PIXEL_Y; y < m_frame_buffer.height; y++)
//    {
//        for (int x = DEBUG_PIXEL_X; x < m_frame_buffer.width; x++)
//#else
//    for (int y = m_frame_buffer.height - DEBUG_PIXEL_Y - 1; y < m_frame_buffer.height; y++)
//    {
//        for (int x = DEBUG_PIXEL_X; x < m_frame_buffer.width; x++)
//#endif
//#else
//#pragma omp parallel for schedule(dynamic)
//    for (int y = 0; y < m_framebuffer_height; y++)
//    {
//        for (int x = 0; x < m_framebuffer_width; x++)
//#endif
//            ray_trace_pixel(x, y);
//
//        lines_completed++;
//
//        if (omp_get_thread_num() == 0)
//            if (lines_completed % (m_framebuffer_height / 25))
//                std::cout << lines_completed / (float)m_framebuffer_height * 100 << "%" << std::endl;
//    }
//}
//
//ColorRGB brdf_dispatcher_eval(const RendererMaterial& material, const float3& view_direction, const float3& shading_normal, const float3& to_light_direction, float& pdf)
//{
//    pdf = 0.0f;
//    if (material.brdf_type == BRDF::Disney)
//        return disney_eval(material, view_direction, shading_normal, to_light_direction, pdf);
//
//    return ColorRGB(0.0f);
//}
//
//ColorRGB brdf_dispatcher_sample(const RendererMaterial& material, const float3& view_direction, const float3& shading_normal, const float3& geometric_normal, float3& bounce_direction, float& brdf_pdf, Xorshift32Generator& random_number_generator)
//{
//    return disney_sample(material, view_direction, shading_normal, geometric_normal, bounce_direction, brdf_pdf, random_number_generator);
//}
//
//bool intersect_scene(const Ray& ray, HitInfo& closest_hit_info)
//{
//    closest_hit_info.t = -1.0f;
//
//    for (int i = 0; i < m_triangle_buffer.size(); i++)
//    {
//        const Triangle& triangle = m_triangle_buffer[i];
//
//        HitInfo hit_info;
//        if(triangle.intersect(ray, hit_info))
//        {
//            if (hit_info.t < closest_hit_info.t || closest_hit_info.t == -1.0f)
//            {
//                closest_hit_info = hit_info;
//                closest_hit_info.primitive_index = i;
//            }
//        }
//    }
//
//    for (int i = 0; i < m_sphere_buffer.size(); i++)
//    {
//        const Sphere& sphere = m_sphere_buffer[i];
//
//        HitInfo hit_info;
//        if (sphere.intersect(ray, hit_info))
//            if (hit_info.t < closest_hit_info.t || closest_hit_info.t == -1.0f)
//                closest_hit_info = hit_info;
//    }
//
//    return closest_hit_info.t > 0.0f;
//}
//
//inline bool intersect_scene_bvh(const Ray& ray, HitInfo& closest_hit_info)
//{
//    closest_hit_info.t = -1.0f;
//
//    m_bvh.intersect(ray, closest_hit_info);
//    if (closest_hit_info.t > 0.0f)
//    {
//        // Computing smooth normal
//        int vertex_A_index = m_triangle_indices[closest_hit_info.primitive_index * 3 + 0];
//        if (m_normals_present[vertex_A_index])
//        {
//            // Smooth normal available for the triangle
//
//            int vertex_B_index = m_triangle_indices[closest_hit_info.primitive_index * 3 + 1];
//            int vertex_C_index = m_triangle_indices[closest_hit_info.primitive_index * 3 + 2];
//
//            float3 smooth_normal = m_vertex_normals[vertex_B_index] * closest_hit_info.u
//                + m_vertex_normals[vertex_C_index] * closest_hit_info.v
//                + m_vertex_normals[vertex_A_index] * (1.0f - closest_hit_info.u - closest_hit_info.v);
//
//            closest_hit_info.shading_normal = hippt::normalize(smooth_normal);
//        }
//    }
//
//    for (int i = 0; i < m_sphere_buffer.size(); i++)
//    {
//        const Sphere& sphere = m_sphere_buffer[i];
//
//        HitInfo hit_info;
//        if (sphere.intersect(ray, hit_info))
//            if (hit_info.t < closest_hit_info.t || closest_hit_info.t == -1.0f)
//                closest_hit_info = hit_info;
//    }
//
//
//    return closest_hit_info.t > 0.0f;
//}
//
//inline bool INTERSECT_SCENE(const Ray& ray, HitInfo& hit_info)
//{
//    return intersect_scene_bvh(ray, hit_info);
//}
//
//float power_heuristic(float pdf_a, float pdf_b)
//{
//    float pdf_a_squared = pdf_a * pdf_a;
//
//    return pdf_a_squared / (pdf_a_squared + pdf_b * pdf_b);
//}
//
//ColorRGB sample_environment_map_from_direction(const float3& direction)
//{
//    float u, v;
//    u = 0.5f + std::atan2(direction.z, direction.x) / (2.0f * (float)M_PI);
//    v = 0.5f + std::asin(direction.y) / (float)M_PI;
//
//    int x = hippt::max(hippt::min((int)(u * m_environment_map.width), m_environment_map.width - 1), 0);
//    int y = hippt::max(hippt::min((int)(v * m_environment_map.height), m_environment_map.height - 1), 0);
//
//    return m_environment_map[y * m_environment_map.width + x];
//}
//
//void env_map_cdf_search(float value, int& x, int& y)
//{
//    //First searching a line to sample
//    int lower = 0;
//    int upper = m_environment_map.height - 1;
//
//    int x_index = m_environment_map.width - 1;
//    while (lower < upper)
//    {
//        int y_index = (lower + upper) / 2;
//        int env_map_index = y_index * m_environment_map.width + x_index;
//
//        if (value < m_environment_map.cdf()[env_map_index])
//            upper = y_index;
//        else
//            lower = y_index + 1;
//    }
//    y = hippt::max(hippt::min(lower, m_environment_map.height), 0);
//
//    //Then sampling the line itself
//    lower = 0;
//    upper = m_environment_map.width - 1;
//
//    int y_index = y;
//    while (lower < upper)
//    {
//        int x_index = (lower + upper) / 2;
//        int env_map_index = y_index * m_environment_map.width + x_index;
//
//        if (value < m_environment_map.cdf()[env_map_index])
//            upper = x_index;
//        else
//            lower = x_index + 1;
//    }
//    x = hippt::max(hippt::min(lower, m_environment_map.width), 0);
//}
//
//ColorRGB sample_environment_map(const Ray& ray, HitInfo& closest_hit_info, const RendererMaterial& material, Xorshift32Generator& random_number_generator)
//{
//    if (material.brdf_type == BRDF::SpecularFresnel)
//        // No sampling for perfectly specular materials
//        return ColorRGB(0.0f);
//
//    const std::vector<float>& cdf = m_environment_map.cdf();
//
//    int x, y;
//    float env_map_total_sum = cdf[cdf.size() - 1];
//    env_map_cdf_search(random_number_generator() * env_map_total_sum, x, y);
//
//    float u = (float)x / m_environment_map.width;
//    float v = (float)y / m_environment_map.height;
//    float phi = u * 2.0f * M_PI;
//    // Clamping to avoid theta = 0 which would imply a skysphere direction straight up
//    // which leads to a pdf of infinity since it is a singularity
//    float theta = hippt::max(1.0e-5f, v * (float)M_PI);
//
//    ColorRGB env_sample;
//    float sin_theta = std::sin(theta);
//    float cos_theta = std::cos(theta);
//
//    // Convert to cartesian coordinates
//    float3 sampled_direction = float3(-sin_theta * cos(phi), -cos_theta, -sin_theta * sin(phi));
//
//    float cosine_term = hippt::dot(closest_hit_info.shading_normal, sampled_direction);
//    if  (cosine_term > 0.0f)
//    {
//        HitInfo trash;
//        if (!INTERSECT_SCENE(Ray(closest_hit_info.inter_point + closest_hit_info.shading_normal * 1.0e-4f, sampled_direction), trash))
//        {
//            float env_map_pdf = m_environment_map.luminance_of_pixel(x, y) / env_map_total_sum;
//            env_map_pdf = (env_map_pdf * m_environment_map.width * m_environment_map.height) / (2.0f * M_PI * M_PI * sin_theta);
//
//            ColorRGB env_map_radiance = m_environment_map[y * m_environment_map.width + x];
//            float pdf;
//            ColorRGB brdf = brdf_dispatcher_eval(material, -ray.direction, closest_hit_info.shading_normal, sampled_direction, pdf);
//
//            float mis_weight = power_heuristic(env_map_pdf, pdf);
//            env_sample = brdf * cosine_term * mis_weight * env_map_radiance / env_map_pdf;
//        }
//    }
//
//    float brdf_sample_pdf;
//    float3 brdf_sampled_dir;
//    ColorRGB brdf_imp_sampling = brdf_dispatcher_sample(material, -ray.direction, closest_hit_info.shading_normal, closest_hit_info.geometric_normal, brdf_sampled_dir, brdf_sample_pdf, random_number_generator);
//
//    cosine_term = hippt::max(hippt::dot(closest_hit_info.shading_normal, brdf_sampled_dir), 0.0f);
//    ColorRGB brdf_sample;
//    if (brdf_sample_pdf != 0.0f && cosine_term > 0.0f)
//    {
//        HitInfo trash;
//        if (!INTERSECT_SCENE(Ray(closest_hit_info.inter_point + closest_hit_info.shading_normal * 1.0e-5f, brdf_sampled_dir), trash))
//        {
//            ColorRGB skysphere_color = sample_environment_map_from_direction(brdf_sampled_dir);
//            float theta_brdf_dir = std::acos(brdf_sampled_dir.z);
//            float sin_theta_bdrf_dir = std::sin(theta_brdf_dir);
//            float env_map_pdf = skysphere_color.luminance() / env_map_total_sum;
//
//            env_map_pdf *= m_environment_map.width * m_environment_map.height;
//            env_map_pdf /= (2.0f * M_PI * M_PI * sin_theta_bdrf_dir);
//
//            float mis_weight = power_heuristic(brdf_sample_pdf, env_map_pdf);
//            brdf_sample = skysphere_color * mis_weight * cosine_term * brdf_imp_sampling / brdf_sample_pdf;
//        }
//    }
//
//    return brdf_sample + env_sample;
//}
//
//ColorRGB sample_light_sources(const float3& view_direction, const HitInfo& closest_hit_info, const RendererMaterial& material, Xorshift32Generator& random_number_generator)
//{
//    if (m_emissive_triangle_indices_buffer.size() == 0)
//        // No emmisive geometry in the scene to sample
//        return ColorRGB(0.0f);
//
//    if (material.emission.r != 0.0f || material.emission.g != 0.0f || material.emission.b != 0.0f)
//        // We're not sampling direct lighting if we're already on an
//        // emissive surface
//        return ColorRGB(0.0f);
//
//    if (hippt::dot(view_direction, closest_hit_info.geometric_normal) < 0.0f)
//        // We're not direct sampling if we're inside a surface
//        // 
//        // Note that we're also taking the geometric normal into account here and not only the 
//        // shading normal because we want to make sure we're actually inside a surface and not just
//        // inside a black fringe cause by smooth normals with microfacet BRDFs
//        // There's a slightly more thorough explanation of what we're doing with the dot products here
//        // in the disney brdf sampling method, in the glass lobe part
//        return ColorRGB(0.0f);
//
//    ColorRGB light_source_radiance_mis;
//    float light_sample_pdf;
//    LightSourceInformation light_source_info;
//    float3 random_light_point = sample_random_point_on_lights(random_number_generator, light_sample_pdf, light_source_info);
//
//    float3 shadow_ray_origin = closest_hit_info.inter_point + closest_hit_info.shading_normal * 1.0e-4f;
//    float3 shadow_ray_direction = random_light_point - shadow_ray_origin;
//    float distance_to_light = length(shadow_ray_direction);
//    float3 shadow_ray_direction_normalized = shadow_ray_direction / distance_to_light;
//
//    Ray shadow_ray(shadow_ray_origin, shadow_ray_direction_normalized);
//
//    float dot_light_source = std::abs(hippt::dot(light_source_info.light_source_normal, -shadow_ray.direction));
//    if (dot_light_source > 0.0f)
//    {
//        bool in_shadow = evaluate_shadow_ray(shadow_ray, distance_to_light);
//
//        if (!in_shadow && hippt::dot(closest_hit_info.shading_normal, shadow_ray_direction_normalized) > 0)
//        {
//            const RendererMaterial& emissive_triangle_material = m_materials_buffer[m_materials_indices_buffer[light_source_info.emissive_triangle_index]];
//
//            light_sample_pdf *= distance_to_light * distance_to_light;
//            light_sample_pdf /= dot_light_source;
//
//            float pdf;
//            ColorRGB brdf = brdf_dispatcher_eval(material, view_direction, closest_hit_info.shading_normal, shadow_ray.direction, pdf);
//            if (pdf != 0.0f)
//            {
//                float mis_weight = power_heuristic(light_sample_pdf, pdf);
//
//                ColorRGB Li = emissive_triangle_material.emission;
//                float cosine_term = hippt::max(hippt::dot(closest_hit_info.shading_normal, shadow_ray.direction), 0.0f);
//
//                light_source_radiance_mis = Li * cosine_term * brdf * mis_weight / light_sample_pdf;
//            }
//        }
//    }
//
//
//    ColorRGB brdf_radiance_mis;
//
//    float3 sampled_brdf_direction;
//    float direction_pdf;
//    ColorRGB brdf = brdf_dispatcher_sample(material, view_direction, closest_hit_info.shading_normal, closest_hit_info.geometric_normal, sampled_brdf_direction, direction_pdf, random_number_generator);
//    if (direction_pdf > 0)
//    {
//        Ray new_ray = Ray(closest_hit_info.inter_point + closest_hit_info.shading_normal * 1.0e-4f, sampled_brdf_direction);
//        HitInfo new_ray_hit_info;
//        bool inter_found = INTERSECT_SCENE(new_ray, new_ray_hit_info);
//
//        if (inter_found)
//        {
//            float cos_angle = hippt::max(hippt::dot(new_ray_hit_info.shading_normal, -sampled_brdf_direction), 0.0f);
//            if (cos_angle > 0.0f)
//            {
//                int material_index = m_materials_indices_buffer[new_ray_hit_info.primitive_index];
//                RendererMaterial material = m_materials_buffer[material_index];
//
//                ColorRGB emission = material.emission;
//                if (emission.r > 0 || emission.g > 0 || emission.b > 0)
//                {
//                    float distance_squared = new_ray_hit_info.t * new_ray_hit_info.t;
//                    float light_area = m_triangle_buffer[new_ray_hit_info.primitive_index].area();
//
//                    float light_pdf = distance_squared / (light_area * cos_angle);
//
//                    float mis_weight = power_heuristic(direction_pdf, light_pdf);
//                    float cosine_term = hippt::max(0.0f, hippt::dot(closest_hit_info.shading_normal, sampled_brdf_direction));
//
//                    brdf_radiance_mis = brdf * cosine_term * emission * mis_weight / direction_pdf;
//                }
//            }
//        }
//    }
//
//    return light_source_radiance_mis + brdf_radiance_mis;
//}
//
//inline float3 sample_random_point_on_lights(Xorshift32Generator& random_number_generator, float& pdf, LightSourceInformation& light_info)
//{
//    light_info.emissive_triangle_index = random_number_generator.random_index(m_emissive_triangle_indices_buffer.size());
//    light_info.emissive_triangle_index = m_emissive_triangle_indices_buffer[light_info.emissive_triangle_index];
//    Triangle random_emissive_triangle = m_triangle_buffer[light_info.emissive_triangle_index];
//
//    float rand_1 = random_number_generator();
//    float rand_2 = random_number_generator();
//
//    float sqrt_r1 = std::sqrt(rand_1);
//    float u = 1.0f - sqrt_r1;
//    float v = (1.0f - rand_2) * sqrt_r1;
//
//    float3 AB = random_emissive_triangle.m_b - random_emissive_triangle.m_a;
//    float3 AC = random_emissive_triangle.m_c - random_emissive_triangle.m_a;
//
//    float3 random_point_on_triangle = random_emissive_triangle.m_a + AB * u + AC * v;
//
//    float3 normal = hippt::cross(AB, AC);
//    float length_normal = length(normal);
//    light_info.light_source_normal = normal / length_normal; // Normalized
//    float triangle_area = length_normal * 0.5f;
//    float nb_emissive_triangles = m_emissive_triangle_indices_buffer.size();
//
//    pdf = 1.0f / (nb_emissive_triangles * triangle_area);
//
//    return random_point_on_triangle;
//}
//
//bool evaluate_shadow_ray(const Ray& ray, float t_max)
//{
//    HitInfo hit_info;
//    bool inter_found = INTERSECT_SCENE(ray, hit_info);
//
//    if (inter_found)
//    {
//        if (hit_info.t + 1.0e-4f < t_max)
//            // There is something in between the light and the origin of the ray
//            return true;
//        else
//            return false;
//    }
//
//    return false;
//}
