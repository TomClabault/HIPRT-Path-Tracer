#include "render_kernel.h"

#include "triangle.h"

void branchlessONB(const Vector& n, Vector& b1, Vector& b2)
{
    float sign = std::copysign(1.0f, n.z);
    const float a = -1.0f / (sign + n.z);
    const float b = n.x * n.y * a;
    b1 = Vector(1.0f + sign * n.x * n.x * a, sign * b, -sign * n.x);
    b2 = Vector(b, sign + n.y * n.y * a, -n.y);
}

Vector RenderKernel::rotate_vector_around_normal(const Vector& normal, const Vector& random_dir_local_space) const
{
    Vector tangent, bitangent;
    branchlessONB(normal, tangent, bitangent);

    //Transforming from the random_direction in its local space to the space around the normal
    //given in parameter (the space with the given normal as the Z up vector)
    return random_dir_local_space.x * tangent + random_dir_local_space.y * bitangent + random_dir_local_space.z * normal;
}

Vector RenderKernel::uniform_direction_around_normal(const Vector& normal, float& pdf, xorshift32_generator& random_number_generator) const
{
    float rand_1 = random_number_generator();
    float rand_2 = random_number_generator();

    float phi = 2.0f * (float)M_PI * rand_1;
    float root = std::sqrt(1.0f - rand_2 * rand_2);

    pdf = 1.0f / (2.0f * (float)M_PI);

    //Generating a random direction in a local space with Z as the Up vector
    Vector random_dir_local_space(std::cos(phi) * root, std::sin(phi) * root, rand_2);
    return rotate_vector_around_normal(normal, random_dir_local_space);
}

Vector RenderKernel::cosine_weighted_direction_around_normal(const Vector& normal, float& pdf, xorshift32_generator& random_number_generator) const
{
    float rand_1 = random_number_generator();
    float rand_2 = random_number_generator();

    float sqrt_rand_2 = std::sqrt(rand_2);
    float phi = 2.0f * (float)M_PI * rand_1;
    float cos_theta = sqrt_rand_2;
    float sin_theta = std::sqrt(std::max(0.0f, 1.0f - cos_theta * cos_theta));

    pdf = sqrt_rand_2 / (float)M_PI;

    //Generating a random direction in a local space with Z as the Up vector
    Vector random_dir_local_space = Vector(std::cos(phi) * sin_theta, std::sin(phi) * sin_theta, sqrt_rand_2);
    return rotate_vector_around_normal(normal, random_dir_local_space);
}

Ray RenderKernel::get_camera_ray(float x, float y) const
{
    float x_ndc_space = x / m_width * 2 - 1;
    x_ndc_space *= (float)m_width / m_height; //Aspect ratio
    float y_ndc_space = y / m_height * 2 - 1;


    Point ray_origin_view_space(0.0f, 0.0f, 0.0f);
    Point ray_origin = m_camera.view_matrix(ray_origin_view_space);

    Point ray_point_direction_ndc_space = Point(x_ndc_space, y_ndc_space, m_camera.fov_dist);
    Point ray_point_direction_world_space = m_camera.view_matrix(ray_point_direction_ndc_space);

    Vector ray_direction = normalize(ray_point_direction_world_space - ray_origin);
    Ray ray(ray_origin, ray_direction);

    return ray;
}

void RenderKernel::ray_trace_pixel(int x, int y) const
{
    xorshift32_generator random_number_generator(31 + x * y * m_render_samples);
    //Generating some numbers to make sure the generators of each thread spread apart
    //If not doing this, the generator shows clear artifacts until it has generated
    //a few numbers
    for (int i = 0; i < 10; i++)
        random_number_generator();

    Color final_color = Color(0.0f, 0.0f, 0.0f);
    for (int sample = 0; sample < m_render_samples; sample++)
    {
        //Jittered around the center
        float x_jittered = (x + 0.5f) + random_number_generator() - 1.0f;
        float y_jittered = (y + 0.5f) + random_number_generator() - 1.0f;

        //TODO area sampling triangles
        Ray ray = get_camera_ray(x_jittered, y_jittered);

        Color throughput = Color(1.0f, 1.0f, 1.0f);
        Color sample_color = Color(0.0f, 0.0f, 0.0f);
        RayState next_ray_state = RayState::BOUNCE;

        for (int bounce = 0; bounce < m_max_bounces; bounce++)
        {
            if (next_ray_state == BOUNCE)
            {
                HitInfo closest_hit_info;
                bool intersection_found = INTERSECT_SCENE(ray, closest_hit_info);

                if (intersection_found)
                {
                    int material_index = m_materials_indices_buffer[closest_hit_info.primitive_index];
                    SimpleMaterial material = m_materials_buffer_access[material_index];

                    // --------------------------------------------------- //
                    // ----------------- Direct lighting ----------------- //
                    // --------------------------------------------------- //
                    Color light_sample_radiance = sample_light_sources(ray, closest_hit_info, material, random_number_generator);
                    Color env_map_radiance = sample_environment_map(ray, closest_hit_info, material, random_number_generator);
                    //Color env_map_radiance = Color(0.0f);

                    // --------------------------------------- //
                    // ---------- Indirect lighting ---------- //
                    // --------------------------------------- //

                    float brdf_pdf;
                    Vector random_bounce_direction;// = cosine_weighted_direction_around_normal(closest_hit_info.normal_at_intersection, brdf_pdf, random_number_generator);
                    Color brdf = cook_torrance_brdf_importance_sample(material, -ray.direction, closest_hit_info.normal_at_intersection, random_bounce_direction, brdf_pdf, random_number_generator);
                    //Color brdf = cook_torrance_brdf(material, random_bounce_direction, -ray.direction, closest_hit_info.normal_at_intersection);
                    
                    if (bounce == 0)
                        sample_color += material.emission;
                    sample_color += (light_sample_radiance + env_map_radiance) * throughput;

                    if ((brdf.r == 0.0f && brdf.g == 0.0f && brdf.b == 0.0f) || brdf_pdf < 1.0e-8f || std::isinf(brdf_pdf))
                    {
                        next_ray_state = RayState::TERMINATED;

                        break;
                    }

                    throughput *= brdf * std::max(0.0f, dot(random_bounce_direction, closest_hit_info.normal_at_intersection)) / brdf_pdf;

                    Point new_ray_origin = closest_hit_info.inter_point + closest_hit_info.normal_at_intersection * 1.0e-4f;
                    ray = Ray(new_ray_origin, random_bounce_direction);
                    next_ray_state = RayState::BOUNCE;
                }
                else
                    next_ray_state = RayState::MISSED;
            }
            else if (next_ray_state == MISSED)
            {
                if (bounce == 1)
                {
                    //We're only getting the skysphere radiance for the first rays because the
                    //syksphere is importance sampled

                    Color skysphere_color = sample_environment_map_from_direction(ray.direction);

                    sample_color += skysphere_color * throughput;
                }

                break;
            }
            else if (next_ray_state == TERMINATED)
                break;
        }

        final_color += sample_color;
    }

    final_color /= m_render_samples;
    final_color.a = 0.0f;
    m_frame_buffer[y * m_width + x] += final_color;

    const float gamma = 2.2f;
    const float exposure = 1.5f;
    Color hdrColor = m_frame_buffer[y * m_width + x];

    //Exposure tone mapping
    Color tone_mapped = Color(1.0f, 1.0f, 1.0f) - exp(-hdrColor * exposure);
    // Gamma correction
    Color gamma_corrected = pow(tone_mapped, 1.0f / gamma);

    m_frame_buffer[y * m_width + x] = gamma_corrected;
}

#include <atomic>
#include <omp.h>

#define DEBUG_PIXEL 0
#define PIXEL_X 104
#define PIXEL_Y 216
void RenderKernel::render()
{
    std::atomic<int> lines_completed = 0;

#if DEBUG_PIXEL
    for (int y = m_frame_buffer.height() - PIXEL_Y - 1; y < m_frame_buffer.height(); y++)
    {
        for (int x = PIXEL_X; x < m_frame_buffer.width(); x++)
#else
#pragma omp parallel for schedule(dynamic)
    for (int y = 0; y < m_frame_buffer.height(); y++)
    {
        for (int x = 0; x < m_frame_buffer.width(); x++)
#endif
            ray_trace_pixel(x, y);

        lines_completed++;

        if (omp_get_thread_num() == 0)
            if (lines_completed % (m_frame_buffer.height() / 25))
                std::cout << lines_completed / (float)m_frame_buffer.height() * 100 << "%" << std::endl;
    }
}

Color RenderKernel::lambertian_brdf(const SimpleMaterial& material, const Vector& to_light_direction, const Vector& view_direction, const Vector& surface_normal) const
{
    return material.diffuse * M_1_PI;
}

Color fresnel_schlick(Color F0, float NoV)
{
    return F0 + (Color(1.0f) - F0) * std::pow((1.0f - NoV), 5.0f);
}

float GGX_normal_distribution(float alpha, float NoH)
{
    //To avoid numerical instability when NoH basically == 1, i.e when the
    //material is a perfect mirror and the normal distribution function is a Dirac

    NoH = std::min(NoH, 0.999999f);
    float alpha2 = alpha * alpha;
    float NoH2 = NoH * NoH;
    float b = (NoH2 * (alpha2 - 1.0f) + 1.0f);
    return alpha2 * M_1_PI / (b * b);// std::max(b * b, 1.0e-18f);
}

float G1_schlick_ggx(float k, float dot_prod)
{
    return dot_prod / (dot_prod * (1.0f - k) + k);
}

float GGX_smith_masking_shadowing(float roughness_squared, float NoV, float NoL)
{
    float k = roughness_squared / 2.0f;

    return G1_schlick_ggx(k, NoL) * G1_schlick_ggx(k, NoV);
}

float RenderKernel::cook_torrance_brdf_pdf(const SimpleMaterial& material, const Vector& view_direction, const Vector& to_light_direction, const Vector& surface_normal) const
{
    Vector microfacet_normal = normalize(view_direction + to_light_direction);

    float alpha = material.roughness * material.roughness;

    float VoH = std::max(0.0f, dot(view_direction, microfacet_normal));
    float NoH = std::max(0.0f, dot(surface_normal, microfacet_normal));
    float D = GGX_normal_distribution(alpha, NoH);

    return D * NoH / (4.0f * VoH);
}

inline Color RenderKernel::cook_torrance_brdf(const SimpleMaterial& material, const Vector& to_light_direction, const Vector& view_direction, const Vector& surface_normal) const
{
    Color brdf_color = Color(0.0f, 0.0f, 0.0f);
    Color base_color = material.diffuse;

    Vector halfway_vector = normalize(view_direction + to_light_direction);

    float NoV = std::max(0.0f, dot(surface_normal, view_direction));
    float NoL = std::max(0.0f, dot(surface_normal, to_light_direction));
    float NoH = std::max(0.0f, dot(surface_normal, halfway_vector));
    float VoH = std::max(0.0f, dot(halfway_vector, view_direction));

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
        kD *= Color(1.0f) - F;//Only the transmitted light is diffused

        Color diffuse_part = kD * base_color / (float)M_PI;
        Color specular_part = (F * D * G) / (4.0f * NoV * NoL);

        brdf_color = diffuse_part + specular_part;
    }

    return brdf_color;
}

Vector SampleVndf_Hemisphere(xorshift32_generator& random_number_generator, Vector wi)
{
    float u1 = random_number_generator();
    float u2 = random_number_generator();

    // sample a spherical cap in (-wi.z, 1]
    float phi = 2.0f * M_PI * u1;
    //float z = fma((1.0f - u.y), (1.0f + wi.z), -wi.z);
    float z = 1.0f - u2 - u2 * wi.z;
    float sinTheta = std::sqrt(std::min(1.0f, std::max(0.0f, (1.0f - z * z))));
    float x = sinTheta * std::cos(phi);
    float y = sinTheta * std::sin(phi);
    Vector c = Vector(x, y, z);
    // compute halfway direction;
    Vector h = c + wi;
    // return without normalization as this is done later (see line 25)
    return h;
}

//From https://schuttejoe.github.io/post/ggximportancesamplingpart2/
Vector RenderKernel::cook_torrance_brdf_sample_visible_normal(const SimpleMaterial& material, const Vector& wo, const Vector& surface_normal, float& pdf, xorshift32_generator& random_number_generator) const
{
//    // -- Stretch the view vector so we are sampling as though
//    // -- roughness==1
//    float roughness = material.roughness;
//    float alpha = roughness * roughness;

//    Vector v = normalize(Vector(wo.x * roughness, wo.y, wo.z * roughness));

//    // -- Build an orthonormal basis with v, t1, and t2
//    Vector t1, t2;
//    branchlessONB(wo, t1, t2);

//    // -- Choose a point on a disk with each half of the disk weighted
//    // -- proportionally to its projection onto direction v
//    float u1 = random_number_generator();
//    float u2 = random_number_generator();
//    float a = 1.0f / (1.0f + v.y);
//    float r = std::sqrt(u1);
//    float phi = (u2 < a) ? (u2 / a) * M_PI : M_PI + (u2 - a) / (1.0f - a) * M_PI;
//    float p1 = r * std::cos(phi);
//    float p2 = r * std::sin(phi) * ((u2 < a) ? 1.0f : v.y);

//    // -- Calculate the normal in this stretched tangent space
//    Vector n = p1 * t1 + p2 * t2 + std::sqrt(std::max(0.0f, 1.0f - p1 * p1 - p2 * p2)) * v;

//    // -- unstretch and normalize the normal
//    Vector microfacet_normal = normalize(Vector(roughness * n.x, std::max(0.0f, n.y), roughness * n.z));
//    //microfacet_normal = rotate_vector_around_normal(surface_normal, microfacet_normal);


    float roughness = material.roughness;
    float alpha = roughness * roughness;
    // warp to the hemisphere configuration
    Vector wiStd = normalize(Vector(wo.x * roughness, wo.y * roughness, wo.z));
    // sample the hemisphere
    Vector wmStd = SampleVndf_Hemisphere(random_number_generator, wiStd);
    // warp back to the ellipsoid configuration
    Vector microfacet_normal = normalize(Vector(wmStd.x * roughness, wmStd.y * roughness, wmStd.z));
    microfacet_normal = rotate_vector_around_normal(surface_normal, microfacet_normal);

    float VoH = std::max(dot(wo, microfacet_normal), 0.0f);
    float NoH = std::max(dot(microfacet_normal, surface_normal), 0.0f);
    float NoV = std::max(dot(wo, surface_normal), 0.0f);
    pdf = G1_schlick_ggx(VoH, alpha * alpha) * VoH * GGX_normal_distribution(alpha * alpha, NoH) / (NoV * 4 * VoH);

    return microfacet_normal;
}

float SmithGGXMasking(Vector normal, Vector wo, float a2)
{
    float dotNV = dot(normal, wo);
    float denomC = std::sqrt(a2 + (1.0f - a2) * dotNV * dotNV) + dotNV;

    return 2.0f * dotNV / denomC;
}

//====================================================================
float SmithGGXMaskingShadowing(Vector normal, Vector wi, Vector wo, float a2)
{
    float dotNL = dot(normal, wi);
    float dotNV = dot(normal, wo);

    float denomA = dotNV * std::sqrt(a2 + (1.0f - a2) * dotNL * dotNL);
    float denomB = dotNL * std::sqrt(a2 + (1.0f - a2) * dotNV * dotNV);

    return 2.0f * dotNL * dotNV / (denomA + denomB);
}

Color RenderKernel::cook_torrance_brdf_importance_sample(const SimpleMaterial& material, const Vector& view_direction, const Vector& surface_normal, Vector& output_direction, float& pdf, xorshift32_generator& random_number_generator) const
{
    pdf = 0.0f;

    float metalness = material.metalness;
    float roughness = material.roughness;
    float alpha = roughness * roughness;

    float rand1 = random_number_generator();
    float rand2 = random_number_generator();

    float phi = 2.0f * (float)M_PI * rand1;
    float theta = std::acos((1.0f - rand2) / (rand2 * (alpha * alpha - 1.0f) + 1.0f));
    float sin_theta = std::sin(theta);

    Vector microfacet_normal_local_space = Vector(std::cos(phi) * sin_theta, std::sin(phi) * sin_theta, std::cos(theta));
    Vector microfacet_normal = rotate_vector_around_normal(surface_normal, microfacet_normal_local_space);
    if (dot(microfacet_normal, surface_normal) < 0.0f)
        //The microfacet normal that we sampled was under the surface, this can happen
        return Color(0.0f);
    Vector to_light_direction = normalize(2.0f * dot(microfacet_normal, view_direction) * microfacet_normal - view_direction);
    Vector halfway_vector = microfacet_normal;
    output_direction = to_light_direction;

    Color brdf_color = Color(0.0f, 0.0f, 0.0f);
    Color base_color = material.diffuse;

    float NoV = std::max(0.0f, dot(surface_normal, view_direction));
    float NoL = std::max(0.0f, dot(surface_normal, to_light_direction));
    float NoH = std::max(0.0f, dot(surface_normal, halfway_vector));
    float VoH = std::max(0.0f, dot(halfway_vector, view_direction));

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
        kD *= Color(1.0f) - F;//Only the transmitted light is diffused

        Color diffuse_part = kD * base_color / (float)M_PI;
        Color specular_part = (F * D * G) / (4.0f * NoV * NoL);

        pdf = D * NoH / (4.0f * VoH);

        brdf_color = diffuse_part + specular_part;
    }

    return brdf_color;
}

bool RenderKernel::intersect_scene(const Ray& ray, HitInfo& closest_hit_info) const
{
    closest_hit_info.t = -1.0f;

    for (int i = 0; i < m_triangle_buffer_access.size(); i++)
    {
        const Triangle& triangle = m_triangle_buffer_access[i];

        HitInfo hit_info;
        if(triangle.intersect(ray, hit_info))
        {
            if (hit_info.t < closest_hit_info.t || closest_hit_info.t == -1.0f)
            {
                closest_hit_info = hit_info;
                closest_hit_info.primitive_index = i;
            }
        }
    }

    for (int i = 0; i < m_sphere_buffer.size(); i++)
    {
        const Sphere& sphere = m_sphere_buffer[i];

        HitInfo hit_info;
        if (sphere.intersect(ray, hit_info))
            if (hit_info.t < closest_hit_info.t || closest_hit_info.t == -1.0f)
                closest_hit_info = hit_info;
    }

    return closest_hit_info.t > 0.0f;
}

inline bool RenderKernel::intersect_scene_bvh(const Ray& ray, HitInfo& closest_hit_info) const
{
    closest_hit_info.t == -1.0f;

    m_bvh.intersect(ray, closest_hit_info);

    for (int i = 0; i < m_sphere_buffer.size(); i++)
    {
        const Sphere& sphere = m_sphere_buffer[i];

        HitInfo hit_info;
        if (sphere.intersect(ray, hit_info))
            if (hit_info.t < closest_hit_info.t || closest_hit_info.t == -1.0f)
                closest_hit_info = hit_info;
    }

    return closest_hit_info.t > 0.0f;
}

inline bool RenderKernel::INTERSECT_SCENE(const Ray& ray, HitInfo& hit_info) const
{
#if USE_BVH
    return intersect_scene_bvh(ray, hit_info);
#else
    return intersect_scene(ray, hit_info);
#endif
}

float power_heuristic(float pdf_a, float pdf_b)
{
    float pdf_a_squared = pdf_a * pdf_a;

    return pdf_a_squared / (pdf_a_squared + pdf_b * pdf_b);
}

Color RenderKernel::sample_environment_map_from_direction(const Vector& direction) const
{
    float u, v;
    u = 0.5f + std::atan2(direction.z, direction.x) / (2.0f * (float)M_PI);
    v = 0.5f + std::asin(direction.y) / (float)M_PI;

    int x = std::max(std::min((int)(u * m_environment_map.width()), m_environment_map.width() - 1), 0);
    int y = std::max(std::min((int)(v * m_environment_map.height()), m_environment_map.height() - 1), 0);

    return m_environment_map[y * m_environment_map.width() + x];
}

void RenderKernel::env_map_cdf_search(float value, int& x, int& y) const
{
    //First searching a line to sample
    int lower = 0;
    int upper = m_environment_map.height() - 1;

    int x_index = m_environment_map.width() - 1;
    while (lower < upper)
    {
        int y_index = (lower + upper) / 2;
        int env_map_index = y_index * m_environment_map.width() + x_index;

        if (value < m_env_map_cdf[env_map_index])
            upper = y_index;
        else
            lower = y_index + 1;
    }
    y = std::max(std::min(lower, m_environment_map.height()), 0);

    //Then sampling the line itself
    lower = 0;
    upper = m_environment_map.width() - 1;

    int y_index = y;
    while (lower < upper)
    {
        int x_index = (lower + upper) / 2;
        int env_map_index = y_index * m_environment_map.width() + x_index;

        if (value < m_env_map_cdf[env_map_index])
            upper = x_index;
        else
            lower = x_index + 1;
    }
    x = std::max(std::min(lower, m_environment_map.width()), 0);
}

Color RenderKernel::sample_environment_map(const Ray& ray, const HitInfo& closest_hit_info, const SimpleMaterial& material, xorshift32_generator& random_number_generator) const
{
    float env_map_total_sum = m_env_map_cdf[m_env_map_cdf.size() - 1];

    int x, y;
    env_map_cdf_search(random_number_generator() * env_map_total_sum, x, y);

    float u = (float)x / m_environment_map.width();
    float v = (float)y / m_environment_map.height();
    float phi = u * 2.0f * M_PI;
    float theta = v * M_PI;

    Color env_sample;
    float sin_theta = std::sin(theta);
    float cos_theta = std::cos(theta);

    // Convert to cartesian coordinates
    Vector sampled_direction = Vector(-sin_theta * cos(phi), -cos_theta, -sin_theta * sin(phi));

    float cosine_term = dot(closest_hit_info.normal_at_intersection, sampled_direction);
    if  (cosine_term > 0.0f)
    {
        HitInfo trash;
        if (!INTERSECT_SCENE(Ray(closest_hit_info.inter_point + closest_hit_info.normal_at_intersection * 1.0e-4f, sampled_direction), trash))
        {
            float env_map_pdf = m_environment_map.luminance_of_pixel(x, y) / env_map_total_sum;
            env_map_pdf = (env_map_pdf * m_environment_map.width() * m_environment_map.height()) / (2.0f * M_PI * M_PI * sin_theta);

            Color env_map_radiance = m_environment_map(x, y);
            Color brdf = cook_torrance_brdf(material, sampled_direction, -ray.direction, closest_hit_info.normal_at_intersection);
            float brdf_pdf = cook_torrance_brdf_pdf(material, -ray.direction, sampled_direction, closest_hit_info.normal_at_intersection);

            float mis_weight = power_heuristic(env_map_pdf, brdf_pdf);
            env_sample = brdf * cosine_term * mis_weight * env_map_radiance / env_map_pdf;
        }
    }

    float brdf_sample_pdf;
    Vector brdf_sampled_dir;
    Color brdf_imp_sampling = cook_torrance_brdf_importance_sample(material, -ray.direction, closest_hit_info.normal_at_intersection, brdf_sampled_dir, brdf_sample_pdf, random_number_generator);

    cosine_term = std::max(dot(closest_hit_info.normal_at_intersection, brdf_sampled_dir), 0.0f);
    Color brdf_sample;
    if (brdf_sample_pdf != 0.0f && cosine_term > 0.0f)
    {
        HitInfo trash;
        if (!INTERSECT_SCENE(Ray(closest_hit_info.inter_point + closest_hit_info.normal_at_intersection * 1.0e-5f, brdf_sampled_dir), trash))
        {
            Color skysphere_color = sample_environment_map_from_direction(brdf_sampled_dir);
            float theta_brdf_dir = std::acos(brdf_sampled_dir.z);
            float sin_theta_bdrf_dir = std::sin(theta_brdf_dir);
            float env_map_pdf = skysphere_color.luminance() / m_env_map_cdf[m_env_map_cdf.size() - 1];

            env_map_pdf *= m_environment_map.width() * m_environment_map.height();
            env_map_pdf /= (2.0f * M_PI * M_PI * sin_theta_bdrf_dir);

            float mis_weight = power_heuristic(brdf_sample_pdf, env_map_pdf);
            brdf_sample = skysphere_color * mis_weight * cosine_term * brdf_imp_sampling / brdf_sample_pdf;
        }
    }

    return brdf_sample + env_sample;
}

Color RenderKernel::sample_light_sources(const Ray& ray, const HitInfo& closest_hit_info, const SimpleMaterial& material, xorshift32_generator& random_number_generator) const
{
    Color light_source_radiance_mis;
    if (m_emissive_triangle_indices_buffer.size() > 0)
    {
        float light_sample_pdf;
        LightSourceInformation light_source_info;
        Point random_light_point = sample_random_point_on_lights(random_number_generator, light_sample_pdf, light_source_info);

        Point shadow_ray_origin = closest_hit_info.inter_point + closest_hit_info.normal_at_intersection * 1.0e-4f;
        Vector shadow_ray_direction = random_light_point - shadow_ray_origin;
        float distance_to_light = length(shadow_ray_direction);
        Vector shadow_ray_direction_normalized = normalize(shadow_ray_direction);

        Ray shadow_ray(shadow_ray_origin, shadow_ray_direction_normalized);

        float dot_light_source = std::max(dot(light_source_info.light_source_normal, -shadow_ray_direction_normalized), 0.0f);
        if (dot_light_source > 0.0f)
        {
            bool in_shadow = evaluate_shadow_ray(shadow_ray, distance_to_light);

            if (!in_shadow)
            {
                const SimpleMaterial& emissive_triangle_material = m_materials_buffer_access[m_materials_indices_buffer[light_source_info.emissive_triangle_index]];

                light_sample_pdf *= distance_to_light * distance_to_light;
                light_sample_pdf /= dot_light_source;

                Color brdf = cook_torrance_brdf(material, shadow_ray.direction, -ray.direction, closest_hit_info.normal_at_intersection);

                float cook_torrance_pdf = cook_torrance_brdf_pdf(material, -ray.direction, shadow_ray_direction_normalized, closest_hit_info.normal_at_intersection);
                if (cook_torrance_pdf != 0.0f)
                {
                    float mis_weight = power_heuristic(light_sample_pdf, cook_torrance_pdf);

                    Color Li = emissive_triangle_material.emission;
                    float cosine_term = dot(closest_hit_info.normal_at_intersection, shadow_ray_direction_normalized);

                    light_source_radiance_mis = Li * cosine_term * brdf * mis_weight / light_sample_pdf;
                }
            }
        }
    }

    Color brdf_radiance_mis;

    Vector sampled_brdf_direction;
    float direction_pdf;
    Color brdf = cook_torrance_brdf_importance_sample(material, -ray.direction, closest_hit_info.normal_at_intersection, sampled_brdf_direction, direction_pdf, random_number_generator);
    if (brdf != Color::Black())
    {
        Ray new_ray = Ray(closest_hit_info.inter_point + closest_hit_info.normal_at_intersection * 1.0e-5f, sampled_brdf_direction);
        HitInfo new_ray_hit_info;
        bool inter_found = INTERSECT_SCENE(new_ray, new_ray_hit_info);

        if (inter_found)
        {
            float cos_angle = std::max(dot(new_ray_hit_info.normal_at_intersection, -sampled_brdf_direction), 0.0f);
            if (cos_angle > 0.0f)
            {
                int material_index = m_materials_indices_buffer[new_ray_hit_info.primitive_index];
                SimpleMaterial material = m_materials_buffer_access[material_index];

                Color emission = material.emission;
                if (emission.r > 0 || emission.g > 0 || emission.b > 0)
                {
                    float distance_squared = new_ray_hit_info.t * new_ray_hit_info.t;
                    float light_area = m_triangle_buffer_access[new_ray_hit_info.primitive_index].area();

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

inline Point RenderKernel::sample_random_point_on_lights(xorshift32_generator& random_number_generator, float& pdf, LightSourceInformation& light_info) const
{
    light_info.emissive_triangle_index = random_number_generator() * m_emissive_triangle_indices_buffer.size();
    light_info.emissive_triangle_index = m_emissive_triangle_indices_buffer[light_info.emissive_triangle_index];
    Triangle random_emissive_triangle = m_triangle_buffer_access[light_info.emissive_triangle_index];

    float rand_1 = random_number_generator();
    float rand_2 = random_number_generator();

    float sqrt_r1 = std::sqrt(rand_1);
    float u = 1.0f - sqrt_r1;
    float v = (1.0f - rand_2) * sqrt_r1;

    Vector AB = random_emissive_triangle.m_b - random_emissive_triangle.m_a;
    Vector AC = random_emissive_triangle.m_c - random_emissive_triangle.m_a;

    Point random_point_on_triangle = random_emissive_triangle.m_a + AB * u + AC * v;

    Vector normal = cross(AB, AC);
    float length_normal = length(normal);
    light_info.light_source_normal = normal / length_normal; //Normalized
    float triangle_area = length_normal * 0.5f;
    float nb_emissive_triangles = m_emissive_triangle_indices_buffer.size();

    pdf = 1.0f / (nb_emissive_triangles * triangle_area);

    return random_point_on_triangle;
}

bool RenderKernel::evaluate_shadow_ray(const Ray& ray, float t_max) const
{
    HitInfo hit_info;
    bool inter_found = INTERSECT_SCENE(ray, hit_info);

    if (inter_found)
    {
        if (hit_info.t + 1.0e-4f < t_max)
            //There is something in between the light and the origin of the ray
            return true;
        else
            return false;
    }

    return false;
}
