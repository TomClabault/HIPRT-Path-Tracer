#include "HostDeviceCommon/hit_info.h"
#include "render_kernel.h"
#include "triangle.h"

#define DEBUG_PIXEL 0
#define DEBUG_EXACT_COORDINATE 0
#define DEBUG_PIXEL_X 105
#define DEBUG_PIXEL_Y 508

Point point_mat4x4(const glm::mat4x4& mat, const Point& p)
{
    glm::vec4 pt = mat * (glm::vec4(p.x, p.y, p.z, 1.0f));
    return Point(pt.x / pt.w, pt.y / pt.w, pt.z / pt.w);
}

Vector vec4_mat4x4(const glm::mat4x4& mat, const Vector& v)
{
    glm::vec4 vt = mat * (glm::vec4(v.x, v.y, v.z, 0.0f));
    return Vector(vt.x / vt.w, vt.y / vt.w, vt.z / vt.w);
}

void build_ONB(const Vector& N, Vector& T, Vector& B)
{
    Vector up = abs(N.z) < 0.9999999 ? Vector(0, 0, 1) : Vector(1, 0, 0);
    T = normalize(cross(up, N));
    B = cross(N, T);
}

void build_rotated_ONB(const Vector& N, Vector& T, Vector& B, float basis_rotation)
{
    Vector up = abs(N.z) < 0.9999999 ? Vector(0, 0, 1) : Vector(1, 0, 0);
    T = normalize(cross(up, N));

    // Rodrigues' rotation
    T = T * cos(basis_rotation) + cross(N, T) * sin(basis_rotation) + N * dot(N, T) * (1.0f - cos(basis_rotation));
    B = cross(N, T);
}

/*
 * Transforms V from its local space to the space around the normal
 */
Vector local_to_world_frame(const Vector& N, const Vector& V)
{
    Vector T, B;
    build_ONB(N, T, B);

    return normalize(V.x * T + V.y * B + V.z * N);
}

Vector local_to_world_frame(const Vector& T, const Vector& B, const Vector& N, const Vector& V)
{
    return normalize(V.x * T + V.y * B + V.z * N);
}

/*
 * Transforms V from its space to the local space around the normal
 * The given normal is the Z axis of the local frame around the normal
 */
Vector world_to_local_frame(const Vector& N, const Vector& V)
{
    Vector T, B;
    build_ONB(N, T, B);

    return normalize(Vector(dot(V, T), dot(V, B), dot(V, N)));
}

Vector world_to_local_frame(const Vector& T, const Vector& B, const Vector& N, const Vector& V)
{
    return normalize(Vector(dot(V, T), dot(V, B), dot(V, N)));
}

// TODO rename render_kernel to renderer_cpu

/**
 * Reflects a ray about a normal. This function requires that dot(ray_direction, surface_normal) > 0 i.e.
 * ray_direction and surface_normal are in the same hemisphere
 */
Vector reflect_ray(const Vector& ray_direction, const Vector& surface_normal)
{
    return -ray_direction + 2.0f * dot(ray_direction, surface_normal) * surface_normal;
}

float fresnel_dielectric(float cos_theta_i, float relative_eta)
{
    // Computing cos_theta_t
    float sin_theta_i2 = 1.0f - cos_theta_i * cos_theta_i;
    float sin_theta_t2 = sin_theta_i2 / (relative_eta * relative_eta);

    if (sin_theta_t2 >= 1.0f)
        // Total internal reflection, 0% refraction, all reflection
        return 1.0f;

    float cos_theta_t = sqrt(1.0f - sin_theta_t2);
    float r_parallel = (relative_eta * cos_theta_i - cos_theta_t) / (relative_eta * cos_theta_i + cos_theta_t);
    float r_perpendicular = (cos_theta_i - relative_eta * cos_theta_t) / (cos_theta_i + relative_eta * cos_theta_t);
    return (r_parallel * r_parallel + r_perpendicular * r_perpendicular) / 2;
}

float fresnel_dielectric(float cos_theta_i, float eta_i, float eta_t)
{
    return fresnel_dielectric(cos_theta_i, eta_t / eta_i);
}

/**
 * Reflects a ray about a normal. This function requires that dot(ray_direction, surface_normal) > 0 i.e.
 * ray_direction and surface_normal are in the same hemisphere
 */
bool refract_ray(const Vector& ray_direction, const Vector& surface_normal, Vector& refract_direction, float relative_eta)
{
    float NoI = dot(ray_direction, surface_normal);

    float sin_theta_i_2 = 1.0f - NoI * NoI;
    float root_term = 1.0f - sin_theta_i_2 / (relative_eta * relative_eta);
    if (root_term < 0.0f)
        return false;
     
    float cos_theta_t = std::sqrt(root_term);
    refract_direction = -ray_direction / relative_eta + (NoI / relative_eta - cos_theta_t) * surface_normal;

    return true;
}

Vector RenderKernel::uniform_direction_around_normal(const Vector& normal, float& pdf, Xorshift32Generator& random_number_generator)
{
    float rand_1 = random_number_generator();
    float rand_2 = random_number_generator();

    float phi = 2.0f * (float)M_PI * rand_1;
    float root = std::sqrt(1.0f - rand_2 * rand_2);

    pdf = 1.0f / (2.0f * (float)M_PI);

    //Generating a random direction in a local space with Z as the Up vector
    Vector random_dir_local_space(std::cos(phi) * root, std::sin(phi) * root, rand_2);
    return local_to_world_frame(normal, random_dir_local_space);
}

Vector RenderKernel::cosine_weighted_sample(const Vector& normal, float& pdf, Xorshift32Generator& random_number_generator)
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
    return local_to_world_frame(normal, random_dir_local_space);
}

void RenderKernel::cosine_weighted_eval(const Vector& normal, const Vector& direction, float& pdf)
{
    pdf = dot(normal, direction) / M_PI;
}

Ray RenderKernel::get_camera_ray(float x, float y)
{
    float x_ndc_space = x / m_framebuffer_width * 2 - 1;
    float y_ndc_space = y / m_framebuffer_height * 2 - 1;

    Point ray_origin_view_space(0.0f, 0.0f, 0.0f);
    Point ray_origin = point_mat4x4(glm::inverse(m_camera.get_view_matrix()), ray_origin_view_space);

    Point ray_point_direction_ndc_space = Point(x_ndc_space, y_ndc_space, 1.0f);
    Point ray_point_direction_view_space = point_mat4x4(glm::inverse(m_camera.projection_matrix), ray_point_direction_ndc_space);
    Point ray_point_direction_world_space = point_mat4x4(glm::inverse(m_camera.get_view_matrix()), ray_point_direction_view_space);

    Vector ray_direction = normalize(ray_point_direction_world_space - ray_origin);

    Ray ray(ray_origin, ray_direction);
    return ray;
}

void RenderKernel::debug_set_final_color(int x, int y, Color final_color)
{
    m_frame_buffer[x + y * m_framebuffer_width] = final_color;
}

unsigned int wang_hash(unsigned int seed)
{
    seed = (seed ^ 61) ^ (seed >> 16);
    seed *= 9;
    seed = seed ^ (seed >> 4);
    seed *= 0x27d4eb2d;
    seed = seed ^ (seed >> 15);
    return seed;
}

void RenderKernel::ray_trace_pixel(int x, int y)
{
    Xorshift32Generator random_number_generator(wang_hash(((x + y * m_framebuffer_width) + 1) * (m_render_samples + 1)));

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
        BRDF last_brdf_hit_type = BRDF::Uninitialized;

        for (int bounce = 0; bounce < m_max_bounces; bounce++)
        {
            if (next_ray_state == RayState::BOUNCE)
            {
                HitInfo closest_hit_info;
                bool intersection_found = INTERSECT_SCENE(ray, closest_hit_info);

                if (intersection_found)
                {
                    int material_index = m_materials_indices_buffer[closest_hit_info.primitive_index];
                    RendererMaterial material = m_materials_buffer[material_index];
                    last_brdf_hit_type = material.brdf_type;

                    // For the BRDF calculations, bounces, ... to be correct, we need the normal to be in the same hemisphere as
                    // the view direction. One thing that can go wrong is when we have an emissive quad (typical area light)
                    // and a ray hits the back of the quad. The normal will not be facing the view direction in this
                    // case and this will cause issues later in the BRDF.
                    // Because we want to allow backfacing emissive geometry (making the emissive geometry double sided
                    // and emitting light in both directions of the surface), we're negating the normal to make
                    // it face the view direction (but only for emissive geometry)
                    if (material.is_emissive() && dot(-ray.direction, closest_hit_info.geometric_normal) < 0)
                    {
                        closest_hit_info.geometric_normal = -closest_hit_info.geometric_normal;
                        closest_hit_info.shading_normal = -closest_hit_info.shading_normal;
                    }

                    // --------------------------------------------------- //
                    // ----------------- Direct lighting ----------------- //
                    // --------------------------------------------------- //
                    Color light_sample_radiance = sample_light_sources(-ray.direction, closest_hit_info, material, random_number_generator);
                    Color env_map_radiance = Color(0.0f);// sample_environment_map(ray, closest_hit_info, material, random_number_generator);

                    // --------------------------------------- //
                    // ---------- Indirect lighting ---------- //
                    // --------------------------------------- //

                    float brdf_pdf;
                    Vector bounce_direction;
                    Color brdf = brdf_dispatcher_sample(material, -ray.direction, closest_hit_info.shading_normal, closest_hit_info.geometric_normal, bounce_direction, brdf_pdf, random_number_generator);
                    
                    if (bounce == 0)
                        sample_color += material.emission;
                    sample_color += (light_sample_radiance + env_map_radiance) * throughput;

                    if ((brdf.r == 0.0f && brdf.g == 0.0f && brdf.b == 0.0f) || brdf_pdf <= 0.0f)
                        break;

                    throughput *= brdf * std::abs(dot(bounce_direction, closest_hit_info.shading_normal)) / brdf_pdf;

                    int outside_surface = dot(bounce_direction, closest_hit_info.shading_normal) < 0 ? -1.0f : 1.0;
                    Point new_ray_origin = closest_hit_info.inter_point + closest_hit_info.shading_normal * 3.0e-3f * outside_surface;
                    ray = Ray(new_ray_origin, bounce_direction);
                    next_ray_state = RayState::BOUNCE;
                }
                else
                    next_ray_state = RayState::MISSED;
            }
            else if (next_ray_state == RayState::MISSED)
            {
                //if (bounce == 1 || last_brdf_hit_type == BRDF::SpecularFresnel)
                {
                    //We're only getting the skysphere radiance for the first rays because the
                    //syksphere is importance sampled
                    // We're also getting the skysphere radiance for perfectly specular BRDF since those
                    // are not importance sampled

                    Color skysphere_color = Color(1.0f);
                    //Color skysphere_color = sample_environment_map_from_direction(ray.direction);

                    sample_color += skysphere_color * throughput;
                }

                break;
            }
        }

        if (sample_color.r < 0 || sample_color.g < 0 || sample_color.b < 0)
        {
            std::cerr << "Sample color < 0" << std::endl;
            std::cerr << "Exact_X, Exact_Y, Sample: " << x << ", " << y << ", " << sample << std::endl;
            sample_color = Color(1000000.0f, 0.0f, 1000000.0f);
        }
        else if (std::isnan(sample_color.r) || std::isnan(sample_color.g) || std::isnan(sample_color.b))
        {
            std::cerr << "Sample color NaN" << std::endl;
            std::cerr << "Exact_X, Exact_Y, Sample: " << x << ", " << y << ", " << sample << std::endl;
            sample_color = Color(1000000.0f, 1000000.0f, 0.0f);
        }

        final_color += sample_color;
    }

    final_color /= m_render_samples;
    m_frame_buffer[y * m_framebuffer_width + x] += final_color;

    const float gamma = 2.2f;
    const float exposure = 1.0f;
    Color hdrColor = m_frame_buffer[y * m_framebuffer_width + x];

    //Exposure tone mapping
    Color tone_mapped = Color(1.0f, 1.0f, 1.0f) - exp(-hdrColor * exposure);
    // Gamma correction
    Color gamma_corrected = pow(tone_mapped, 1.0f / gamma);

    m_frame_buffer[y * m_framebuffer_width + x] = gamma_corrected;
}

#include <atomic>
#include <omp.h>

void RenderKernel::render()
{
    std::atomic<int> lines_completed = 0;
#if DEBUG_PIXEL
#if DEBUG_EXACT_COORDINATE
    for (int y = DEBUG_PIXEL_Y; y < m_frame_buffer.height; y++)
    {
        for (int x = DEBUG_PIXEL_X; x < m_frame_buffer.width; x++)
#else
    for (int y = m_frame_buffer.height - DEBUG_PIXEL_Y - 1; y < m_frame_buffer.height; y++)
    {
        for (int x = DEBUG_PIXEL_X; x < m_frame_buffer.width; x++)
#endif
#else
#pragma omp parallel for schedule(dynamic)
    for (int y = 0; y < m_framebuffer_height; y++)
    {
        for (int x = 0; x < m_framebuffer_width; x++)
#endif
            ray_trace_pixel(x, y);

        lines_completed++;

        if (omp_get_thread_num() == 0)
            if (lines_completed % (m_framebuffer_height / 25))
                std::cout << lines_completed / (float)m_framebuffer_height * 100 << "%" << std::endl;
    }
}

Color RenderKernel::lambertian_brdf(const RendererMaterial& material, const Vector& to_light_direction, const Vector& view_direction, const Vector& shading_normal)
{
    return material.base_color * M_1_PI;
}

Color fresnel_schlick(Color F0, float NoV)
{
    return F0 + (Color(1.0f) - F0) * std::pow((1.0f - NoV), 5.0f);
}

Vector GGXVNDF_sample(Vector local_view_direction, float alpha_x, float alpha_y, Xorshift32Generator& random_number_generator)
{
    float r1 = random_number_generator();
    float r2 = random_number_generator();

    Vector Vh = normalize(Vector(alpha_x * local_view_direction.x, alpha_y * local_view_direction.y, local_view_direction.z));

    float lensq = Vh.x * Vh.x + Vh.y * Vh.y;
    Vector T1 = lensq > 0.0f ? Vector(-Vh.y, Vh.x, 0) * 1.0f / std::sqrt(lensq) : Vector(1.0f, 0.0f, 0.0f);
    Vector T2 = cross(Vh, T1);

    float r = sqrt(r1);
    float phi = 2.0f * M_PI * r2;
    float t1 = r * cos(phi);
    float t2 = r * sin(phi);
    float s = 0.5f * (1.0f + Vh.z);
    t2 = (1.0f - s) * sqrt(1.0f - t1 * t1) + s * t2;

    Vector Nh = t1 * T1 + t2 * T2 + std::sqrt(std::max(0.0f, 1.0f - t1 * t1 - t2 * t2)) * Vh;

    return normalize(Vector(alpha_x * Nh.x, alpha_y * Nh.y, std::max(0.0f, Nh.z)));
}

float GTR2_anisotropic(const RendererMaterial& material, const Vector& local_half_vector)
{
    float denom = (local_half_vector.x * local_half_vector.x) / (material.alpha_x * material.alpha_x);
    denom += (local_half_vector.y * local_half_vector.y) / (material.alpha_y * material.alpha_y);
    denom += (local_half_vector.z * local_half_vector.z);

    return 1.0f / (M_PI * material.alpha_x * material.alpha_y * denom * denom);
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

float GTR1(float alpha_g, float local_halfway_z)
{
    float alpha_g_2 = alpha_g * alpha_g;

    float num = alpha_g_2 - 1.0f;
    float denom = M_PI * log(alpha_g_2) * (1.0f + (alpha_g_2 - 1.0f) * local_halfway_z * local_halfway_z);

    return num / denom;
}

float G1(float alpha_x, float alpha_y, const Vector& local_direction)
{
    float ax = local_direction.x * alpha_x;
    float ay = local_direction.y * alpha_y;

    float denom = (std::sqrt(1.0f + (ax * ax + ay * ay) / (local_direction.z * local_direction.z)) - 1.0f) * 0.5f;

    return 1.0f / (1.0f + denom);
}

float disney_clearcoat_masking_shadowing(const Vector& direction)
{
    return G1(0.25f, 0.25f, direction);
}

float GGX_masking_shadowing_anisotropic(const RendererMaterial& material, const Vector& local_view_direction, const Vector& local_to_light_direction)
{
    return G1(material.alpha_x, material.alpha_y, local_view_direction) 
        * G1(material.alpha_x, material.alpha_y, local_to_light_direction);
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

inline Color RenderKernel::cook_torrance_brdf_eval(const RendererMaterial& material, const Vector& view_direction, const Vector& shading_normal, const Vector& to_light_direction, float& pdf)
{
    Color brdf_color = Color(0.0f, 0.0f, 0.0f);
    Color base_color = material.base_color;

    Vector halfway_vector = normalize(view_direction + to_light_direction);

    float NoV = std::max(0.0f, dot(shading_normal, view_direction));
    float NoL = std::max(0.0f, dot(shading_normal, to_light_direction));
    float NoH = std::max(0.0f, dot(shading_normal, halfway_vector));
    float VoH = std::max(0.0f, dot(halfway_vector, view_direction));

    if (NoV > 0.0f && NoL > 0.0f && NoH > 0.0f)
    {
        float metallic = material.metallic;
        float roughness = material.roughness;

        float alpha = roughness * roughness;

        ////////// Cook Torrance BRDF //////////
        Color F;
        float D, G;

        //F0 = 0.04 for dielectrics, 1.0 for metals (approximation)
        Color F0 = Color(0.04f * (1.0f - metallic)) + metallic * base_color;

        //GGX Distribution function
        F = fresnel_schlick(F0, VoH);
        D = GGX_normal_distribution(alpha, NoH);
        G = GGX_smith_masking_shadowing(alpha, NoV, NoL);

        Color kD = Color(1.0f - metallic); //Metals do not have a base_color part
        kD *= Color(1.0f) - F;//Only the transmitted light is diffused

        Color diffuse_part = kD * base_color / (float)M_PI;
        Color specular_part = (F * D * G) / (4.0f * NoV * NoL);

        brdf_color = diffuse_part + specular_part;
        pdf = D * NoH / (4.0f * VoH);
    }

    return brdf_color;
}

Color RenderKernel::cook_torrance_brdf_sample(const RendererMaterial& material, const Vector& view_direction, const Vector& shading_normal, Vector& output_direction, float& pdf, Xorshift32Generator& random_number_generator)
{
    pdf = 0.0f;

    float metallic = material.metallic;
    float roughness = material.roughness;
    float alpha = roughness * roughness;

    float rand1 = random_number_generator();
    float rand2 = random_number_generator();

    float phi = 2.0f * (float)M_PI * rand1;
    float theta = std::acos((1.0f - rand2) / (rand2 * (alpha * alpha - 1.0f) + 1.0f));
    float sin_theta = std::sin(theta);

    Vector microfacet_normal_local_space = Vector(std::cos(phi) * sin_theta, std::sin(phi) * sin_theta, std::cos(theta));
    Vector microfacet_normal = local_to_world_frame(shading_normal, microfacet_normal_local_space);
    if (dot(microfacet_normal, shading_normal) < 0.0f)
        //The microfacet normal that we sampled was under the surface, this can happen
        return Color(0.0f);
    Vector to_light_direction = normalize(2.0f * dot(microfacet_normal, view_direction) * microfacet_normal - view_direction);
    Vector halfway_vector = microfacet_normal;
    output_direction = to_light_direction;

    Color brdf_color = Color(0.0f, 0.0f, 0.0f);
    Color base_color = material.base_color;

    float NoV = std::max(0.0f, dot(shading_normal, view_direction));
    float NoL = std::max(0.0f, dot(shading_normal, to_light_direction));
    float NoH = std::max(0.0f, dot(shading_normal, halfway_vector));
    float VoH = std::max(0.0f, dot(halfway_vector, view_direction));

    if (NoV > 0.0f && NoL > 0.0f && NoH > 0.0f)
    {
        /////////// Cook Torrance BRDF //////////
        Color F;
        float D, G;


        //GGX Distribution function
        D = GGX_normal_distribution(alpha, NoH);

        //F0 = 0.04 for dielectrics, 1.0 for metals (approximation)
        Color F0 = Color(0.04f * (1.0f - metallic)) + metallic * base_color;
        F = fresnel_schlick(F0, VoH);
        G = GGX_smith_masking_shadowing(alpha, NoV, NoL);

        Color kD = Color(1.0f - metallic); //Metals do not have a base_color part
        kD *= Color(1.0f) - F;//Only the transmitted light is diffused

        Color diffuse_part = kD * base_color / (float)M_PI;
        Color specular_part = (F * D * G) / (4.0f * NoV * NoL);

        pdf = D * NoH / (4.0f * VoH);

        brdf_color = diffuse_part + specular_part;
    }

    return brdf_color;
}

Color RenderKernel::smooth_glass_bsdf(const RendererMaterial& material, Vector& out_bounce_direction, const Vector& ray_direction, Vector& shading_normal, float eta_i, float eta_t, float& pdf, Xorshift32Generator& random_generator)
{
    // Clamping here because the dot product can eventually returns values less
    // than -1 or greater than 1 because of precision errors in the vectors
    // (in previous calculations)
    float cos_theta_i = std::min(std::max(-1.0f, dot(shading_normal, -ray_direction)), 1.0f);

    if (cos_theta_i < 0.0f)
    {
        // We're inside the surface, we're going to flip the eta and the normal for
        // the calculations that follow
        // Note that this also flips the normal for the caller of this function
        // since the normal is passed by reference. This is useful since the normal
        // will be used for offsetting the new ray origin for example
        cos_theta_i = -cos_theta_i;
        shading_normal = -shading_normal;
        std::swap(eta_i, eta_t);
    }

    // Computing the proportion of reflected light using fresnel equations
    // We're going to use the result to decide whether to refract or reflect the
    // ray
    float fresnel_reflect = fresnel_dielectric(cos_theta_i, eta_i, eta_t);
    if (random_generator() <= fresnel_reflect)
    {
        // Reflect the ray

        out_bounce_direction = reflect_ray(-ray_direction, shading_normal);
        pdf = fresnel_reflect;

        return Color(fresnel_reflect) / dot(shading_normal, out_bounce_direction);
    }
    else
    {
        // Refract the ray

        Vector refract_direction;
        bool can_refract = refract_ray(-ray_direction, shading_normal, refract_direction, eta_t / eta_i);
        if (!can_refract)
            // Shouldn't happen but can because of floating point imprecisions
            return Color(0.0f);

        out_bounce_direction = refract_direction;
        shading_normal = -shading_normal;
        pdf = 1.0f - fresnel_reflect;

        return Color(1.0f - fresnel_reflect) * material.base_color / dot(out_bounce_direction, shading_normal);
    }
}

Color RenderKernel::oren_nayar_eval(const RendererMaterial& material, const Vector& view_direction, const Vector& shading_normal, const Vector& to_light_direction)
{
    Vector T, B;
    build_ONB(shading_normal, T, B);

    // Using local view and light directions to simply following computations
    Vector local_view_direction = world_to_local_frame(T, B, shading_normal, view_direction);
    Vector local_to_light_direction = world_to_local_frame(T, B, shading_normal, to_light_direction);

    // sin(theta) = 1.0 - cos(theta)^2
    float sin_theta_i = sqrt(1.0f - local_to_light_direction.z * local_to_light_direction.z);
    float sin_theta_o = sqrt(1.0f - local_view_direction.z * local_view_direction.z);

    // max_cos here is going to be cos(phi_to_light - phi_view_direction)
    // but computed as cos(phi_light) * cos(phi_view) + sin(phi_light) * sin(phi_view)
    // according to cos(a - b) = cos(a) * cos(b) + sin(a) * sin(b)
    float max_cos = 0;
    if (sin_theta_i > 1.0e-4f && sin_theta_o > 1.0e-4f)
    {
        float sin_phi_i = local_to_light_direction.y / sin_theta_i;
        float cos_phi_i = local_to_light_direction.x / sin_theta_i;

        float sin_phi_o = local_view_direction.y / sin_theta_o;
        float cos_phi_o = local_view_direction.x / sin_theta_o;

        float d_cos = cos_phi_i * cos_phi_o + sin_phi_i * sin_phi_o;

        max_cos = RT_MAX(0.0f, d_cos);
    }

    float sin_alpha, tan_beta;
    if (abs(local_to_light_direction.z) > abs(local_view_direction.z))
    {
        sin_alpha = sin_theta_o;
        tan_beta = sin_theta_i / abs(local_to_light_direction.z);
    }
    else
    {
        sin_alpha = sin_theta_i;
        tan_beta = sin_theta_o / abs(local_view_direction.z);
    }

    return material.base_color / M_PI * (material.oren_nayar_A + material.oren_nayar_B * max_cos * sin_alpha * tan_beta);
}

float RenderKernel::disney_schlick_weight(float f0, float abs_cos_angle)
{
    return 1.0f + (f0 - 1.0f) * pow(1.0f - abs_cos_angle, 5.0f);
}

Color RenderKernel::disney_diffuse_eval(const RendererMaterial& material, const Vector& view_direction, const Vector& shading_normal, const Vector& to_light_direction, float& pdf)
{
    Vector half_vector = normalize(to_light_direction + view_direction);

    float LoH = clamp(0.0f, 1.0f, abs(dot(to_light_direction, half_vector)));
    float NoL = clamp(0.0f, 1.0f, abs(dot(shading_normal, to_light_direction)));
    float NoV = clamp(0.0f, 1.0f, abs(dot(shading_normal, view_direction)));

    pdf = NoL / M_PI;

    Color diffuse_part;
    float diffuse_90 = 0.5f + 2.0f * material.roughness * LoH * LoH;
    // Lambertian base_color
    //diffuse_part = material.base_color / M_PI;
    // Disney base_color
    diffuse_part = material.base_color / M_PI * disney_schlick_weight(diffuse_90, NoL) * disney_schlick_weight(diffuse_90, NoV) * NoL;
    // Oren nayar base_color
    //diffuse_part = oren_nayar_eval(material, view_direction, shading_normal, to_light_direction);

    Color fake_subsurface_part;
    float subsurface_90 = material.roughness * LoH * LoH;
    fake_subsurface_part = 1.25f * material.base_color / M_PI *
        (disney_schlick_weight(subsurface_90, NoL) * disney_schlick_weight(subsurface_90, NoV) * (1.0f / (NoL + NoV) - 0.5f) + 0.5f) * NoL;

    return (1.0f - material.subsurface) * diffuse_part + material.subsurface * fake_subsurface_part;
}

Vector RenderKernel::disney_diffuse_sample(const RendererMaterial& material, const Vector& view_direction, const Vector& shading_normal, Xorshift32Generator& random_number_generator)
{
    float trash_pdf;
    Vector sampled_direction = cosine_weighted_sample(shading_normal, trash_pdf, random_number_generator);

    return sampled_direction;
}

Color RenderKernel::disney_metallic_fresnel(const RendererMaterial& material, const Vector& local_half_vector, const Vector& local_to_light_direction)
{
    // The summary of what is below is the following:
    //
    // If the material is 100% metallic, then the fresnel term color is going to be 
    // the base_color of the material i.e. typical conductor response.
    // 
    // If the material is 0% metallic, then the fresnel term color is going to be
    // material.specular_color modulated by the material.specular_tint coefficient (which blends 
    // between white and material.specular_color) and the material.specular coefficient which
    // dictates whether we have a specular at all
    Color Ks = Color(1.0f - material.specular_tint) + material.specular_tint * material.specular_color;
    float R0 = ((material.ior - 1.0f) * (material.ior - 1.0f)) / ((material.ior + 1.0f) * (material.ior + 1.0f));
    Color C0 = material.specular * R0 * (1.0f - material.metallic) * Ks + material.metallic * material.base_color;

    return C0 + (Color(1.0f) - C0) * pow(1.0f - dot(local_half_vector, local_to_light_direction), 5.0f);
}

Color RenderKernel::disney_metallic_eval(const RendererMaterial& material, const Vector& view_direction, const Vector& surface_normal, const Vector& to_light_direction, Color F, float& pdf)
{
    // Building the local shading frame
    Vector T, B;
    build_rotated_ONB(surface_normal, T, B, material.anisotropic_rotation * M_PI);

    Vector local_view_direction = world_to_local_frame(T, B, surface_normal, view_direction);
    Vector local_to_light_direction = world_to_local_frame(T, B, surface_normal, to_light_direction);
    Vector local_half_vector = normalize(local_to_light_direction + local_view_direction);

    float NoV = abs(local_view_direction.z);
    float NoL = abs(local_to_light_direction.z);
    float HoL = abs(dot(local_half_vector, local_to_light_direction));

    // TODO remove
    //{
    //    // F = (-2.0f, -2.0f, -2.0f) is the default argument when the overload without the 'Color F' argument
    //    // of disney_metallic_eval() was called. Thus, if no F was passed, we're computing it here.
    //    // Otherwise, we're going to use the given one
    //    if (F.r == -2.0f)
    //        F = fresnel_schlick(material.base_color, NoL);
    //}

    float D = GTR2_anisotropic(material, local_half_vector);
    float G1_V = G1(material.alpha_x, material.alpha_y, local_view_direction);
    float G1_L = G1(material.alpha_x, material.alpha_y, local_to_light_direction);
    float G = G1_V * G1_L;

    pdf = D * G1_V / (4.0f * NoV);
    return F * D * G / (4.0 * NoL * NoV);
}

Vector RenderKernel::disney_metallic_sample(const RendererMaterial& material, const Vector& view_direction, const Vector& surface_normal, Xorshift32Generator& random_number_generator)
{
    Vector local_view_direction = world_to_local_frame(surface_normal, view_direction);

    // The view direction can sometimes be below the shading normal hemisphere
    // because of normal mapping
    int below_normal = (local_view_direction.z < 0) ? -1 : 1;
    Vector microfacet_normal = GGXVNDF_sample(local_view_direction * below_normal, material.alpha_x, material.alpha_y, random_number_generator);
    Vector sampled_direction = reflect_ray(view_direction, local_to_world_frame(surface_normal, microfacet_normal * below_normal));

    return sampled_direction;
}

Color RenderKernel::disney_clearcoat_eval(const RendererMaterial& material, const Vector& view_direction, const Vector& surface_normal, const Vector& to_light_direction, float& pdf)
{
    Vector T, B;
    build_ONB(surface_normal, T, B);

    Vector local_view_direction = world_to_local_frame(T, B, surface_normal, view_direction);
    Vector local_to_light_direction = world_to_local_frame(T, B, surface_normal, to_light_direction);
    Vector local_halfway_vector = normalize(local_view_direction + local_to_light_direction);

    if (local_view_direction.z * local_to_light_direction.z < 0)
        return Color(0.0f);

    float num = material.clearcoat_ior - 1.0f;
    float denom = material.clearcoat_ior + 1.0f;
    Color R0 = Color((num * num) / (denom * denom));

    float HoV = dot(local_halfway_vector, local_to_light_direction);
    float clearcoat_gloss = 1.0f - material.clearcoat_roughness;
    float alpha_g = (1.0f - clearcoat_gloss) * 0.1f + clearcoat_gloss * 0.001f;

    Color F_clearcoat = fresnel_schlick(R0, HoV);
    float D_clearcoat = GTR1(alpha_g, abs(local_halfway_vector.z));
    float G_clearcoat = disney_clearcoat_masking_shadowing(local_view_direction) * disney_clearcoat_masking_shadowing(local_to_light_direction);

    pdf = D_clearcoat * abs(local_halfway_vector.z) / (4.0f * dot(local_halfway_vector, local_to_light_direction));
    return F_clearcoat * D_clearcoat * G_clearcoat / (4.0f * local_view_direction.z);
}

Vector RenderKernel::disney_clearcoat_sample(const RendererMaterial& material, const Vector& view_direction, const Vector& surface_normal, Xorshift32Generator& random_number_generator)
{
    float clearcoat_gloss = 1.0f - material.clearcoat_roughness;
    float alpha_g = (1.0f - clearcoat_gloss) * 0.1f + clearcoat_gloss * 0.001f;
    float alpha_g_2 = alpha_g * alpha_g;

    float rand_1 = random_number_generator();
    float rand_2 = random_number_generator();

    float cos_theta = sqrt((1.0f - pow(alpha_g_2, 1.0f - rand_1)) / (1.0f - alpha_g_2));
    float sin_theta = sqrt(1.0f - cos_theta * cos_theta);

    float phi = 2.0f * M_PI * rand_2;
    float cos_phi = cos(phi);
    float sin_phi = sqrt(1.0f - cos_phi * cos_phi);

    Vector microfacet_normal = normalize(Vector{ sin_theta * cos_phi, sin_theta * sin_phi, cos_theta });
    Vector sampled_direction = reflect_ray(view_direction, local_to_world_frame(surface_normal, microfacet_normal));

    return sampled_direction;
}

// TOOD can use local_view dir and light_dir here
Color RenderKernel::disney_glass_eval(const RendererMaterial& material, const Vector& view_direction, Vector surface_normal, const Vector& to_light_direction, float& pdf)
{
    float start_NoV = dot(surface_normal, view_direction);
    if (start_NoV < 0.0f)
        surface_normal = -surface_normal;

    Vector T, B;
    build_rotated_ONB(surface_normal, T, B, material.anisotropic_rotation * M_PI);

    Vector local_to_light_direction = world_to_local_frame(T, B, surface_normal, to_light_direction);
    Vector local_view_direction = world_to_local_frame(T, B, surface_normal, view_direction);

    float NoV = local_view_direction.z;
    float NoL = local_to_light_direction.z;

    // We're in the case of reflection if the view direction and the bounced ray (light direction) are in the same hemisphere
    bool reflecting = NoL * NoV > 0;

    // Relative eta = eta_t / eta_i and we're assuming here that the eta of the incident light is air, 1.0f
    float relative_eta = material.ior;

    // Computing the generalized (that takes refraction into account) half vector
    Vector local_half_vector;
    if (reflecting)
        local_half_vector = local_to_light_direction + local_view_direction;
    else
    {
        // We want relative eta to always be eta_transmitted / eta_incident
        // so if we're refracting OUT of the surface, we're transmitting into
        // the air which has an eta of 1.0f so transmitted / incident
        // = 1.0f / material.ior (which relative_eta is equal to here)
        relative_eta = start_NoV > 0 ? material.ior : (1.0f / relative_eta);

        // We need to take the relative_eta into account when refracting to compute
        // the half vector (this is the "generalized" part of the half vector computation)
        local_half_vector = local_to_light_direction * relative_eta + local_view_direction;
    }

    local_half_vector = normalize(local_half_vector);
    if (local_half_vector.z < 0.0f)
        // Because the rest of the function we're going to compute here assume
        // that the microfacet normal is in the same hemisphere as the surface
        // normal, we're going to flip it if needed
        local_half_vector = -local_half_vector;

    float HoL = dot(local_to_light_direction, local_half_vector);
    float HoV = dot(local_view_direction, local_half_vector);

    // TODO to test removing that
    if (HoL * NoL < 0.0f || HoV * NoV < 0.0f)
        // Backfacing microfacets
        return Color(0.0f);

    Color color;
    float F = fresnel_dielectric(dot(local_view_direction, local_half_vector), relative_eta);
    if (reflecting)
    {
        color = disney_metallic_eval(material, view_direction, surface_normal, to_light_direction, Color(F), pdf);

        // Scaling the PDF by the probability of being here (reflection of the ray and not transmission)
        pdf *= F;
    }
    else
    {
        float dot_prod = HoL + HoV / relative_eta;
        float dot_prod2 = dot_prod * dot_prod;
        float denom = dot_prod2 * NoL * NoV;
        float D = GTR2_anisotropic(material, local_half_vector);
        float G1_V = G1(material.alpha_x, material.alpha_y, local_view_direction);
        float G = G1_V * G1(material.alpha_x, material.alpha_y, local_to_light_direction);

        float dwm_dwi = abs(HoL) / dot_prod2;
        float D_pdf = G1_V / abs(NoV) * D * abs(HoV);
        pdf = dwm_dwi * D_pdf * (1.0f - F);

        color = sqrt(material.base_color) * D * (1 - F) * G * abs(HoL * HoV / denom);
    }

    return color;
}

Vector RenderKernel::disney_glass_sample(const RendererMaterial& material, const Vector& view_direction, Vector surface_normal, Xorshift32Generator& random_number_generator)
{
    Vector T, B;
    build_rotated_ONB(surface_normal, T, B, material.anisotropic_rotation * M_PI);

    float relative_eta = material.ior;
    if (dot(surface_normal, view_direction) < 0)
    {
        // We want the surface normal in the same hemisphere as 
        // the view direction for the rest of the calculations
        surface_normal = -surface_normal;
        relative_eta = 1.0f / relative_eta;
    }

    Vector local_view_direction = world_to_local_frame(T, B, surface_normal, view_direction);
    Vector microfacet_normal = GGXVNDF_sample(local_view_direction, material.alpha_x, material.alpha_y, random_number_generator);
    if (microfacet_normal.z < 0)
        microfacet_normal = -microfacet_normal;

    float F = fresnel_dielectric(dot(local_view_direction, microfacet_normal), relative_eta);
    float rand_1 = random_number_generator();

    Vector sampled_direction;
    if (rand_1 < F)
    {
        // Reflection
        sampled_direction = reflect_ray(local_view_direction, microfacet_normal);
    }
    else
    {
        // Refraction

        if (dot(microfacet_normal, local_view_direction) < 0.0f)
            // For the refraction operation that follows, we want the direction to refract (the view
            // direction here) to be in the same hemisphere as the normal (the microfacet normal here)
            // so we're flipping the microfacet normal in case it wasn't in the same hemisphere as
            // the view direction
            // Relative_eta as already been flipped above in the code
            microfacet_normal = -microfacet_normal;

        refract_ray(local_view_direction, microfacet_normal, sampled_direction, relative_eta);
    }

    return local_to_world_frame(T, B, surface_normal, sampled_direction);
}

Color RenderKernel::disney_sheen_eval(const RendererMaterial& material, const Vector& view_direction, Vector surface_normal, const Vector& to_light_direction, float& pdf)
{
    Color sheen_color = Color(1.0f - material.sheen_tint) + material.sheen_color * material.sheen_tint;

    float base_color_luminance = material.base_color.luminance();
    Color specular_color = base_color_luminance > 0 ? material.base_color / base_color_luminance : Color(1.0f);

    Vector half_vector = normalize(view_direction + to_light_direction);

    float NoL = abs(dot(surface_normal, to_light_direction));
    pdf = NoL / M_PI;

    // Clamping here because floating point errors can give us a dot > 1 sometimes
    // leading to 1.0f - dot being negative and the BRDF returns a negative color
    float HoL = clamp(0.0f, 1.0f, dot(half_vector, to_light_direction));
    return sheen_color * pow(1.0f - HoL, 5.0f) * NoL;
}

Vector RenderKernel::disney_sheen_sample(const RendererMaterial& material, const Vector& view_direction, Vector surface_normal, Xorshift32Generator& random_number_generator)
{
    float trash_pdf;
    return cosine_weighted_sample(surface_normal, trash_pdf, random_number_generator);
}

Color RenderKernel::disney_eval(const RendererMaterial& material, const Vector& view_direction, const Vector& shading_normal, const Vector& to_light_direction, float& pdf)
{
    pdf = 0.0f;

    Vector T, B;
    build_ONB(shading_normal, T, B);

    Vector local_view_direction = world_to_local_frame(T, B, shading_normal, view_direction);
    Vector local_to_light_direction = world_to_local_frame(T, B, shading_normal, to_light_direction);
    Vector local_half_vector = normalize(local_view_direction + local_to_light_direction);

    Color final_color = Color(0.0f);
    // We're only going to compute the diffuse, metallic, clearcoat and sheen lobes if we're 
    // outside of the object. Said otherwise, only the glass lobe is considered while traveling 
    // inside the object
    bool outside_object = dot(view_direction, shading_normal) > 0;
    float tmp_pdf = 0.0f, tmp_weight = 0.0f;

    // Diffuse
    tmp_weight = (1.0f - material.metallic) * (1.0f - material.specular_transmission);
    final_color += tmp_weight > 0 && outside_object ? tmp_weight * disney_diffuse_eval(material, view_direction, shading_normal, to_light_direction, tmp_pdf) : Color(0.0f);
    pdf += tmp_pdf * tmp_weight;
    tmp_pdf = 0.0f;

    // Metallic
    // Computing a custom fresnel term based on the material specular, specular tint, ... coefficients
    Color metallic_fresnel = disney_metallic_fresnel(material, local_half_vector, local_to_light_direction);
    tmp_weight = (1.0f - material.specular_transmission * (1.0f - material.metallic));
    final_color += tmp_weight > 0 && outside_object ? tmp_weight * disney_metallic_eval(material, view_direction, shading_normal, to_light_direction, metallic_fresnel, tmp_pdf) : Color(0.0f);
    pdf += tmp_pdf * tmp_weight;
    tmp_pdf = 0.0f;

    // Clearcoat
    tmp_weight = 0.25f * material.clearcoat;
    final_color += tmp_weight > 0 && outside_object ? tmp_weight * disney_clearcoat_eval(material, view_direction, shading_normal, to_light_direction, tmp_pdf) : Color(0.0f);
    pdf += tmp_pdf * tmp_weight;
    tmp_pdf = 0.0f;

    // Glass
    tmp_weight = (1.0f - material.metallic) * material.specular_transmission;
    final_color += tmp_weight > 0 ? tmp_weight * disney_glass_eval(material, view_direction, shading_normal, to_light_direction, tmp_pdf) : Color(0.0f);
    pdf += tmp_pdf * tmp_weight;
    tmp_pdf = 0.0f;

    // Sheen
    tmp_weight = (1.0f - material.metallic) * material.sheen;
    Color sheen_color = tmp_weight > 0 && outside_object ? tmp_weight * disney_sheen_eval(material, view_direction, shading_normal, to_light_direction, tmp_pdf) : Color(0.0f);
    final_color += sheen_color;
    pdf += tmp_pdf * tmp_weight;

    return final_color;
}

Color RenderKernel::disney_sample(const RendererMaterial& material, const Vector& view_direction, const Vector& shading_normal, const Vector& geometric_normal, Vector& output_direction, float& pdf, Xorshift32Generator& random_number_generator)
{
    pdf = 0.0f;

    Vector normal = shading_normal;

    float glass_weight = (1.0f - material.metallic) * material.specular_transmission;
    bool outside_object = dot(view_direction, normal) > 0;
    if (glass_weight == 0.0f && !outside_object)
    {
        // If we're not sampling the glass lobe so we're checking
        // whether the view direction is below the upper hemisphere around the shading
        // normal or not. This may be the case mainly due to normal mapping / smooth vertex normals. 
        // See Microfacet-based Normal Mapping for Robust Monte Carlo Path Tracing, Eric Heitz, 2017
        // for some illustrations of the problem and a solution (not implemented here because
        // it requires quite a bit of code and overhead). 
        // 
        // We're flipping the normal instead which is a quick dirty fix solution mentioned
        // in the above mentioned paper.
        // 
        // The Position-free Multiple-bounce Computations for Smith Microfacet BSDFs by 
        // Wang et al. 2022 proposes an alternative position-free solution that even solves
        // the multi-scattering issue of microfacet BRDFs on top of the dark fringes issue we're
        // having here

        normal = reflect_ray(shading_normal, geometric_normal);
        outside_object = true;
    }

    float diffuse_weight = (1.0f - material.metallic) * (1.0f - material.specular_transmission) * outside_object;
    float metal_weight = (1.0f - material.specular_transmission * (1.0f - material.metallic)) * outside_object;
    float clearcoat_weight = 0.25f * material.clearcoat * outside_object;

    float normalize_factor = 1.0f / (diffuse_weight + metal_weight + clearcoat_weight + glass_weight);
    diffuse_weight *= normalize_factor;
    metal_weight *= normalize_factor;
    clearcoat_weight *= normalize_factor;
    glass_weight *= normalize_factor;

    float cdf[4];
    cdf[0] = diffuse_weight;
    cdf[1] = cdf[0] + metal_weight;
    cdf[2] = cdf[1] + clearcoat_weight;
    cdf[3] = cdf[2] + glass_weight;

    float rand_1 = random_number_generator();
    if (rand_1 > cdf[2])
    {
        // We're going to sample the glass lobe

        float dot_shading = dot(view_direction, shading_normal);
        float dot_geometric = dot(view_direction, geometric_normal);
        if (dot_shading * dot_geometric < 0)
        {
            // The view direction is below the surface normal because of normal mapping / smooth normals.
            // We're going to flip the normal for the same reason as explained above to avoid black fringes
            // the reason we're also checking for the dot product with the geometric normal here
            // is because in the case of the glass lobe of the BRDF, we could be legitimately having
            // the dot product between the shading normal and the view direction be negative when we're
            // currently travelling inside the surface. To make sure that we're in the case of the black fringes
            // caused by normal mapping and microfacet BRDFs, we're also checking with the geometric normal.
            // If the view direction isn't below the geometric normal but is below the shading normal, this
            // indicates that we're in the case of the black fringes and we can flip the normal
            // If both dot products are negative, this means that we're travelling inside the surface
            // and we shouldn't flip the normal
            normal = reflect_ray(shading_normal, geometric_normal);
        }
    }

    if (rand_1 < cdf[0])
        output_direction = disney_diffuse_sample(material, view_direction, normal, random_number_generator);
    else if (rand_1 < cdf[1])
        output_direction = disney_metallic_sample(material, view_direction, normal, random_number_generator);
    else if (rand_1 < cdf[2])
        output_direction = disney_clearcoat_sample(material, view_direction, normal, random_number_generator);
    else
        output_direction = disney_glass_sample(material, view_direction, normal, random_number_generator);

    if (dot(output_direction, shading_normal) < 0 && !rand_1 > cdf[2])
        // It can happen that the light direction sampled is below the surface. 
        // We return 0.0 in this case because the glass lobe wasn't sampled
        // so we can't have a bounce direction below the surface
        // 
        // We're also checking that we're not sampling the glass lobe because this
        // is a valid configuration for the glass lobe
        return Color(0.0f);

    return disney_eval(material, view_direction, normal, output_direction, pdf);
}

Color RenderKernel::brdf_dispatcher_eval(const RendererMaterial& material, const Vector& view_direction, const Vector& shading_normal, const Vector& to_light_direction, float& pdf)
{
    pdf = 0.0f;
    if (material.brdf_type == BRDF::Disney)
        return disney_eval(material, view_direction, shading_normal, to_light_direction, pdf);

    return Color(0.0f);
}

Color RenderKernel::brdf_dispatcher_sample(const RendererMaterial& material, const Vector& view_direction, const Vector& shading_normal, const Vector& geometric_normal, Vector& bounce_direction, float& brdf_pdf, Xorshift32Generator& random_number_generator)
{
    return disney_sample(material, view_direction, shading_normal, geometric_normal, bounce_direction, brdf_pdf, random_number_generator);
}

bool RenderKernel::intersect_scene(const Ray& ray, HitInfo& closest_hit_info)
{
    closest_hit_info.t = -1.0f;

    for (int i = 0; i < m_triangle_buffer.size(); i++)
    {
        const Triangle& triangle = m_triangle_buffer[i];

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

inline bool RenderKernel::intersect_scene_bvh(const Ray& ray, HitInfo& closest_hit_info)
{
    closest_hit_info.t = -1.0f;

    m_bvh.intersect(ray, closest_hit_info);
    if (closest_hit_info.t > 0.0f)
    {
        // Computing smooth normal
        int vertex_A_index = m_triangle_indices[closest_hit_info.primitive_index * 3 + 0];
        if (m_normals_present[vertex_A_index])
        {
            // Smooth normal available for the triangle

            int vertex_B_index = m_triangle_indices[closest_hit_info.primitive_index * 3 + 1];
            int vertex_C_index = m_triangle_indices[closest_hit_info.primitive_index * 3 + 2];

            Vector smooth_normal = m_vertex_normals[vertex_B_index] * closest_hit_info.u
                + m_vertex_normals[vertex_C_index] * closest_hit_info.v
                + m_vertex_normals[vertex_A_index] * (1.0f - closest_hit_info.u - closest_hit_info.v);

            closest_hit_info.shading_normal = normalize(smooth_normal);
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

inline bool RenderKernel::INTERSECT_SCENE(const Ray& ray, HitInfo& hit_info)
{
    return intersect_scene_bvh(ray, hit_info);
}

float power_heuristic(float pdf_a, float pdf_b)
{
    float pdf_a_squared = pdf_a * pdf_a;

    return pdf_a_squared / (pdf_a_squared + pdf_b * pdf_b);
}

Color RenderKernel::sample_environment_map_from_direction(const Vector& direction)
{
    float u, v;
    u = 0.5f + std::atan2(direction.z, direction.x) / (2.0f * (float)M_PI);
    v = 0.5f + std::asin(direction.y) / (float)M_PI;

    int x = std::max(std::min((int)(u * m_environment_map.width), m_environment_map.width - 1), 0);
    int y = std::max(std::min((int)(v * m_environment_map.height), m_environment_map.height - 1), 0);

    return m_environment_map[y * m_environment_map.width + x];
}

void RenderKernel::env_map_cdf_search(float value, int& x, int& y)
{
    //First searching a line to sample
    int lower = 0;
    int upper = m_environment_map.height - 1;

    int x_index = m_environment_map.width - 1;
    while (lower < upper)
    {
        int y_index = (lower + upper) / 2;
        int env_map_index = y_index * m_environment_map.width + x_index;

        if (value < m_environment_map.cdf()[env_map_index])
            upper = y_index;
        else
            lower = y_index + 1;
    }
    y = std::max(std::min(lower, m_environment_map.height), 0);

    //Then sampling the line itself
    lower = 0;
    upper = m_environment_map.width - 1;

    int y_index = y;
    while (lower < upper)
    {
        int x_index = (lower + upper) / 2;
        int env_map_index = y_index * m_environment_map.width + x_index;

        if (value < m_environment_map.cdf()[env_map_index])
            upper = x_index;
        else
            lower = x_index + 1;
    }
    x = std::max(std::min(lower, m_environment_map.width), 0);
}

Color RenderKernel::sample_environment_map(const Ray& ray, HitInfo& closest_hit_info, const RendererMaterial& material, Xorshift32Generator& random_number_generator)
{
    if (material.brdf_type == BRDF::SpecularFresnel)
        // No sampling for perfectly specular materials
        return Color(0.0f);

    const std::vector<float>& cdf = m_environment_map.cdf();

    int x, y;
    float env_map_total_sum = cdf[cdf.size() - 1];
    env_map_cdf_search(random_number_generator() * env_map_total_sum, x, y);

    float u = (float)x / m_environment_map.width;
    float v = (float)y / m_environment_map.height;
    float phi = u * 2.0f * M_PI;
    // Clamping to avoid theta = 0 which would imply a skysphere direction straight up
    // which leads to a pdf of infinity since it is a singularity
    float theta = std::max(1.0e-5f, v * (float)M_PI);

    Color env_sample;
    float sin_theta = std::sin(theta);
    float cos_theta = std::cos(theta);

    // Convert to cartesian coordinates
    Vector sampled_direction = Vector(-sin_theta * cos(phi), -cos_theta, -sin_theta * sin(phi));

    float cosine_term = dot(closest_hit_info.shading_normal, sampled_direction);
    if  (cosine_term > 0.0f)
    {
        HitInfo trash;
        if (!INTERSECT_SCENE(Ray(closest_hit_info.inter_point + closest_hit_info.shading_normal * 1.0e-4f, sampled_direction), trash))
        {
            float env_map_pdf = m_environment_map.luminance_of_pixel(x, y) / env_map_total_sum;
            env_map_pdf = (env_map_pdf * m_environment_map.width * m_environment_map.height) / (2.0f * M_PI * M_PI * sin_theta);

            Color env_map_radiance = m_environment_map[y * m_environment_map.width + x];
            float pdf;
            Color brdf = brdf_dispatcher_eval(material, -ray.direction, closest_hit_info.shading_normal, sampled_direction, pdf);

            float mis_weight = power_heuristic(env_map_pdf, pdf);
            env_sample = brdf * cosine_term * mis_weight * env_map_radiance / env_map_pdf;
        }
    }

    float brdf_sample_pdf;
    Vector brdf_sampled_dir;
    Color brdf_imp_sampling = brdf_dispatcher_sample(material, -ray.direction, closest_hit_info.shading_normal, closest_hit_info.geometric_normal, brdf_sampled_dir, brdf_sample_pdf, random_number_generator);

    cosine_term = std::max(dot(closest_hit_info.shading_normal, brdf_sampled_dir), 0.0f);
    Color brdf_sample;
    if (brdf_sample_pdf != 0.0f && cosine_term > 0.0f)
    {
        HitInfo trash;
        if (!INTERSECT_SCENE(Ray(closest_hit_info.inter_point + closest_hit_info.shading_normal * 1.0e-5f, brdf_sampled_dir), trash))
        {
            Color skysphere_color = sample_environment_map_from_direction(brdf_sampled_dir);
            float theta_brdf_dir = std::acos(brdf_sampled_dir.z);
            float sin_theta_bdrf_dir = std::sin(theta_brdf_dir);
            float env_map_pdf = skysphere_color.luminance() / env_map_total_sum;

            env_map_pdf *= m_environment_map.width * m_environment_map.height;
            env_map_pdf /= (2.0f * M_PI * M_PI * sin_theta_bdrf_dir);

            float mis_weight = power_heuristic(brdf_sample_pdf, env_map_pdf);
            brdf_sample = skysphere_color * mis_weight * cosine_term * brdf_imp_sampling / brdf_sample_pdf;
        }
    }

    return brdf_sample + env_sample;
}

Color RenderKernel::sample_light_sources(const Vector& view_direction, const HitInfo& closest_hit_info, const RendererMaterial& material, Xorshift32Generator& random_number_generator)
{
    if (m_emissive_triangle_indices_buffer.size() == 0)
        // No emmisive geometry in the scene to sample
        return Color(0.0f);

    if (material.emission.r != 0.0f || material.emission.g != 0.0f || material.emission.b != 0.0f)
        // We're not sampling direct lighting if we're already on an
        // emissive surface
        return Color(0.0f);

    if (dot(view_direction, closest_hit_info.geometric_normal) < 0.0f)
        // We're not direct sampling if we're inside a surface
        // 
        // Note that we're also taking the geometric normal into account here and not only the 
        // shading normal because we want to make sure we're actually inside a surface and not just
        // inside a black fringe cause by smooth normals with microfacet BRDFs
        // There's a slightly more thorough explanation of what we're doing with the dot products here
        // in the disney brdf sampling method, in the glass lobe part
        return Color(0.0f);

    Color light_source_radiance_mis;
    float light_sample_pdf;
    LightSourceInformation light_source_info;
    Point random_light_point = sample_random_point_on_lights(random_number_generator, light_sample_pdf, light_source_info);

    Point shadow_ray_origin = closest_hit_info.inter_point + closest_hit_info.shading_normal * 1.0e-4f;
    Vector shadow_ray_direction = random_light_point - shadow_ray_origin;
    float distance_to_light = length(shadow_ray_direction);
    Vector shadow_ray_direction_normalized = shadow_ray_direction / distance_to_light;

    Ray shadow_ray(shadow_ray_origin, shadow_ray_direction_normalized);

    float dot_light_source = std::abs(dot(light_source_info.light_source_normal, -shadow_ray.direction));
    if (dot_light_source > 0.0f)
    {
        bool in_shadow = evaluate_shadow_ray(shadow_ray, distance_to_light);

        if (!in_shadow && dot(closest_hit_info.shading_normal, shadow_ray_direction_normalized) > 0)
        {
            const RendererMaterial& emissive_triangle_material = m_materials_buffer[m_materials_indices_buffer[light_source_info.emissive_triangle_index]];

            light_sample_pdf *= distance_to_light * distance_to_light;
            light_sample_pdf /= dot_light_source;

            float pdf;
            Color brdf = brdf_dispatcher_eval(material, view_direction, closest_hit_info.shading_normal, shadow_ray.direction, pdf);
            if (pdf != 0.0f)
            {
                float mis_weight = power_heuristic(light_sample_pdf, pdf);

                Color Li = emissive_triangle_material.emission;
                float cosine_term = std::max(dot(closest_hit_info.shading_normal, shadow_ray.direction), 0.0f);

                light_source_radiance_mis = Li * cosine_term * brdf * mis_weight / light_sample_pdf;
            }
        }
    }


    Color brdf_radiance_mis;

    Vector sampled_brdf_direction;
    float direction_pdf;
    Color brdf = brdf_dispatcher_sample(material, view_direction, closest_hit_info.shading_normal, closest_hit_info.geometric_normal, sampled_brdf_direction, direction_pdf, random_number_generator);
    if (direction_pdf > 0)
    {
        Ray new_ray = Ray(closest_hit_info.inter_point + closest_hit_info.shading_normal * 1.0e-4f, sampled_brdf_direction);
        HitInfo new_ray_hit_info;
        bool inter_found = INTERSECT_SCENE(new_ray, new_ray_hit_info);

        if (inter_found)
        {
            float cos_angle = std::max(dot(new_ray_hit_info.shading_normal, -sampled_brdf_direction), 0.0f);
            if (cos_angle > 0.0f)
            {
                int material_index = m_materials_indices_buffer[new_ray_hit_info.primitive_index];
                RendererMaterial material = m_materials_buffer[material_index];

                Color emission = material.emission;
                if (emission.r > 0 || emission.g > 0 || emission.b > 0)
                {
                    float distance_squared = new_ray_hit_info.t * new_ray_hit_info.t;
                    float light_area = m_triangle_buffer[new_ray_hit_info.primitive_index].area();

                    float light_pdf = distance_squared / (light_area * cos_angle);

                    float mis_weight = power_heuristic(direction_pdf, light_pdf);
                    float cosine_term = std::max(0.0f, dot(closest_hit_info.shading_normal, sampled_brdf_direction));

                    brdf_radiance_mis = brdf * cosine_term * emission * mis_weight / direction_pdf;
                }
            }
        }
    }

    return light_source_radiance_mis + brdf_radiance_mis;
}

inline Point RenderKernel::sample_random_point_on_lights(Xorshift32Generator& random_number_generator, float& pdf, LightSourceInformation& light_info)
{
    light_info.emissive_triangle_index = random_number_generator.random_index(m_emissive_triangle_indices_buffer.size());
    light_info.emissive_triangle_index = m_emissive_triangle_indices_buffer[light_info.emissive_triangle_index];
    Triangle random_emissive_triangle = m_triangle_buffer[light_info.emissive_triangle_index];

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
    light_info.light_source_normal = normal / length_normal; // Normalized
    float triangle_area = length_normal * 0.5f;
    float nb_emissive_triangles = m_emissive_triangle_indices_buffer.size();

    pdf = 1.0f / (nb_emissive_triangles * triangle_area);

    return random_point_on_triangle;
}

bool RenderKernel::evaluate_shadow_ray(const Ray& ray, float t_max)
{
    HitInfo hit_info;
    bool inter_found = INTERSECT_SCENE(ray, hit_info);

    if (inter_found)
    {
        if (hit_info.t + 1.0e-4f < t_max)
            // There is something in between the light and the origin of the ray
            return true;
        else
            return false;
    }

    return false;
}
