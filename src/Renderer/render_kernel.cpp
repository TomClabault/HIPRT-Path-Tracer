#include "HostDeviceCommon/hit_info.h"
#include "render_kernel.h"
#include "triangle.h"

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

void branchlessONB(const Vector& n, Vector& b1, Vector& b2)
{
    float sign = std::copysign(1.0f, n.z);
    const float a = -1.0f / (sign + n.z);
    const float b = n.x * n.y * a;
    b1 = Vector(1.0f + sign * n.x * n.x * a, sign * b, -sign * n.x);
    b2 = Vector(b, sign + n.y * n.y * a, -n.y);
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

float fresnel_dielectric(float cos_theta_i, float eta_i, float eta_t)
{
	// Computing cos_theta_t
	float sinThetaI = std::sqrt(1 - cos_theta_i * cos_theta_i);
	float sin_theta_t = eta_i / eta_t * sinThetaI;

	if (sin_theta_t >= 1.0f) 
	    // Total internal reflection, 0% refraction, all reflection
	    return 1.0f;

    float cos_theta_t = std::sqrt(1.0f - sin_theta_t * sin_theta_t);
    float r_parallel = ((eta_t * cos_theta_i) - (eta_i * cos_theta_t)) / ((eta_t * cos_theta_i) + (eta_i * cos_theta_t));
    float r_perpendicular = ((eta_i * cos_theta_i) - (eta_t * cos_theta_t)) / ((eta_i * cos_theta_i) + (eta_t * cos_theta_t));
    return (r_parallel * r_parallel + r_perpendicular * r_perpendicular) / 2;
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

                    // --------------------------------------------------- //
                    // ----------------- Direct lighting ----------------- //
                    // --------------------------------------------------- //
                    Color light_sample_radiance = sample_light_sources(ray, closest_hit_info, material, random_number_generator);
                    Color env_map_radiance = sample_environment_map(ray, closest_hit_info, material, random_number_generator);

                    // --------------------------------------- //
                    // ---------- Indirect lighting ---------- //
                    // --------------------------------------- //

                    float brdf_pdf;
                    Vector bounce_direction;
                    Color brdf = brdf_dispatcher_sample(material, bounce_direction, ray.direction, closest_hit_info.normal_at_intersection, brdf_pdf, random_number_generator); //TODO relative IOR in the RayData rather than two incident and output ior values
                    
                    if (bounce == 0)
                        sample_color += material.emission;
                    sample_color += (light_sample_radiance + env_map_radiance) * throughput;

                    if ((brdf.r == 0.0f && brdf.g == 0.0f && brdf.b == 0.0f) || brdf_pdf < 1.0e-8f || std::isinf(brdf_pdf))
                    {
                        next_ray_state = RayState::TERMINATED;

                        break;
                    }

                    throughput *= brdf * std::max(0.0f, dot(bounce_direction, closest_hit_info.normal_at_intersection)) / brdf_pdf;

                    //TODO RayData rather than having the normal, ray direction, is inside surface, ... as free variables in the code
                    Point new_ray_origin = closest_hit_info.inter_point + closest_hit_info.normal_at_intersection * 1.0e-4f;
                    ray = Ray(new_ray_origin, bounce_direction);
                    next_ray_state = RayState::BOUNCE;
                }
                else
                    next_ray_state = RayState::MISSED;
            }
            else if (next_ray_state == RayState::MISSED)
            {
                if (bounce == 1 || last_brdf_hit_type == BRDF::SpecularFresnel)
                {
                    //We're only getting the skysphere radiance for the first rays because the
                    //syksphere is importance sampled
                    // We're also getting the skysphere radiance for perfectly specular BRDF since those
                    // are not importance sampled

                    Color skysphere_color = sample_environment_map_from_direction(ray.direction);

                    sample_color += skysphere_color * throughput;
                }

                break;
            }
            else if (next_ray_state == RayState::TERMINATED)
                break;
        }

        final_color += sample_color;
    }

    final_color /= m_render_samples;
    m_frame_buffer[y * m_framebuffer_width + x] += final_color;

    const float gamma = 2.2f;
    const float exposure = 2.0f;
    Color hdrColor = m_frame_buffer[y * m_framebuffer_width + x];

    //Exposure tone mapping
    Color tone_mapped = Color(1.0f, 1.0f, 1.0f) - exp(-hdrColor * exposure);
    // Gamma correction
    Color gamma_corrected = pow(tone_mapped, 1.0f / gamma);

    m_frame_buffer[y * m_framebuffer_width + x] = gamma_corrected;
}

#include <atomic>
#include <omp.h>

#define DEBUG_PIXEL 0
#define DEBUG_PIXEL_X 450
#define DEBUG_PIXEL_Y 399

void RenderKernel::render()
{
    std::atomic<int> lines_completed = 0;

#if DEBUG_PIXEL
    for (int y = m_frame_buffer.height - DEBUG_PIXEL_Y - 1; y < m_frame_buffer.height; y++)
    {
        for (int x = DEBUG_PIXEL_X; x < m_frame_buffer.width; x++)
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

Color RenderKernel::lambertian_brdf(const RendererMaterial& material, const Vector& to_light_direction, const Vector& view_direction, const Vector& surface_normal) const
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

float RenderKernel::cook_torrance_brdf_pdf(const RendererMaterial& material, const Vector& view_direction, const Vector& to_light_direction, const Vector& surface_normal) const
{
    Vector microfacet_normal = normalize(view_direction + to_light_direction);

    float alpha = material.roughness * material.roughness;

    float VoH = std::max(0.0f, dot(view_direction, microfacet_normal));
    float NoH = std::max(0.0f, dot(surface_normal, microfacet_normal));
    float D = GGX_normal_distribution(alpha, NoH);

    return D * NoH / (4.0f * VoH);
}

inline Color RenderKernel::cook_torrance_brdf(const RendererMaterial& material, const Vector& to_light_direction, const Vector& view_direction, const Vector& surface_normal) const
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

Color RenderKernel::cook_torrance_brdf_importance_sample(const RendererMaterial& material, const Vector& view_direction, const Vector& surface_normal, Vector& output_direction, float& pdf, xorshift32_generator& random_number_generator) const
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

Color RenderKernel::smooth_glass_bsdf(const RendererMaterial& material, Vector& out_bounce_direction, const Vector& ray_direction, Vector& surface_normal, float eta_i, float eta_t, float& pdf, xorshift32_generator& random_generator) const
{
    // Clamping here because the dot product can eventually returns values less
    // than -1 or greater than 1 because of precision errors in the vectors
    // (in previous calculations)
    float cos_theta_i = std::min(std::max(-1.0f, dot(surface_normal, -ray_direction)), 1.0f);

    if (cos_theta_i < 0.0f)
    {
        // We're inside the surface, we're going to flip the eta and the normal for
        // the calculations that follow
        // Note that this also flips the normal for the caller of this function
        // since the normal is passed by reference. This is useful since the normal
        // will be used for offsetting the new ray origin for example
        cos_theta_i = -cos_theta_i;
        surface_normal = -surface_normal;
        std::swap(eta_i, eta_t);
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

        Vector refract_direction;
        bool can_refract = refract_ray(-ray_direction, surface_normal, refract_direction, eta_t / eta_i);
        if (!can_refract)
        {
            // Shouldn't happen (?)
            std::cout << "cannot refract" << std::endl;
            std::exit(1);
        }

        out_bounce_direction = refract_direction;
        surface_normal = -surface_normal;
        pdf = 1.0f - fresnel_reflect;

        return Color(1.0f - fresnel_reflect) * material.diffuse / dot(out_bounce_direction, surface_normal);
    }
}

Color RenderKernel::brdf_dispatcher_sample(const RendererMaterial& material, Vector& bounce_direction, const Vector& ray_direction, Vector& surface_normal, float& brdf_pdf, xorshift32_generator& random_number_generator) const
{
    if (material.brdf_type == BRDF::SpecularFresnel)
        return smooth_glass_bsdf(material, bounce_direction, ray_direction, surface_normal, 1.0f, material.ior, brdf_pdf, random_number_generator); //TODO relative IOR in the RayData rather than two incident and output ior values
    else if (material.brdf_type == BRDF::CookTorrance)
        return cook_torrance_brdf_importance_sample(material, -ray_direction, surface_normal, bounce_direction, brdf_pdf, random_number_generator);

    return Color(0.0f);
}

bool RenderKernel::intersect_scene(const Ray& ray, HitInfo& closest_hit_info) const
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

inline bool RenderKernel::intersect_scene_bvh(const Ray& ray, HitInfo& closest_hit_info) const
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

            closest_hit_info.normal_at_intersection = normalize(smooth_normal);
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

inline bool RenderKernel::INTERSECT_SCENE(const Ray& ray, HitInfo& hit_info) const
{
    return intersect_scene_bvh(ray, hit_info);
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

    int x = std::max(std::min((int)(u * m_environment_map.width), m_environment_map.width - 1), 0);
    int y = std::max(std::min((int)(v * m_environment_map.height), m_environment_map.height - 1), 0);

    return m_environment_map[y * m_environment_map.width + x];
}

void RenderKernel::env_map_cdf_search(float value, int& x, int& y) const
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

Color RenderKernel::sample_environment_map(const Ray& ray, const HitInfo& closest_hit_info, const RendererMaterial& material, xorshift32_generator& random_number_generator) const
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

    float cosine_term = dot(closest_hit_info.normal_at_intersection, sampled_direction);
    if  (cosine_term > 0.0f)
    {
        HitInfo trash;
        if (!INTERSECT_SCENE(Ray(closest_hit_info.inter_point + closest_hit_info.normal_at_intersection * 1.0e-4f, sampled_direction), trash))
        {
            float env_map_pdf = m_environment_map.luminance_of_pixel(x, y) / env_map_total_sum;
            env_map_pdf = (env_map_pdf * m_environment_map.width * m_environment_map.height) / (2.0f * M_PI * M_PI * sin_theta);

            Color env_map_radiance = m_environment_map[y * m_environment_map.width + x];
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
            float env_map_pdf = skysphere_color.luminance() / env_map_total_sum;

            env_map_pdf *= m_environment_map.width * m_environment_map.height;
            env_map_pdf /= (2.0f * M_PI * M_PI * sin_theta_bdrf_dir);

            float mis_weight = power_heuristic(brdf_sample_pdf, env_map_pdf);
            brdf_sample = skysphere_color * mis_weight * cosine_term * brdf_imp_sampling / brdf_sample_pdf;
        }
    }

    return brdf_sample + env_sample;
}

Color RenderKernel::sample_light_sources(const Ray& ray, const HitInfo& closest_hit_info, const RendererMaterial& material, xorshift32_generator& random_number_generator) const
{
    if (material.brdf_type == BRDF::SpecularFresnel)
        // No sampling for perfectly specular materials
        return Color(0.0f);

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
                const RendererMaterial& emissive_triangle_material = m_materials_buffer[m_materials_indices_buffer[light_source_info.emissive_triangle_index]];

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
    if (brdf != Color(0.0f, 0.0f, 0.0f))
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
                RendererMaterial material = m_materials_buffer[material_index];

                Color emission = material.emission;
                if (emission.r > 0 || emission.g > 0 || emission.b > 0)
                {
                    float distance_squared = new_ray_hit_info.t * new_ray_hit_info.t;
                    float light_area = m_triangle_buffer[new_ray_hit_info.primitive_index].area();

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

