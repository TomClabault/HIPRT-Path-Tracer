/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef RENDER_KERNEL_H
#define RENDER_KERNEL_H

#include "Scene/camera.h"
#include "Image/envmap.h"
#include "Image/image.h"
#include "Renderer/bvh.h"
#include "Renderer/sphere.h"
#include "Renderer/triangle.h"
#include "HostDeviceCommon/material.h"
#include "HostDeviceCommon/xorshift.h"
#include "Utils/utils.h"

class RenderKernel
{
public:
    RenderKernel(int width, int height,
                    int render_samples, int max_bounces,
                    Image& image_buffer,
                    const std::vector<Triangle>& triangle_buffer,
                    const std::vector<int>& triangle_indices,
                    const std::vector<unsigned char>& normals_present,
                    const std::vector<Vector>& vertex_normals,
                    const std::vector<RendererMaterial>& materials_buffer,
                    const std::vector<int>& emissive_triangle_indices_buffer,
                    const std::vector<int>& materials_indices_buffer,
                    const std::vector<Sphere>& analytic_spheres_buffer,
                    BVH& bvh,
                    const EnvironmentMap& env_map) : 
        m_framebuffer_width(width), m_framebuffer_height(height),
        m_pixels_sample_count(width * height, 0),
        m_pixels_squared_luminance(width * height, 0.0f),
        m_render_samples(render_samples),
        m_max_bounces(max_bounces),
        m_frame_buffer(image_buffer),
        m_triangle_buffer(triangle_buffer),
        m_triangle_indices(triangle_indices),
        m_normals_present(normals_present),
        m_vertex_normals(vertex_normals),
        m_materials_buffer(materials_buffer),
        m_emissive_triangle_indices_buffer(emissive_triangle_indices_buffer),
        m_materials_indices_buffer(materials_indices_buffer),
        m_sphere_buffer(analytic_spheres_buffer),
        m_bvh(bvh),
        m_environment_map(env_map) {}

    void set_camera(Camera camera) { m_camera = camera; }

    Ray get_camera_ray(float x, float y);

    /*
     * Projects the given vector in the frame oriented around the given normal.
     * The given normal represents the forward z axis of the frame the vector
     * is going to be projected in
     */
    Vector uniform_direction_around_normal(const Vector& normal, float& pdf, Xorshift32Generator& random_number_generator);
    Vector cosine_weighted_sample(const Vector& normal, float& pdf, Xorshift32Generator& random_number_generator);
    void cosine_weighted_eval(const Vector& normal, const Vector& direction, float& pdf);

    void debug_set_final_color(int x, int y, ColorRGB final_color);
    void ray_trace_pixel(int x, int y);
    void render();

    ColorRGB brdf_dispatcher_eval(const RendererMaterial& material, const Vector& view_direction, const Vector& shading_normal, const Vector& to_light_direction, float& pdf);
    ColorRGB brdf_dispatcher_sample(const RendererMaterial& material, const Vector& view_direction, const Vector& shading_normal, const Vector& geometric_normal, Vector& bounce_direction, float& brdf_pdf, Xorshift32Generator& random_number_generator);
    ColorRGB lambertian_brdf(const RendererMaterial& material, const Vector& to_light_direction, const Vector& view_direction, const Vector& shading_normal);
    ColorRGB cook_torrance_brdf_eval(const RendererMaterial& material, const Vector& view_direction, const Vector& shading_normal, const Vector& to_light_direction, float& pdf);
    ColorRGB cook_torrance_brdf_sample(const RendererMaterial& material, const Vector& view_direction, const Vector& shading_normal, Vector& output_direction, float& pdf, Xorshift32Generator& random_number_generator);
    ColorRGB smooth_glass_bsdf(const RendererMaterial& material, Vector& out_bounce_direction, const Vector& ray_direction, Vector& shading_normal, float eta_I, float eta_O, float& pdf, Xorshift32Generator& random_generator);
    ColorRGB oren_nayar_eval(const RendererMaterial& material, const Vector& view_direction, const Vector& shading_normal, const Vector& to_light_direction);

    float disney_schlick_weight(float f0, float abs_cos_angle);
    ColorRGB disney_eval(const RendererMaterial& material, const Vector& view_direction, const Vector& shading_normal, const Vector& to_light_direction, float& pdf);
    ColorRGB disney_sample(const RendererMaterial& material, const Vector& view_direction, const Vector& shading_normal, const Vector& geometric_normal, Vector& output_direction, float& pdf, Xorshift32Generator& random_number_generator);
    ColorRGB disney_diffuse_eval(const RendererMaterial& material, const Vector& view_direction, const Vector& shading_normal, const Vector& to_light_direction, float& pdf);
    Vector disney_diffuse_sample(const RendererMaterial& material, const Vector& view_direction, const Vector& shading_normal, Xorshift32Generator& random_number_generator);
    ColorRGB disney_metallic_eval(const RendererMaterial& material, const Vector& view_direction, const Vector& shading_normal, const Vector& to_light_direction, float& pdf);
    ColorRGB disney_metallic_fresnel(const RendererMaterial& material, const Vector& local_half_vector, const Vector& local_to_light_direction);
    ColorRGB disney_metallic_eval(const RendererMaterial& material, const Vector& view_direction, const Vector& surface_normal, const Vector& to_light_direction, ColorRGB F, float& pdf);
    Vector disney_metallic_sample(const RendererMaterial& material, const Vector& view_direction, const Vector& shading_normal, Xorshift32Generator& random_number_generator);
    ColorRGB disney_clearcoat_eval(const RendererMaterial& material, const Vector& view_direction, const Vector& shading_normal, const Vector& to_light_direction, float& pdf);
    Vector disney_clearcoat_sample(const RendererMaterial& material, const Vector& view_direction, const Vector& shading_normal, Xorshift32Generator& random_number_generator);
    ColorRGB disney_glass_eval(const RendererMaterial& material, const Vector& view_direction, Vector shading_normal, const Vector& to_light_direction, float& pdf);
    Vector disney_glass_sample(const RendererMaterial& material, const Vector& view_direction, Vector shading_normal, Xorshift32Generator& random_number_generator);
    ColorRGB disney_sheen_eval(const RendererMaterial& material, const Vector& view_direction, Vector shading_normal, const Vector& to_light_direction, float& pdf);
    Vector disney_sheen_sample(const RendererMaterial& material, const Vector& view_direction, Vector shading_normal, Xorshift32Generator& random_number_generator);

    bool intersect_scene(const Ray& ray, HitInfo& closest_hit_info);
    bool intersect_scene_bvh(const Ray& ray, HitInfo& closest_hit_info);
    bool INTERSECT_SCENE(const Ray& ray, HitInfo& hit_info);

    void env_map_cdf_search(float value, int& x, int& y);
    ColorRGB sample_environment_map_from_direction(const Vector& direction);
    ColorRGB sample_environment_map(const Ray& ray, HitInfo& closest_hit_info, const RendererMaterial& material, Xorshift32Generator& random_number_generator);
    ColorRGB sample_light_sources(const Vector& view_direction, const HitInfo& closest_hit_info, const RendererMaterial& material, Xorshift32Generator& random_number_generator);
    Point sample_random_point_on_lights(Xorshift32Generator& random_number_generator, float& pdf, LightSourceInformation& light_info);
    bool evaluate_shadow_ray(const Ray& ray, float t_max);

private:
    int m_framebuffer_width, m_framebuffer_height;

    int m_render_samples; 
    int m_max_bounces;

    Image& m_frame_buffer;
    const std::vector<int> m_pixels_sample_count;
    const std::vector<float> m_pixels_squared_luminance;

    const std::vector<Triangle>& m_triangle_buffer;
    const std::vector<int>& m_triangle_indices;
    const std::vector<unsigned char>& m_normals_present;
    const std::vector<Vector>& m_vertex_normals;
    const std::vector<RendererMaterial>& m_materials_buffer;
    const std::vector<int>& m_emissive_triangle_indices_buffer;
    const std::vector<int>& m_materials_indices_buffer;

    const std::vector<Sphere>& m_sphere_buffer;

    const BVH& m_bvh;

    const EnvironmentMap& m_environment_map;

    Camera m_camera;
};

#endif
