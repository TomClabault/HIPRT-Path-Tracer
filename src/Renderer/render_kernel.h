#ifndef RENDER_KERNEL_H
#define RENDER_KERNEL_H

#include "Scene/camera.h"
#include "Image/color.h"
#include "Image/envmap.h"
#include "Image/image.h"
#include "Renderer/bvh.h"
#include "Renderer/renderer_material.h"
#include "Renderer/sphere.h"
#include "Renderer/triangle.h"
#include "Utils/xorshift.h"
#include "Utils/utils.h"

#define USE_BVH 1

struct LightSourceInformation
{
    int emissive_triangle_index = -1;
    Vector light_source_normal;
};

class RenderKernel
{
public:
    RenderKernel(int width, int height,
                    int render_samples, int max_bounces,
                    std::vector<HIPRTColor>& image_buffer,
                    const std::vector<Triangle>& triangle_buffer,
                    const std::vector<RendererMaterial>& materials_buffer,
                    const std::vector<int>& emissive_triangle_indices_buffer,
                    const std::vector<int>& materials_indices_buffer,
                    const std::vector<Sphere>& analytic_spheres_buffer,
                    BVH& bvh,
                    const EnvironmentMap& env_map) : 
        m_framebuffer_width(width), m_framebuffer_height(height),
        m_render_samples(render_samples),
        m_max_bounces(max_bounces),
        m_frame_buffer(image_buffer),
        m_triangle_buffer(triangle_buffer),
        m_materials_buffer(materials_buffer),
        m_emissive_triangle_indices_buffer(emissive_triangle_indices_buffer),
        m_materials_indices_buffer(materials_indices_buffer),
        m_sphere_buffer(analytic_spheres_buffer),
        m_bvh(bvh),
        m_environment_map(env_map) {}

    void set_camera(Camera camera) { m_camera = camera; }

    Ray get_camera_ray(float x, float y) const;

    Vector rotate_vector_around_normal(const Vector& normal, const Vector& random_dir_local_space) const;
    Vector uniform_direction_around_normal(const Vector& normal, float& pdf, xorshift32_generator& random_number_generator) const;
    Vector cosine_weighted_direction_around_normal(const Vector& normal, float& pdf, xorshift32_generator& random_number_generator) const;

    void ray_trace_pixel(int x, int y) const;
    void render();

    HIPRTColor brdf_dispatcher_sample(const RendererMaterial& material, Vector& bounce_direction, const Vector& ray_direction, Vector& surface_normal, float& brdf_pdf, xorshift32_generator& random_number_generator) const;
    HIPRTColor lambertian_brdf(const RendererMaterial& material, const Vector& to_light_direction, const Vector& view_direction, const Vector& surface_normal) const;
    float cook_torrance_brdf_pdf(const RendererMaterial& material, const Vector& view_direction, const Vector& to_light_direction, const Vector& surface_normal) const;
    HIPRTColor cook_torrance_brdf(const RendererMaterial& material, const Vector& to_light_direction, const Vector& view_direction, const Vector& surface_normal) const;
    HIPRTColor cook_torrance_brdf_importance_sample(const RendererMaterial& material, const Vector& view_direction, const Vector& surface_normal, Vector& output_direction, float& pdf, xorshift32_generator& random_number_generator) const;
    HIPRTColor smooth_glass_bsdf(const RendererMaterial& material, Vector& out_bounce_direction, const Vector& ray_direction, Vector& surface_normal, float eta_I, float eta_O, float& pdf, xorshift32_generator& random_generator) const;

    bool intersect_scene(const Ray& ray, HitInfo& closest_hit_info) const;
    bool intersect_scene_bvh(const Ray& ray, HitInfo& closest_hit_info) const;
    bool INTERSECT_SCENE(const Ray& ray, HitInfo& hit_info)const ;

    void env_map_cdf_search(float value, int& x, int& y) const;
    HIPRTColor sample_environment_map_from_direction(const Vector& direction) const;
    HIPRTColor sample_environment_map(const Ray& ray, const HitInfo& closest_hit_info, const RendererMaterial& material, xorshift32_generator& random_number_generator) const;
    HIPRTColor sample_light_sources(const Ray& ray, const HitInfo& closest_hit_info, const RendererMaterial& material, xorshift32_generator& random_number_generator) const;
    Point sample_random_point_on_lights(xorshift32_generator& random_number_generator, float& pdf, LightSourceInformation& light_info) const;
    bool evaluate_shadow_ray(const Ray& ray, float t_max) const;

private:
    int m_framebuffer_width, m_framebuffer_height;

    int m_render_samples; 
    int m_max_bounces;

    std::vector<HIPRTColor>& m_frame_buffer;

    const std::vector<Triangle>& m_triangle_buffer;
    const std::vector<RendererMaterial>& m_materials_buffer;
    const std::vector<int>& m_emissive_triangle_indices_buffer;
    const std::vector<int>& m_materials_indices_buffer;

    const std::vector<Sphere>& m_sphere_buffer;

    const BVH& m_bvh;

    const EnvironmentMap& m_environment_map;

    Camera m_camera;
};

#endif
