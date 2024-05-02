///*
// * Copyright 2024 Tom Clabault. GNU GPL3 license.
// * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
// */
//
//#ifndef RENDER_KERNEL_H
//#define RENDER_KERNEL_H
//
//#include "Scene/Camera.h"
//#include "Image/Envmap.h"
//#include "Image/Image.h"
//#include "Renderer/BVH.h"
//#include "Renderer/Sphere.h"
//#include "Renderer/Triangle.h"
//#include "HostDeviceCommon/Material.h"
//#include "HostDeviceCommon/Xorshift.h"
//#include "Utils/Utils.h"
//
//class RenderKernel
//{
//public:
//    RenderKernel(int width, int height,
//                    int render_samples, int max_bounces,
//                    Image& image_buffer,
//                    const std::vector<Triangle>& triangle_buffer,
//                    const std::vector<int>& triangle_indices,
//                    const std::vector<unsigned char>& normals_present,
//                    const std::vector<float3>& vertex_normals,
//                    const std::vector<RendererMaterial>& materials_buffer,
//                    const std::vector<int>& emissive_triangle_indices_buffer,
//                    const std::vector<int>& materials_indices_buffer,
//                    const std::vector<Sphere>& analytic_spheres_buffer,
//                    BVH& bvh,
//                    const EnvironmentMap& env_map) : 
//        m_framebuffer_width(width), m_framebuffer_height(height),
//        m_pixels_sample_count(width * height, 0),
//        m_pixels_squared_luminance(width * height, 0.0f),
//        m_render_samples(render_samples),
//        m_max_bounces(max_bounces),
//        m_frame_buffer(image_buffer),
//        m_triangle_buffer(triangle_buffer),
//        m_triangle_indices(triangle_indices),
//        m_normals_present(normals_present),
//        m_vertex_normals(vertex_normals),
//        m_materials_buffer(materials_buffer),
//        m_emissive_triangle_indices_buffer(emissive_triangle_indices_buffer),
//        m_materials_indices_buffer(materials_indices_buffer),
//        m_sphere_buffer(analytic_spheres_buffer),
//        m_bvh(bvh),
//        m_environment_map(env_map) {}
//
//    void set_camera(Camera camera) { m_camera = camera; }
//
//    Ray get_camera_ray(float x, float y);
//
//    void debug_set_final_color(int x, int y, ColorRGB final_color);
//    void ray_trace_pixel(int x, int y);
//    void render();
//
//    ColorRGB brdf_dispatcher_eval(const RendererMaterial& material, const float3& view_direction, const float3& shading_normal, const float3& to_light_direction, float& pdf);
//    ColorRGB brdf_dispatcher_sample(const RendererMaterial& material, const float3& view_direction, const float3& shading_normal, const float3& geometric_normal, float3& bounce_direction, float& brdf_pdf, Xorshift32Generator& random_number_generator);
//    ColorRGB lambertian_brdf(const RendererMaterial& material, const float3& to_light_direction, const float3& view_direction, const float3& shading_normal);
//    ColorRGB cook_torrance_brdf_eval(const RendererMaterial& material, const float3& view_direction, const float3& shading_normal, const float3& to_light_direction, float& pdf);
//    ColorRGB cook_torrance_brdf_sample(const RendererMaterial& material, const float3& view_direction, const float3& shading_normal, float3& output_direction, float& pdf, Xorshift32Generator& random_number_generator);
//    ColorRGB smooth_glass_bsdf(const RendererMaterial& material, float3& out_bounce_direction, const float3& ray_direction, float3& shading_normal, float eta_I, float eta_O, float& pdf, Xorshift32Generator& random_generator);
//    ColorRGB oren_nayar_eval(const RendererMaterial& material, const float3& view_direction, const float3& shading_normal, const float3& to_light_direction);
//
//    float disney_schlick_weight(float f0, float abs_cos_angle);
//    ColorRGB disney_eval(const RendererMaterial& material, const float3& view_direction, const float3& shading_normal, const float3& to_light_direction, float& pdf);
//    ColorRGB disney_sample(const RendererMaterial& material, const float3& view_direction, const float3& shading_normal, const float3& geometric_normal, float3& output_direction, float& pdf, Xorshift32Generator& random_number_generator);
//    ColorRGB disney_diffuse_eval(const RendererMaterial& material, const float3& view_direction, const float3& shading_normal, const float3& to_light_direction, float& pdf);
//    float3 disney_diffuse_sample(const RendererMaterial& material, const float3& view_direction, const float3& shading_normal, Xorshift32Generator& random_number_generator);
//    ColorRGB disney_metallic_eval(const RendererMaterial& material, const float3& view_direction, const float3& shading_normal, const float3& to_light_direction, float& pdf);
//    ColorRGB disney_metallic_fresnel(const RendererMaterial& material, const float3& local_half_vector, const float3& local_to_light_direction);
//    ColorRGB disney_metallic_eval(const RendererMaterial& material, const float3& view_direction, const float3& surface_normal, const float3& to_light_direction, ColorRGB F, float& pdf);
//    float3 disney_metallic_sample(const RendererMaterial& material, const float3& view_direction, const float3& shading_normal, Xorshift32Generator& random_number_generator);
//    ColorRGB disney_clearcoat_eval(const RendererMaterial& material, const float3& view_direction, const float3& shading_normal, const float3& to_light_direction, float& pdf);
//    float3 disney_clearcoat_sample(const RendererMaterial& material, const float3& view_direction, const float3& shading_normal, Xorshift32Generator& random_number_generator);
//    ColorRGB disney_glass_eval(const RendererMaterial& material, const float3& view_direction, float3 shading_normal, const float3& to_light_direction, float& pdf);
//    float3 disney_glass_sample(const RendererMaterial& material, const float3& view_direction, float3 shading_normal, Xorshift32Generator& random_number_generator);
//    ColorRGB disney_sheen_eval(const RendererMaterial& material, const float3& view_direction, float3 shading_normal, const float3& to_light_direction, float& pdf);
//    float3 disney_sheen_sample(const RendererMaterial& material, const float3& view_direction, float3 shading_normal, Xorshift32Generator& random_number_generator);
//
//    bool intersect_scene(const Ray& ray, HitInfo& closest_hit_info);
//    bool intersect_scene_bvh(const Ray& ray, HitInfo& closest_hit_info);
//    bool INTERSECT_SCENE(const Ray& ray, HitInfo& hit_info);
//
//    void env_map_cdf_search(float value, int& x, int& y);
//    ColorRGB sample_environment_map_from_direction(const float3& direction);
//    ColorRGB sample_environment_map(const Ray& ray, HitInfo& closest_hit_info, const RendererMaterial& material, Xorshift32Generator& random_number_generator);
//    ColorRGB sample_light_sources(const float3& view_direction, const HitInfo& closest_hit_info, const RendererMaterial& material, Xorshift32Generator& random_number_generator);
//    Point sample_random_point_on_lights(Xorshift32Generator& random_number_generator, float& pdf, LightSourceInformation& light_info);
//    bool evaluate_shadow_ray(const Ray& ray, float t_max);
//
//private:
//    int m_framebuffer_width, m_framebuffer_height;
//
//    int m_render_samples; 
//    int m_max_bounces;
//
//    Image& m_frame_buffer;
//    const std::vector<int> m_pixels_sample_count;
//    const std::vector<float> m_pixels_squared_luminance;
//
//    const std::vector<Triangle>& m_triangle_buffer;
//    const std::vector<int>& m_triangle_indices;
//    const std::vector<unsigned char>& m_normals_present;
//    const std::vector<float3>& m_vertex_normals;
//    const std::vector<RendererMaterial>& m_materials_buffer;
//    const std::vector<int>& m_emissive_triangle_indices_buffer;
//    const std::vector<int>& m_materials_indices_buffer;
//
//    const std::vector<Sphere>& m_sphere_buffer;
//
//    const BVH& m_bvh;
//
//    const EnvironmentMap& m_environment_map;
//
//    Camera m_camera;
//};
//
//#endif
