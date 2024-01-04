#ifndef RENDER_KERNEL_H
#define RENDER_KERNEL_H

#include "bvh.h"
#include "camera.h"
#include "color.h"
#include "image.h"
#include "simple_material.h"
#include "sphere.h"
#include "triangle.h"
#include "xorshift.h"

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
                 Image& image_buffer,
                 const std::vector<Triangle>& triangle_buffer_accessor,
                 const std::vector<SimpleMaterial>& materials_buffer_accessor,
                 const std::vector<int>& emissive_triangle_indices_buffer_accessor,
                 const std::vector<int>& materials_indices_buffer_accessor,
                 const std::vector<Sphere>& analytic_spheres_buffer,
                 BVH& bvh,
                 const Image& skysphere,
                 const std::vector<float>& env_map_cdf) : 
        m_width(width), m_height(height),
        m_render_samples(render_samples),
        m_max_bounces(max_bounces),
        m_frame_buffer(image_buffer),
        m_triangle_buffer_access(triangle_buffer_accessor),
        m_materials_buffer_access(materials_buffer_accessor),
        m_emissive_triangle_indices_buffer(emissive_triangle_indices_buffer_accessor),
        m_materials_indices_buffer(materials_indices_buffer_accessor),
        m_sphere_buffer(analytic_spheres_buffer),
        m_bvh(bvh),
        m_environment_map(skysphere),
        m_env_map_cdf(env_map_cdf) {}

    void set_camera(Camera camera) { m_camera = camera; }

    Ray get_camera_ray(float x, float y) const;

    Vector rotate_vector_around_normal(const Vector& normal, const Vector& random_dir_local_space) const;
    Vector uniform_direction_around_normal(const Vector& normal, float& pdf, xorshift32_generator& random_number_generator) const;
    Vector cosine_weighted_direction_around_normal(const Vector& normal, float& pdf, xorshift32_generator& random_number_generator) const;

    void ray_trace_pixel(int x, int y) const;
    void render();

    Color lambertian_brdf(const SimpleMaterial& material, const Vector& to_light_direction, const Vector& view_direction, const Vector& surface_normal) const;
    float cook_torrance_brdf_pdf(const SimpleMaterial& material, const Vector& view_direction, const Vector& to_light_direction, const Vector& surface_normal) const;
    Color cook_torrance_brdf(const SimpleMaterial& material, const Vector& to_light_direction, const Vector& view_direction, const Vector& surface_normal) const;
    Vector cook_torrance_brdf_sample_visible_normal(const SimpleMaterial& material, const Vector& wo, const Vector& surface_normal, float& pdf, xorshift32_generator& random_number_generator) const;
    Color cook_torrance_brdf_importance_sample(const SimpleMaterial& material, const Vector& view_direction, const Vector& surface_normal, Vector& output_direction, float& pdf, xorshift32_generator& random_number_generator) const;

    bool intersect_scene(const Ray& ray, HitInfo& closest_hit_info) const;
    bool intersect_scene_bvh(const Ray& ray, HitInfo& closest_hit_info) const;
    bool INTERSECT_SCENE(const Ray& ray, HitInfo& hit_info)const ;

    void env_map_cdf_search(float value, int& x, int& y) const;
    Color sample_environment_map_from_direction(const Vector& direction) const;
    Color sample_environment_map(const Ray& ray, const HitInfo& closest_hit_info, const SimpleMaterial& material, xorshift32_generator& random_number_generator) const;
    Color sample_light_sources(const Ray& ray, const HitInfo& closest_hit_info, const SimpleMaterial& material, xorshift32_generator& random_number_generator) const;
    Point sample_random_point_on_lights(xorshift32_generator& random_number_generator, float& pdf, LightSourceInformation& light_info) const;
    bool evaluate_shadow_ray(const Ray& ray, float t_max) const;

private:
    int m_width, m_height;
    int m_render_samples; 
    int m_max_bounces;

    Image& m_frame_buffer;

    const std::vector<Triangle>& m_triangle_buffer_access;
    const std::vector<SimpleMaterial>& m_materials_buffer_access;
    const std::vector<int>& m_emissive_triangle_indices_buffer;
    const std::vector<int>& m_materials_indices_buffer;

    const std::vector<Sphere>& m_sphere_buffer;

    const BVH& m_bvh;

    const Image& m_environment_map;
    const std::vector<float>& m_env_map_cdf;

    Camera m_camera;
};

#endif

/*
* Flat BVH intersection for the GPU
*/
//inline bool RenderKernel::intersect_scene_bvh(const Ray& ray, HitInfo& closest_hit_info) const
//{
//    closest_hit_info.t = -1.0f;
//
//    FlattenedBVH::Stack stack;
//    stack.push(0);//Pushing the root of the BVH
//
//    std::array<float, BVHConstants::PLANES_COUNT> denoms;
//    std::array<float, BVHConstants::PLANES_COUNT> numers;
//
//    for (int i = 0; i < BVHConstants::PLANES_COUNT; i++)
//    {
//        denoms[i] = dot(BoundingVolume::PLANE_NORMALS[i], ray.direction);
//        numers[i] = dot(BoundingVolume::PLANE_NORMALS[i], Vector(ray.origin));
//    }
//
//    float closest_intersection_distance = -1;
//    while (!stack.empty())
//    {
//        int node_index = stack.pop();
//        const FlattenedBVH::FlattenedNode& node = m_bvh_nodes[node_index];
//
//        if (node.intersect_volume(denoms, numers))
//        {
//            if (node.is_leaf)
//            {
//                for (int i = 0; i < node.nb_triangles; i++)
//                {
//                    int triangle_index = node.triangles_indices[i];
//
//                    HitInfo local_hit_info;
//                    if (m_triangle_buffer_access[triangle_index].intersect(ray, local_hit_info))
//                    {
//                        if (closest_intersection_distance > local_hit_info.t || closest_intersection_distance == -1)
//                        {
//                            closest_intersection_distance = local_hit_info.t;
//                            closest_hit_info = local_hit_info;
//                            closest_hit_info.material_index = m_materials_indices_buffer[triangle_index];
//                        }
//                    }
//                }
//            }
//            else
//            {
//                stack.push(node.children[0]);
//                stack.push(node.children[1]);
//                stack.push(node.children[2]);
//                stack.push(node.children[3]);
//                stack.push(node.children[4]);
//                stack.push(node.children[5]);
//                stack.push(node.children[6]);
//                stack.push(node.children[7]);
//            }
//        }
//    }
//
//    for (int i = 0; i < m_sphere_buffer.size(); i++)
//    {
//        const Sphere& sphere = m_sphere_buffer[i];
//        HitInfo hit_info;
//        if (sphere.intersect(ray, hit_info))
//        {
//            if (hit_info.t < closest_intersection_distance || closest_intersection_distance == -1.0f)
//            {
//                closest_intersection_distance = hit_info.t;
//                closest_hit_info = hit_info;
//            }
//        }
//    }
//
//    return closest_hit_info.t > -1.0f;
//}