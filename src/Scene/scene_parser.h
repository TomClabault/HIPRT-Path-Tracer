#ifndef SCENE_PARSER_H
#define SCENE_PARSER_H

#include "Scene/camera.h"
#include "Renderer/renderer_material.h"
#include "Renderer/sphere.h"
#include "Renderer/triangle.h"

#include <vector>

struct Scene
{
    std::vector<Triangle> triangles;
    std::vector<RendererMaterial> materials;

    std::vector<int> emissive_triangle_indices;
    std::vector<int> material_indices;

    bool has_camera = false;
    Camera camera;

    Sphere add_sphere(const Point& center, float radius, const RendererMaterial& material, int primitive_index)
    {
        int material_index = materials.size();

        materials.push_back(material);
        material_indices.push_back(material_index);

        Sphere sphere(center, radius, primitive_index);

        return sphere;
    }
};

class SceneParser
{
public:
    static Scene parse_scene_file(const std::string& filepath);
};

#endif
