/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef SCENE_PARSER_H
#define SCENE_PARSER_H

#include "assimp/Importer.hpp"
#include "assimp/scene.h"
#include "assimp/postprocess.h"

#include "HostDeviceCommon/Material.h"
#include "Scene/Camera.h"
#include "Renderer/Sphere.h"
#include "Renderer/Triangle.h"

#include <vector>

struct Scene
{
    std::vector<RendererMaterial> materials;

    std::vector<int> triangle_indices;
    std::vector<hiprtFloat3> vertices_positions;
    std::vector<unsigned char> normals_present;
    std::vector<hiprtFloat3> vertex_normals;
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

    std::vector<Triangle> get_triangles()
    {
        std::vector<Triangle> triangles;

        for (int i = 0; i < triangle_indices.size(); i += 3)
        {
            triangles.push_back(Triangle(*reinterpret_cast<Point*>(&vertices_positions[triangle_indices[i + 0]]),
                                         *reinterpret_cast<Point*>(&vertices_positions[triangle_indices[i + 1]]),
                                         *reinterpret_cast<Point*>(&vertices_positions[triangle_indices[i + 2]])));
        }

        return triangles;
    }
};

class SceneParser
{
public:
    static RendererMaterial ai_mat_to_renderer_mat(aiMaterial* mesh_material);

    /**
     * Parses the scene file at @filepath and returns a scene appropriate for the renderer.
     * All formats supported by the ASSIMP library are supported by the renderer
     * 
     * If provided, the @frame_aspect_override parameter is meant to override the aspect ratio of the camera
     * of the scene file (if any). This is useful because the renderer uses a default aspect ratio
     * of 16:9 but the camera of the scene file ma not use the same aspect. Without this parameter,
     * this would result in rendering the scene with an aspect different of 16:9 in the defualt 
     * framebuffer of the renderer which is 16:9, resulting in deformations.
     */
    static Scene parse_scene_file(const std::string& filepath, float frame_aspect_override = -1.0f);
};

#endif
